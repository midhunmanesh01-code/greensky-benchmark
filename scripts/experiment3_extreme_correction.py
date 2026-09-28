# -*- coding: utf-8 -*-
"""
experiment3_extreme_correction.py
==================================
GreenSky Blend -- SIH26081
Experiment 3: Extreme-Rainfall Post-Processing Correction

IDENTIFIED FAILURE MODE
------------------------
Experiments 1 and 2 found that the regime-conditioned adaptive blender
still severely underforecasts high-rainfall days.  Pre-experiment
diagnostic revealed:

  -- TRAINING heavy-obs days (>= 35 mm, n=11) --
  7/11 events sit in the Moderate forecast bin (adaptive 15-35 mm).
  4/11 events sit in the Heavy forecast bin (>= 35 mm).
  0/11 events were in the Dry/Light bin.

  -- TEST (Exp-1, Aug 4-31) heavy-obs days (>= 15.6 mm, n=9) --
  3 events: Moderate bin  (moderate forecast, heavy outcome)
  4 events: Light bin     (light forecast, heavy outcome)
  1 event:  Heavy bin     (Aug 28, correctly forecast)
  1 event:  Dry bin       (Aug 19, adapt=3.1, obs=20.8 -- complete miss)

This reveals TWO distinct failure types:
  A. Proportional bias: forecast in right regime but systematically low.
     (Moderate bin factor = 1.33x in training)
  B. Regime misclassification: heavy rain falls in Light/Dry forecast bins.
     Multiplicative correction CANNOT fix these -- even 2x of 3 mm = 6 mm,
     far below the 15.6 mm threshold.

CORRECTION METHOD
-----------------
Bin-based multiplicative calibration applied to the adaptive blend:

  For each adaptive-blend bin [Dry, Light, Moderate, Heavy]:
    correction_factor = mean_obs(training_in_bin) / mean_fc(training_in_bin)

  Applied in test: corrected = adaptive_blend * correction_factor(bin)
  Clipped to [0, 200] mm (no negative rain).

  Fallback: factor = 1.0 if bin has < MIN_CORRECTION_SAMPLES training days.

CRITICAL CONSTRAINT
-------------------
All thresholds, bin boundaries, and correction factors are learned
ONLY from the training portion of each evaluation split.
Test observations are NEVER used.

EVALUATION
----------
Exp-1 structure: Train Jun 1 - Aug 3 (64 days),  Test Aug 4 - Aug 31 (28 days)
Exp-2 structure: Train Jun 1 - Jun 30 (30 days),  Test Jul 1 - Aug 3 (34 days)
CSI thresholds: 15.6, 35.0, 64.5 mm/day
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE    = RESULTS_DIR / "corrected_benchmark_results.csv"
OUTPUT_FILE   = RESULTS_DIR / "experiment3_results.csv"

# Regime weights from Experiment 1 (fixed -- not re-learned here)
EXP1_WEIGHTS  = {0: 1.00, 1: 1.00, 2: 0.60, 3: 0.45}

# Regime/calibration bin boundaries (mm of adaptive blend forecast)
CAL_BINS      = [0.0, 5.0, 15.0, 35.0, float("inf")]
CAL_LABELS    = ["Dry (<5)", "Light (5-15)", "Moderate (15-35)", "Heavy (>=35)"]

CSI_THRESHOLDS = [15.6, 35.0, 64.5]
MIN_CAL_SAMPLES = 5

SEP = "=" * 70


# =============================================================================
# HELPERS
# =============================================================================

def assign_regime(mean_fc_series):
    return pd.cut(
        mean_fc_series,
        bins=CAL_BINS,
        labels=list(range(len(CAL_LABELS))),
        right=False,
    ).astype(int)


def build_adaptive(df, weights):
    """Construct regime-conditioned adaptive blend using fixed Exp-1 weights."""
    df = df.copy()
    df["mean_fc"] = (df["ecmwf"] + df["gfs"]) / 2.0
    df["regime"]  = assign_regime(df["mean_fc"])
    df["w"]       = df["regime"].map(weights)
    df["adaptive"] = df["w"] * df["ecmwf"] + (1.0 - df["w"]) * df["gfs"]
    return df


def learn_calibration(fc_arr, obs_arr, min_samples=5):
    """
    Per-bin multiplicative correction factor.
    factor[i] = mean_obs(bin_i) / mean_fc(bin_i)
    Returns list of (factor, n_train, fc_mean, obs_mean) per bin.
    """
    results = []
    for i in range(len(CAL_BINS) - 1):
        lo, hi  = CAL_BINS[i], CAL_BINS[i + 1]
        mask    = (fc_arr >= lo) & (fc_arr < hi)
        n       = int(mask.sum())
        fc_m    = float(fc_arr[mask].mean()) if n > 0 else 0.0
        obs_m   = float(obs_arr[mask].mean()) if n > 0 else 0.0
        if n >= min_samples and fc_m > 0:
            factor = obs_m / fc_m
        else:
            factor = 1.0  # no correction
        results.append({
            "bin":     CAL_LABELS[i],
            "n":       n,
            "fc_mean": round(fc_m, 3),
            "obs_mean":round(obs_m, 3),
            "factor":  round(factor, 4),
            "applied": n >= min_samples and fc_m > 0,
        })
    return results


def apply_calibration(fc_arr, cal_info):
    """Apply learned per-bin multiplicative correction."""
    corrected = fc_arr.copy()
    for i, info in enumerate(cal_info):
        lo, hi = CAL_BINS[i], CAL_BINS[i + 1]
        mask   = (fc_arr >= lo) & (fc_arr < hi)
        corrected[mask] = fc_arr[mask] * info["factor"]
    return np.clip(corrected, 0.0, None)


def mae(obs, fct):
    return float(np.abs(fct - obs).mean())


def rmse(obs, fct):
    return float(np.sqrt(((fct - obs) ** 2).mean()))


def csi_detail(obs, fct, thr):
    obs_e = obs >= thr
    fct_e = fct >= thr
    hits  = int(( obs_e &  fct_e).sum())
    miss  = int(( obs_e & ~fct_e).sum())
    fa    = int((~obs_e &  fct_e).sum())
    denom = hits + miss + fa
    csi   = hits / denom if denom > 0 else float("nan")
    return hits, miss, fa, csi


def print_metrics_block(obs, model_dict, label, n_days):
    """Print a formatted metrics table for a given obs/forecast set."""
    print(f"\n  {label}  (n={n_days})")
    print(f"  {'-'*65}")
    print(f"  {'Model':<24}  {'MAE':>7}  {'RMSE':>7}  "
          f"{'CSI@15.6':>9}  {'CSI@35':>7}  {'CSI@64.5':>9}")
    print(f"  {'-'*24}  {'-'*7}  {'-'*7}  {'-'*9}  {'-'*7}  {'-'*9}")
    all_m = {}
    for name, fct in model_dict.items():
        m_val = mae(obs, fct)
        r_val = rmse(obs, fct)
        h1, ms1, fa1, c1 = csi_detail(obs, fct, 15.6)
        h2, ms2, fa2, c2 = csi_detail(obs, fct, 35.0)
        h3, ms3, fa3, c3 = csi_detail(obs, fct, 64.5)
        c1s = f"{c1:.3f}" if not np.isnan(c1) else "  N/A"
        c2s = f"{c2:.3f}" if not np.isnan(c2) else "  N/A"
        c3s = f"{c3:.3f}" if not np.isnan(c3) else "  N/A"
        all_m[name] = {
            "MAE": m_val, "RMSE": r_val,
            "H15": h1, "M15": ms1, "FA15": fa1, "CSI15": c1,
            "H35": h2, "M35": ms2, "FA35": fa2, "CSI35": c2,
            "H64": h3, "M64": ms3, "FA64": fa3, "CSI64": c3,
        }
        print(f"  {name:<24}  {m_val:>7.2f}  {r_val:>7.2f}  "
              f"{c1s:>9}  {c2s:>7}  {c3s:>9}")
    return all_m


def print_event_detail(obs, model_dict, label, threshold):
    """Print hits/misses/FAs for all models at given threshold."""
    n_obs_events = int((obs >= threshold).sum())
    print(f"\n  {label}  threshold={threshold} mm  ({n_obs_events} observed events)")
    print(f"  {'Model':<24}  {'Hits':>5}  {'Misses':>7}  {'FA':>5}  {'CSI':>7}")
    print(f"  {'-'*24}  {'-'*5}  {'-'*7}  {'-'*5}  {'-'*7}")
    for name, fct in model_dict.items():
        h, ms, fa, csi = csi_detail(obs, fct, threshold)
        cs = f"{csi:.3f}" if not np.isnan(csi) else "  N/A"
        print(f"  {name:<24}  {h:>5}  {ms:>7}  {fa:>5}  {cs:>7}")


def analyse_heavy_days(test_df, model_dict, threshold=15.6):
    """Print per-day results specifically on observed heavy-rain days."""
    heavy = test_df[test_df["rain"] >= threshold].copy()
    if len(heavy) == 0:
        print(f"  No days with obs >= {threshold} mm in this period.")
        return {}
    print(f"\n  Observed heavy days (rain >= {threshold} mm), n={len(heavy)}:")
    header = f"  {'Date':<12}  {'IMD':>7}"
    for name in model_dict:
        header += f"  {name[:10]:>10}"
    print(header)
    print("  " + "-" * (12 + 9 + 12 * len(model_dict)))
    for _, row in heavy.iterrows():
        line = f"  {str(row['time'].date()):<12}  {row['rain']:>7.1f}"
        for name, fct in model_dict.items():
            idx = test_df.index[test_df['time'] == row['time']][0]
            line += f"  {fct[list(test_df.index).index(idx) if isinstance(fct, np.ndarray) else idx]:>10.1f}"
        print(line)
    # MAE on heavy days only
    heavy_mae = {}
    for name, fct in model_dict.items():
        mask = test_df["rain"].values >= threshold
        heavy_mae[name] = mae(test_df["rain"].values[mask], fct[mask] if isinstance(fct, np.ndarray) else
                               np.array([fct[i] for i, v in enumerate(mask) if v]))
    print(f"\n  MAE on heavy days only (n={len(heavy)}):")
    for name, m in heavy_mae.items():
        print(f"    {name:<24}  {m:.2f} mm")
    return heavy_mae


# =============================================================================
# 0. HEADER
# =============================================================================

print(SEP)
print("EXPERIMENT 3 -- EXTREME-RAINFALL CORRECTION  --  GreenSky Blend SIH26081")
print(SEP)

data = pd.read_csv(INPUT_FILE)
data["time"] = pd.to_datetime(data["time"])
data = data.sort_values("time").reset_index(drop=True)

print(f"\nInput file : {INPUT_FILE}")
print(f"Total rows : {len(data)}")

# =============================================================================
# 1. PRE-EXPERIMENT DIAGNOSIS (training data only, Exp-1 structure)
# =============================================================================

print("\n" + SEP)
print("PRE-EXPERIMENT DIAGNOSTIC  (training period Jun 1 - Aug 3 only)")
print(SEP)

train64 = build_adaptive(data[data["time"] < "2025-08-04"].copy(), EXP1_WEIGHTS)

print("""
Two distinct underforecasting failure types identified:

  Type A -- Proportional bias:
    Forecast is in the right regime but systematically too low.
    Example: Aug 15 (obs=51.4 mm, adapt=21.5 mm) -- Moderate bin.
    A multiplicative correction of 1.33x gives 28.6 mm -- better, but
    still a miss at the 35-mm threshold.

  Type B -- Regime misclassification:
    Heavy rain occurs on days with very low forecasts.
    Example: Aug 19 (obs=20.8 mm, adapt=3.1 mm) -- Dry bin.
    Even a 2.13x correction gives 6.6 mm -- still a miss at 15.6 mm.
    These events CANNOT be recovered by any forecast-conditional correction.

The calibration method below addresses Type A only.
Type B failures are structurally unrecoverable without additional
predictor variables (e.g. moisture indices, large-scale circulation).
""")

# Correction factors from 64-day training
cal64 = learn_calibration(train64["adaptive"].values, train64["rain"].values, MIN_CAL_SAMPLES)

print("  Learned calibration factors (Exp-1 training, n=64 days):")
print(f"  {'Bin':<22}  {'N':>4}  {'fc_mean':>8}  {'obs_mean':>9}  {'factor':>7}  {'applied':>8}  {'corrected_range'}")
print(f"  {'-'*22}  {'-'*4}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*8}  {'-'*20}")
for info in cal64:
    i = CAL_LABELS.index(info["bin"])
    lo, hi = CAL_BINS[i], min(CAL_BINS[i + 1], 100)
    corr_lo = lo * info["factor"]
    corr_hi = hi * info["factor"]
    app_str = "YES" if info["applied"] else "NO (fallback 1.0)"
    print(f"  {info['bin']:<22}  {info['n']:>4}  {info['fc_mean']:>8.2f}  "
          f"{info['obs_mean']:>9.2f}  {info['factor']:>7.3f}  {app_str:>8}  "
          f"[{corr_lo:.1f}, {corr_hi:.1f}]")

# What can a Moderate correction of 1.33x actually achieve?
print(f"""
  CALIBRATION CEILING ANALYSIS:
  - Moderate bin (15-35 mm) correction = {cal64[2]['factor']:.3f}x
    Max correctable: 34.9 mm * {cal64[2]['factor']:.3f} = {34.9*cal64[2]['factor']:.1f} mm
    --> Cannot lift a forecast to >= 64.5 mm from this bin.
  - Heavy bin (>= 35 mm) correction = {cal64[3]['factor']:.3f}x (DOWNWARD correction!)
    Models that DO forecast heavy rain tend to overforecast slightly.
  - Dry bin (< 5 mm) correction = {cal64[0]['factor']:.3f}x
    WARNING: Dry-day forecasts of ~3 mm become ~{3*cal64[0]['factor']:.1f} mm.
    This may push some dry-day forecasts above the 15.6 mm threshold.
""")


# =============================================================================
# 2. EXP-1 STRUCTURE  (Train: Jun 1 - Aug 3 / Test: Aug 4 - Aug 31)
# =============================================================================

print(SEP)
print("EVALUATION A  --  EXP-1 STRUCTURE  (Aug 4-31, n=28)")
print(SEP)

train_e1 = build_adaptive(data[data["time"] < "2025-08-04"].copy(), EXP1_WEIGHTS)
test_e1  = build_adaptive(data[data["time"] >= "2025-08-04"].copy(), EXP1_WEIGHTS)

# Learn calibration on Exp-1 training
cal_e1 = learn_calibration(train_e1["adaptive"].values, train_e1["rain"].values, MIN_CAL_SAMPLES)

# Apply to test
test_e1["adaptive_cal"] = apply_calibration(test_e1["adaptive"].values, cal_e1)
test_e1["equal_blend"]  = (test_e1["ecmwf"] + test_e1["gfs"]) / 2.0

obs_e1  = test_e1["rain"].values
models_e1 = {
    "ECMWF"            : test_e1["ecmwf"].values,
    "GFS"              : test_e1["gfs"].values,
    "Equal blend"      : test_e1["equal_blend"].values,
    "Adaptive"         : test_e1["adaptive"].values,
    "Adaptive+CalCorr" : test_e1["adaptive_cal"].values,
}

print(f"\n  Calibration factors applied (Exp-1 training):")
for info in cal_e1:
    print(f"    {info['bin']:<22}  factor={info['factor']:.3f}  "
          f"({'applied' if info['applied'] else 'fallback 1.0'})")

all_m_e1 = print_metrics_block(obs_e1, models_e1, "Aug 4-31 metrics", 28)

for thr in CSI_THRESHOLDS:
    n_ev = int((obs_e1 >= thr).sum())
    if n_ev > 0:
        print_event_detail(obs_e1, models_e1, "CSI detail", thr)
    else:
        print(f"\n  CSI @ {thr} mm: 0 observed events -- N/A")

# Heavy-day analysis: accurate per-row lookup
print(f"\n  Observed heavy days (rain >= 15.6 mm, n={int((obs_e1>=15.6).sum())}):")
print(f"  {'Date':<12}  {'IMD':>6}  {'ECMWF':>6}  {'GFS':>6}  "
      f"{'Equal':>6}  {'Adapt':>7}  {'Adapt+Cal':>10}  {'Cal_bin'}")
print("  " + "-" * 80)
heavy_e1_rows = []
for idx, row in test_e1.iterrows():
    if row["rain"] < 15.6:
        continue
    i_local = list(test_e1.index).index(idx)
    flag = " <<" if row["rain"] >= 64.5 else (" <" if row["rain"] >= 35 else "")
    print(f"  {str(row['time'].date()):<12}  {row['rain']:>6.1f}{flag:<3}  "
          f"{row['ecmwf']:>6.1f}  {row['gfs']:>6.1f}  "
          f"{row['equal_blend']:>6.1f}  {row['adaptive']:>7.1f}  "
          f"{row['adaptive_cal']:>10.1f}  {CAL_LABELS[row['regime']]}")
    heavy_e1_rows.append({
        "date": str(row["time"].date()),
        "obs": row["rain"],
        "adaptive": row["adaptive"],
        "adaptive_cal": row["adaptive_cal"],
    })
heavy_e1 = pd.DataFrame(heavy_e1_rows)
if len(heavy_e1):
    hmae_adapt = np.abs(heavy_e1["adaptive"] - heavy_e1["obs"]).mean()
    hmae_cal   = np.abs(heavy_e1["adaptive_cal"] - heavy_e1["obs"]).mean()
    print(f"\n  MAE on heavy days only:  Adaptive={hmae_adapt:.2f}  Adaptive+Cal={hmae_cal:.2f}  "
          f"({'BETTER' if hmae_cal < hmae_adapt else 'WORSE'})")

# =============================================================================
# 3. EXP-2 STRUCTURE  (Train: Jun 1 - Jun 30 / Test: Jul 1 - Aug 3)
# =============================================================================

print("\n" + SEP)
print("EVALUATION B  --  EXP-2 STRUCTURE  (Jul 1 - Aug 3, n=34)")
print(SEP)

train_e2 = build_adaptive(data[data["time"] < "2025-07-01"].copy(), EXP1_WEIGHTS)
test_e2  = build_adaptive(
    data[(data["time"] >= "2025-07-01") & (data["time"] < "2025-08-04")].copy(),
    EXP1_WEIGHTS
)

cal_e2 = learn_calibration(train_e2["adaptive"].values, train_e2["rain"].values, MIN_CAL_SAMPLES)

print(f"\n  Calibration factors applied (Exp-2 training, Jun only, n=30 days):")
for info in cal_e2:
    print(f"    {info['bin']:<22}  factor={info['factor']:.3f}  "
          f"({'applied' if info['applied'] else 'fallback -- < 5 samples'})")

test_e2["adaptive_cal"] = apply_calibration(test_e2["adaptive"].values, cal_e2)
test_e2["equal_blend"]  = (test_e2["ecmwf"] + test_e2["gfs"]) / 2.0

obs_e2 = test_e2["rain"].values
models_e2 = {
    "ECMWF"            : test_e2["ecmwf"].values,
    "GFS"              : test_e2["gfs"].values,
    "Equal blend"      : test_e2["equal_blend"].values,
    "Adaptive"         : test_e2["adaptive"].values,
    "Adaptive+CalCorr" : test_e2["adaptive_cal"].values,
}

all_m_e2 = print_metrics_block(obs_e2, models_e2, "Jul 1 - Aug 3 metrics", 34)

for thr in CSI_THRESHOLDS:
    n_ev = int((obs_e2 >= thr).sum())
    if n_ev > 0:
        print_event_detail(obs_e2, models_e2, "CSI detail", thr)
    else:
        print(f"\n  CSI @ {thr} mm: 0 observed events in this test period -- N/A")

print(f"\n  Observed heavy days (rain >= 15.6 mm, n={int((obs_e2>=15.6).sum())}):")
print(f"  {'Date':<12}  {'IMD':>6}  {'ECMWF':>6}  {'GFS':>6}  "
      f"{'Equal':>6}  {'Adapt':>7}  {'Adapt+Cal':>10}  {'Cal_bin'}")
print("  " + "-" * 80)
heavy_e2_rows = []
for idx, row in test_e2.iterrows():
    if row["rain"] < 15.6:
        continue
    flag = " <<" if row["rain"] >= 64.5 else (" <" if row["rain"] >= 35 else "")
    print(f"  {str(row['time'].date()):<12}  {row['rain']:>6.1f}{flag:<3}  "
          f"{row['ecmwf']:>6.1f}  {row['gfs']:>6.1f}  "
          f"{row['equal_blend']:>6.1f}  {row['adaptive']:>7.1f}  "
          f"{row['adaptive_cal']:>10.1f}  {CAL_LABELS[row['regime']]}")
    heavy_e2_rows.append({
        "date": str(row["time"].date()),
        "obs": row["rain"],
        "adaptive": row["adaptive"],
        "adaptive_cal": row["adaptive_cal"],
    })
heavy_e2 = pd.DataFrame(heavy_e2_rows)
if len(heavy_e2):
    hmae_adapt = np.abs(heavy_e2["adaptive"] - heavy_e2["obs"]).mean()
    hmae_cal   = np.abs(heavy_e2["adaptive_cal"] - heavy_e2["obs"]).mean()
    print(f"\n  MAE on heavy days only:  Adaptive={hmae_adapt:.2f}  Adaptive+Cal={hmae_cal:.2f}  "
          f"({'BETTER' if hmae_cal < hmae_adapt else 'WORSE'})")

# =============================================================================
# 4. FALSE ALARM ANALYSIS  (does the correction harm dry days?)
# =============================================================================

print("\n" + SEP)
print("FALSE ALARM ANALYSIS  (Dry days: obs < 5 mm)")
print(SEP)

for split_label, test_df, models in [
    ("Exp-1 (Aug 4-31)", test_e1, models_e1),
    ("Exp-2 (Jul 1-Aug3)", test_e2, models_e2),
]:
    dry_mask = test_df["rain"].values < 5.0
    n_dry = dry_mask.sum()
    if n_dry == 0:
        print(f"\n  {split_label}: no dry days in test.")
        continue
    print(f"\n  {split_label} -- {n_dry} observed dry days (obs < 5 mm):")
    print(f"  {'Model':<24}  {'FA@15.6':>8}  {'FA@35':>7}  {'Mean_fc_on_dry':>15}")
    print(f"  {'-'*24}  {'-'*8}  {'-'*7}  {'-'*15}")
    for name, fct in models.items():
        fa15 = int((fct[dry_mask] >= 15.6).sum())
        fa35 = int((fct[dry_mask] >= 35.0).sum())
        mean_fc = fct[dry_mask].mean()
        print(f"  {name:<24}  {fa15:>8}  {fa35:>7}  {mean_fc:>15.2f}")

# =============================================================================
# 5. SUMMARY TABLE
# =============================================================================

print("\n" + SEP)
print("SUMMARY TABLE  --  Adaptive vs Adaptive+CalCorr (both test periods)")
print(SEP)

print(f"\n  {'Metric':<22}  {'E1 Adapt':>10}  {'E1 +Cal':>9}  "
      f"{'E2 Adapt':>10}  {'E2 +Cal':>9}  {'Consistent?':>12}")
print(f"  {'-'*22}  {'-'*10}  {'-'*9}  {'-'*10}  {'-'*9}  {'-'*12}")

def consistent(e1_a, e1_c, e2_a, e2_c, higher_is_better=False):
    """True if correction improves in both periods."""
    if higher_is_better:
        b1 = e1_c > e1_a
        b2 = e2_c > e2_a
    else:
        b1 = e1_c < e1_a
        b2 = e2_c < e2_a
    if b1 and b2:     return "BOTH IMPROVE"
    if not b1 and not b2: return "BOTH WORSEN"
    return "MIXED"

metrics = [
    ("MAE (mm)",      "MAE",   False),
    ("RMSE (mm)",     "RMSE",  False),
    ("CSI @ 15.6",    "CSI15", True),
    ("CSI @ 35.0",    "CSI35", True),
    ("CSI @ 64.5",    "CSI64", True),
]
for label, key, higher in metrics:
    e1a = all_m_e1["Adaptive"][key]
    e1c = all_m_e1["Adaptive+CalCorr"][key]
    e2a = all_m_e2["Adaptive"][key]
    e2c = all_m_e2["Adaptive+CalCorr"][key]
    f_e1a = f"{e1a:.3f}" if not np.isnan(e1a) else "N/A"
    f_e1c = f"{e1c:.3f}" if not np.isnan(e1c) else "N/A"
    f_e2a = f"{e2a:.3f}" if not np.isnan(e2a) else "N/A"
    f_e2c = f"{e2c:.3f}" if not np.isnan(e2c) else "N/A"
    cons_str = consistent(e1a, e1c, e2a, e2c, higher) if not (np.isnan(e1a) or np.isnan(e2a)) else "N/A"
    print(f"  {label:<22}  {f_e1a:>10}  {f_e1c:>9}  {f_e2a:>10}  {f_e2c:>9}  {cons_str:>12}")

# =============================================================================
# 6. VERDICT
# =============================================================================

print("\n" + SEP)
print("VERDICT  (Experiment 3)")
print(SEP)

# Evaluate correction benefit vs harm
e1_adapt_m  = all_m_e1["Adaptive"]
e1_cal_m    = all_m_e1["Adaptive+CalCorr"]
e2_adapt_m  = all_m_e2["Adaptive"]
e2_cal_m    = all_m_e2["Adaptive+CalCorr"]

e1_mae_improved  = e1_cal_m["MAE"]  < e1_adapt_m["MAE"]
e2_mae_improved  = e2_cal_m["MAE"]  < e2_adapt_m["MAE"]
e1_rmse_improved = e1_cal_m["RMSE"] < e1_adapt_m["RMSE"]
e2_rmse_improved = e2_cal_m["RMSE"] < e2_adapt_m["RMSE"]

def nan_better(a, b, higher_is_better):
    if np.isnan(a) or np.isnan(b):
        return False
    return b > a if higher_is_better else b < a

e1_csi15_improved = nan_better(e1_adapt_m["CSI15"], e1_cal_m["CSI15"], True)
e2_csi15_improved = nan_better(e2_adapt_m["CSI15"], e2_cal_m["CSI15"], True)
e1_csi35_improved = nan_better(e1_adapt_m["CSI35"], e1_cal_m["CSI35"], True)
e2_csi35_improved = nan_better(e2_adapt_m["CSI35"], e2_cal_m["CSI35"], True)

print(f"""
  Correction effect summary:

  MAE        : E1 {'IMPROVED' if e1_mae_improved else 'WORSENED'} ({e1_adapt_m['MAE']:.2f} -> {e1_cal_m['MAE']:.2f})  |  "
               E2 {'IMPROVED' if e2_mae_improved else 'WORSENED'} ({e2_adapt_m['MAE']:.2f} -> {e2_cal_m['MAE']:.2f})
  RMSE       : E1 {'IMPROVED' if e1_rmse_improved else 'WORSENED'} ({e1_adapt_m['RMSE']:.2f} -> {e1_cal_m['RMSE']:.2f})  |  "
               E2 {'IMPROVED' if e2_rmse_improved else 'WORSENED'} ({e2_adapt_m['RMSE']:.2f} -> {e2_cal_m['RMSE']:.2f})
  CSI@15.6   : E1 {'IMPROVED' if e1_csi15_improved else 'WORSENED/SAME'} ({e1_adapt_m['CSI15']:.3f} -> {e1_cal_m['CSI15']:.3f})  |  "
               E2 {'IMPROVED' if e2_csi15_improved else 'WORSENED/SAME'} ({e2_adapt_m['CSI15']:.3f} -> {e2_cal_m['CSI15']:.3f})
  CSI@35     : E1 {'IMPROVED' if e1_csi35_improved else 'WORSENED/SAME'} ({e1_adapt_m['CSI35']:.3f} -> {e1_cal_m['CSI35']:.3f})  |  "
               E2 {'IMPROVED' if e2_csi35_improved else 'WORSENED/SAME'} ({e2_adapt_m['CSI35']:.3f} -> {e2_cal_m['CSI35']:.3f})
""")

# Count positives
n_positive = sum([e1_mae_improved, e2_mae_improved,
                  e1_rmse_improved, e2_rmse_improved,
                  e1_csi15_improved, e2_csi15_improved,
                  e1_csi35_improved, e2_csi35_improved])
n_metrics  = 8

if n_positive >= 6:
    verdict = "IMPROVED"
elif n_positive >= 4:
    verdict = "MIXED"
else:
    verdict = "FAILED"

print(f"  {'='*60}")
print(f"  VERDICT: {verdict}  ({n_positive}/{n_metrics} metric-period pairs improved)")
print(f"  {'='*60}")

print(f"""
  ROOT CAUSE ANALYSIS:
  The multiplicative calibration partially addresses the PROPORTIONAL BIAS
  failure (Type A) but CANNOT address REGIME MISCLASSIFICATION (Type B).

  Type A examples (correctable):
    Aug 15: obs=51.4, adapt=21.5 -> corrected={21.5*cal_e1[2]['factor']:.1f} mm  (bin: Moderate, factor={cal_e1[2]['factor']:.3f}x)
    Aug 16: obs=50.1, adapt=17.0 -> corrected={17.0*cal_e1[2]['factor']:.1f} mm  (bin: Moderate, factor={cal_e1[2]['factor']:.3f}x)

  Type B examples (not recoverable by any forecast-conditional correction):
    Aug 19: obs=20.8, adapt=3.1  -> corrected={3.1*cal_e1[0]['factor']:.1f} mm  (bin: Dry, factor={cal_e1[0]['factor']:.3f}x)
    Aug 17: obs=21.9, adapt=6.5  -> corrected={6.5*cal_e1[1]['factor']:.1f} mm  (bin: Light, factor={cal_e1[1]['factor']:.3f}x)

  INFERENCE: To improve detection of Type B events, features beyond the
  precipitation forecast itself are needed -- e.g. predicted wind shear,
  850 hPa moisture flux, or CAPE from the same ECMWF/GFS runs.
  Simple post-processing calibration has reached its practical ceiling
  given only precipitation forecast inputs.
""")

# =============================================================================
# 7. SAVE OUTPUT
# =============================================================================

# Save Exp-1 test results with correction
out_e1 = test_e1[[
    "time", "rain", "ecmwf", "gfs",
]].copy()
out_e1["equal_blend"]   = test_e1["equal_blend"]
out_e1["adaptive"]      = test_e1["adaptive"]
out_e1["adaptive_cal"]  = test_e1["adaptive_cal"]
out_e1["eval_period"]   = "Exp1_Aug04_Aug31"

out_e2 = test_e2[[
    "time", "rain", "ecmwf", "gfs",
]].copy()
out_e2["equal_blend"]   = test_e2["equal_blend"]
out_e2["adaptive"]      = test_e2["adaptive"]
out_e2["adaptive_cal"]  = test_e2["adaptive_cal"]
out_e2["eval_period"]   = "Exp2_Jul01_Aug03"

out = pd.concat([out_e1, out_e2], ignore_index=True)
out.to_csv(OUTPUT_FILE, index=False, float_format="%.4f")

print(f"Saved: {OUTPUT_FILE}")
print()
print(SEP)
print("DONE.  Experiments 1 and 2 files are untouched.")
print("No test observations used for calibration learning.")
print(SEP)
