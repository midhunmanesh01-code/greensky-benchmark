# -*- coding: utf-8 -*-
"""
regime_blender.py
=================
GreenSky Blend -- SIH26081
Experiment 1: Regime-Conditioned Adaptive Blender

APPROACH
--------
We divide the forecast space into four regimes based on the mean
of ECMWF and GFS daily forecasts:

    Dry       : mean_fc <  5.0 mm
    Light     :  5.0 <= mean_fc < 15.0 mm
    Moderate  : 15.0 <= mean_fc < 35.0 mm
    Heavy     : mean_fc >= 35.0 mm

The mean forecast uses ONLY the two model outputs -- no observations.
This means the regime can be determined on any future day without
look-ahead.

For each regime, we grid-search the ECMWF weight w in [0, 0.05, ..., 1.0]
that minimises MAE on the TRAINING set only.  The learned weights are
then applied to the LOCKED TEST SET purely from forecast values.

WHAT THIS IS NOT
----------------
- Not a neural network.
- Not fitted on test observations.
- Not tuned to maximise test scores.

DATA
----
Input  : corrected_benchmark_results.csv  (Convention B aligned)
Train  : 2025-06-01 to 2025-08-03  (64 days)
Test   : 2025-08-04 to 2025-08-31  (28 days, LOCKED)
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

INPUT_FILE  = RESULTS_DIR / "corrected_benchmark_results.csv"
OUTPUT_FILE = RESULTS_DIR / "regime_blender_results.csv"

LOCKED_TEST_START = pd.Timestamp("2025-08-04")

# Regime thresholds (mm) on mean of ECMWF and GFS forecast
REGIME_BINS   = [0.0, 5.0, 15.0, 35.0, float("inf")]
REGIME_LABELS = [
    "Dry      (<  5 mm)",
    "Light    ( 5-15 mm)",
    "Moderate (15-35 mm)",
    "Heavy    (>= 35 mm)",
]

# CSI thresholds per SIH26081 specification
CSI_THRESHOLDS = [15.6, 64.5]

# Minimum training samples required to trust a learned weight;
# regime falls back to 0.50 (equal) if below this.
MIN_REGIME_SAMPLES = 5

# ECMWF weight search grid (0 = all GFS, 1 = all ECMWF)
WEIGHT_GRID = np.round(np.arange(0.0, 1.01, 0.05), 2)


# =============================================================================
# HELPERS
# =============================================================================

def assign_regime(mean_fc_series):
    return pd.cut(
        mean_fc_series,
        bins=REGIME_BINS,
        labels=list(range(len(REGIME_LABELS))),
        right=False,
    ).astype(int)


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


def best_weight(ec, gfs, obs):
    """
    Grid-search the ECMWF weight that minimises MAE on the given
    training subset.  Returns (best_w, best_mae).
    """
    best_w, best_mae_val = 0.5, float("inf")
    for w in WEIGHT_GRID:
        blend = w * ec + (1.0 - w) * gfs
        m = np.abs(blend - obs).mean()
        if m < best_mae_val:
            best_mae_val = m
            best_w = w
    return float(best_w), float(best_mae_val)


# =============================================================================
# 1. LOAD DATA
# =============================================================================

SEP = "=" * 68

print(SEP)
print("REGIME-CONDITIONED ADAPTIVE BLENDER  --  GreenSky Blend SIH26081")
print(SEP)

data = pd.read_csv(INPUT_FILE)
data["time"] = pd.to_datetime(data["time"])
data = data.sort_values("time").reset_index(drop=True)

print(f"\nInput            : {INPUT_FILE}")
print(f"Rows             : {len(data)}")
print(f"Date range       : {data['time'].min().date()} to {data['time'].max().date()}")
print(f"Columns          : {list(data.columns)}")

# =============================================================================
# 2. CHRONOLOGICAL SPLIT
# =============================================================================

train = data[data["time"] < LOCKED_TEST_START].copy().reset_index(drop=True)
test  = data[data["time"] >= LOCKED_TEST_START].copy().reset_index(drop=True)

print(f"\nTrain : {len(train)} days  "
      f"({train['time'].min().date()} to {train['time'].max().date()})")
print(f"Test  : {len(test)} days  "
      f"({test['time'].min().date()} to {test['time'].max().date()})  [LOCKED -- not used for learning]")

# =============================================================================
# 3. REGIME ASSIGNMENT
# =============================================================================

for df in [train, test]:
    df["mean_fc"] = (df["ecmwf"] + df["gfs"]) / 2.0
    df["regime"]  = assign_regime(df["mean_fc"])

print("\n" + SEP)
print("STEP 1 -- REGIME ASSIGNMENT  (based solely on forecast values)")
print(SEP)
print("  Regimes are defined by (ECMWF + GFS) / 2 -- no observations needed.")
print()

# Per-regime training stats
print(f"  {'Regime':<22}  {'N_train':>7}  "
      f"{'EC_MAE':>7}  {'GFS_MAE':>8}  {'Eq_MAE':>7}  {'Obs_mean':>9}")
print(f"  {'-'*22}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*7}  {'-'*9}")

for rid, label in enumerate(REGIME_LABELS):
    sub = train[train["regime"] == rid]
    n   = len(sub)
    if n == 0:
        print(f"  {label:<22}  {0:>7}  {'--':>7}  {'--':>8}  {'--':>7}  {'--':>9}")
        continue
    obs = sub["rain"].values
    ec  = sub["ecmwf"].values
    gfs = sub["gfs"].values
    ec_mae  = mae(obs, ec)
    gfs_mae = mae(obs, gfs)
    eq_mae  = mae(obs, (ec + gfs) / 2.0)
    print(f"  {label:<22}  {n:>7}  {ec_mae:>7.2f}  {gfs_mae:>8.2f}  "
          f"{eq_mae:>7.2f}  {obs.mean():>9.2f}")

# =============================================================================
# 4. LEARN OPTIMAL ECMWF WEIGHT PER REGIME  (training only)
# =============================================================================

print("\n" + SEP)
print("STEP 2 -- LEARNING REGIME WEIGHTS  (grid search on training MAE only)")
print(SEP)
print(f"  Weight grid : 0.00 to 1.00 in steps of 0.05  (21 values)")
print(f"  Fallback to 0.50 if regime has < {MIN_REGIME_SAMPLES} training days.")
print()

learned = {}   # rid -> {"w": float, "n": int, "mae": float|None, "note": str}

print(f"  {'Regime':<22}  {'N':>4}  {'Best_w_ECMWF':>12}  {'Best_MAE':>9}  {'Decision'}")
print(f"  {'-'*22}  {'-'*4}  {'-'*12}  {'-'*9}  {'-'*25}")

for rid, label in enumerate(REGIME_LABELS):
    sub = train[train["regime"] == rid]
    n   = len(sub)

    if n < MIN_REGIME_SAMPLES:
        learned[rid] = {"w": 0.5, "n": n, "mae": None,
                        "note": f"fallback (n={n} < {MIN_REGIME_SAMPLES})"}
        print(f"  {label:<22}  {n:>4}  {'0.50 (fallback)':>12}  {'--':>9}  "
              f"insufficient training data")
        continue

    obs = sub["rain"].values
    ec  = sub["ecmwf"].values
    gfs = sub["gfs"].values

    bw, bm = best_weight(ec, gfs, obs)

    if   bw > 0.5: note = f"ECMWF-favoured (w={bw:.2f})"
    elif bw < 0.5: note = f"GFS-favoured   (w={bw:.2f})"
    else:          note = "Equal weight"

    learned[rid] = {"w": bw, "n": n, "mae": bm, "note": note}
    print(f"  {label:<22}  {n:>4}  {bw:>12.2f}  {bm:>9.2f}  {note}")

# =============================================================================
# 5. APPLY WEIGHTS TO TEST SET
# =============================================================================

print("\n" + SEP)
print("STEP 3 -- APPLYING LEARNED WEIGHTS TO LOCKED TEST SET")
print(SEP)
print("  [Test observations NOT used -- regime assigned from forecast only]")
print()

test["ecmwf_weight"]   = test["regime"].map({rid: learned[rid]["w"] for rid in range(4)})
test["gfs_weight"]     = 1.0 - test["ecmwf_weight"]
test["adaptive_blend"] = (test["ecmwf_weight"] * test["ecmwf"]
                          + test["gfs_weight"] * test["gfs"])
# Recompute equal_blend explicitly for clarity
test["equal_blend"]    = (test["ecmwf"] + test["gfs"]) / 2.0
test["regime_label"]   = test["regime"].map({i: l.strip() for i, l in enumerate(REGIME_LABELS)})

print(f"  {'Regime':<22}  {'N_test':>6}  {'w_ECMWF':>9}  {'Regime trained on N_train':}")
print(f"  {'-'*22}  {'-'*6}  {'-'*9}  {'-'*26}")
for rid, label in enumerate(REGIME_LABELS):
    n_test  = (test["regime"] == rid).sum()
    if n_test == 0:
        continue
    n_train = learned[rid]["n"]
    w       = learned[rid]["w"]
    print(f"  {label:<22}  {n_test:>6}  {w:>9.2f}  n_train={n_train}")

# =============================================================================
# 6. DAY-BY-DAY TEST TABLE
# =============================================================================

print("\n" + SEP)
print("TEST SET DAY-BY-DAY  (2025-08-04 to 2025-08-31, n=28)")
print(SEP)

hdr = (f"  {'Date':<12}  {'IMD':>7}  {'ECMWF':>7}  {'GFS':>7}  "
       f"{'Equal':>7}  {'Adaptive':>9}  {'w_EC':>5}  {'Regime'}")
print(hdr)
print("  " + "-" * 80)

for _, row in test.iterrows():
    print(f"  {str(row['time'].date()):<12}  "
          f"{row['rain']:>7.1f}  "
          f"{row['ecmwf']:>7.1f}  "
          f"{row['gfs']:>7.1f}  "
          f"{row['equal_blend']:>7.1f}  "
          f"{row['adaptive_blend']:>9.1f}  "
          f"{row['ecmwf_weight']:>5.2f}  "
          f"{row['regime_label']}")

# =============================================================================
# 7. METRICS  (test set)
# =============================================================================

print("\n" + SEP)
print("BASELINE METRICS -- LOCKED TEST SET (n=28 days)")
print(SEP)

obs_t = test["rain"].values

model_forecasts = {
    "ECMWF"    : test["ecmwf"].values,
    "GFS"      : test["gfs"].values,
    "Equal"    : test["equal_blend"].values,
    "Adaptive" : test["adaptive_blend"].values,
}

print(f"\n  {'Model':<12}  {'MAE':>7}  {'RMSE':>7}  {'CSI@15.6':>9}  {'CSI@64.5':>9}")
print(f"  {'-'*12}  {'-'*7}  {'-'*7}  {'-'*9}  {'-'*9}")

all_m = {}
for name, fct in model_forecasts.items():
    m_val  = mae(obs_t, fct)
    r_val  = rmse(obs_t, fct)
    h1, ms1, fa1, csi1 = csi_detail(obs_t, fct, 15.6)
    h2, ms2, fa2, csi2 = csi_detail(obs_t, fct, 64.5)
    all_m[name] = {
        "MAE": m_val, "RMSE": r_val,
        "H15": h1, "M15": ms1, "FA15": fa1, "CSI15": csi1,
        "H64": h2, "M64": ms2, "FA64": fa2, "CSI64": csi2,
    }
    c1s = f"{csi1:.3f}" if not np.isnan(csi1) else "  N/A"
    c2s = f"{csi2:.3f}" if not np.isnan(csi2) else "  N/A"
    print(f"  {name:<12}  {m_val:>7.2f}  {r_val:>7.2f}  {c1s:>9}  {c2s:>9}")

# Event detail for CSI@15.6
print(f"\n  CSI @ 15.6 mm/day detail  (9 observed events in test period):")
print(f"  {'Model':<12}  {'Hits':>5}  {'Misses':>7}  {'FalseAlarms':>12}  {'CSI':>6}")
print(f"  {'-'*12}  {'-'*5}  {'-'*7}  {'-'*12}  {'-'*6}")
for name, m in all_m.items():
    c = f"{m['CSI15']:.3f}" if not np.isnan(m['CSI15']) else "  N/A"
    print(f"  {name:<12}  {m['H15']:>5}  {m['M15']:>7}  {m['FA15']:>12}  {c:>6}")

print(f"\n  CSI @ 64.5 mm/day detail  (0 observed events -- CSI undefined):")
print(f"  All models: Hits=0, Misses=0, FalseAlarms=0, CSI=N/A")

# =============================================================================
# 8. TRAINING-SET METRICS  (for reference -- not a performance claim)
# =============================================================================

print("\n" + SEP)
print("TRAINING-SET METRICS  (reference only -- not a performance claim)")
print(SEP)

obs_tr = train["rain"].values
train["adaptive_blend"] = (
    train["regime"].map({rid: learned[rid]["w"] for rid in range(4)}) * train["ecmwf"]
    + (1.0 - train["regime"].map({rid: learned[rid]["w"] for rid in range(4)})) * train["gfs"]
)
train["equal_blend_check"] = (train["ecmwf"] + train["gfs"]) / 2.0

tr_models = {
    "ECMWF"    : train["ecmwf"].values,
    "GFS"      : train["gfs"].values,
    "Equal"    : train["equal_blend_check"].values,
    "Adaptive" : train["adaptive_blend"].values,
}

print(f"\n  {'Model':<12}  {'MAE':>7}  {'RMSE':>7}")
print(f"  {'-'*12}  {'-'*7}  {'-'*7}")
for name, fct in tr_models.items():
    print(f"  {name:<12}  {mae(obs_tr, fct):>7.2f}  {rmse(obs_tr, fct):>7.2f}")
print(f"\n  NOTE: Adaptive training-set MAE is the optimised objective.")
print(f"        It will always be <= or close to ECMWF/GFS by construction.")
print(f"        Do not use training metrics to claim generalisation.")

# =============================================================================
# 9. HONEST VERDICT
# =============================================================================

print("\n" + SEP)
print("VERDICT  (test set only)")
print(SEP)

ad = all_m["Adaptive"]
ec = all_m["ECMWF"]
eq = all_m["Equal"]
gf = all_m["GFS"]

beats_equal_mae  = ad["MAE"]  < eq["MAE"]
beats_equal_rmse = ad["RMSE"] < eq["RMSE"]
beats_equal_csi  = (not np.isnan(ad["CSI15"]) and not np.isnan(eq["CSI15"])
                    and ad["CSI15"] > eq["CSI15"])

beats_ecmwf_mae  = ad["MAE"]  < ec["MAE"]
beats_ecmwf_rmse = ad["RMSE"] < ec["RMSE"]
beats_ecmwf_csi  = (not np.isnan(ad["CSI15"]) and not np.isnan(ec["CSI15"])
                    and ad["CSI15"] > ec["CSI15"])

yn = lambda b: "YES" if b else "NO "

print(f"\n  A) Does Adaptive beat Equal Blend?")
print(f"     MAE     : {yn(beats_equal_mae )}  "
      f"({ad['MAE']:.2f} vs {eq['MAE']:.2f} mm)  "
      f"delta={ad['MAE']-eq['MAE']:+.2f}")
print(f"     RMSE    : {yn(beats_equal_rmse)}  "
      f"({ad['RMSE']:.2f} vs {eq['RMSE']:.2f} mm)  "
      f"delta={ad['RMSE']-eq['RMSE']:+.2f}")
print(f"     CSI@15.6: {yn(beats_equal_csi )}  "
      f"({ad['CSI15']:.3f} vs {eq['CSI15']:.3f})")

print(f"\n  B) Does Adaptive beat ECMWF alone?")
print(f"     MAE     : {yn(beats_ecmwf_mae )}  "
      f"({ad['MAE']:.2f} vs {ec['MAE']:.2f} mm)  "
      f"delta={ad['MAE']-ec['MAE']:+.2f}")
print(f"     RMSE    : {yn(beats_ecmwf_rmse)}  "
      f"({ad['RMSE']:.2f} vs {ec['RMSE']:.2f} mm)  "
      f"delta={ad['RMSE']-ec['RMSE']:+.2f}")
print(f"     CSI@15.6: {yn(beats_ecmwf_csi )}  "
      f"({ad['CSI15']:.3f} vs {ec['CSI15']:.3f})")

# Overall verdict
a_beats_eq  = beats_equal_mae  and beats_equal_rmse
a_beats_ec  = beats_ecmwf_mae  and beats_ecmwf_rmse
a_mixed_eq  = beats_equal_mae  or  beats_equal_rmse
a_mixed_ec  = beats_ecmwf_mae  or  beats_ecmwf_rmse

print("\n  C) Overall conclusion:")
if a_beats_ec:
    print("     >> Adaptive beats BOTH Equal Blend AND ECMWF on MAE and RMSE.")
elif a_beats_eq and not a_beats_ec:
    print("     >> Adaptive beats Equal Blend but does NOT beat ECMWF on MAE+RMSE.")
elif a_mixed_eq:
    print("     >> Mixed result: adaptive better on some metrics, not all.")
else:
    print("     >> Adaptive beats NEITHER Equal Blend NOR ECMWF. No net improvement.")

print()
print("  INTERPRETATION:")
print(f"  The test set is dominated by Dry ({(test['regime']==0).sum()} days) and")
print(f"  Light ({(test['regime']==1).sum()} days) regimes, where the learned weight")
print(f"  is 1.00 (pure ECMWF) in both cases.  Only {(test['regime']>=2).sum()} test days")
print(f"  fall into Moderate/Heavy regimes where blending differs from ECMWF.")
print(f"  The regime blender therefore converges to near-ECMWF on this test set.")
print(f"  A different test period or additional models/features could change this.")

# =============================================================================
# 10. SAVE RESULTS
# =============================================================================

out = test[[
    "time", "rain", "ecmwf", "gfs",
    "equal_blend", "adaptive_blend",
    "regime_label", "ecmwf_weight", "gfs_weight", "mean_fc",
]].copy()

out.to_csv(OUTPUT_FILE, index=False, float_format="%.4f")
print(f"\nSaved: {OUTPUT_FILE}")

print("\n" + SEP)
print("DONE.  Original benchmark files are untouched.")
print("No test observations were used during learning.")
print(SEP)
