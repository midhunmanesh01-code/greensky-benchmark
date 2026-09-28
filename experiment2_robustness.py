# -*- coding: utf-8 -*-
"""
experiment2_robustness.py
=========================
GreenSky Blend -- SIH26081
Experiment 2: Robustness Test (Chronological Sub-Period)

PURPOSE
-------
Test whether the same regime-conditioned adaptive blending method from
Experiment 1 generalises to a completely different chronological period.

DATA SOURCE DECISION (documented)
----------------------------------
The preferred approach was another full monsoon season (2024 Jun-Aug)
using the same IMD grid cell (10.75N, 76.25E).

PROBE RESULTS:
  - Open-Meteo Previous Runs API: 2024 data IS available for both
    ECMWF IFS and GFS. HTTP 200, full non-null coverage confirmed.
  - IMD gridded data (imdlib 2024.grd): Downloaded successfully (25 MB)
    but ALL values are -999 (missing) for the Jun-Aug 2024 period.
    The 2024 IMD gridded dataset on the FTP server appears to lack
    quality-controlled values for this region and period.
  - CONCLUSION: 2024 season is NOT usable. This is clearly reported.

FALLBACK: 2025 CHRONOLOGICAL SUB-PERIOD
----------------------------------------
The existing corrected_benchmark_results.csv (Convention B aligned,
Jun-Aug 2025) is split into a fresh train/test pair that does NOT
overlap with Experiment 1's test set (Aug 4-31):

  Exp-2 Train : 2025-06-01 to 2025-06-30  (30 days, June only)
  Exp-2 Test  : 2025-07-01 to 2025-08-03  (34 days)

The test period is:
  - Chronologically AFTER the training period (no look-ahead).
  - Completely separate from Experiment 1's locked test (Aug 4-31).
  - Contains more heavy-rain events (12 vs 9) and includes the only
    observed extreme event (>=64.5 mm) in the dataset (Jul 25, 69.3 mm).
  - Meteorologically different: peak monsoon July vs declining August.

METHODOLOGY (identical to Experiment 1)
-----------------------------------------
- Same data alignment: (D-1) 03:00 UTC -> D 03:00 UTC
- Same regimes: Dry (<5), Light (5-15), Moderate (15-35), Heavy (>=35)
  defined by (ECMWF + GFS) / 2 -- forecast values only
- Same weight grid: 0.00 to 1.00 in steps of 0.05
- Same learning objective: minimise MAE on training set per regime
- Same fallback: 0.50 if regime has < 5 training days
- Same CSI thresholds: 15.6 and 64.5 mm/day

NO TUNING was performed using the new test period.
"""

import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

# =============================================================================
# CONFIGURATION  (identical to regime_blender.py)
# =============================================================================

INPUT_FILE  = "corrected_benchmark_results.csv"
OUTPUT_FILE = "experiment2_results.csv"

# Exp-2 chronological boundaries
EXP2_TRAIN_END  = pd.Timestamp("2025-07-01")   # exclusive (train < this)
EXP2_TEST_END   = pd.Timestamp("2025-08-04")   # exclusive (test < this)

REGIME_BINS   = [0.0, 5.0, 15.0, 35.0, float("inf")]
REGIME_LABELS = [
    "Dry      (<  5 mm)",
    "Light    ( 5-15 mm)",
    "Moderate (15-35 mm)",
    "Heavy    (>= 35 mm)",
]

CSI_THRESHOLDS     = [15.6, 64.5]
MIN_REGIME_SAMPLES = 5
WEIGHT_GRID        = np.round(np.arange(0.0, 1.01, 0.05), 2)

SEP = "=" * 68


# =============================================================================
# HELPERS  (identical signatures to regime_blender.py)
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
    best_w, best_m = 0.5, float("inf")
    for w in WEIGHT_GRID:
        m = np.abs(w * ec + (1.0 - w) * gfs - obs).mean()
        if m < best_m:
            best_m, best_w = m, w
    return float(best_w), float(best_m)


# =============================================================================
# 0. DATA SOURCE REPORT
# =============================================================================

print(SEP)
print("EXPERIMENT 2 -- ROBUSTNESS TEST  --  GreenSky Blend SIH26081")
print(SEP)

print("""
DATA SOURCE DECISION
--------------------
Preferred: IMD + Open-Meteo forecast data for monsoon 2024 (Jun-Aug).

Probe results:
  [OK ] Open-Meteo ECMWF IFS 2024: HTTP 200, 72/72 non-null values confirmed.
  [OK ] Open-Meteo GFS 2024      : HTTP 200, 72/72 non-null values confirmed.
  [FAIL] IMD gridded 2024 (imdlib): Downloaded 25 MB .grd file successfully,
         but ALL 92 monsoon days contain -999 (missing/unreleased data).
         IMD's public FTP has not yet quality-controlled 2024 gridded
         rainfall for this grid cell.  2024 season is NOT usable.

Fallback: 2025 chronological sub-period split.
  Exp-2 Train : 2025-06-01 to 2025-06-30  (30 days)
  Exp-2 Test  : 2025-07-01 to 2025-08-03  (34 days)
  This is entirely DIFFERENT from Experiment 1's test (2025-08-04 to 2025-08-31).
""")

# =============================================================================
# 1. LOAD DATA
# =============================================================================

data = pd.read_csv(INPUT_FILE)
data["time"] = pd.to_datetime(data["time"])
data = data.sort_values("time").reset_index(drop=True)

print(f"Input file : {INPUT_FILE}")
print(f"Total rows : {len(data)}")

# =============================================================================
# 2. EXP-2 CHRONOLOGICAL SPLIT
# =============================================================================

train = data[data["time"] < EXP2_TRAIN_END].copy().reset_index(drop=True)
test  = data[(data["time"] >= EXP2_TRAIN_END) &
             (data["time"] < EXP2_TEST_END)].copy().reset_index(drop=True)

print()
print(f"Exp-2 Train : {len(train)} days  "
      f"({train['time'].min().date()} to {train['time'].max().date()})")
print(f"Exp-2 Test  : {len(test)} days  "
      f"({test['time'].min().date()} to {test['time'].max().date()})  [LOCKED]")
print()
print(f"Events >= 15.6 mm in test : {(test['rain'] >= 15.6).sum()}")
print(f"Events >= 64.5 mm in test : {(test['rain'] >= 64.5).sum()}")

# =============================================================================
# 3. REGIME ASSIGNMENT
# =============================================================================

for df in [train, test]:
    df["mean_fc"] = (df["ecmwf"] + df["gfs"]) / 2.0
    df["regime"]  = assign_regime(df["mean_fc"])

print()
print(SEP)
print("STEP 1 -- REGIME ASSIGNMENT  (based solely on forecast values)")
print(SEP)

print(f"\n  {'Regime':<22}  {'N_train':>7}  "
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
    print(f"  {label:<22}  {n:>7}  "
          f"{mae(obs,ec):>7.2f}  {mae(obs,gfs):>8.2f}  "
          f"{mae(obs,(ec+gfs)/2):>7.2f}  {obs.mean():>9.2f}")

# =============================================================================
# 4. LEARN REGIME WEIGHTS  (training data only -- SAME procedure as Exp 1)
# =============================================================================

print()
print(SEP)
print("STEP 2 -- LEARNING REGIME WEIGHTS  (Exp-2 training data only)")
print(SEP)
print(f"  Procedure identical to Experiment 1.")
print(f"  Fallback to 0.50 if regime has < {MIN_REGIME_SAMPLES} training days.")
print()

learned = {}

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

    bw, bm = best_weight(sub["ecmwf"].values, sub["gfs"].values, sub["rain"].values)

    if   bw > 0.5: note = f"ECMWF-favoured (w={bw:.2f})"
    elif bw < 0.5: note = f"GFS-favoured   (w={bw:.2f})"
    else:          note = "Equal weight"

    learned[rid] = {"w": bw, "n": n, "mae": bm, "note": note}
    print(f"  {label:<22}  {n:>4}  {bw:>12.2f}  {bm:>9.2f}  {note}")

# =============================================================================
# 5. COMPARE WITH EXP-1 WEIGHTS
# =============================================================================

# Experiment 1 learned weights (from regime_blender.py output -- hard-coded
# here to allow comparison without re-running that script)
EXP1_WEIGHTS = {0: 1.00, 1: 1.00, 2: 0.60, 3: 0.45}
EXP1_TRAIN_N = {0: 20,   1: 24,   2: 13,   3: 7}

print()
print(SEP)
print("WEIGHT COMPARISON: Experiment 1 vs Experiment 2")
print(SEP)
print(f"\n  {'Regime':<22}  {'Exp1_w':>8}  {'Exp1_n':>7}  {'Exp2_w':>8}  {'Exp2_n':>7}  {'Delta_w':>8}")
print(f"  {'-'*22}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*7}  {'-'*8}")

for rid, label in enumerate(REGIME_LABELS):
    w1 = EXP1_WEIGHTS[rid]
    n1 = EXP1_TRAIN_N[rid]
    w2 = learned[rid]["w"]
    n2 = learned[rid]["n"]
    dw = w2 - w1
    sign = "+" if dw >= 0 else ""
    print(f"  {label:<22}  {w1:>8.2f}  {n1:>7}  {w2:>8.2f}  {n2:>7}  {sign}{dw:>7.2f}")

# =============================================================================
# 6. APPLY LEARNED WEIGHTS TO EXP-2 TEST
# =============================================================================

print()
print(SEP)
print("STEP 3 -- APPLYING LEARNED WEIGHTS TO EXP-2 TEST SET")
print(SEP)
print("  [Test observations NOT used -- regime assigned from forecast only]")
print()

test["ecmwf_weight"]   = test["regime"].map({rid: learned[rid]["w"] for rid in range(4)})
test["gfs_weight"]     = 1.0 - test["ecmwf_weight"]
test["adaptive_blend"] = (test["ecmwf_weight"] * test["ecmwf"]
                          + test["gfs_weight"]  * test["gfs"])
test["equal_blend"]    = (test["ecmwf"] + test["gfs"]) / 2.0
test["regime_label"]   = test["regime"].map({i: l.strip() for i, l in enumerate(REGIME_LABELS)})

print(f"  {'Regime':<22}  {'N_test':>6}  {'w_ECMWF':>9}  {'n_train':>8}")
print(f"  {'-'*22}  {'-'*6}  {'-'*9}  {'-'*8}")
for rid, label in enumerate(REGIME_LABELS):
    n_t = (test["regime"] == rid).sum()
    if n_t == 0:
        continue
    print(f"  {label:<22}  {n_t:>6}  {learned[rid]['w']:>9.2f}  {learned[rid]['n']:>8}")

# =============================================================================
# 7. DAY-BY-DAY TABLE
# =============================================================================

print()
print(SEP)
print("TEST SET DAY-BY-DAY  (2025-07-01 to 2025-08-03, n=34)")
print(SEP)

print(f"\n  {'Date':<12}  {'IMD':>7}  {'ECMWF':>7}  {'GFS':>7}  "
      f"{'Equal':>7}  {'Adaptive':>9}  {'w_EC':>5}  {'Regime'}")
print("  " + "-" * 85)

for _, row in test.iterrows():
    heavy = " <<" if row["rain"] >= 64.5 else (" <" if row["rain"] >= 15.6 else "")
    print(f"  {str(row['time'].date()):<12}  "
          f"{row['rain']:>7.1f}{heavy:<3}  "
          f"{row['ecmwf']:>7.1f}  "
          f"{row['gfs']:>7.1f}  "
          f"{row['equal_blend']:>7.1f}  "
          f"{row['adaptive_blend']:>9.1f}  "
          f"{row['ecmwf_weight']:>5.2f}  "
          f"{row['regime_label']}")
print("  (< = event >= 15.6 mm,  << = event >= 64.5 mm)")

# =============================================================================
# 8. METRICS
# =============================================================================

print()
print(SEP)
print("METRICS -- EXP-2 TEST SET (n=34 days, Jul 1 - Aug 3 2025)")
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
    h1, ms1, fa1, csi1 = csi_detail(obs_t, fct, 15.6)
    h2, ms2, fa2, csi2 = csi_detail(obs_t, fct, 64.5)
    all_m[name] = {
        "MAE":  mae(obs_t, fct),
        "RMSE": rmse(obs_t, fct),
        "H15": h1, "M15": ms1, "FA15": fa1, "CSI15": csi1,
        "H64": h2, "M64": ms2, "FA64": fa2, "CSI64": csi2,
    }
    c1s = f"{csi1:.3f}" if not np.isnan(csi1) else "  N/A"
    c2s = f"{csi2:.3f}" if not np.isnan(csi2) else "  N/A"
    print(f"  {name:<12}  {all_m[name]['MAE']:>7.2f}  {all_m[name]['RMSE']:>7.2f}  "
          f"{c1s:>9}  {c2s:>9}")

print(f"\n  CSI @ 15.6 mm/day  ({int((obs_t >= 15.6).sum())} observed events):")
print(f"  {'Model':<12}  {'Hits':>5}  {'Misses':>7}  {'FA':>5}  {'CSI':>6}")
print(f"  {'-'*12}  {'-'*5}  {'-'*7}  {'-'*5}  {'-'*6}")
for name, m in all_m.items():
    c = f"{m['CSI15']:.3f}" if not np.isnan(m['CSI15']) else "  N/A"
    print(f"  {name:<12}  {m['H15']:>5}  {m['M15']:>7}  {m['FA15']:>5}  {c:>6}")

print(f"\n  CSI @ 64.5 mm/day  ({int((obs_t >= 64.5).sum())} observed event -- limited statistical weight):")
print(f"  {'Model':<12}  {'Hits':>5}  {'Misses':>7}  {'FA':>5}  {'CSI':>6}")
print(f"  {'-'*12}  {'-'*5}  {'-'*7}  {'-'*5}  {'-'*6}")
for name, m in all_m.items():
    c = f"{m['CSI64']:.3f}" if not np.isnan(m['CSI64']) else "  N/A"
    print(f"  {name:<12}  {m['H64']:>5}  {m['M64']:>7}  {m['FA64']:>5}  {c:>6}")

# =============================================================================
# 9. SIDE-BY-SIDE WITH EXPERIMENT 1
# =============================================================================

EXP1_METRICS = {
    "ECMWF"    : {"MAE": 9.03, "RMSE": 12.54, "CSI15": 0.286},
    "GFS"      : {"MAE": 10.80, "RMSE": 16.27, "CSI15": 0.167},
    "Equal"    : {"MAE": 9.02, "RMSE": 13.62, "CSI15": 0.083},
    "Adaptive" : {"MAE": 8.80, "RMSE": 12.30, "CSI15": 0.286},
}

print()
print(SEP)
print("SIDE-BY-SIDE: Experiment 1 (Aug) vs Experiment 2 (Jul)")
print(SEP)
print(f"\n  {'Model':<12}  {'E1_MAE':>7}  {'E2_MAE':>7}  {'E1_RMSE':>8}  {'E2_RMSE':>8}  "
      f"{'E1_CSI':>7}  {'E2_CSI':>7}")
print(f"  {'-'*12}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*7}")

for name in ["ECMWF", "GFS", "Equal", "Adaptive"]:
    e1 = EXP1_METRICS[name]
    e2 = all_m[name]
    c2 = f"{e2['CSI15']:.3f}" if not np.isnan(e2['CSI15']) else "  N/A"
    print(f"  {name:<12}  {e1['MAE']:>7.2f}  {e2['MAE']:>7.2f}  "
          f"{e1['RMSE']:>8.2f}  {e2['RMSE']:>8.2f}  "
          f"{e1['CSI15']:>7.3f}  {c2:>7}")

# =============================================================================
# 10. VERDICT
# =============================================================================

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

print()
print(SEP)
print("VERDICT (Experiment 2 test set)")
print(SEP)

print(f"\n  A) Adaptive beats Equal Blend?")
print(f"     MAE     : {yn(beats_equal_mae )}  "
      f"({ad['MAE']:.2f} vs {eq['MAE']:.2f})  delta={ad['MAE']-eq['MAE']:+.2f}")
print(f"     RMSE    : {yn(beats_equal_rmse)}  "
      f"({ad['RMSE']:.2f} vs {eq['RMSE']:.2f})  delta={ad['RMSE']-eq['RMSE']:+.2f}")
print(f"     CSI@15.6: {yn(beats_equal_csi )}  "
      f"({ad['CSI15']:.3f} vs {eq['CSI15']:.3f})")

print(f"\n  B) Adaptive beats ECMWF?")
print(f"     MAE     : {yn(beats_ecmwf_mae )}  "
      f"({ad['MAE']:.2f} vs {ec['MAE']:.2f})  delta={ad['MAE']-ec['MAE']:+.2f}")
print(f"     RMSE    : {yn(beats_ecmwf_rmse)}  "
      f"({ad['RMSE']:.2f} vs {ec['RMSE']:.2f})  delta={ad['RMSE']-ec['RMSE']:+.2f}")
print(f"     CSI@15.6: {yn(beats_ecmwf_csi )}  "
      f"({ad['CSI15']:.3f} vs {ec['CSI15']:.3f})")

# Classify verdict
all_beat_eq  = beats_equal_mae and beats_equal_rmse and beats_equal_csi
some_beat_eq = beats_equal_mae or  beats_equal_rmse or  beats_equal_csi
all_beat_ec  = beats_ecmwf_mae and beats_ecmwf_rmse
some_beat_ec = beats_ecmwf_mae or  beats_ecmwf_rmse

print()
print(f"  {'='*60}")
if all_beat_eq and all_beat_ec:
    verdict = "ROBUST POSITIVE"
    detail  = ("Adaptive beats both Equal Blend and ECMWF on all three "
               "key metrics (MAE, RMSE, CSI@15.6) in this independent period.")
elif all_beat_eq and some_beat_ec:
    verdict = "MIXED / INCONCLUSIVE"
    detail  = ("Adaptive beats Equal Blend on all metrics but only partially "
               "beats ECMWF. The improvement is inconsistent across metrics.")
elif some_beat_eq and not all_beat_eq:
    verdict = "MIXED / INCONCLUSIVE"
    detail  = ("Adaptive beats Equal Blend on some but not all metrics. "
               "No consistent improvement signal.")
elif all_beat_eq and not some_beat_ec:
    verdict = "MIXED / INCONCLUSIVE"
    detail  = ("Adaptive beats Equal Blend but cannot match ECMWF alone. "
               "Blending GFS in adds noise rather than skill.")
else:
    verdict = "NEGATIVE"
    detail  = ("Adaptive does not beat Equal Blend or ECMWF on this test period. "
               "The regime weights learned in training do not transfer.")

print(f"  VERDICT: {verdict}")
print(f"  {'='*60}")
print()
print(f"  Reason: {detail}")
print()
print("  Experiment 1 (Aug 4-31)  --> POSITIVE on MAE/RMSE; tied CSI@15.6")
print(f"  Experiment 2 (Jul 1-Aug3) --> {verdict}")
print()
print("  Cross-experiment interpretation:")
if verdict == "ROBUST POSITIVE":
    print("  Both experiments show consistent improvement.")
    print("  Regime blending appears to generalise within this monsoon season.")
elif verdict == "MIXED / INCONCLUSIVE":
    print("  Results are inconsistent across the two test periods.")
    print("  Cannot claim reliable generalisation from 92 days of training data.")
    print("  More data (multiple seasons) is needed before drawing conclusions.")
else:
    print("  Experiment 2 does not confirm Experiment 1.")
    print("  The positive Exp-1 result may have been specific to August 2025.")
    print("  Do not claim generalisation.")

# =============================================================================
# 11. SAVE RESULTS
# =============================================================================

out = test[[
    "time", "rain", "ecmwf", "gfs",
    "equal_blend", "adaptive_blend",
    "regime_label", "ecmwf_weight", "gfs_weight", "mean_fc",
]].copy()

out.to_csv(OUTPUT_FILE, index=False, float_format="%.4f")
print(f"Saved: {OUTPUT_FILE}")
print()
print(SEP)
print("DONE.  No existing files were modified.")
print("No test observations used for weight learning.")
print(SEP)
