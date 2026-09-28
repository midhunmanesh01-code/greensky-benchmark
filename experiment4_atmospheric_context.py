# -*- coding: utf-8 -*-
"""
experiment4_atmospheric_context.py
====================================
GreenSky Blend -- SIH26081
Experiment 4: Atmospheric-Context Adaptive Blending

MOTIVATION
----------
Experiments 1-3 showed that when both ECMWF and GFS precipitation forecasts
substantially underforecast an event (Type B: regime misclassification), no
precipitation-only weighting or calibration can recover the missing signal.

This experiment adds atmospheric-context predictors -- from the SAME historical
forecast runs at the SAME forecast lead -- to a gradient boosting model to
determine whether the atmospheric state can help identify underforecast days.

FEASIBILITY CHECK (CONDUCTED BEFORE WRITING THIS CODE)
-------------------------------------------------------
API: https://previous-runs-api.open-meteo.com/v1/forecast
Models: ecmwf_ifs025 and gfs_seamless (same as Experiments 1-3)
Suffix: _previous_day1 (same lead time as precipitation_previous_day1)

CONFIRMED AVAILABLE (non-null for Jun-Aug 2025):
  ECMWF + GFS: temperature_2m, relative_humidity_2m, dew_point_2m,
               wind_speed_10m, wind_direction_10m, surface_pressure, cloud_cover
  GFS only:    cape, wind_speed_80m, lifted_index

CONFIRMED NULL / REJECTED:
  ECMWF: cape, wind_speed_80m, lifted_index, convective_inhibition
  Both:  convective_inhibition
  Reanalysis variables: PROHIBITED (would use observed atmospheric state)

TEMPORAL ALIGNMENT
------------------
All _previous_day1 variables use the corrected IMD convention:
    IMD date D = (D-1) 03:00 UTC to D 03:00 UTC (24 hourly rows)
This is identical to the precipitation window in Experiments 1-3.
No leakage: the forecast was issued before the accumulation period ends.

CRITICAL CONSTRAINTS
--------------------
- Training-only fitting and preprocessing: all scalers/encoders fit on train set
- No test observations used as features
- Experiments 1-3 outputs are NOT modified
- No fabricated or reanalysis variables

EVALUATION STRUCTURE
--------------------
Exp-1: Train Jun 1 - Aug 3 (64 days), Test Aug 4 - Aug 31 (28 days)
Exp-2: Train Jun 1 - Jun 30 (30 days), Test Jul 1 - Aug 3 (34 days)
CSI thresholds: 15.6, 35.0, 64.5 mm/day

HONEST REPORTING
----------------
This is an exploratory experiment on a small dataset (n=64 / n=30 training days).
Any improvement may not generalize. Overfitting risk is explicitly discussed.
"""

import sys
import time
import requests
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

sys.stdout.reconfigure(encoding="utf-8")

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE   = "corrected_benchmark_results.csv"
OUTPUT_FILE  = "experiment4_results.csv"
ATMOS_CACHE  = "experiment4_atmos_cache.csv"  # cached atmospheric data (avoid re-fetching)

LAT, LON     = 10.75, 76.25
API_URL      = "https://previous-runs-api.open-meteo.com/v1/forecast"
API_START    = "2025-05-31"   # one day before IMD start for 03:00 UTC window
API_END      = "2025-08-31"
IMD_WIN_H    = 3              # window offset: 03:00 UTC

# Regime weights from Experiment 1 (used for adaptive baseline comparison)
EXP1_WEIGHTS = {0: 1.00, 1: 1.00, 2: 0.60, 3: 0.45}
EXP1_BINS    = [0.0, 5.0, 15.0, 35.0, float("inf")]

CSI_THRESHOLDS = [15.6, 35.0, 64.5]
MIN_TRAIN_SAMPLES = 5    # minimum training samples before allowing atmos model
RANDOM_STATE = 42

SEP = "=" * 72

# Atmospheric variables to fetch (from API probe -- verified non-null)
ECMWF_ATMOS_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "cloud_cover",
]

GFS_ATMOS_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "cloud_cover",
    "cape",           # GFS only -- ECMWF serves null for this variable
    "wind_speed_80m", # GFS only
    "lifted_index",   # GFS only
]


# =============================================================================
# 1. HEADER
# =============================================================================

print(SEP)
print("EXPERIMENT 4 -- ATMOSPHERIC-CONTEXT ADAPTIVE BLENDING  GreenSky SIH26081")
print(SEP)

print("""
FEASIBILITY SUMMARY (pre-verified by API probe):
  Source  : Open-Meteo Previous Runs API (same as Experiments 1-3)
  Models  : ecmwf_ifs025, gfs_seamless
  Lead    : _previous_day1 (same lead as precipitation)
  Archive : Jan 2024+ -- covers Jun-Aug 2025

  CONFIRMED AVAILABLE (both models):
    temperature_2m, relative_humidity_2m, dew_point_2m,
    wind_speed_10m, wind_direction_10m, surface_pressure, cloud_cover

  CONFIRMED AVAILABLE (GFS only, ECMWF returns null):
    cape, wind_speed_80m, lifted_index

  REJECTED (all null): convective_inhibition (both); ECMWF cape/lifted_index
  REJECTED (category): all reanalysis variables (ERA5 etc.)

  Temporal alignment: same (D-1) 03:00 UTC -> D 03:00 UTC window.
  No leakage introduced beyond existing precipitation forecasts.
""")


# =============================================================================
# 2. LOAD PRECIPITATION BENCHMARK DATA
# =============================================================================

data = pd.read_csv(INPUT_FILE)
data["time"] = pd.to_datetime(data["time"])
data = data.sort_values("time").reset_index(drop=True)

print(f"Loaded: {INPUT_FILE}  ({len(data)} rows, "
      f"{data['time'].min().date()} to {data['time'].max().date()})")


# =============================================================================
# 3. FETCH ATMOSPHERIC DATA
# =============================================================================

def detect_triplet(vals: list) -> bool:
    """Return True if >= 90% of value-pairs 2 steps apart are equal (3h repeat)."""
    arr = [v for v in vals if v is not None]
    if len(arr) < 6:
        return False
    eq = sum(1 for i in range(len(arr) - 2) if arr[i] == arr[i + 2])
    return eq / (len(arr) - 2) >= 0.90


def fetch_atmos(model: str, var_list: list) -> pd.DataFrame:
    """
    Fetch atmospheric hourly data for all variables in var_list.
    Returns DataFrame indexed by UTC time with one column per variable.
    """
    hourly_str = ",".join(f"{v}_previous_day1" for v in var_list)
    params = {
        "latitude":      LAT,
        "longitude":     LON,
        "start_date":    API_START,
        "end_date":      API_END,
        "previous_day1": True,
        "models":        model,
        "hourly":        hourly_str,
    }
    for attempt in range(3):
        try:
            r = requests.get(API_URL, params=params, timeout=90)
            if r.status_code == 200:
                break
            print(f"  [{model}] HTTP {r.status_code}: {r.text[:200]}")
            time.sleep(5)
        except Exception as e:
            print(f"  [{model}] attempt {attempt+1} error: {e}")
            time.sleep(10)
    else:
        raise RuntimeError(f"Failed to fetch {model} atmospheric data after 3 attempts")

    d = r.json()
    times = pd.to_datetime(d["hourly"]["time"])
    df = pd.DataFrame({"time": times})

    for v in var_list:
        key = f"{v}_previous_day1"
        raw = d["hourly"].get(key, [None] * len(times))
        is_triplet = detect_triplet(raw)
        if v == "wind_direction_10m":
            # Direction: never divide by 3, always take circular mean per window
            df[v] = raw
        elif is_triplet and v not in ("wind_direction_10m",):
            # Divide by 3 to recover physical units (same as precipitation)
            df[v] = [x / 3.0 if x is not None else None for x in raw]
        else:
            df[v] = raw

    return df


def aggregate_atmos_window(hourly_df: pd.DataFrame, imd_dates: pd.Series,
                             var_list: list, model_prefix: str) -> pd.DataFrame:
    """
    Aggregate hourly atmospheric data over the corrected IMD window:
        (D-1) 03:00 UTC to D 03:00 UTC
    For each variable, compute:
        - mean (default for all continuous variables)
        - max  (additional for cape, useful for convective events)
    For wind_direction_10m, use circular (sin/cos) mean.
    """
    rows = []
    for imd_date in imd_dates:
        win_start = (imd_date - timedelta(days=1)) + timedelta(hours=IMD_WIN_H)
        win_end   = imd_date + timedelta(hours=IMD_WIN_H)
        mask = (hourly_df["time"] >= win_start) & (hourly_df["time"] < win_end)
        row = {"imd_date": imd_date.date()}

        for v in var_list:
            if v not in hourly_df.columns:
                continue
            window = hourly_df.loc[mask, v].dropna()
            if len(window) == 0:
                row[f"{model_prefix}_{v}_mean"] = np.nan
                if v == "cape":
                    row[f"{model_prefix}_{v}_max"] = np.nan
                continue

            if v == "wind_direction_10m":
                # Circular mean via sin/cos components
                angles_rad = np.deg2rad(window.values.astype(float))
                row[f"{model_prefix}_wind_dir_sin"] = float(np.sin(angles_rad).mean())
                row[f"{model_prefix}_wind_dir_cos"] = float(np.cos(angles_rad).mean())
            else:
                row[f"{model_prefix}_{v}_mean"] = float(window.mean())
                if v == "cape":
                    row[f"{model_prefix}_{v}_max"] = float(window.max())

        rows.append(row)

    return pd.DataFrame(rows)


# Check for cached atmospheric data to avoid re-fetching
import os
if os.path.exists(ATMOS_CACHE):
    print(f"\nLoading cached atmospheric data from {ATMOS_CACHE}")
    atmos_daily = pd.read_csv(ATMOS_CACHE)
    atmos_daily["imd_date"] = pd.to_datetime(atmos_daily["imd_date"]).dt.date
    print(f"  Cached rows: {len(atmos_daily)}")
else:
    print(f"\nFetching atmospheric data from Open-Meteo Previous Runs API ...")
    print(f"  ECMWF IFS025 variables: {ECMWF_ATMOS_VARS}")
    print(f"  GFS Seamless variables:  {GFS_ATMOS_VARS}")

    ecmwf_h = fetch_atmos("ecmwf_ifs025", ECMWF_ATMOS_VARS)
    print(f"  ECMWF hourly rows: {len(ecmwf_h)}")

    gfs_h = fetch_atmos("gfs_seamless", GFS_ATMOS_VARS)
    print(f"  GFS hourly rows:   {len(gfs_h)}")

    imd_dates = data["time"]
    ecmwf_agg = aggregate_atmos_window(ecmwf_h, imd_dates, ECMWF_ATMOS_VARS, "ecmwf")
    gfs_agg   = aggregate_atmos_window(gfs_h,   imd_dates, GFS_ATMOS_VARS,   "gfs")

    ecmwf_agg["imd_date"] = pd.to_datetime(ecmwf_agg["imd_date"].astype(str)).dt.date
    gfs_agg["imd_date"]   = pd.to_datetime(gfs_agg["imd_date"].astype(str)).dt.date

    atmos_daily = ecmwf_agg.merge(gfs_agg, on="imd_date", how="inner")
    atmos_daily.to_csv(ATMOS_CACHE, index=False)
    print(f"  Saved atmospheric cache: {ATMOS_CACHE} ({len(atmos_daily)} rows)")


# =============================================================================
# 4. MERGE INTO FULL DATASET
# =============================================================================

data["imd_date"] = data["time"].dt.date
full = data.merge(atmos_daily, on="imd_date", how="inner")
print(f"\nMerged dataset: {len(full)} rows "
      f"({full['time'].min().date()} to {full['time'].max().date()})")

# Check for NaN coverage in atmos columns
atmos_cols = [c for c in full.columns if c.startswith(("ecmwf_", "gfs_")) and c not in
              ("ecmwf", "gfs")]
nan_summary = full[atmos_cols].isnull().sum()
if nan_summary.any():
    print("\n  NaN counts in atmospheric features:")
    for col, n in nan_summary[nan_summary > 0].items():
        print(f"    {col}: {n}")
else:
    print("  No NaN values in atmospheric features. ✓")


# =============================================================================
# 5. FEATURE ENGINEERING
# =============================================================================

def build_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean_fc and regime bin (same logic as Exp 1-3)."""
    df = df.copy()
    df["mean_fc"] = (df["ecmwf"] + df["gfs"]) / 2.0
    df["regime"]  = pd.cut(
        df["mean_fc"],
        bins=EXP1_BINS,
        labels=list(range(len(EXP1_BINS) - 1)),
        right=False,
    ).astype(int)
    df["adaptive"] = df["regime"].map(EXP1_WEIGHTS) * df["ecmwf"] + \
                     (1 - df["regime"].map(EXP1_WEIGHTS)) * df["gfs"]
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all feature columns from a merged row.
    Uses only forecast variables (no observations).
    All computed from training-only statistics when scaling is needed.
    """
    feats = pd.DataFrame(index=df.index)

    # --- Precipitation forecast features ---
    feats["ecmwf_precip"]    = df["ecmwf"]
    feats["gfs_precip"]      = df["gfs"]
    feats["mean_precip"]     = (df["ecmwf"] + df["gfs"]) / 2.0
    feats["precip_disagree"] = (df["ecmwf"] - df["gfs"]).abs()
    feats["precip_ratio"]    = df["ecmwf"] / (df["gfs"] + 1.0)
    feats["log_mean_precip"] = np.log1p(feats["mean_precip"])
    feats["regime"]          = df["regime"].astype(float)

    # --- ECMWF atmospheric ---
    for v in ["temperature_2m", "relative_humidity_2m", "dew_point_2m",
              "wind_speed_10m", "surface_pressure", "cloud_cover"]:
        col = f"ecmwf_{v}_mean"
        if col in df.columns:
            feats[f"ecmwf_{v}"] = df[col]

    # Wind direction circular components (ECMWF)
    if "ecmwf_wind_dir_sin" in df.columns:
        feats["ecmwf_wind_dir_sin"] = df["ecmwf_wind_dir_sin"]
        feats["ecmwf_wind_dir_cos"] = df["ecmwf_wind_dir_cos"]

    # --- GFS atmospheric ---
    for v in ["temperature_2m", "relative_humidity_2m", "dew_point_2m",
              "wind_speed_10m", "surface_pressure", "cloud_cover",
              "wind_speed_80m"]:
        col = f"gfs_{v}_mean"
        if col in df.columns:
            feats[f"gfs_{v}"] = df[col]

    # CAPE and Lifted Index (GFS only -- ECMWF null for these)
    if "gfs_cape_mean" in df.columns:
        feats["gfs_cape_mean"] = df["gfs_cape_mean"]
        if "gfs_cape_max" in df.columns:
            feats["gfs_cape_max"]  = df["gfs_cape_max"]
    if "gfs_lifted_index_mean" in df.columns:
        feats["gfs_lifted_index"] = df["gfs_lifted_index_mean"]

    # Wind direction circular components (GFS)
    if "gfs_wind_dir_sin" in df.columns:
        feats["gfs_wind_dir_sin"] = df["gfs_wind_dir_sin"]
        feats["gfs_wind_dir_cos"] = df["gfs_wind_dir_cos"]

    # --- Cross-model disagreement features ---
    if "ecmwf_temperature_2m_mean" in df.columns and "gfs_temperature_2m_mean" in df.columns:
        feats["temp_disagree"] = (
            df["ecmwf_temperature_2m_mean"] - df["gfs_temperature_2m_mean"]
        ).abs()
    if "ecmwf_relative_humidity_2m_mean" in df.columns and "gfs_relative_humidity_2m_mean" in df.columns:
        feats["rh_disagree"] = (
            df["ecmwf_relative_humidity_2m_mean"] - df["gfs_relative_humidity_2m_mean"]
        ).abs()
    if "ecmwf_surface_pressure_mean" in df.columns and "gfs_surface_pressure_mean" in df.columns:
        feats["pressure_disagree"] = (
            df["ecmwf_surface_pressure_mean"] - df["gfs_surface_pressure_mean"]
        ).abs()

    # --- Seasonal context (not observation-derived) ---
    feats["month"] = df["time"].dt.month
    feats["day_of_year"] = df["time"].dt.dayofyear

    return feats.fillna(0.0)  # fill residual NaN with 0 (safe: only affects null atmos cols)


full = build_regime(full)


# =============================================================================
# 6. METRIC HELPERS
# =============================================================================

def mae(obs, fct):
    return float(np.abs(np.array(fct) - np.array(obs)).mean())

def rmse(obs, fct):
    return float(np.sqrt(((np.array(fct) - np.array(obs)) ** 2).mean()))

def csi_detail(obs, fct, thr):
    obs_e = np.array(obs) >= thr
    fct_e = np.array(fct) >= thr
    hits  = int(( obs_e &  fct_e).sum())
    miss  = int(( obs_e & ~fct_e).sum())
    fa    = int((~obs_e &  fct_e).sum())
    denom = hits + miss + fa
    return hits, miss, fa, (hits / denom if denom > 0 else float("nan"))

def print_metrics_block(obs, model_dict, label, n_days):
    print(f"\n  {label}  (n={n_days})")
    print(f"  {'-'*68}")
    print(f"  {'Model':<26}  {'MAE':>7}  {'RMSE':>7}  "
          f"{'CSI@15.6':>9}  {'CSI@35':>7}  {'CSI@64.5':>9}")
    print(f"  {'-'*26}  {'-'*7}  {'-'*7}  {'-'*9}  {'-'*7}  {'-'*9}")
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
        all_m[name] = {"MAE": m_val, "RMSE": r_val,
                       "H15": h1, "M15": ms1, "FA15": fa1, "CSI15": c1,
                       "H35": h2, "M35": ms2, "FA35": fa2, "CSI35": c2,
                       "H64": h3, "M64": ms3, "FA64": fa3, "CSI64": c3}
        print(f"  {name:<26}  {m_val:>7.2f}  {r_val:>7.2f}  "
              f"{c1s:>9}  {c2s:>7}  {c3s:>9}")
    return all_m


def print_csi_detail_table(obs, model_dict, thr):
    n_ev = int((np.array(obs) >= thr).sum())
    print(f"\n  CSI @ {thr} mm  ({n_ev} observed events)")
    print(f"  {'Model':<26}  {'Hits':>5}  {'Misses':>7}  {'FA':>5}  {'CSI':>7}")
    print(f"  {'-'*26}  {'-'*5}  {'-'*7}  {'-'*5}  {'-'*7}")
    for name, fct in model_dict.items():
        h, ms, fa, csi = csi_detail(obs, fct, thr)
        cs = f"{csi:.3f}" if not np.isnan(csi) else "  N/A"
        print(f"  {name:<26}  {h:>5}  {ms:>7}  {fa:>5}  {cs:>7}")


def print_heavy_days(test_df, model_dict, thr=15.6):
    heavy = test_df[test_df["rain"] >= thr]
    if len(heavy) == 0:
        print(f"  No observed days >= {thr} mm.")
        return {}
    print(f"\n  Observed days >= {thr} mm  (n={len(heavy)}):")
    names = list(model_dict.keys())
    hdr = f"  {'Date':<12}  {'IMD':>6}"
    for n in names:
        hdr += f"  {n[:11]:>11}"
    print(hdr)
    print("  " + "-" * (12 + 8 + 13 * len(names)))

    heavy_mae = {n: [] for n in names}
    for idx, row in heavy.iterrows():
        flag = " <<" if row["rain"] >= 64.5 else (" <" if row["rain"] >= 35 else "")
        line = f"  {str(row['time'].date()):<12}  {row['rain']:>6.1f}{flag:<3}"
        i_local = list(test_df.index).index(idx)
        for n, fct in model_dict.items():
            val = fct[i_local] if isinstance(fct, np.ndarray) else float(fct.iloc[i_local])
            heavy_mae[n].append(abs(val - row["rain"]))
            line += f"  {val:>11.1f}"
        print(line)

    print(f"\n  MAE on heavy days only (n={len(heavy)}):")
    for n, errs in heavy_mae.items():
        print(f"    {n:<26}  {np.mean(errs):.2f} mm")
    return heavy_mae


# =============================================================================
# 7. EXPERIMENT EVALUATION FUNCTION
# =============================================================================

def run_evaluation(full_df: pd.DataFrame, train_end_date: str,
                   test_start_date: str, test_end_date: str,
                   label: str) -> dict:
    """
    Run one evaluation period.
    train_end_date: exclusive upper bound for training set
    test_start_date, test_end_date: inclusive bounds for test set
    """
    print(f"\n{SEP}")
    print(f"EVALUATION -- {label}")
    print(SEP)

    train_df = full_df[full_df["time"] < train_end_date].copy()
    test_df  = full_df[
        (full_df["time"] >= test_start_date) &
        (full_df["time"] <= test_end_date)
    ].copy()
    test_df = test_df.reset_index(drop=True)

    print(f"\n  Train: {len(train_df)} days  "
          f"({train_df['time'].min().date()} to {train_df['time'].max().date()})")
    print(f"  Test : {len(test_df)} days  "
          f"({test_df['time'].min().date()} to {test_df['time'].max().date()})")

    # Safety check
    if len(train_df) < MIN_TRAIN_SAMPLES:
        print(f"\n  [!] Insufficient training samples ({len(train_df)} < {MIN_TRAIN_SAMPLES}). Skipping ML.")
        return {}

    # Build features (fit nothing on test)
    X_train = build_features(train_df)
    X_test  = build_features(test_df)
    y_train = train_df["rain"].values
    obs     = test_df["rain"].values

    feature_names = list(X_train.columns)

    print(f"\n  Features: {len(feature_names)}")
    print(f"  Training target (rain) stats: "
          f"mean={y_train.mean():.1f} max={y_train.max():.1f} "
          f"n_heavy(>=15.6)={int((y_train>=15.6).sum())}")

    # -------------------------------------------------------------------------
    # GRADIENT BOOSTING (primary model)
    # Regularized: max_depth=2, min_samples_leaf=5, subsample=0.8
    # These are conservative settings to limit overfitting on small dataset
    # -------------------------------------------------------------------------
    gb = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=2,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )
    gb.fit(X_train.values, y_train)
    gb_pred = np.clip(gb.predict(X_test.values), 0.0, None)

    # -------------------------------------------------------------------------
    # RANDOM FOREST (secondary model for comparison)
    # -------------------------------------------------------------------------
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=5,
        max_features=0.5,
        random_state=RANDOM_STATE,
    )
    rf.fit(X_train.values, y_train)
    rf_pred = np.clip(rf.predict(X_test.values), 0.0, None)

    # -------------------------------------------------------------------------
    # TRAINING-SET PERFORMANCE (for overfitting check)
    # -------------------------------------------------------------------------
    gb_train_pred = np.clip(gb.predict(X_train.values), 0.0, None)
    rf_train_pred = np.clip(rf.predict(X_train.values), 0.0, None)

    print(f"\n  Training-set metrics (OVERFITTING CHECK -- these are IN-SAMPLE):")
    print(f"  {'Model':<26}  {'MAE_train':>10}  {'MAE_test':>10}  {'Ratio':>6}")
    print(f"  {'-'*26}  {'-'*10}  {'-'*10}  {'-'*6}")
    for name, trp, tep in [
        ("Gradient Boosting", gb_train_pred, gb_pred),
        ("Random Forest",     rf_train_pred, rf_pred),
    ]:
        mtr = mae(y_train, trp)
        mte = mae(obs, tep)
        ratio = mte / mtr if mtr > 0 else float("inf")
        print(f"  {name:<26}  {mtr:>10.2f}  {mte:>10.2f}  {ratio:>6.2f}x")

    print(f"  (Ratio > 1.5 may indicate overfitting on this small dataset)")

    # -------------------------------------------------------------------------
    # BASELINES
    # -------------------------------------------------------------------------
    test_df["equal_blend"] = (test_df["ecmwf"] + test_df["gfs"]) / 2.0

    models = {
        "ECMWF"           : test_df["ecmwf"].values,
        "GFS"             : test_df["gfs"].values,
        "Equal blend"     : test_df["equal_blend"].values,
        "Adaptive (Exp1)" : test_df["adaptive"].values,
        "GradBoost+Atmos" : gb_pred,
        "RandForest+Atmos": rf_pred,
    }

    all_m = print_metrics_block(obs, models, f"{label} metrics", len(test_df))

    for thr in CSI_THRESHOLDS:
        n_ev = int((obs >= thr).sum())
        if n_ev > 0:
            print_csi_detail_table(obs, models, thr)
        else:
            print(f"\n  CSI @ {thr} mm: 0 observed events in this period -- N/A")

    print_heavy_days(test_df, models)

    # -------------------------------------------------------------------------
    # FEATURE IMPORTANCE
    # -------------------------------------------------------------------------
    print(f"\n  Feature importances (Gradient Boosting -- built-in impurity-based):")
    gb_imp = gb.feature_importances_
    sorted_idx = np.argsort(gb_imp)[::-1]
    for i in sorted_idx[:15]:  # top-15
        print(f"    {feature_names[i]:<36}  {gb_imp[i]:.4f}")

    # -------------------------------------------------------------------------
    # FALSE ALARM ANALYSIS
    # -------------------------------------------------------------------------
    dry_mask = obs < 5.0
    n_dry = dry_mask.sum()
    if n_dry > 0:
        print(f"\n  False alarm analysis  ({n_dry} dry observed days, obs < 5 mm):")
        print(f"  {'Model':<26}  {'FA@15.6':>8}  {'FA@35':>7}  {'Mean_fc':>8}")
        print(f"  {'-'*26}  {'-'*8}  {'-'*7}  {'-'*8}")
        for name, fct in models.items():
            fa15 = int((fct[dry_mask] >= 15.6).sum())
            fa35 = int((fct[dry_mask] >= 35.0).sum())
            mfc  = fct[dry_mask].mean()
            print(f"  {name:<26}  {fa15:>8}  {fa35:>7}  {mfc:>8.2f}")

    # Store results
    result = {
        "label"   : label,
        "n_train" : len(train_df),
        "n_test"  : len(test_df),
        "metrics" : all_m,
        "gb_pred" : gb_pred,
        "rf_pred" : rf_pred,
        "test_df" : test_df,
        "feature_names": feature_names,
        "gb_importances": gb_imp,
    }
    return result


# =============================================================================
# 8. RUN BOTH EVALUATION SPLITS
# =============================================================================

res_e1 = run_evaluation(
    full, "2025-08-04", "2025-08-04", "2025-08-31",
    "Exp-1 Structure (Aug 4-31, n=28)"
)

res_e2 = run_evaluation(
    full, "2025-07-01", "2025-07-01", "2025-08-03",
    "Exp-2 Structure (Jul 1-Aug 3, n=34)"
)


# =============================================================================
# 9. CROSS-SPLIT SUMMARY TABLE
# =============================================================================

print(f"\n{SEP}")
print("CROSS-SPLIT SUMMARY  --  Adaptive(Exp1) vs GradBoost+Atmos")
print(SEP)

def nan_better(a, b, higher_is_better):
    if np.isnan(a) or np.isnan(b):
        return None
    return b > a if higher_is_better else b < a

metrics_spec = [
    ("MAE (mm)",   "MAE",   False),
    ("RMSE (mm)",  "RMSE",  False),
    ("CSI @ 15.6", "CSI15", True),
    ("CSI @ 35.0", "CSI35", True),
    ("CSI @ 64.5", "CSI64", True),
]

print(f"\n  {'Metric':<22}  {'E1 Adapt':>10}  {'E1 Atmos':>9}  "
      f"{'E2 Adapt':>10}  {'E2 Atmos':>9}  {'Consistent?':>12}")
print(f"  {'-'*22}  {'-'*10}  {'-'*9}  {'-'*10}  {'-'*9}  {'-'*12}")

for label, key, higher in metrics_spec:
    try:
        e1a = res_e1["metrics"]["Adaptive (Exp1)"][key]
        e1c = res_e1["metrics"]["GradBoost+Atmos"][key]
        e2a = res_e2["metrics"]["Adaptive (Exp1)"][key]
        e2c = res_e2["metrics"]["GradBoost+Atmos"][key]
    except (KeyError, TypeError):
        print(f"  {label:<22}  {'N/A':>10}  {'N/A':>9}  {'N/A':>10}  {'N/A':>9}  {'N/A':>12}")
        continue

    f = lambda v: f"{v:.3f}" if not np.isnan(v) else "  N/A"
    b1 = nan_better(e1a, e1c, higher)
    b2 = nan_better(e2a, e2c, higher)
    if b1 is None or b2 is None:
        cons = "N/A"
    elif b1 and b2:
        cons = "BOTH IMPROVE"
    elif not b1 and not b2:
        cons = "BOTH WORSEN"
    else:
        cons = "MIXED"
    print(f"  {label:<22}  {f(e1a):>10}  {f(e1c):>9}  {f(e2a):>10}  {f(e2c):>9}  {cons:>12}")


# =============================================================================
# 10. HONEST VERDICT
# =============================================================================

print(f"\n{SEP}")
print("VERDICT AND HONEST ASSESSMENT  (Experiment 4)")
print(SEP)

print(f"""
  DATASET SIZE WARNING:
  Exp-1 training: n=64 days  |  ML features: ~{len(res_e1.get('feature_names', [])) if res_e1 else '?'}
  Exp-2 training: n=30 days  |  ML features: ~{len(res_e2.get('feature_names', [])) if res_e2 else '?'}

  A gradient boosting model with ~20 features and 30-64 training samples
  has high overfitting risk. The conservative hyperparameters
  (max_depth=2, min_samples_leaf=5, lr=0.03) mitigate but cannot eliminate
  this risk. The training/test MAE ratio printed above is the primary
  overfitting diagnostic.

  KEY QUESTION: Does CAPE / Lifted Index (GFS-only) help identify Type B
  events (heavy rain on low-precip forecast days)?

  These variables are GFS model output, not observations. They carry
  GFS-specific biases and cannot be verified against observations here.

  GENERALIZATION CAVEAT:
  The test periods (28 days / 34 days) are too small to make strong
  statistical claims. Any result should be reported as hypothesis-generating,
  not as established benchmark performance.

  NO TEST OBSERVATIONS WERE USED FOR FITTING.
  Experiments 1-3 outputs are unchanged.
""")

# Count improvements
n_imp = 0
n_tot = 0
for label, key, higher in metrics_spec:
    try:
        e1a = res_e1["metrics"]["Adaptive (Exp1)"][key]
        e1c = res_e1["metrics"]["GradBoost+Atmos"][key]
        e2a = res_e2["metrics"]["Adaptive (Exp1)"][key]
        e2c = res_e2["metrics"]["GradBoost+Atmos"][key]
    except (KeyError, TypeError):
        continue
    b1 = nan_better(e1a, e1c, higher)
    b2 = nan_better(e2a, e2c, higher)
    if b1 is not None:
        n_imp += int(b1)
        n_tot += 1
    if b2 is not None:
        n_imp += int(b2)
        n_tot += 1

if n_tot > 0:
    ratio = n_imp / n_tot
    if ratio >= 0.75:
        verdict = "PROMISING (but overfitting risk is high -- see training/test ratios)"
    elif ratio >= 0.5:
        verdict = "MIXED"
    else:
        verdict = "NOT IMPROVED by atmospheric context on this dataset"

    print(f"  {'='*65}")
    print(f"  VERDICT: {verdict}")
    print(f"  GradBoost+Atmos improved {n_imp}/{n_tot} metric-period pairs vs Adaptive(Exp1)")
    print(f"  {'='*65}")


# =============================================================================
# 11. SAVE OUTPUT CSV
# =============================================================================

out_frames = []
for res, eval_label in [(res_e1, "Exp1_Aug04_Aug31"), (res_e2, "Exp2_Jul01_Aug03")]:
    if not res:
        continue
    df = res["test_df"][["time", "rain", "ecmwf", "gfs"]].copy()
    df["equal_blend"]      = res["test_df"]["equal_blend"].values
    df["adaptive_exp1"]    = res["test_df"]["adaptive"].values
    df["gradboost_atmos"]  = res["gb_pred"]
    df["randforest_atmos"] = res["rf_pred"]
    df["eval_period"]      = eval_label
    out_frames.append(df)

if out_frames:
    out = pd.concat(out_frames, ignore_index=True)
    out.to_csv(OUTPUT_FILE, index=False, float_format="%.4f")
    print(f"\nSaved: {OUTPUT_FILE}")

print()
print(SEP)
print("DONE. Experiments 1-3 files are untouched.")
print("No test observations used for feature fitting or model training.")
print("Atmospheric data source: Open-Meteo Previous Runs API (same as Exp 1-3).")
print(SEP)
