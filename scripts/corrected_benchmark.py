# -*- coding: utf-8 -*-
"""
corrected_benchmark.py
======================
GreenSky Blend – SIH26081
Corrected Data-Alignment Benchmark (baseline models only)

METHODOLOGY CORRECTION
----------------------
IMD gridded daily rainfall is measured for the 24-hour period ending at
08:30 IST = 03:00 UTC.  Therefore the correct accumulation window for
IMD date D is:

    D 03:00 UTC  →  (D+1) 03:00 UTC   [WRONG — Convention A, now corrected]

CORRECTED WINDOW (Convention B — standard IMD definition):
    IMD date D = rainfall accumulated from (D-1) 03:00 UTC → D 03:00 UTC

The date label on an IMD row is the END of the accumulation period, not the
start.  Official source: Pai et al. (2014) and IMD API docs state
"past 24 hrs rainfall = 08:30 IST (D-1) to 08:30 IST (D)" = 03:00 UTC (D-1)
to 03:00 UTC (D).

The original benchmark.py used UTC-calendar-day grouping
(D 00:00 UTC → D 23:00 UTC, i.e. dt.date), which shifts the accumulation
window by 3 hours relative to the IMD observation.

This script:
  1. Fetches the same Open-Meteo hourly data as the original.
  2. Aggregates over the corrected 03:00–03:00 UTC windows.
  3. Flags every IMD date where the window does not contain exactly 24
     hourly values (data gap / model dropout).
  4. Reports baseline metrics (ECMWF, GFS, equal-weight blend).
  5. Prints a side-by-side comparison against the original UTC-day results.
  6. Saves corrected_benchmark_results.csv and corrected_diagnostic.csv.
  7. Does NOT train any adaptive / ML model.

IMPORTANT CAVEATS ABOUT precipitation_previous_day1
----------------------------------------------------
Open-Meteo serves ECMWF IFS at 3-hourly native resolution and repeats
each 3-hour accumulation value across 3 consecutive hourly slots
(e.g. hours 00, 01, 02 all carry the same mm value which represents
the accumulation for that 3-hour bucket).

This means:
  * Summing 24 hourly rows TRIPLE-COUNTS every native 3-hour bucket.
  * The corrected window still sums 24 rows, so both old and new scripts
    triple-count consistently within their own window – the comparison
    between them is therefore fair on a relative basis.
  * The ABSOLUTE totals are 3x the true model accumulation.

This script detects the repeat pattern, warns clearly, and outputs
BOTH the raw sum (24-row) and a de-duplicated sum (dividing by 3 only
when the triplet pattern is confirmed) so the comparison table is
unambiguous.

NOTE on GFS: GFS outputs at 1-hourly or 3-hourly depending on the run
and lead time. We check empirically whether each model's values repeat
in triplets and report the finding.
"""

import sys
import requests
import pandas as pd
import numpy as np
from datetime import timedelta

# Ensure UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
sys.stdout.reconfigure(encoding="utf-8")


# =============================================================================
# CONFIGURATION
# =============================================================================

LAT = 10.75
LON = 76.25

# Open-Meteo returns data in UTC (utc_offset_seconds = 0)
API_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

# IMD measurement period: 2025-06-01 through 2025-08-31 (92 days)
IMD_START = pd.Timestamp("2025-06-01")
IMD_END   = pd.Timestamp("2025-08-31")

# Convention B: IMD date D window = (D-1) 03:00 UTC to D 03:00 UTC.
# First IMD date = 2025-06-01, so first window starts 2025-05-31 03:00 UTC.
# Last IMD date  = 2025-08-31, so last window ends   2025-08-31 03:00 UTC.
# We must request from 2025-05-31 (one day before IMD_START).
# We do NOT need September data — the last window closes within Aug 31.
API_START_DATE = "2025-05-31"   # one day before IMD_START
API_END_DATE   = "2025-08-31"   # last IMD window ends at 2025-08-31 03:00 UTC

# IMD accumulation window offset (hours past midnight UTC)
IMD_WINDOW_OFFSET_H = 3   # 03:00 UTC = 08:30 IST ≈ gauge read time

# Locked test split (last 28 days, matching original benchmark)
LOCKED_TEST_START = pd.Timestamp("2025-08-04")

# CSI thresholds (mm/day) – as per SIH26081 specification
CSI_THRESHOLDS = [15.6, 64.5]


# =============================================================================
# 1. LOAD IMD OBSERVATIONS
# =============================================================================

print("=" * 60)
print("CORRECTED BENCHMARK  –  GreenSky Blend SIH26081")
print("=" * 60)

imd_raw = pd.read_csv("imdweb_hMXvt0uE.csv")
imd_raw["time"] = pd.to_datetime(imd_raw["time"])

imd = imd_raw[
    (imd_raw["rain_cell_lat"] == 10.75) &
    (imd_raw["rain_cell_lon"] == 76.25)
].copy()

imd = imd.sort_values("time").reset_index(drop=True)

print(f"\nIMD observations loaded : {len(imd)} days")
print(f"IMD date range          : {imd['time'].min().date()} to {imd['time'].max().date()}")


# =============================================================================
# 2. FETCH HOURLY FORECAST DATA
# =============================================================================

def fetch_hourly(model_name: str) -> pd.DataFrame:
    """
    Fetch precipitation_previous_day1 for the given model.

    Returns a DataFrame with columns [time, rainfall] where 'time' is UTC
    and 'rainfall' is the hourly precipitation value in mm.
    """
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "precipitation_previous_day1",
        "models": model_name,
        "start_date": API_START_DATE,
        "end_date": API_END_DATE,
        "previous_day1": True,
    }

    print(f"\nFetching {model_name} from Open-Meteo ...")
    response = requests.get(API_URL, params=params, timeout=60)
    print(f"  HTTP status : {response.status_code}")

    if response.status_code != 200:
        print(f"  ERROR: non-200 response for {model_name}. Aborting.")
        sys.exit(1)

    data = response.json()
    df = pd.DataFrame({
        "time": pd.to_datetime(data["hourly"]["time"]),
        "rainfall": data["hourly"]["precipitation_previous_day1"],
    })

    print(f"  Rows received : {len(df)}")
    print(f"  Time range    : {df['time'].min()} to {df['time'].max()}")

    return df


ecmwf_h = fetch_hourly("ecmwf_ifs025")
gfs_h   = fetch_hourly("gfs_seamless")


# =============================================================================
# 3. DETECT TEMPORAL REPEAT PATTERN
#    (Are hourly values actually 3-hourly buckets repeated in triplets?)
# =============================================================================

def detect_triplet_pattern(df: pd.DataFrame, model_name: str) -> bool:
    """
    Check whether consecutive hourly rows repeat values in groups of 3.
    This happens when Open-Meteo interpolates 3-hourly native output to
    hourly by repeating each bucket value 3 times.

    Returns True if >=90 % of hour-triplets are identical, False otherwise.
    """
    vals = df["rainfall"].dropna().values
    # Compare each value with the one 2 steps ahead (same triplet bucket)
    triplet_equal_count = np.sum(vals[:-2] == vals[2:])
    total_pairs = len(vals) - 2

    frac = triplet_equal_count / total_pairs if total_pairs > 0 else 0.0

    print(f"\n  [{model_name}] Triplet-repeat fraction : {frac:.3f}")

    if frac >= 0.90:
        print(f"  [!] WARNING: {model_name} hourly values repeat in triplets.")
        print(f"     Each hourly slot carries the 3-hour bucket accumulation.")
        print(f"     Summing 24 hourly rows gives 3x the true daily total.")
        print(f"     We apply a /3 correction to recover physical mm totals.")
        return True
    else:
        print(f"  [OK] {model_name} appears to have genuine hourly resolution.")
        return False


print("\n" + "=" * 60)
print("TEMPORAL RESOLUTION CHECK")
print("=" * 60)

ecmwf_triplet = detect_triplet_pattern(ecmwf_h, "ECMWF IFS")
gfs_triplet   = detect_triplet_pattern(gfs_h,   "GFS")


# =============================================================================
# 4. CORRECTED DAILY AGGREGATION  (Convention B)
#    Window: (D-1) 03:00 UTC → D 03:00 UTC  (24 hourly rows)
#    IMD date D is the END of the accumulation, not the start.
# =============================================================================

def aggregate_corrected(hourly_df: pd.DataFrame,
                         imd_dates: pd.Series,
                         is_triplet: bool,
                         model_name: str) -> pd.DataFrame:
    """
    For each IMD date D, sum hourly rainfall from (D-1) 03:00 UTC (inclusive)
    to D 03:00 UTC (exclusive) — exactly 24 rows.

    Convention B: the date label is the END of the accumulation period.
    Source: IMD/Pai et al. — "past 24 hrs rainfall = 08:30 IST (D-1) to
    08:30 IST (D)" = 03:00 UTC (D-1) to 03:00 UTC (D).

    If is_triplet is True, the raw sum is divided by 3 to obtain the
    physically correct accumulation in mm.

    Returns a DataFrame with columns:
        imd_date, window_start, window_end, n_hours,
        raw_sum, corrected_total
    """
    rows = []

    for imd_date in imd_dates:
        # Convention B: start = previous day at 03:00 UTC, end = this day at 03:00 UTC
        win_start = (imd_date - timedelta(days=1)) + timedelta(hours=IMD_WINDOW_OFFSET_H)
        win_end   = imd_date + timedelta(hours=IMD_WINDOW_OFFSET_H)

        mask = (hourly_df["time"] >= win_start) & (hourly_df["time"] < win_end)
        window_data = hourly_df.loc[mask, "rainfall"]

        n_hours  = mask.sum()
        raw_sum  = window_data.sum()
        corrected = raw_sum / 3.0 if is_triplet else raw_sum

        rows.append({
            "imd_date"        : imd_date.date(),
            "window_start"    : win_start,
            "window_end"      : win_end,
            "n_hours"         : n_hours,
            "raw_sum"         : round(raw_sum, 4),
            "corrected_total" : round(corrected, 4),
        })

    result = pd.DataFrame(rows)

    # Flag incomplete windows
    incomplete = result["n_hours"] != 24
    if incomplete.any():
        print(f"\n  [!] WARNING [{model_name}]: {incomplete.sum()} IMD date(s) have "
              f"!= 24 hourly rows in the corrected window:")
        print(result[incomplete][["imd_date", "n_hours"]].to_string(index=False))
    else:
        print(f"\n  [OK] [{model_name}] All {len(result)} IMD dates have exactly "
              f"24 hourly rows in the corrected window.")

    return result


print("\n" + "=" * 60)
print("CORRECTED WINDOW AGGREGATION  (D-1) 03:00 UTC -> D 03:00 UTC  [Convention B]")
print("=" * 60)

imd_dates = imd["time"]

ecmwf_corr = aggregate_corrected(ecmwf_h, imd_dates, ecmwf_triplet, "ECMWF IFS")
gfs_corr   = aggregate_corrected(gfs_h,   imd_dates, gfs_triplet,   "GFS")


# =============================================================================
# 5. OLD UTC-CALENDAR-DAY AGGREGATION (for comparison)
#    Reproduces original benchmark.py behaviour (dt.date groupby)
#    Limited to the original date range 2025-06-01 to 2025-08-31
# =============================================================================

def aggregate_old_utc(hourly_df: pd.DataFrame,
                       imd_dates: pd.Series,
                       is_triplet: bool) -> pd.DataFrame:
    """
    Reproduce the original groupby(date).sum() approach.
    Window: D 00:00 UTC → D 23:00 UTC (24 rows, but offset from IMD by 3 h).
    """
    hourly_df = hourly_df.copy()
    hourly_df["date"] = hourly_df["time"].dt.date
    daily = hourly_df.groupby("date")["rainfall"].sum().reset_index()
    daily.rename(columns={"rainfall": "raw_sum"}, inplace=True)

    # Filter to the same IMD date set
    imd_date_set = set(d.date() for d in imd_dates)
    daily = daily[daily["date"].isin(imd_date_set)].copy()

    daily["corrected_total"] = daily["raw_sum"] / 3.0 if is_triplet else daily["raw_sum"]
    daily["corrected_total"] = daily["corrected_total"].round(4)

    return daily.sort_values("date").reset_index(drop=True)


ecmwf_old = aggregate_old_utc(ecmwf_h, imd_dates, ecmwf_triplet)
gfs_old   = aggregate_old_utc(gfs_h,   imd_dates, gfs_triplet)


# =============================================================================
# 6. MERGE INTO A SINGLE ALIGNED DATAFRAME
# =============================================================================

# --- Corrected dataset ---
merged_corr = imd[["time", "rain"]].copy()
merged_corr["imd_date"] = merged_corr["time"].dt.date

ecmwf_corr_lookup = ecmwf_corr.set_index("imd_date")["corrected_total"]
gfs_corr_lookup   = gfs_corr.set_index("imd_date")["corrected_total"]

merged_corr["ecmwf"]       = merged_corr["imd_date"].map(ecmwf_corr_lookup)
merged_corr["gfs"]         = merged_corr["imd_date"].map(gfs_corr_lookup)
merged_corr["equal_blend"] = (merged_corr["ecmwf"] + merged_corr["gfs"]) / 2.0
merged_corr = merged_corr.dropna(subset=["ecmwf", "gfs"]).reset_index(drop=True)

# --- Old dataset (load from existing benchmark_results.csv for reference) ---
old_results = pd.read_csv("benchmark_results.csv")
old_results["time"] = pd.to_datetime(old_results["time"])
old_results = old_results.sort_values("time").reset_index(drop=True)

print(f"\nCorrected dataset rows : {len(merged_corr)}")
print(f"Old (UTC-day) rows     : {len(old_results)}")


# =============================================================================
# 7. CHRONOLOGICAL TEST SPLIT
#    Locked to match original: test = 2025-08-04 through 2025-08-31
# =============================================================================

train_corr = merged_corr[merged_corr["time"] < LOCKED_TEST_START].copy()
test_corr  = merged_corr[merged_corr["time"] >= LOCKED_TEST_START].copy()

train_old  = old_results[old_results["time"] < LOCKED_TEST_START].copy()
test_old   = old_results[old_results["time"] >= LOCKED_TEST_START].copy()

print(f"\nCorrected  train: {len(train_corr)} days  "
      f"({train_corr['time'].min().date()} → {train_corr['time'].max().date()})")
print(f"Corrected  test : {len(test_corr)} days  "
      f"({test_corr['time'].min().date()} → {test_corr['time'].max().date()})")
print(f"Old (UTC)  train: {len(train_old)} days")
print(f"Old (UTC)  test : {len(test_old)} days")


# =============================================================================
# 8. METRIC FUNCTIONS
# =============================================================================

def mae(obs, fct):
    return float(np.abs(fct - obs).mean())

def rmse(obs, fct):
    return float(np.sqrt(((fct - obs) ** 2).mean()))

def csi(obs, fct, threshold):
    obs_evt = obs >= threshold
    fct_evt = fct >= threshold
    hits        = int((obs_evt & fct_evt).sum())
    misses      = int((obs_evt & ~fct_evt).sum())
    false_alarms= int((~obs_evt & fct_evt).sum())
    denom = hits + misses + false_alarms
    csi_val = hits / denom if denom > 0 else float("nan")
    return hits, misses, false_alarms, csi_val

def event_counts(obs, threshold):
    return int((obs >= threshold).sum())


# =============================================================================
# 9. COMPUTE AND DISPLAY METRICS
# =============================================================================

MODELS_CORR = ["ecmwf", "gfs", "equal_blend"]
MODELS_OLD  = ["ecmwf", "gfs", "equal_blend"]


def print_metrics_block(df, label, models):
    obs = df["rain"].values
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"  n = {len(df)} days  "
          f"({df['time'].min().date()} → {df['time'].max().date()})")
    print(f"{'─'*60}")
    print(f"  {'Model':<18}  {'MAE':>7}  {'RMSE':>7}  "
          f"{'CSI@15.6':>9}  {'CSI@64.5':>9}")
    print(f"  {'-'*18}  {'-'*7}  {'-'*7}  {'-'*9}  {'-'*9}")

    rows_out = []
    for col in models:
        fct  = df[col].values
        m    = mae(obs, fct)
        r    = rmse(obs, fct)
        _, _, _, c1 = csi(obs, fct, 15.6)
        _, _, _, c2 = csi(obs, fct, 64.5)
        c1_str = f"{c1:.3f}" if not np.isnan(c1) else "  N/A"
        c2_str = f"{c2:.3f}" if not np.isnan(c2) else "  N/A"
        print(f"  {col:<18}  {m:>7.2f}  {r:>7.2f}  {c1_str:>9}  {c2_str:>9}")
        rows_out.append({"model": col, "MAE": round(m, 4), "RMSE": round(r, 4),
                          "CSI_15.6": round(c1, 4) if not np.isnan(c1) else None,
                          "CSI_64.5": round(c2, 4) if not np.isnan(c2) else None})
    return rows_out


def print_event_table(df, label):
    obs = df["rain"].values
    print(f"\n  {label} — Event Counts")
    print(f"  {'Threshold':<14}  {'Observed':>9}  ", end="")
    models = [c for c in ["ecmwf", "gfs", "equal_blend"] if c in df.columns]
    for col in models:
        print(f"  {col:>12}", end="")
    print()
    for thr in CSI_THRESHOLDS:
        n_obs = event_counts(obs, thr)
        print(f"  {'>= '+str(thr)+' mm':<14}  {n_obs:>9}", end="")
        for col in models:
            fct = df[col].values
            hits, misses, fa, csi_val = csi(obs, fct, thr)
            print(f"  H={hits:>2} M={misses:>2} FA={fa:>2}", end="")
        print()


print("\n" + "=" * 60)
print("BASELINE METRICS — UNSEEN TEST SET (2025-08-04 to 2025-08-31)")
print("=" * 60)

corr_metrics = print_metrics_block(
    test_corr, "CORRECTED  (D-1) 03:00 UTC -> D 03:00 UTC  [Convention B]", MODELS_CORR)

old_metrics = print_metrics_block(
    test_old, "ORIGINAL   (D 00:00 UTC → D 23:00 UTC)", MODELS_OLD)

print_event_table(test_corr, "CORRECTED")
print_event_table(test_old,  "ORIGINAL ")


# =============================================================================
# 10. SIDE-BY-SIDE COMPARISON (CORRECTED vs ORIGINAL)
# =============================================================================

print("\n" + "=" * 60)
print("SIDE-BY-SIDE COMPARISON  (Corrected vs Original)")
print("=" * 60)
print(f"\n  {'Model':<18}  {'MAE_corr':>9}  {'MAE_old':>9}  "
      f"{'ΔMAE':>7}  {'RMSE_corr':>10}  {'RMSE_old':>9}  {'ΔRMSE':>7}")
print(f"  {'-'*18}  {'-'*9}  {'-'*9}  {'-'*7}  {'-'*10}  {'-'*9}  {'-'*7}")

for c_row, o_row in zip(corr_metrics, old_metrics):
    dm  = c_row["MAE"]  - o_row["MAE"]
    dr  = c_row["RMSE"] - o_row["RMSE"]
    sign_m = "+" if dm >= 0 else ""
    sign_r = "+" if dr >= 0 else ""
    print(f"  {c_row['model']:<18}  "
          f"{c_row['MAE']:>9.2f}  {o_row['MAE']:>9.2f}  {sign_m}{dm:>6.2f}  "
          f"{c_row['RMSE']:>10.2f}  {o_row['RMSE']:>9.2f}  {sign_r}{dr:>6.2f}")


# =============================================================================
# 11. FULL TRAINING + TEST METRICS (for completeness)
# =============================================================================

print("\n" + "=" * 60)
print("FULL PERIOD METRICS  (Train + Test combined, 92 days)")
print("=" * 60)

print_metrics_block(
    merged_corr, "CORRECTED (all 92 days)", MODELS_CORR)

print_metrics_block(
    old_results, "ORIGINAL  (all 92 days)", MODELS_OLD)


# =============================================================================
# 12. SAVE corrected_benchmark_results.csv
# =============================================================================

output_cols = ["time", "rain", "ecmwf", "gfs", "equal_blend"]
merged_corr[output_cols].to_csv(
    "corrected_benchmark_results.csv",
    index=False,
    float_format="%.4f"
)
print("\nSaved: corrected_benchmark_results.csv")


# =============================================================================
# 13. SAVE corrected_diagnostic.csv
#     One row per IMD date showing window bounds, n_hours, both model
#     totals, and the observed IMD rain.
# =============================================================================

diag_ecmwf = ecmwf_corr[["imd_date", "window_start", "window_end",
                            "n_hours", "corrected_total"]].rename(
    columns={"corrected_total": "ecmwf_total"})

diag_gfs = gfs_corr[["imd_date", "corrected_total"]].rename(
    columns={"corrected_total": "gfs_total"})

diag_imd = imd[["time", "rain"]].copy()
diag_imd["imd_date"] = diag_imd["time"].dt.date

diagnostic = (
    diag_ecmwf
    .merge(diag_gfs, on="imd_date", how="inner")
    .merge(diag_imd[["imd_date", "rain"]], on="imd_date", how="left")
    .sort_values("imd_date")
    .reset_index(drop=True)
)

diagnostic.to_csv(
    "corrected_diagnostic.csv",
    index=False,
    float_format="%.4f"
)

print("Saved: corrected_diagnostic.csv")

print("\n" + "=" * 60)
print("DIAGNOSTIC PREVIEW  (first 10 rows)")
print("=" * 60)
print(diagnostic.head(10).to_string(index=False))

print("\n" + "=" * 60)
print("DONE.  No adaptive model was trained in this script.")
print("Alignment correction complete.  Review findings before next ML step.")
print("=" * 60)
