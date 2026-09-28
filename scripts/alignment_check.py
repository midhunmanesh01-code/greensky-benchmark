import pandas as pd

# =========================
# LOAD IMD DATA
# =========================

df = pd.read_csv("imdweb_hMXvt0uE.csv")

df["time"] = pd.to_datetime(df["time"])

# Select our exact IMD grid cell
imd = df[
    (df["rain_cell_lat"] == 10.75) &
    (df["rain_cell_lon"] == 76.25)
].copy()

imd = imd.sort_values("time")

# =========================
# CHECK IMD DAILY DATA
# =========================

print("========== IMD DATA ==========")

print("Rows:", len(imd))

print("\nFirst timestamps:")
print(imd["time"].head(10).to_string(index=False))

print("\nLast timestamps:")
print(imd["time"].tail(10).to_string(index=False))

print("\nDate range:")
print(imd["time"].min())
print("to")
print(imd["time"].max())

# =========================
# CHECK TIME DIFFERENCES
# =========================

time_diff = imd["time"].diff().dropna()

print("\n========== IMD TIME INTERVALS ==========")

print(
    time_diff.value_counts().head(10)
)

# =========================
# CHECK DUPLICATES
# =========================

print("\nDuplicate timestamps:", imd["time"].duplicated().sum())

# =========================
# SHOW OBSERVATIONS
# =========================

print("\n========== FIRST 20 IMD OBSERVATIONS ==========")

print(
    imd[
        [
            "time",
            "rain",
            "rain_cell_lat",
            "rain_cell_lon"
        ]
    ].head(20).to_string(index=False)
)