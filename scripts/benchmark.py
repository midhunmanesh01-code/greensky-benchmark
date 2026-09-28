from pathlib import Path
import pandas as pd
import requests

from sklearn.ensemble import GradientBoostingRegressor

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# =========================
# 1. LOAD IMD OBSERVATIONS
# =========================

df = pd.read_csv(DATA_DIR / "imdweb_hMXvt0uE.csv")
df["time"] = pd.to_datetime(df["time"])

lat = 10.75
lon = 76.25

url = "https://previous-runs-api.open-meteo.com/v1/forecast"


# =========================
# 2. GET MODEL DATA
# =========================

def get_model(model):

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "precipitation_previous_day1",
        "models": model,
        "start_date": "2025-06-01",
        "end_date": "2025-08-31",
        "previous_day1": True
    }

    response = requests.get(url, params=params)

    print(model, "API status:", response.status_code)

    data = response.json()

    return pd.DataFrame({
        "time": pd.to_datetime(data["hourly"]["time"]),
        "rainfall": data["hourly"]["precipitation_previous_day1"]
    })


ecmwf = get_model("ecmwf_ifs025")
gfs = get_model("gfs_seamless")


# =========================
# 3. DAILY AGGREGATION
# =========================

ecmwf["date"] = ecmwf["time"].dt.date
gfs["date"] = gfs["time"].dt.date

ecmwf_daily = (
    ecmwf.groupby("date")["rainfall"]
    .sum()
    .reset_index()
)

gfs_daily = (
    gfs.groupby("date")["rainfall"]
    .sum()
    .reset_index()
)

ecmwf_daily.rename(
    columns={"rainfall": "ecmwf"},
    inplace=True
)

gfs_daily.rename(
    columns={"rainfall": "gfs"},
    inplace=True
)

ecmwf_daily["time"] = pd.to_datetime(ecmwf_daily["date"])
gfs_daily["time"] = pd.to_datetime(gfs_daily["date"])


# =========================
# 4. MERGE DATA
# =========================

result = pd.merge(
    df[["time", "rain"]],
    ecmwf_daily[["time", "ecmwf"]],
    on="time",
    how="inner"
)

result = pd.merge(
    result,
    gfs_daily[["time", "gfs"]],
    on="time",
    how="inner"
)

result["equal_blend"] = (
    result["ecmwf"] + result["gfs"]
) / 2

result = result.sort_values("time").reset_index(drop=True)


print("\nMatched rows:", len(result))


# =========================
# 5. CHRONOLOGICAL SPLIT
# =========================

split_index = int(len(result) * 0.70)

train = result.iloc[:split_index].copy()
test = result.iloc[split_index:].copy()

print("\n========== DATA SPLIT ==========")

print("Training days:", len(train))
print("Test days:", len(test))

print(
    "Training:",
    train["time"].min().date(),
    "to",
    train["time"].max().date()
)

print(
    "Test:",
    test["time"].min().date(),
    "to",
    test["time"].max().date()
)


# =========================
# 6. CREATE ADAPTIVE FEATURES
# =========================

def create_features(data):

    features = pd.DataFrame(index=data.index)

    # Forecast values
    features["ecmwf"] = data["ecmwf"]
    features["gfs"] = data["gfs"]

    # Model disagreement
    features["disagreement"] = (
        data["ecmwf"] - data["gfs"]
    ).abs()

    # Forecast ratio
    features["forecast_ratio"] = (
        data["ecmwf"] /
        (data["gfs"] + 1.0)
    )

    # Month / seasonal context
    features["month"] = data["time"].dt.month

    # Day of month
    features["day"] = data["time"].dt.day

    return features


X_train = create_features(train)
X_test = create_features(test)

y_train = train["rain"]


# =========================
# 7. TRAIN ADAPTIVE AI MODEL
# =========================

print("\n========== TRAINING ADAPTIVE BLENDER ==========")

model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=2,
    random_state=42
)

model.fit(X_train, y_train)

print("Training complete.")


# =========================
# 8. PREDICT UNSEEN TEST
# =========================

test["adaptive_blend"] = model.predict(X_test)

# Rainfall cannot physically be negative
test["adaptive_blend"] = test["adaptive_blend"].clip(lower=0)


# =========================
# 9. METRICS
# =========================

def metrics(data, column):

    mae = (
        data[column] - data["rain"]
    ).abs().mean()

    rmse = (
        ((data[column] - data["rain"]) ** 2).mean()
    ) ** 0.5

    return mae, rmse


print("\n========== UNSEEN TEST RESULTS ==========")

for column in [
    "ecmwf",
    "gfs",
    "equal_blend",
    "adaptive_blend"
]:

    mae, rmse = metrics(test, column)

    print(f"\n{column.upper()}")
    print("MAE :", round(mae, 2), "mm")
    print("RMSE:", round(rmse, 2), "mm")


# =========================
# 10. HEAVY-RAIN CSI
# =========================

threshold = 15.0

print("\n========== HEAVY-RAIN CSI ==========")
print("Threshold:", threshold, "mm/day")


def calculate_csi(data, column):

    observed = data["rain"] >= threshold
    forecast = data[column] >= threshold

    hits = ((observed) & (forecast)).sum()
    misses = ((observed) & (~forecast)).sum()
    false_alarms = ((~observed) & (forecast)).sum()

    denominator = hits + misses + false_alarms

    csi = (
        hits / denominator
        if denominator > 0
        else 0
    )

    return hits, misses, false_alarms, csi


for column in [
    "ecmwf",
    "gfs",
    "equal_blend",
    "adaptive_blend"
]:

    hits, misses, false_alarms, csi = calculate_csi(
        test,
        column
    )

    print(f"\n{column.upper()}")
    print("Hits:", hits)
    print("Misses:", misses)
    print("False alarms:", false_alarms)
    print("CSI:", round(csi, 3))


# =========================
# 11. SAVE FINAL RESULTS
# =========================

test.to_csv(
    RESULTS_DIR / "adaptive_benchmark_results.csv",
    index=False
)

print(f"\nSaved: {RESULTS_DIR / 'adaptive_benchmark_results.csv'}")