from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

# =========================
# 1. LOAD DATA
# =========================

data = pd.read_csv(RESULTS_DIR / "benchmark_results.csv")
data["time"] = pd.to_datetime(data["time"])

data = data.sort_values("time").reset_index(drop=True)

print("Total rows:", len(data))


# =========================
# 2. LOCK CHRONOLOGICAL TEST
# =========================

split_index = int(len(data) * 0.70)

train = data.iloc[:split_index].copy()
test = data.iloc[split_index:].copy()

print("\n========== SPLIT ==========")

print("Training:", len(train))
print("Test:", len(test))

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
# 3. DETERMINE BEST MODEL
#    FOR EACH TRAINING DAY
# =========================

train["ecmwf_error"] = (
    train["ecmwf"] - train["rain"]
).abs()

train["gfs_error"] = (
    train["gfs"] - train["rain"]
).abs()

# 1 = ECMWF better
# 0 = GFS better

train["best_model"] = (
    train["ecmwf_error"] <= train["gfs_error"]
).astype(int)


print("\n========== TRAINING MODEL CHOICES ==========")

print(
    train["best_model"]
    .value_counts()
    .rename({
        1: "ECMWF",
        0: "GFS"
    })
)


# =========================
# 4. CREATE CONTEXT FEATURES
# =========================

def create_features(data):

    features = pd.DataFrame(index=data.index)

    features["ecmwf"] = data["ecmwf"]

    features["gfs"] = data["gfs"]

    features["disagreement"] = (
        data["ecmwf"] - data["gfs"]
    ).abs()

    features["difference"] = (
        data["ecmwf"] - data["gfs"]
    )

    features["month"] = (
        data["time"].dt.month
    )

    return features


X_train = create_features(train)
X_test = create_features(test)

y_train = train["best_model"]


# =========================
# 5. TRAIN GATING MODEL
# =========================

print("\n========== TRAINING GATING MODEL ==========")

gate = DecisionTreeClassifier(
    max_depth=2,
    random_state=42
)

gate.fit(
    X_train,
    y_train
)

print("Training complete.")


# =========================
# 6. PREDICT WHICH MODEL
# =========================

test["selected_model"] = gate.predict(X_test)


# =========================
# 7. CREATE GATED FORECAST
# =========================

test["gating_blend"] = np.where(
    test["selected_model"] == 1,
    test["ecmwf"],
    test["gfs"]
)


# =========================
# 8. SHOW MODEL SELECTION
# =========================

print("\n========== GATING DECISIONS ==========")

display_columns = [
    "time",
    "rain",
    "ecmwf",
    "gfs",
    "selected_model",
    "gating_blend"
]

print(
    test[display_columns].to_string(
        index=False
    )
)


# =========================
# 9. METRICS
# =========================

def calculate_metrics(data, column):

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
    "gating_blend"
]:

    mae, rmse = calculate_metrics(
        test,
        column
    )

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

    hits = (
        observed & forecast
    ).sum()

    misses = (
        observed & ~forecast
    ).sum()

    false_alarms = (
        ~observed & forecast
    ).sum()

    denominator = (
        hits +
        misses +
        false_alarms
    )

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
    "gating_blend"
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
# 11. SAVE RESULTS
# =========================

test.to_csv(
    RESULTS_DIR / "gating_benchmark_results.csv",
    index=False
)

print(
    f"\nSaved: {RESULTS_DIR / 'gating_benchmark_results.csv'}"
)