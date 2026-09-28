import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor


# =========================
# 1. LOAD EXISTING DATA
# =========================

data = pd.read_csv("benchmark_results.csv")
data["time"] = pd.to_datetime(data["time"])

data = data.sort_values("time").reset_index(drop=True)

print("Total rows:", len(data))


# =========================
# 2. CHRONOLOGICAL SPLIT
# =========================

split_index = int(len(data) * 0.70)

train = data.iloc[:split_index].copy()
test = data.iloc[split_index:].copy()

print("\n========== SPLIT ==========")

print("Training:", len(train), "days")
print("Test:", len(test), "days")

print(
    "Training period:",
    train["time"].min().date(),
    "to",
    train["time"].max().date()
)

print(
    "Test period:",
    test["time"].min().date(),
    "to",
    test["time"].max().date()
)


# =========================
# 3. FIND OPTIMAL TRAINING WEIGHT
# =========================
#
# For every training day:
#
# forecast = w * ECMWF + (1-w) * GFS
#
# We find the weight that would have
# minimized the error on that training day.
#
# These weights are TARGETS for the
# gating model.
# =========================

def optimal_weight(ecmwf, gfs, actual):

    difference = ecmwf - gfs

    if abs(difference) < 1e-8:
        return 0.5

    w = (actual - gfs) / difference

    # Keep weight between 0 and 1
    return np.clip(w, 0.0, 1.0)


train["optimal_weight"] = train.apply(
    lambda row: optimal_weight(
        row["ecmwf"],
        row["gfs"],
        row["rain"]
    ),
    axis=1
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

    features["month"] = data["time"].dt.month

    return features


X_train = create_features(train)
X_test = create_features(test)

y_train = train["optimal_weight"]


# =========================
# 5. TRAIN WEIGHT MODEL
# =========================

print("\n========== TRAINING WEIGHT MODEL ==========")

model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=2,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

print("Training complete.")


# =========================
# 6. PREDICT ADAPTIVE WEIGHTS
# =========================

test["ecmwf_weight"] = model.predict(X_test)

# Enforce valid blending weights
test["ecmwf_weight"] = test["ecmwf_weight"].clip(
    0.0,
    1.0
)

test["gfs_weight"] = (
    1.0 - test["ecmwf_weight"]
)


# =========================
# 7. CREATE ADAPTIVE BLEND
# =========================

test["adaptive_blend"] = (
    test["ecmwf_weight"] * test["ecmwf"]
    +
    test["gfs_weight"] * test["gfs"]
)


# =========================
# 8. METRICS
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
    "adaptive_blend"
]:

    mae, rmse = calculate_metrics(
        test,
        column
    )

    print(f"\n{column.upper()}")
    print("MAE :", round(mae, 2), "mm")
    print("RMSE:", round(rmse, 2), "mm")


# =========================
# 9. HEAVY-RAIN CSI
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
# 10. SHOW LEARNED WEIGHTS
# =========================

print("\n========== ADAPTIVE WEIGHTS ==========")

print(
    test[
        [
            "time",
            "ecmwf",
            "gfs",
            "ecmwf_weight",
            "gfs_weight",
            "adaptive_blend"
        ]
    ].to_string(index=False)
)


# =========================
# 11. SAVE RESULTS
# =========================

test.to_csv(
    "adaptive_weight_results.csv",
    index=False
)

print(
    "\nSaved: adaptive_weight_results.csv"
)