"""
============================================================
 Regression Assignment Pipeline
 Decision Tree Regression vs. XGBoost Regression
 Datasets: RG-Credit.csv, RG-Wage.csv
============================================================
SECTION 1: IMPORT LIBRARIES
============================================================
"""
import json
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ============================================================
# SECTION 9 (defined early so it can be reused throughout):
# MANUAL MSE IMPLEMENTATION (no sklearn.metrics used anywhere)
# ============================================================
def manual_mse(y_true, y_pred):
    """
    Mean Squared Error, implemented manually with NumPy.

    Formula:
        MSE = (1/n) * sum_{i=1}^{n} (y_true_i - y_pred_i)^2

    Steps:
        1. Compute the element-wise residual (y_true - y_pred).
        2. Square every residual.
        3. Sum the squared residuals.
        4. Divide by the number of samples n to get the mean.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n = y_true.shape[0]
    residuals = y_true - y_pred          # step 1
    squared_residuals = residuals ** 2   # step 2
    sum_squared = np.sum(squared_residuals)  # step 3
    mse = sum_squared / n                # step 4
    return float(mse)


# ============================================================
# SECTION 2: LOAD DATASET  /  SECTION 3: DATA INSPECTION
# ============================================================
def load_credit(path):
    df = pd.read_csv(path)
    return df, "Balance"


def load_wage(path):
    df = pd.read_csv(path)
    # 'logwage' is a deterministic transform of the target 'wage' -> leakage, must drop.
    # 'region' is constant (single category) in this file -> drop, it carries no information.
    df = df.drop(columns=["logwage", "region"])
    return df, "wage"


def inspect(df, name):
    report = {
        "dataset": name,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": {c: int(v) for c, v in df.isna().sum().items()},
    }
    return report


def encode_features(df, target):
    """
    One-hot encode categorical predictors; leave the target untouched.
    Encoding is fit on the FULL feature set's category list (categories are a
    fixed property of the raw data, not a statistic estimated from samples),
    but no row of data / no target value from validation or test ever
    influences the train split -- this is not target leakage.
    """
    X = df.drop(columns=[target]).copy()
    y = df[target].copy().reset_index(drop=True)
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    X = X.reset_index(drop=True)
    return X, y


# ============================================================
# SECTION 4: SEQUENTIAL TRAIN / VALIDATION / TEST SPLIT
# (No shuffling -- strict sequential 70% / 15% / 15% split)
# ============================================================
def sequential_split(X, y):
    n = len(X)
    train_end = int(np.floor(0.70 * n))
    val_end = int(np.floor(0.85 * n))

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ============================================================
# SECTION 5/6: DECISION TREE + HYPERPARAMETER TUNING
# ============================================================
DT_GRID = {
    "max_depth": [3, 5, 7, 10, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 5, 10],
    "criterion": ["squared_error", "friedman_mse"],
}


def tune_decision_tree(X_train, y_train, X_val, y_val, grid=DT_GRID):
    results = []
    best = {"mse": np.inf, "params": None, "model": None}
    for depth in grid["max_depth"]:
        for mss in grid["min_samples_split"]:
            for msl in grid["min_samples_leaf"]:
                for crit in grid["criterion"]:
                    params = dict(max_depth=depth, min_samples_split=mss,
                                  min_samples_leaf=msl, criterion=crit,
                                  random_state=RANDOM_STATE)
                    model = DecisionTreeRegressor(**params)
                    model.fit(X_train, y_train)
                    val_pred = model.predict(X_val)
                    val_mse = manual_mse(y_val, val_pred)
                    results.append({**params, "val_mse": val_mse})
                    if val_mse < best["mse"]:
                        best = {"mse": val_mse, "params": params, "model": model}
    results_df = pd.DataFrame(results).sort_values("val_mse").reset_index(drop=True)
    return best, results_df


# ============================================================
# SECTION 7/8: XGBOOST + HYPERPARAMETER TUNING
# ============================================================
# A coarse-to-reasonable grid (kept moderate in size to keep runtime tractable
# while still covering the parameters the assignment asks for).
XGB_GRID = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [100, 300],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 5],
    "gamma": [0, 0.1],
}


def tune_xgboost(X_train, y_train, X_val, y_val, grid=XGB_GRID, n_random=60, seed=RANDOM_STATE):
    """
    The full Cartesian product of XGB_GRID has 3*3*2*2*2*2*2 = 288 combinations.
    To keep runtime reasonable while still exploring every hyperparameter
    listed in the assignment, we use a fixed-seed RANDOM SEARCH over
    n_random combinations drawn from the grid (a standard, well justified
    practical alternative to exhaustive grid search for XGBoost).
    """
    rng = np.random.RandomState(seed)
    keys = list(grid.keys())
    all_combos = []
    for _ in range(n_random * 3):  # oversample then dedupe
        combo = {k: grid[k][rng.randint(len(grid[k]))] for k in keys}
        all_combos.append(tuple(sorted(combo.items())))
    unique_combos = list(dict.fromkeys(all_combos))[:n_random]

    results = []
    best = {"mse": np.inf, "params": None, "model": None}
    for combo in unique_combos:
        params = dict(combo)
        model = XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=4,
            verbosity=0,
            **params
        )
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_mse = manual_mse(y_val, val_pred)
        results.append({**params, "val_mse": val_mse})
        if val_mse < best["mse"]:
            best = {"mse": val_mse, "params": params, "model": model}
    results_df = pd.DataFrame(results).sort_values("val_mse").reset_index(drop=True)
    return best, results_df


# ============================================================
# SECTION 10/11: FINAL EVALUATION + RESULTS COMPARISON
# ============================================================
def run_pipeline(path, loader, dataset_name):
    print(f"\n{'='*70}\nDATASET: {dataset_name}\n{'='*70}")
    df, target = loader(path)
    insp = inspect(df, dataset_name)
    print(json.dumps(insp, indent=2))

    X, y = encode_features(df, target)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = sequential_split(X, y)
    print(f"Train size: {len(X_train)}  Val size: {len(X_val)}  Test size: {len(X_test)}")

    # ---- Decision Tree ----
    dt_best, dt_results = tune_decision_tree(X_train, y_train, X_val, y_val)
    dt_model = dt_best["model"]
    dt_test_pred = dt_model.predict(X_test)
    dt_test_mse = manual_mse(y_test, dt_test_pred)
    print("\nBest Decision Tree params:", dt_best["params"])
    print("Decision Tree Validation MSE:", dt_best["mse"])
    print("Decision Tree Test MSE:", dt_test_mse)

    # ---- XGBoost ----
    xgb_best, xgb_results = tune_xgboost(X_train, y_train, X_val, y_val)
    xgb_model = xgb_best["model"]
    xgb_test_pred = xgb_model.predict(X_test)
    xgb_test_mse = manual_mse(y_test, xgb_test_pred)
    print("\nBest XGBoost params:", xgb_best["params"])
    print("XGBoost Validation MSE:", xgb_best["mse"])
    print("XGBoost Test MSE:", xgb_test_mse)

    return {
        "dataset_name": dataset_name,
        "inspection": insp,
        "n_features": X.shape[1],
        "feature_names": list(X.columns),
        "split_sizes": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        "dt_best_params": dt_best["params"],
        "dt_val_mse": dt_best["mse"],
        "dt_test_mse": dt_test_mse,
        "dt_top5": dt_results.head(5).to_dict(orient="records"),
        "xgb_best_params": xgb_best["params"],
        "xgb_val_mse": xgb_best["mse"],
        "xgb_test_mse": xgb_test_mse,
        "xgb_top5": xgb_results.head(5).to_dict(orient="records"),
        "y_stats": {
            "train_mean": float(y_train.mean()), "train_std": float(y_train.std()),
            "test_mean": float(y_test.mean()), "test_std": float(y_test.std()),
        },
    }


if __name__ == "__main__":
    all_results = {}
    all_results["credit"] = run_pipeline(
        "/mnt/user-data/uploads/RG-Credit.csv", load_credit, "RG-Credit (Balance)"
    )
    all_results["wage"] = run_pipeline(
        "/mnt/user-data/uploads/RG-Wage.csv", load_wage, "RG-Wage (wage)"
    )

    with open("/home/claude/results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n\nSaved results to /home/claude/results.json")
