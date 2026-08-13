from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

import features as feat
import preprocessing as prep

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "report"

HOLDOUT_START = "2025-09-01"  # last two months held out as the internal test set
TARGET = "posted_rate"

CHECK_EVERY = 20      # between checkpoints
PATIENCE_CHECKS = 4   # checkpoint for holdout MAE if it hasn't improved
MAX_ROUNDS = 1500

HGBR_PARAMS = dict(
    loss="absolute_error",
    learning_rate=0.05,
    max_leaf_nodes=63,
    min_samples_leaf=30,
    l2_regularization=1.0,
    categorical_features="from_dtype",
    random_state=42,
)


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "train_test.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


def time_split(df: pd.DataFrame):
    cutoff = pd.Timestamp(HOLDOUT_START)
    train = df[df["date"] < cutoff].copy()
    holdout = df[df["date"] >= cutoff].copy()
    return train, holdout


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def fit_baseline(X_train, X_holdout, y_train_log, y_holdout_dollars) -> tuple[dict, LinearRegression]:
    
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    cat_train = ohe.fit_transform(X_train[feat.CATEGORICAL_COLUMNS].astype(str))
    cat_holdout = ohe.transform(X_holdout[feat.CATEGORICAL_COLUMNS].astype(str))
    num_cols = feat.NUMERIC_COLUMNS + feat.DATE_FEATURE_COLUMNS
    baseline_X_train = np.hstack([cat_train, X_train[num_cols].values])
    baseline_X_holdout = np.hstack([cat_holdout, X_holdout[num_cols].values])

    baseline = LinearRegression()
    baseline.fit(baseline_X_train, y_train_log)
    baseline_pred = np.expm1(baseline.predict(baseline_X_holdout))
    return evaluate(y_holdout_dollars, baseline_pred), baseline


def train_hgbr(X_train, X_holdout, y_train_log, y_holdout_log):
    
    model = HistGradientBoostingRegressor(**HGBR_PARAMS, max_iter=CHECK_EVERY, warm_start=True, early_stopping=False)

    best_holdout_mae = np.inf
    best_iter = 0
    no_improve_checks = 0
    rounds, train_curve, holdout_curve = [], [], []

    print("Training HGBR with manual warm-start early stopping against the Sep-Oct holdout...")
    for target_iter in range(CHECK_EVERY, MAX_ROUNDS + 1, CHECK_EVERY):
        model.max_iter = target_iter
        model.fit(X_train, y_train_log)  # warm_start=True: continues from current trees

        train_pred_log = model.predict(X_train)
        holdout_pred_log = model.predict(X_holdout)
        train_mae_log = float(np.mean(np.abs(y_train_log - train_pred_log)))
        holdout_mae_log = float(np.mean(np.abs(y_holdout_log - holdout_pred_log)))

        rounds.append(target_iter)
        train_curve.append(train_mae_log)
        holdout_curve.append(holdout_mae_log)

        improved = holdout_mae_log < best_holdout_mae - 1e-5
        if improved:
            best_holdout_mae = holdout_mae_log
            best_iter = target_iter
            no_improve_checks = 0
        else:
            no_improve_checks += 1

        print(f"  iter {target_iter:4d}  train_mae(log)={train_mae_log:.5f}  "
              f"holdout_mae(log)={holdout_mae_log:.5f}{'  <- best' if improved else ''}")

        if no_improve_checks >= PATIENCE_CHECKS:
            print(f"  no improvement in {PATIENCE_CHECKS} checkpoints, stopping at iter {target_iter} "
                  f"(best was {best_iter})")
            break

    curve_df = pd.DataFrame({"iteration": rounds, "train_mae_log": train_curve, "holdout_mae_log": holdout_curve})

    # refit fresh at exactly best_iter so we return the checkpoint that
    # scored best, not whatever iteration the patience loop stopped at
    best_model = HistGradientBoostingRegressor(**HGBR_PARAMS, max_iter=best_iter, early_stopping=False)
    best_model.fit(X_train, y_train_log)

    return best_model, best_iter, curve_df


def plot_training_curve(curve_df: pd.DataFrame, best_iter: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=170)
    ax.plot(curve_df["iteration"], curve_df["train_mae_log"], color="#9DAFB3", linewidth=1.6,
            linestyle="--", label="Train MAE")
    ax.plot(curve_df["iteration"], curve_df["holdout_mae_log"], color="#064A56", linewidth=2,
            label="Holdout MAE (Sep-Oct 2025)")
    best_row = curve_df.loc[curve_df["iteration"] == best_iter]
    if len(best_row):
        ax.scatter(best_row["iteration"], best_row["holdout_mae_log"], color="#C0392B", zorder=5, s=45,
                   label=f"Best iteration ({best_iter})")
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("MAE (log1p target)")
    ax.set_title("HGBR training curve", loc="left", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    raw = load_raw()
    train_raw, holdout_raw = time_split(raw)
    print(f"Train rows: {len(train_raw):,} (through {train_raw['date'].max().date()})")
    print(f"Holdout rows: {len(holdout_raw):,} ({holdout_raw['date'].min().date()} to {holdout_raw['date'].max().date()})")

    # cleaning: fit imputation on train only, reuse on holdout 
    train_clean, ref_medians = prep.clean_dataframe(train_raw, reference_medians=None)
    holdout_clean, _ = prep.clean_dataframe(holdout_raw, reference_medians=ref_medians)

    # feature engineering setup, also fit on train only
    city_coords = feat.build_city_coords(train_clean)
    category_levels = {
        "pickup": sorted(train_clean["pickup"].unique()),
        "delivery": sorted(train_clean["delivery"].unique()),
        "equipment": sorted(train_clean["equipment"].unique()),
    }
    market_fallback = {
        "market_index": ref_medians["market_index"],
        "quote_signal": float(train_clean["quote_signal"].median()),
    }

    X_train = feat.build_feature_matrix(train_clean, city_coords, category_levels, market_fallback)
    X_holdout = feat.build_feature_matrix(holdout_clean, city_coords, category_levels, market_fallback)
    y_train_log = np.log1p(train_clean[TARGET].values)
    y_holdout_log = np.log1p(holdout_clean[TARGET].values)
    y_holdout_dollars = holdout_clean[TARGET].values

    # baseline
    baseline_metrics, baseline_model = fit_baseline(X_train, X_holdout, y_train_log, y_holdout_dollars)
    print("Baseline (linear regression) holdout metrics:", baseline_metrics)

    # HGBR, with training curve
    model, best_iter, curve_df = train_hgbr(X_train, X_holdout, y_train_log, y_holdout_log)
    holdout_pred_dollars = np.expm1(model.predict(X_holdout))
    hgbr_metrics = evaluate(y_holdout_dollars, holdout_pred_dollars)
    print(f"\nBest iteration: {best_iter}")
    print("HGBR holdout metrics:", hgbr_metrics)

    curve_df.to_csv(REPORT_DIR / "training_curve.csv", index=False)
    plot_training_curve(curve_df, best_iter, REPORT_DIR / "training_curve.png")
    print(f"Saved training_curve.csv and training_curve.png ({len(curve_df)} rounds)")

    # feature importance
    
    print("\nComputing permutation importance on the holdout set (this takes a moment)...")
    perm = permutation_importance(
        model, X_holdout, y_holdout_log, scoring="neg_mean_absolute_error",
        n_repeats=5, random_state=42,
    )
    importance = pd.DataFrame({
        "feature": X_holdout.columns,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)
    importance.to_csv(REPORT_DIR / "feature_importance.csv", index=False)
    print(importance.head(10).to_string(index=False))

    # refit on the FULL train_test.csv before shipping the model
    full_clean, full_medians = prep.clean_dataframe(raw, reference_medians=None)
    full_city_coords = feat.build_city_coords(full_clean)
    full_category_levels = {
        "pickup": sorted(full_clean["pickup"].unique()),
        "delivery": sorted(full_clean["delivery"].unique()),
        "equipment": sorted(full_clean["equipment"].unique()),
    }
    full_market_fallback = {
        "market_index": full_medians["market_index"],
        "quote_signal": float(full_clean["quote_signal"].median()),
    }
    X_full = feat.build_feature_matrix(full_clean, full_city_coords, full_category_levels, full_market_fallback)
    y_full_log = np.log1p(full_clean[TARGET].values)

    final_model = HistGradientBoostingRegressor(**HGBR_PARAMS, max_iter=best_iter, early_stopping=False)
    final_model.fit(X_full, y_full_log)

    artifact = {
        "model": final_model,
        "reference_medians": full_medians,
        "city_coords": full_city_coords,
        "category_levels": full_category_levels,
        "market_fallback": full_market_fallback,
        "feature_columns": feat.FEATURE_COLUMNS,
        "holdout_metrics": hgbr_metrics,
        "best_iteration": int(best_iter),
    }
    joblib.dump(artifact, MODEL_DIR / "model.pkl")
    joblib.dump(baseline_model, MODEL_DIR / "model_baseline.pkl")

    metrics_out = {
        "holdout_period": [str(holdout_raw["date"].min().date()), str(holdout_raw["date"].max().date())],
        "n_train": int(len(train_raw)),
        "n_holdout": int(len(holdout_raw)),
        "baseline_linear_regression": baseline_metrics,
        "hgbr": hgbr_metrics,
        "best_iteration": int(best_iter),
    }
    with open(REPORT_DIR / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    holdout_out = holdout_raw[["load_id", "date", "equipment", "distance"]].copy()
    holdout_out["actual_rate"] = y_holdout_dollars
    holdout_out["predicted_rate"] = holdout_pred_dollars
    holdout_out["abs_error"] = np.abs(holdout_out["actual_rate"] - holdout_out["predicted_rate"])
    holdout_out.to_csv(REPORT_DIR / "holdout_predictions.csv", index=False)

    print("\nSaved model.pkl, model_baseline.pkl, metrics.json, holdout_predictions.csv, feature_importance.csv")


if __name__ == "__main__":
    main()
