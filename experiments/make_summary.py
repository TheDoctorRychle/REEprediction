"""Builds results/summary_tables.md - report-ready markdown tables
from the CSV files produced by the experiment scripts.

Run after: compare_models.py, param_study.py, feature_ablation.py
"""

import os
import pandas as pd

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
OUT_MD      = os.path.join(RESULTS_DIR, "summary_tables.md")

METRIC_COLS = ["mae", "mse", "rmse", "direction_accuracy", "time_s"]


def fmt(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].round(4)
    return df.to_markdown(index=False)


def section_comparison(parts):
    path = os.path.join(RESULTS_DIR, "comparison_results.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    parts.append("## 1. Model comparison (all tested configurations)\n")
    parts.append("Metrics on the chronological test split (20% most recent samples).\n")
    parts.append(fmt(df))
    parts.append("")


def section_best(parts):
    path = os.path.join(RESULTS_DIR, "best_configurations.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    df["parameters"] = (df["param_1_name"] + "=" + df["param_1_value"].astype(str)
                        + ", " + df["param_2_name"] + "=" + df["param_2_value"].astype(str))
    cols = ["ticker", "model", "parameters"] + METRIC_COLS
    parts.append("## 2. Best configuration per ticker/model (lowest test RMSE)\n")
    parts.append(fmt(df[cols]))
    parts.append("")


def section_param_study(parts):
    path = os.path.join(RESULTS_DIR, "param_study.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    parts.append("## 3. Hyperparameter study (full grid)\n")
    for model in df["model"].unique():
        sub = df[df["model"] == model].copy()
        p1, p2 = sub["param_1_name"].iloc[0], sub["param_2_name"].iloc[0]
        sub = sub.rename(columns={"param_1_value": p1, "param_2_value": p2})
        cols = ["ticker", p1, p2] + METRIC_COLS
        parts.append(f"### 3.{list(df['model'].unique()).index(model) + 1} {model}\n")
        parts.append(fmt(sub[cols]))
        parts.append("")


def section_ablation(parts):
    path = os.path.join(RESULTS_DIR, "feature_ablation.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    parts.append("## 4. Feature ablation (engineered features: Close_mean_5, Close_q75_5)\n")
    parts.append(fmt(df))
    parts.append("")

    pivot = df.pivot_table(index=["ticker", "model"], columns="feature_set",
                           values="rmse").reset_index()
    for set_name in ["base5+mean", "base5+q75", "all7"]:
        if set_name in pivot.columns:
            pivot[f"delta_{set_name}"] = pivot[set_name] - pivot["base5"]
    parts.append("### 4.1 RMSE change vs base5 (negative = feature helps)\n")
    parts.append(fmt(pivot))
    parts.append("")


def main():
    parts = ["# REEprediction - experiment summary tables\n",
             "Metrics: MAE, MSE, RMSE (prediction error of the next-day price "
             "change), direction_accuracy (% of test days with correctly "
             "predicted change direction), time_s (training time).\n"]

    section_comparison(parts)
    section_best(parts)
    section_param_study(parts)
    section_ablation(parts)

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"Summary written: {OUT_MD}")


if __name__ == "__main__":
    main()
