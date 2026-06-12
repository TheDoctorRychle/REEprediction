import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model.mlp import MLP
from train.train import train_model
from evaluate.metrics import mae as mae_fn, mse as mse_fn, rmse as rmse_fn, direction_accuracy as dir_acc_fn
from utils.preprocessing import load_and_preprocess
from models.random_forest import RandomForestModel
from models.svm_model import SVMModel
from models.hist_gradient_boosting import HistGradientBoostingModel


TICKERS = {
    "REMX":   os.path.join(ROOT, "data", "REMX.csv"),
    "AMG_AS": os.path.join(ROOT, "data", "AMG_AS.csv"),
    "KGH_WA": os.path.join(ROOT, "data", "KGH_WA.csv"),
}

BASE5 = ["Open", "High", "Low", "Close", "Volume"]
FEATURE_SETS = {
    "base5":      BASE5,
    "base5+mean": BASE5 + ["Close_mean_5"],
    "base5+q75":  BASE5 + ["Close_q75_5"],
    "all7":       BASE5 + ["Close_mean_5", "Close_q75_5"],
}

MLP_HIDDEN = [25]
MLP_LR     = 0.0001
MLP_EPOCHS = 300

RESULTS_CSV = os.path.join(ROOT, "results", "feature_ablation.csv")
PLOT_PNG    = os.path.join(ROOT, "results", "plots", "feature_ablation.png")


def train_all_models(X_train, X_test, y_train, y_test, ticker, set_name, rows):
    n_features = X_train.shape[1]

    def add_row(model_name, y_pred, elapsed):
        rows.append({
            "ticker":             ticker,
            "feature_set":        set_name,
            "n_features":         n_features,
            "model":              model_name,
            "mae":                mae_fn(y_test, y_pred),
            "mse":                mse_fn(y_test, y_pred),
            "rmse":               rmse_fn(y_test, y_pred),
            "direction_accuracy": dir_acc_fn(y_test, y_pred),
            "time_s":             round(elapsed, 3),
        })

    np.random.seed(42)
    mlp = MLP(n_features, MLP_HIDDEN, 1)
    t0 = time.time()
    train_model(mlp, X_train, y_train, epochs=MLP_EPOCHS, lr=MLP_LR, verbose=False)
    add_row("MLP", mlp.forward_propagation(X_test), time.time() - t0)

    sk_models = {
        "RandomForest":         RandomForestModel(n_estimators=100),
        "SVM":                  SVMModel(kernel="rbf", C=1.0),
        "HistGradientBoosting": HistGradientBoostingModel(max_iter=300),
    }
    for name, model in sk_models.items():
        t0 = time.time()
        model.train(X_train, y_train)
        add_row(name, model.predict(X_test), time.time() - t0)


def plot_ablation(df):
    tickers   = list(TICKERS.keys())
    models    = df["model"].unique()
    set_names = list(FEATURE_SETS.keys())
    colors    = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, axes = plt.subplots(2, len(tickers), figsize=(16, 9), sharex="col")
    fig.suptitle("Feature ablation - impact of engineered features "
                 "(Close_mean_5, Close_q75_5)", fontsize=14, fontweight="bold")

    x = np.arange(len(models))
    width = 0.2

    for col, ticker in enumerate(tickers):
        st = df[df["ticker"] == ticker]
        for i, set_name in enumerate(set_names):
            sub = st[st["feature_set"] == set_name].set_index("model").loc[models]
            axes[0, col].bar(x + i * width, sub["rmse"], width,
                             label=set_name, color=colors[i], alpha=0.85)
            axes[1, col].bar(x + i * width, sub["direction_accuracy"], width,
                             label=set_name, color=colors[i], alpha=0.85)

        axes[0, col].set_title(ticker, fontsize=12)
        for row in (0, 1):
            axes[row, col].set_xticks(x + width * 1.5)
            axes[row, col].set_xticklabels(
                [m.replace("HistGradientBoosting", "HGB") for m in models],
                fontsize=9)
            axes[row, col].grid(axis="y", alpha=0.3)

    axes[0, 0].set_ylabel("RMSE (test)")
    axes[1, 0].set_ylabel("Direction accuracy [%]")
    axes[0, 0].legend(fontsize=9, title="Feature set")

    plt.tight_layout()
    os.makedirs(os.path.dirname(PLOT_PNG), exist_ok=True)
    plt.savefig(PLOT_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved: {PLOT_PNG}")


def run_feature_ablation():
    rows = []

    for ticker, csv_path in TICKERS.items():
        print(f"\n{'=' * 70}")
        print(f"  Instrument: {ticker}")
        print(f"{'=' * 70}")

        for set_name, columns in FEATURE_SETS.items():
            print(f"  Feature set: {set_name} ({len(columns)} features)")
            data = load_and_preprocess(csv_path, feature_columns=columns)
            train_all_models(*data[:2], *data[2:], ticker, set_name, rows)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False, encoding="utf-8")
    print(f"\nResults saved: {RESULTS_CSV}  ({len(df)} rows)")

    plot_ablation(df)

    print("\nRMSE change vs base5 (negative = engineered features help):")
    pivot = df.pivot_table(index=["ticker", "model"], columns="feature_set", values="rmse")
    for set_name in ["base5+mean", "base5+q75", "all7"]:
        pivot[f"d_{set_name}"] = pivot[set_name] - pivot["base5"]
    print(pivot.round(4).to_string())

    return df


if __name__ == "__main__":
    run_feature_ablation()
