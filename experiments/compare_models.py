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
from evaluate.metrics import evaluate_model, rmse as rmse_fn, mae as mae_fn, mse as mse_fn, direction_accuracy as dir_acc_fn
from utils.preprocessing import load_and_preprocess
from models.random_forest import RandomForestModel
from models.svm_model import SVMModel
from models.hist_gradient_boosting import HistGradientBoostingModel


TICKERS = {
    "REMX":   os.path.join(ROOT, "data", "REMX.csv"),
    "AMG_AS": os.path.join(ROOT, "data", "AMG_AS.csv"),
    "KGH_WA": os.path.join(ROOT, "data", "KGH_WA.csv"),
}

MLP_HIDDEN  = [25]
MLP_LR      = 0.0001
MLP_EPOCHS  = 300
INPUT_SIZE  = 7
OUTPUT_SIZE = 1

RF_N_ESTIMATORS = [50, 100, 200]
SVM_C_VALUES    = [0.1, 1.0, 10.0]
HGB_MAX_ITER = [100, 200, 300]

RESULTS_CSV = os.path.join(ROOT, "results", "comparison_results.csv")
PLOT_PNG    = os.path.join(ROOT, "results", "plots", "model_comparison.png")


def evaluate_sklearn(model_obj, X_test, y_test):
    y_pred = model_obj.predict(X_test)
    return {
        "mae":                mae_fn(y_test, y_pred),
        "mse":                mse_fn(y_test, y_pred),
        "rmse":               rmse_fn(y_test, y_pred),
        "direction_accuracy": dir_acc_fn(y_test, y_pred),
    }


def train_mlp(X_train, y_train, X_test, y_test):
    np.random.seed(42)
    model = MLP(input_size=INPUT_SIZE, hidden_layers=MLP_HIDDEN, output_size=OUTPUT_SIZE)
    t0 = time.time()
    train_model(model, X_train, y_train, epochs=MLP_EPOCHS, lr=MLP_LR, verbose=False)
    elapsed = time.time() - t0
    metrics = evaluate_model(model, X_test, y_test)
    return metrics, elapsed


def train_rf(X_train, y_train, X_test, y_test, n_estimators):
    model = RandomForestModel(n_estimators=n_estimators)
    t0 = time.time()
    model.train(X_train, y_train)
    elapsed = time.time() - t0
    metrics = evaluate_sklearn(model, X_test, y_test)
    return metrics, elapsed


def train_svm(X_train, y_train, X_test, y_test, C):
    model = SVMModel(kernel="rbf", C=C)
    t0 = time.time()
    model.train(X_train, y_train)
    elapsed = time.time() - t0
    metrics = evaluate_sklearn(model, X_test, y_test)
    return metrics, elapsed

def train_hgb(X_train, y_train, X_test, y_test, max_iter):
    model = HistGradientBoostingModel(max_iter=max_iter)

    t0 = time.time()
    model.train(X_train, y_train)
    elapsed = time.time() - t0

    metrics = evaluate_sklearn(model, X_test, y_test)
    return metrics, elapsed
def print_table(rows):
    width = 100
    print("\n" + "=" * width)
    print(f"{'TICKER':<10} {'MODEL':<15} {'PARAMETERS':<22} "
          f"{'MAE':>9} {'RMSE':>9} {'DIR%':>7} {'TIME':>7}")
    print("=" * width)

    prev = None
    for r in rows:
        if r["ticker"] != prev and prev is not None:
            print("-" * width)
        prev = r["ticker"]
        print(f"{r['ticker']:<10} {r['model']:<15} {r['parameters']:<22} "
              f"{r['mae']:>9.4f} {r['rmse']:>9.4f} "
              f"{r['direction_accuracy']:>6.1f}% "
              f"{r['time_s']:>6.2f}s")

    print("=" * width)


def draw_plot(best_rmse, tickers, models):
    os.makedirs(os.path.dirname(PLOT_PNG), exist_ok=True)

    x      = np.arange(len(tickers))
    width  = 0.2
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, model_name in enumerate(models):
        values = [best_rmse[t][model_name] for t in tickers]
        bars = ax.bar(x + i * width, values, width,
                      label=model_name, color=colors[i], alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Stock instrument", fontsize=12)
    ax.set_ylabel("RMSE (best configuration)", fontsize=12)
    ax.set_title("Model comparison - RMSE on test set\n"
                 "(MLP vs Random Forest vs SVM vs HGB)", fontsize=13, fontweight="bold")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(tickers, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(PLOT_PNG, dpi=150)
    plt.close()
    print(f"\nPlot saved: {PLOT_PNG}")


def compare_models():
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)

    all_rows = []
    best_rmse = {t: {} for t in TICKERS}

    for ticker, csv_path in TICKERS.items():
        print(f"\n{'=' * 70}")
        print(f"  Instrument: {ticker}")
        print(f"{'=' * 70}")

        X_train, X_test, y_train, y_test = load_and_preprocess(csv_path)
        print(f"  Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

        print(f"\n  [MLP] hidden={MLP_HIDDEN}, lr={MLP_LR}, epochs={MLP_EPOCHS} ...", end=" ", flush=True)
        metrics, elapsed = train_mlp(X_train, y_train, X_test, y_test)
        print(f"RMSE={metrics['rmse']:.4f}  Dir={metrics['direction_accuracy']:.1f}%")

        mlp_row = {
            "ticker":             ticker,
            "model":              "MLP",
            "parameters":         f"hidden={MLP_HIDDEN} lr={MLP_LR}",
            "mae":                metrics["mae"],
            "mse":                metrics["mse"],
            "rmse":               metrics["rmse"],
            "direction_accuracy": metrics["direction_accuracy"],
            "time_s":             round(elapsed, 3),
        }
        all_rows.append(mlp_row)
        best_rmse[ticker]["MLP"] = metrics["rmse"]

        rf_rmse_min = float("inf")
        for n in RF_N_ESTIMATORS:
            print(f"  [RF]  n_estimators={n:<3} ...", end=" ", flush=True)
            metrics, elapsed = train_rf(X_train, y_train, X_test, y_test, n)
            print(f"RMSE={metrics['rmse']:.4f}  Dir={metrics['direction_accuracy']:.1f}%")

            all_rows.append({
                "ticker":             ticker,
                "model":              "RandomForest",
                "parameters":         f"n_est={n}",
                "mae":                metrics["mae"],
                "mse":                metrics["mse"],
                "rmse":               metrics["rmse"],
                "direction_accuracy": metrics["direction_accuracy"],
                "time_s":             round(elapsed, 3),
            })
            if metrics["rmse"] < rf_rmse_min:
                rf_rmse_min = metrics["rmse"]

        best_rmse[ticker]["RandomForest"] = rf_rmse_min

        svm_rmse_min = float("inf")
        for c in SVM_C_VALUES:
            print(f"  [SVM] C={c:<5} ...", end=" ", flush=True)
            metrics, elapsed = train_svm(X_train, y_train, X_test, y_test, c)
            print(f"RMSE={metrics['rmse']:.4f}  Dir={metrics['direction_accuracy']:.1f}%")

            all_rows.append({
                "ticker":             ticker,
                "model":              "SVM",
                "parameters":         f"rbf C={c}",
                "mae":                metrics["mae"],
                "mse":                metrics["mse"],
                "rmse":               metrics["rmse"],
                "direction_accuracy": metrics["direction_accuracy"],
                "time_s":             round(elapsed, 3),
            })
            if metrics["rmse"] < svm_rmse_min:
                svm_rmse_min = metrics["rmse"]

        best_rmse[ticker]["SVM"] = svm_rmse_min

        hgb_rmse_min = float("inf")
        for it in HGB_MAX_ITER:
            print(f"  [HGB] max_iter={it:<3} ...", end=" ", flush=True)
            metrics, elapsed = train_hgb(X_train, y_train, X_test, y_test, it)
            print(f"RMSE={metrics['rmse']:.4f}  Dir={metrics['direction_accuracy']:.1f}%")

            all_rows.append({
                "ticker": ticker,
                "model": "HistGradientBoosting",
                "parameters": f"max_iter={it}",
                "mae": metrics["mae"],
                "mse": metrics["mse"],
                "rmse": metrics["rmse"],
                "direction_accuracy": metrics["direction_accuracy"],
                "time_s": round(elapsed, 3),
            })

            hgb_rmse_min = min(hgb_rmse_min, metrics["rmse"])

        best_rmse[ticker]["HistGradientBoosting"] = hgb_rmse_min

    df_results = pd.DataFrame(all_rows)
    df_results.to_csv(RESULTS_CSV, index=False, encoding="utf-8")
    print(f"\nResults saved: {RESULTS_CSV}")

    print_table(all_rows)

    print("\n" + "=" * 55)
    print("  SUMMARY - Best RMSE per instrument/model")
    print("=" * 55)
    for ticker in TICKERS:
        print(f"\n  {ticker}:")
        for model_name, rmse_val in best_rmse[ticker].items():
            print(f"    {model_name:<15} RMSE = {rmse_val:.4f}")

    draw_plot(best_rmse, list(TICKERS.keys()),
              ["MLP", "RandomForest", "SVM", "HistGradientBoosting"])

    return all_rows, best_rmse


if __name__ == "__main__":
    compare_models()
