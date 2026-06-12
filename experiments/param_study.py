"""Parameter study for all models (MLP, RandomForest, SVM, HistGradientBoosting).

For each ticker and each model a grid of hyperparameters is evaluated on the
chronological test split. Results (MAE, MSE, RMSE, direction accuracy, time)
are stored in results/param_study.csv and one figure per model is generated
(RMSE and direction accuracy vs. parameter, one subplot column per ticker).
"""

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

INPUT_SIZE  = 7
OUTPUT_SIZE = 1

# parameter grids
MLP_HIDDEN = [[5], [10], [25], [10, 5]]
MLP_LRS    = [0.01, 0.001, 0.0001]
MLP_EPOCHS = 300

RF_N_ESTIMATORS = [25, 50, 100, 200, 400]
RF_MAX_DEPTHS   = [None, 5, 10]

SVM_C_VALUES  = [0.1, 1.0, 10.0, 100.0]
SVM_EPSILONS  = [0.01, 0.1, 0.5]

HGB_MAX_ITERS = [50, 100, 200, 300, 500]
HGB_LRS       = [0.01, 0.05, 0.1]

RESULTS_CSV = os.path.join(ROOT, "results", "param_study.csv")
BEST_CSV    = os.path.join(ROOT, "results", "best_configurations.csv")
PLOTS_DIR   = os.path.join(ROOT, "results", "plots")


def compute_metrics(y_test, y_pred, elapsed):
    return {
        "mae":                mae_fn(y_test, y_pred),
        "mse":                mse_fn(y_test, y_pred),
        "rmse":               rmse_fn(y_test, y_pred),
        "direction_accuracy": dir_acc_fn(y_test, y_pred),
        "time_s":             round(elapsed, 3),
    }


def run_mlp(data, ticker, rows):
    X_train, X_test, y_train, y_test = data
    for hidden in MLP_HIDDEN:
        for lr in MLP_LRS:
            np.random.seed(42)
            model = MLP(INPUT_SIZE, hidden, OUTPUT_SIZE)
            t0 = time.time()
            train_model(model, X_train, y_train, epochs=MLP_EPOCHS, lr=lr, verbose=False)
            y_pred = model.forward_propagation(X_test)
            m = compute_metrics(y_test, y_pred, time.time() - t0)
            hidden_str = "-".join(str(h) for h in hidden)
            rows.append({
                "ticker": ticker, "model": "MLP",
                "param_1_name": "hidden", "param_1_value": hidden_str,
                "param_2_name": "lr",     "param_2_value": lr,
                **m,
            })
            print(f"  [MLP] hidden={hidden_str:<5} lr={lr:<7} RMSE={m['rmse']:.4f} Dir={m['direction_accuracy']:.1f}%")


def run_rf(data, ticker, rows):
    X_train, X_test, y_train, y_test = data
    for n in RF_N_ESTIMATORS:
        for depth in RF_MAX_DEPTHS:
            model = RandomForestModel(n_estimators=n, max_depth=depth)
            t0 = time.time()
            model.train(X_train, y_train)
            m = compute_metrics(y_test, model.predict(X_test), time.time() - t0)
            rows.append({
                "ticker": ticker, "model": "RandomForest",
                "param_1_name": "n_estimators", "param_1_value": n,
                "param_2_name": "max_depth",    "param_2_value": depth,
                **m,
            })
            print(f"  [RF]  n_est={n:<4} depth={str(depth):<5} RMSE={m['rmse']:.4f} Dir={m['direction_accuracy']:.1f}%")


def run_svm(data, ticker, rows):
    X_train, X_test, y_train, y_test = data
    for C in SVM_C_VALUES:
        for eps in SVM_EPSILONS:
            model = SVMModel(kernel="rbf", C=C, epsilon=eps)
            t0 = time.time()
            model.train(X_train, y_train)
            m = compute_metrics(y_test, model.predict(X_test), time.time() - t0)
            rows.append({
                "ticker": ticker, "model": "SVM",
                "param_1_name": "C",       "param_1_value": C,
                "param_2_name": "epsilon", "param_2_value": eps,
                **m,
            })
            print(f"  [SVM] C={C:<6} eps={eps:<5} RMSE={m['rmse']:.4f} Dir={m['direction_accuracy']:.1f}%")


def run_hgb(data, ticker, rows):
    X_train, X_test, y_train, y_test = data
    for max_iter in HGB_MAX_ITERS:
        for lr in HGB_LRS:
            model = HistGradientBoostingModel(max_iter=max_iter, learning_rate=lr)
            t0 = time.time()
            model.train(X_train, y_train)
            m = compute_metrics(y_test, model.predict(X_test), time.time() - t0)
            rows.append({
                "ticker": ticker, "model": "HistGradientBoosting",
                "param_1_name": "max_iter", "param_1_value": max_iter,
                "param_2_name": "lr",       "param_2_value": lr,
                **m,
            })
            print(f"  [HGB] max_iter={max_iter:<4} lr={lr:<5} RMSE={m['rmse']:.4f} Dir={m['direction_accuracy']:.1f}%")


def plot_model_study(df, model_name, x_label, line_label, png_name,
                     log_x=False, categorical_x=False):
    """One figure per model: rows = (RMSE, direction acc.), cols = tickers.

    x axis = param_1, one line per value of param_2.
    """
    sub = df[df["model"] == model_name]
    tickers = list(TICKERS.keys())
    line_values = sorted(sub["param_2_value"].astype(str).unique())

    fig, axes = plt.subplots(2, len(tickers), figsize=(15, 8), sharex="col")
    fig.suptitle(f"{model_name} - hyperparameter study "
                 f"({x_label} vs. {line_label})", fontsize=14, fontweight="bold")

    for col, ticker in enumerate(tickers):
        st = sub[sub["ticker"] == ticker]
        for lv in line_values:
            sl = st[st["param_2_value"].astype(str) == lv]
            if categorical_x:
                x = np.arange(len(sl))
                x_ticks = sl["param_1_value"].astype(str).tolist()
            else:
                sl = sl.sort_values("param_1_value")
                x = sl["param_1_value"].astype(float).values

            axes[0, col].plot(x, sl["rmse"].values, marker="o", label=f"{line_label}={lv}")
            axes[1, col].plot(x, sl["direction_accuracy"].values, marker="s", label=f"{line_label}={lv}")

            if categorical_x:
                axes[1, col].set_xticks(x)
                axes[1, col].set_xticklabels(x_ticks)

        axes[0, col].set_title(ticker, fontsize=12)
        axes[1, col].set_xlabel(x_label)
        if log_x:
            axes[0, col].set_xscale("log")
            axes[1, col].set_xscale("log")
        for row in (0, 1):
            axes[row, col].grid(True, alpha=0.3)

    axes[0, 0].set_ylabel("RMSE (test)")
    axes[1, 0].set_ylabel("Direction accuracy [%]")
    axes[0, 0].legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, png_name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {path}")


def save_best_configurations(df):
    """For each ticker/model pick the configuration with the lowest test RMSE."""
    idx = df.groupby(["ticker", "model"])["rmse"].idxmin()
    best = df.loc[idx].sort_values(["ticker", "model"])
    best.to_csv(BEST_CSV, index=False, encoding="utf-8")
    print(f"Best configurations saved: {BEST_CSV}")

    print("\n" + "=" * 95)
    print(f"{'TICKER':<10} {'MODEL':<22} {'PARAMETERS':<28} {'MAE':>8} {'RMSE':>8} {'DIR%':>7}")
    print("=" * 95)
    for _, r in best.iterrows():
        params = f"{r['param_1_name']}={r['param_1_value']} {r['param_2_name']}={r['param_2_value']}"
        print(f"{r['ticker']:<10} {r['model']:<22} {params:<28} "
              f"{r['mae']:>8.4f} {r['rmse']:>8.4f} {r['direction_accuracy']:>6.1f}%")
    print("=" * 95)
    return best


def run_param_study():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    rows = []

    for ticker, csv_path in TICKERS.items():
        print(f"\n{'=' * 70}")
        print(f"  Instrument: {ticker}")
        print(f"{'=' * 70}")
        data = load_and_preprocess(csv_path)
        print(f"  Train: {data[0].shape[0]} samples | Test: {data[1].shape[0]} samples")

        run_mlp(data, ticker, rows)
        run_rf(data, ticker, rows)
        run_svm(data, ticker, rows)
        run_hgb(data, ticker, rows)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False, encoding="utf-8")
    print(f"\nResults saved: {RESULTS_CSV}  ({len(df)} rows)")

    plot_model_study(df, "MLP", "hidden layers", "lr",
                     "param_study_MLP.png", categorical_x=True)
    plot_model_study(df, "RandomForest", "n_estimators", "max_depth",
                     "param_study_RandomForest.png")
    plot_model_study(df, "SVM", "C", "epsilon",
                     "param_study_SVM.png", log_x=True)
    plot_model_study(df, "HistGradientBoosting", "max_iter", "lr",
                     "param_study_HistGradientBoosting.png")

    save_best_configurations(df)
    return df


if __name__ == "__main__":
    run_param_study()
