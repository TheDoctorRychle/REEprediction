import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model.mlp import MLP
from train.train import train_model
from evaluate.metrics import rmse as rmse_fn, direction_accuracy as dir_acc_fn
from utils.preprocessing import load_and_preprocess
from models.random_forest import RandomForestModel
from models.svm_model import SVMModel
from models.hist_gradient_boosting import HistGradientBoostingModel


TICKERS = {
    "REMX":   os.path.join(ROOT, "data", "REMX.csv"),
    "AMG_AS": os.path.join(ROOT, "data", "AMG_AS.csv"),
    "KGH_WA": os.path.join(ROOT, "data", "KGH_WA.csv"),
}

INPUT_SIZE = 7
MLP_HIDDEN = [25]
MLP_LR     = 0.0001
MLP_EPOCHS = 300

PLOTS_DIR = os.path.join(ROOT, "results", "plots")


def get_predictions(X_train, y_train, X_test):
    preds = {}

    np.random.seed(42)
    mlp = MLP(INPUT_SIZE, MLP_HIDDEN, 1)
    train_model(mlp, X_train, y_train, epochs=MLP_EPOCHS, lr=MLP_LR, verbose=False)
    preds["MLP"] = mlp.forward_propagation(X_test)

    for name, model in {
        "RandomForest":         RandomForestModel(n_estimators=100),
        "SVM":                  SVMModel(kernel="rbf", C=1.0),
        "HistGradientBoosting": HistGradientBoostingModel(max_iter=300),
    }.items():
        model.train(X_train, y_train)
        preds[name] = model.predict(X_test)

    return preds


def plot_all_models(y_test, preds, ticker):
    y_true = y_test.flatten()
    idx = np.arange(len(y_true))

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True, sharey=True)
    fig.suptitle(f"Predictions vs actual price change (test set) - {ticker}",
                 fontsize=14, fontweight="bold")

    for ax, (name, y_pred) in zip(axes.flat, preds.items()):
        y_p = y_pred.flatten()
        ax.plot(idx, y_true, color="#2196F3", linewidth=1.0, alpha=0.8,
                label="Actual")
        ax.plot(idx, y_p, color="#F44336", linewidth=0.9, alpha=0.75,
                linestyle="--", label="Prediction")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")

        score = (f"RMSE={rmse_fn(y_test, y_pred):.4f}  "
                 f"Dir={dir_acc_fn(y_test, y_pred):.1f}%")
        ax.set_title(f"{name}\n{score}", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    for ax in axes[1]:
        ax.set_xlabel("Test sample (business day)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Price change Close[t+1] - Close[t]")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"predictions_all_models_{ticker}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {path}")


def run_prediction_plots():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    for ticker, csv_path in TICKERS.items():
        print(f"\n  Instrument: {ticker}")
        X_train, X_test, y_train, y_test = load_and_preprocess(csv_path)
        preds = get_predictions(X_train, y_train, X_test)
        plot_all_models(y_test, preds, ticker)


if __name__ == "__main__":
    run_prediction_plots()
