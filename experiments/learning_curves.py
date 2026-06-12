import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model.mlp import MLP
from evaluate.metrics import rmse as rmse_fn
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

# fixed configurations used for the curves (defaults from the comparison study)
MLP_HIDDEN = [25]
MLP_LR     = 0.0001
MLP_EPOCHS = 500

TRAIN_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

PLOTS_DIR       = os.path.join(ROOT, "results", "plots")
EPOCH_CSV       = os.path.join(ROOT, "results", "mlp_epoch_curves.csv")
SIZE_CSV        = os.path.join(ROOT, "results", "learning_curves.csv")


def mlp_epoch_curve(X_train, y_train, X_test, y_test, ticker):
    """Train the MLP epoch by epoch, logging train loss and test RMSE."""
    np.random.seed(42)
    model = MLP(INPUT_SIZE, MLP_HIDDEN, OUTPUT_SIZE)

    rows = []
    for epoch in range(1, MLP_EPOCHS + 1):
        y_pred = model.forward_propagation(X_train)
        train_loss = float(((y_pred - y_train) ** 2).mean())
        grads_W, grads_b = model.backward_propagation(y_train)
        model.update(grads_W, grads_b, MLP_LR)

        test_rmse = rmse_fn(y_test, model.forward_propagation(X_test))
        rows.append({
            "ticker":     ticker,
            "epoch":      epoch,
            "train_mse":  train_loss,
            "test_rmse":  test_rmse,
        })
    return rows


def plot_mlp_epoch_curve(rows, ticker):
    epochs     = [r["epoch"] for r in rows]
    train_mse  = [r["train_mse"] for r in rows]
    test_rmse  = [r["test_rmse"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(epochs, train_mse, color="#4C72B0", linewidth=1.5, label="Train MSE loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train MSE loss", color="#4C72B0")
    ax1.tick_params(axis="y", labelcolor="#4C72B0")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(epochs, test_rmse, color="#C44E52", linewidth=1.5, linestyle="--",
             label="Test RMSE")
    ax2.set_ylabel("Test RMSE", color="#C44E52")
    ax2.tick_params(axis="y", labelcolor="#C44E52")

    best_epoch = int(np.argmin(test_rmse)) + 1
    ax2.scatter([best_epoch], [min(test_rmse)], color="black", zorder=5, s=50,
                label=f"Best test RMSE = {min(test_rmse):.4f} (epoch {best_epoch})")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")

    plt.title(f"MLP learning curve - {ticker}\n"
              f"hidden={MLP_HIDDEN} lr={MLP_LR} epochs={MLP_EPOCHS}",
              fontsize=12, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, f"learning_curve_mlp_{ticker}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {path}")


def make_models():
    """Factories so every train-size step starts from a fresh model."""
    def make_mlp():
        np.random.seed(42)
        return MLP(INPUT_SIZE, MLP_HIDDEN, OUTPUT_SIZE)

    return {
        "MLP":                  make_mlp,
        "RandomForest":         lambda: RandomForestModel(n_estimators=100),
        "SVM":                  lambda: SVMModel(kernel="rbf", C=1.0),
        "HistGradientBoosting": lambda: HistGradientBoostingModel(max_iter=300),
    }


def fit_predict(name, model, X_tr, y_tr, X_te):
    if name == "MLP":
        for _ in range(MLP_EPOCHS):
            model.forward_propagation(X_tr)
            grads_W, grads_b = model.backward_propagation(y_tr)
            model.update(grads_W, grads_b, MLP_LR)
        return model.forward_propagation(X_tr), model.forward_propagation(X_te)
    model.train(X_tr, y_tr)
    return model.predict(X_tr), model.predict(X_te)


def train_size_curves(X_train, y_train, X_test, y_test, ticker):
    rows = []
    factories = make_models()

    for frac in TRAIN_FRACTIONS:
        n = max(int(len(X_train) * frac), 30)
        # most recent samples - the train window always ends right before test
        X_tr, y_tr = X_train[-n:], y_train[-n:]

        for name, factory in factories.items():
            model = factory()
            pred_tr, pred_te = fit_predict(name, model, X_tr, y_tr, X_test)
            rows.append({
                "ticker":      ticker,
                "model":       name,
                "train_frac":  frac,
                "train_size":  n,
                "rmse_train":  rmse_fn(y_tr, pred_tr),
                "rmse_test":   rmse_fn(y_test, pred_te),
            })
        print(f"  frac={frac:.1f} ({n} samples) done")
    return rows


def plot_train_size_curves(rows, ticker):
    df = pd.DataFrame(rows)
    models = df["model"].unique()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    fig.suptitle(f"Learning curves (train set size) - {ticker}",
                 fontsize=14, fontweight="bold")

    for ax, name in zip(axes.flat, models):
        sub = df[df["model"] == name].sort_values("train_size")
        ax.plot(sub["train_size"], sub["rmse_train"], marker="o",
                color="#4C72B0", label="Train RMSE")
        ax.plot(sub["train_size"], sub["rmse_test"], marker="s",
                color="#C44E52", label="Test RMSE")
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("Training samples")
        ax.set_ylabel("RMSE")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"learning_curve_sizes_{ticker}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {path}")


def run_learning_curves():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    epoch_rows, size_rows = [], []

    for ticker, csv_path in TICKERS.items():
        print(f"\n{'=' * 70}")
        print(f"  Instrument: {ticker}")
        print(f"{'=' * 70}")
        X_train, X_test, y_train, y_test = load_and_preprocess(csv_path)

        print("  MLP epoch curve...")
        rows = mlp_epoch_curve(X_train, y_train, X_test, y_test, ticker)
        plot_mlp_epoch_curve(rows, ticker)
        epoch_rows.extend(rows)

        print("  Train-size curves (all models)...")
        rows = train_size_curves(X_train, y_train, X_test, y_test, ticker)
        plot_train_size_curves(rows, ticker)
        size_rows.extend(rows)

    pd.DataFrame(epoch_rows).to_csv(EPOCH_CSV, index=False, encoding="utf-8")
    print(f"\nEpoch curves saved: {EPOCH_CSV}")
    pd.DataFrame(size_rows).to_csv(SIZE_CSV, index=False, encoding="utf-8")
    print(f"Train-size curves saved: {SIZE_CSV}")


if __name__ == "__main__":
    run_learning_curves()
