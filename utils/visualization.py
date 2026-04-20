"""
Visualization module for REEprediction model results.

Functions:
  - plot_predictions    -- predictions vs. actual values
  - plot_learning_curve -- MSE loss across epochs
  - save_all_plots      -- generates both plots for the best configuration
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless backend (save to PNG)
import matplotlib.pyplot as plt
import os

# Plot output directory
PLOTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "plots"
)


def plot_predictions(y_true, y_pred, ticker, png_path=None, show=False):
    """
    Creates a plot comparing model predictions with actual values.

    Parameters:
      y_true   -- array of actual price changes
      y_pred   -- array of model predictions
      ticker   -- instrument name (for the title)
      png_path -- PNG save path (None = auto)
      show     -- whether to call plt.show() (False when saving to file)
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    n = len(y_true)
    indices = np.arange(n)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f"Predictions vs Actual — {ticker}", fontsize=14, fontweight="bold")

    # Top panel: time series
    ax1 = axes[0]
    ax1.plot(indices, y_true, label="Actual price change",
             color="#2196F3", linewidth=1.2, alpha=0.85)
    ax1.plot(indices, y_pred, label="Model prediction",
             color="#F44336", linewidth=1.0, alpha=0.75, linestyle="--")
    ax1.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax1.set_xlabel("Sample (business day)")
    ax1.set_ylabel("Price change Close[t+1] − Close[t]")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Bottom panel: residuals
    ax2 = axes[1]
    residuals = y_pred - y_true
    ax2.bar(indices, residuals, color="#9C27B0", alpha=0.5, width=1.0)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Sample")
    ax2.set_ylabel("Residual")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    if png_path is None:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        png_path = os.path.join(PLOTS_DIR, f"predictions_{ticker}.png")

    plt.savefig(png_path, dpi=120, bbox_inches="tight")
    print(f"  Saved predictions plot: {png_path}")

    if show:
        plt.show()
    plt.close()

    return png_path


def plot_learning_curve(loss_history, ticker, config="",
                        png_path=None, show=False):
    """
    Plots the learning curve (MSE loss vs. epoch).

    Parameters:
      loss_history -- list of MSE loss values per epoch (returned by train_model)
      ticker       -- instrument name
      config       -- configuration description (for the title)
      png_path     -- PNG save path (None = auto)
      show         -- whether to call plt.show()
    """
    epochs = np.arange(1, len(loss_history) + 1)
    losses = np.array(loss_history)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, losses, color="#4CAF50", linewidth=1.5, label="MSE loss (train)")

    # Mark minimum
    idx_min = np.argmin(losses)
    ax.scatter(epochs[idx_min], losses[idx_min],
               color="red", zorder=5, s=60,
               label=f"Min loss = {losses[idx_min]:.4f} (epoch {epochs[idx_min]})")

    title = f"Learning curve — {ticker}"
    if config:
        title += f"  |  {config}"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    if png_path is None:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        png_path = os.path.join(PLOTS_DIR, f"learning_curve_{ticker}.png")

    plt.savefig(png_path, dpi=120, bbox_inches="tight")
    print(f"  Saved learning curve: {png_path}")

    if show:
        plt.show()
    plt.close()

    return png_path


def save_all_plots(model, X_test, y_test, loss_history,
                   ticker, hidden, lr, epochs):
    """
    Generates and saves both plots for a given model configuration.

    Returns a list of saved PNG file paths.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    y_pred = model.forward_propagation(X_test)
    description = f"hidden={'-'.join(str(h) for h in hidden)} lr={lr} epochs={epochs}"

    paths = [
        plot_predictions(y_test, y_pred, ticker),
        plot_learning_curve(loss_history, ticker, config=description),
    ]
    return paths


# ── Quick module test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, ROOT)

    from model.mlp import MLP
    from train.train import train_model
    from utils.preprocessing import load_and_preprocess

    ticker  = "AMG_AS"
    hidden  = [10]
    lr      = 0.01
    epochs  = 500
    path    = os.path.join(ROOT, "data", f"{ticker}.csv")

    print(f"Generating plots for {ticker} (hidden={hidden}, lr={lr}, epochs={epochs})")
    np.random.seed(42)

    X_train, X_test, y_train, y_test = load_and_preprocess(path)
    model = MLP(input_size=5, hidden_layers=hidden, output_size=1)
    history = train_model(model, X_train, y_train, epochs=epochs, lr=lr)

    files = save_all_plots(model, X_test, y_test, history,
                           ticker, hidden, lr, epochs)
    print(f"\nAll plots ({len(files)}) saved in: {PLOTS_DIR}")
