"""
Grid search experiments for the MLP model on REE data.

Tested configurations:
  - hidden_sizes : [5], [10], [10, 5]
  - learning_rate: 0.01, 0.001
  - epochs       : 100, 300, 500

For each of the 3 instruments (REMX, AMG_AS, KGH_WA).
Results are saved to results/results.csv and printed as a table.
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import csv

# Add the project root to the import path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model.mlp import MLP
from train.train import train_model
from evaluate.metrics import evaluate_model
from utils.preprocessing import load_and_preprocess

# ── Grid search configuration ────────────────────────────────────────────────

TICKERS = {
    "REMX":   os.path.join(ROOT, "data", "REMX.csv"),
    "AMG_AS": os.path.join(ROOT, "data", "AMG_AS.csv"),
    "KGH_WA": os.path.join(ROOT, "data", "KGH_WA.csv"),
}

HIDDEN_SIZES   = [[5], [10], [10, 5]]
LEARNING_RATES = [0.01, 0.001]
EPOCHS_LIST    = [100, 300, 500]

INPUT_SIZE  = 5   # Open, High, Low, Close, Volume
OUTPUT_SIZE = 1   # price_change

RESULTS_CSV = os.path.join(ROOT, "results", "results.csv")
CSV_HEADERS = [
    "ticker", "hidden_sizes", "learning_rate", "epochs",
    "mae", "mse", "rmse", "direction_accuracy", "time_s"
]

# ── Helpers ──────────────────────────────────────────────────────────────────

def hidden_str(hidden):
    """Converts a list to a readable string, e.g. [10, 5] → '10-5'."""
    return "-".join(str(h) for h in hidden)


def print_table(results):
    """Prints results as a formatted table in the console."""
    print("\n" + "=" * 90)
    print(f"{'TICKER':<10} {'HIDDEN':<8} {'LR':<7} {'EPOCHS':<7} "
          f"{'MAE':>8} {'RMSE':>8} {'DIR%':>7} {'TIME':>6}")
    print("=" * 90)

    prev_ticker = None
    for r in results:
        if r["ticker"] != prev_ticker and prev_ticker is not None:
            print("-" * 90)
        prev_ticker = r["ticker"]

        print(f"{r['ticker']:<10} "
              f"{hidden_str(r['hidden_sizes']):<8} "
              f"{r['learning_rate']:<7} "
              f"{r['epochs']:<7} "
              f"{r['mae']:>8.4f} "
              f"{r['rmse']:>8.4f} "
              f"{r['direction_accuracy']:>6.1f}% "
              f"{r['time_s']:>5.1f}s")

    print("=" * 90)


def best_configuration(results):
    """Returns the row with the lowest RMSE."""
    return min(results, key=lambda r: r["rmse"])


# ── Main function ────────────────────────────────────────────────────────────

def run_experiments():
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)

    all_results = []
    total = len(TICKERS) * len(HIDDEN_SIZES) * len(LEARNING_RATES) * len(EPOCHS_LIST)
    counter = 0

    print(f"\nStarting grid search: {total} configurations × {len(TICKERS)} tickers\n")

    for ticker, csv_path in TICKERS.items():
        print(f"\n{'━' * 60}")
        print(f"  Instrument: {ticker}")
        print(f"{'━' * 60}")

        # Load and preprocess data once per ticker
        X_train, X_test, y_train, y_test = load_and_preprocess(csv_path)
        print(f"  Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

        for hidden in HIDDEN_SIZES:
            for lr in LEARNING_RATES:
                for epochs in EPOCHS_LIST:
                    counter += 1
                    description = (f"  [{counter:>3}/{total}] "
                                   f"hidden={hidden_str(hidden)} lr={lr} epochs={epochs}")
                    print(description, end="  ", flush=True)

                    # Initialize model with random weights (seed for reproducibility)
                    np.random.seed(42)
                    model = MLP(
                        input_size=INPUT_SIZE,
                        hidden_layers=hidden,
                        output_size=OUTPUT_SIZE
                    )

                    # Training
                    t0 = time.time()
                    train_model(model, X_train, y_train,
                                epochs=epochs, lr=lr, verbose=False)
                    elapsed = time.time() - t0

                    # Evaluation
                    metrics = evaluate_model(model, X_test, y_test)

                    result = {
                        "ticker":             ticker,
                        "hidden_sizes":       hidden,
                        "learning_rate":      lr,
                        "epochs":             epochs,
                        "mae":                metrics["mae"],
                        "mse":                metrics["mse"],
                        "rmse":               metrics["rmse"],
                        "direction_accuracy": metrics["direction_accuracy"],
                        "time_s":             round(elapsed, 2),
                    }
                    all_results.append(result)

                    print(f"RMSE={metrics['rmse']:.4f}  Dir={metrics['direction_accuracy']:.1f}%")

    # Save to CSV
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for r in all_results:
            row = dict(r)
            row["hidden_sizes"] = hidden_str(r["hidden_sizes"])
            writer.writerow(row)

    print(f"\nResults saved: {RESULTS_CSV}")

    # Print table
    print_table(all_results)

    # Best configuration
    best = best_configuration(all_results)
    print(f"\nBest configuration (lowest RMSE on test):")
    print(f"   Ticker: {best['ticker']}")
    print(f"   Hidden: {hidden_str(best['hidden_sizes'])}")
    print(f"   LR:     {best['learning_rate']}")
    print(f"   Epochs: {best['epochs']}")
    print(f"   MAE:    {best['mae']:.4f}")
    print(f"   RMSE:   {best['rmse']:.4f}")
    print(f"   Dir.Acc:{best['direction_accuracy']:.1f}%")

    return all_results


if __name__ == "__main__":
    run_experiments()
