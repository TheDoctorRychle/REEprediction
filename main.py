"""
Main script of the REEprediction project.
Predicts price changes of stocks related to Rare Earth Elements (REE).
Available models: MLP (NumPy), Random Forest, SVM (scikit-learn).

Usage:
  python main.py --ticker REMX --model mlp --epochs 300 --lr 0.01 --hidden 10 5
  python main.py --ticker KGH_WA --model rf --n-estimators 100
  python main.py --ticker AMG_AS --model svm --C 1.0
  python main.py --ticker REMX   (defaults: mlp, parameters from config.py)
"""

import argparse
import os
import sys
import numpy as np

# Add project directory to path (allows running from any directory)
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from model.mlp import MLP
from models.random_forest import RandomForestModel
from models.svm_model import SVMModel
from train.train import train_model
from evaluate.metrics import evaluate_model, mae as mae_fn, rmse as rmse_fn, direction_accuracy as dir_acc_fn
from utils.preprocessing import load_and_preprocess
from utils.visualization import save_all_plots
import config as cfg


# Available tickers and their CSV files
AVAILABLE_TICKERS = {
    "REMX":   os.path.join(ROOT, "data", "REMX.csv"),
    "AMG_AS": os.path.join(ROOT, "data", "AMG_AS.csv"),
    "KGH_WA": os.path.join(ROOT, "data", "KGH_WA.csv"),
}


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="REEprediction — stock price prediction for REE companies (MLP / RF / SVM)"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="REMX",
        choices=list(AVAILABLE_TICKERS.keys()),
        help="Stock instrument: REMX, AMG_AS, KGH_WA (default: REMX)"
    )
    # ── Model selection ─────────────────────────────────────────────────────
    parser.add_argument(
        "--model",
        type=str,
        default="mlp",
        choices=["mlp", "rf", "svm"],
        help="Model to train: mlp | rf (Random Forest) | svm (default: mlp)"
    )
    # ── MLP parameters ──────────────────────────────────────────────────────
    parser.add_argument(
        "--epochs",
        type=int,
        default=cfg.EPOCHS,
        help=f"[MLP] Number of training epochs (default: {cfg.EPOCHS})"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=cfg.LEARNING_RATE,
        help=f"[MLP] Learning rate (default: {cfg.LEARNING_RATE})"
    )
    parser.add_argument(
        "--hidden",
        type=int,
        nargs="+",
        default=cfg.HIDDEN_LAYERS,
        help=f"[MLP] Hidden layer sizes (default: {cfg.HIDDEN_LAYERS}). "
             f"Example: --hidden 10 5  →  [10, 5]"
    )
    # ── Random Forest parameters ────────────────────────────────────────────
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="[RF] Number of trees in the forest (default: 100)"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="[RF] Maximum tree depth (default: None = no limit)"
    )
    # ── SVM parameters ──────────────────────────────────────────────────────
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="[SVM] Regularization parameter C (default: 1.0)"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="[SVM] Epsilon — width of the no-penalty tube (default: 0.1)"
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="rbf",
        choices=["rbf", "linear", "poly"],
        help="[SVM] Kernel type (default: rbf)"
    )
    # ── Other ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--no-plot",
        action="store_true",
        default=False,
        help="Do not generate PNG plots"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("=" * 60)
    print("  REEprediction — REE stock price prediction")
    print("=" * 60)
    print(f"  Ticker  : {args.ticker}")
    print(f"  Model   : {args.model.upper()}")

    # 1. Load and preprocess data
    csv_path = AVAILABLE_TICKERS[args.ticker]
    if not os.path.exists(csv_path):
        print(f"ERROR: Data file not found: {csv_path}")
        print("Run first: python data/download_data.py")
        sys.exit(1)

    print("\nLoading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess(csv_path)
    print(f"  Train: {X_train.shape[0]} samples  |  Test: {X_test.shape[0]} samples\n")

    history = None   # only MLP produces a loss history

    # 2. Initialize and train the model
    if args.model == "mlp":
        print(f"  Hidden  : {args.hidden}")
        print(f"  LR      : {args.lr}")
        print(f"  Epochs  : {args.epochs}")
        np.random.seed(42)
        model = MLP(
            input_size=cfg.INPUT_SIZE,
            hidden_layers=args.hidden,
            output_size=cfg.OUTPUT_SIZE
        )
        print(f"\nMLP model: {cfg.INPUT_SIZE} → {' → '.join(str(h) for h in args.hidden)} → {cfg.OUTPUT_SIZE}")
        print(f"Training ({args.epochs} epochs, lr={args.lr})...")
        history = train_model(model, X_train, y_train,
                              epochs=args.epochs, lr=args.lr, verbose=True)
        print(f"  Final MSE loss (train): {history[-1]:.6f}\n")
        metrics = evaluate_model(model, X_test, y_test)

    elif args.model == "rf":
        print(f"  n_estimators: {args.n_estimators}")
        print(f"  max_depth   : {args.max_depth}")
        model = RandomForestModel(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth
        )
        print(f"\nTraining Random Forest ({args.n_estimators} trees)...")
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = {
            "mae":                mae_fn(y_test, y_pred),
            "mse":                float(np.mean((y_test - y_pred) ** 2)),
            "rmse":               rmse_fn(y_test, y_pred),
            "direction_accuracy": dir_acc_fn(y_test, y_pred),
        }

    elif args.model == "svm":
        print(f"  kernel  : {args.kernel}")
        print(f"  C       : {args.C}")
        print(f"  epsilon : {args.epsilon}")
        model = SVMModel(kernel=args.kernel, C=args.C, epsilon=args.epsilon)
        print(f"\nTraining SVM (kernel={args.kernel}, C={args.C})...")
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = {
            "mae":                mae_fn(y_test, y_pred),
            "mse":                float(np.mean((y_test - y_pred) ** 2)),
            "rmse":               rmse_fn(y_test, y_pred),
            "direction_accuracy": dir_acc_fn(y_test, y_pred),
        }

    # 3. Results
    print("Results on TEST set:")
    print(f"  MAE              : {metrics['mae']:.4f}")
    print(f"  MSE              : {metrics['mse']:.4f}")
    print(f"  RMSE             : {metrics['rmse']:.4f}")
    print(f"  Direction acc.   : {metrics['direction_accuracy']:.1f}%")

    # 4. Plots (only for MLP — they use the loss history)
    if not args.no_plot and args.model == "mlp" and history is not None:
        print("\nGenerating plots...")
        files = save_all_plots(
            model, X_test, y_test, history,
            ticker=args.ticker,
            hidden=args.hidden,
            lr=args.lr,
            epochs=args.epochs
        )
        print(f"  Plots saved ({len(files)} PNG files)")

    print("\nDone.")


if __name__ == "__main__":
    main()
