"""
Module for loading and preprocessing stock market data.

Pipeline:
  1. Load CSV (Date, Open, High, Low, Close, Volume)
  2. Compute target: price_change = Close[t+1] - Close[t]
  3. MinMax normalization (manual NumPy) — fit ONLY on the train set
  4. Chronological 80/20 split (no shuffling!)
"""

import numpy as np
import pandas as pd
import os


# Input feature column names
FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def load_csv(path):
    """Loads a CSV file with OHLCV data and returns a DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df


def compute_target(df):
    """
    Computes the closing price change as the target.
    price_change[t] = Close[t+1] - Close[t]
    The last row is dropped (no next day available).
    """
    changes = df["Close"].diff(1).shift(-1)  # forward difference
    df = df.copy()
    df["price_change"] = changes
    df = df.dropna()                         # drop last row without target
    return df


def minmax_normalize(X_train, X_test):
    """
    MinMax normalization implemented manually in NumPy.
    Fit is performed ONLY on training data — prevents data leakage.

    Returns:
      X_train_norm, X_test_norm, min_val, max_val
    """
    min_val = X_train.min(axis=0)
    max_val = X_train.max(axis=0)
    range_ = max_val - min_val

    # Protection against division by zero (constant feature)
    range_[range_ == 0] = 1.0

    X_train_norm = (X_train - min_val) / range_
    X_test_norm  = (X_test  - min_val) / range_

    return X_train_norm, X_test_norm, min_val, max_val


def chronological_split(X, y, train_ratio=0.8):
    """
    Splits data into train/test preserving temporal order.
    No shuffling — important for time series!
    """
    n = len(X)
    n_train = int(n * train_ratio)

    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    return X_train, X_test, y_train, y_test


def load_and_preprocess(path, train_ratio=0.8):
    """
    Full data preprocessing pipeline.

    Arguments:
      path        -- path to the CSV file with OHLCV data
      train_ratio -- fraction of training data (default 0.8)

    Returns:
      X_train, X_test, y_train, y_test  -- NumPy arrays ready for training
    """
    # 1. Load data
    df = load_csv(path)

    # 2. Compute target (price change)
    df = compute_target(df)

    # 3. Extract feature matrix and target vector
    X = df[FEATURE_COLUMNS].values.astype(np.float64)
    y = df["price_change"].values.astype(np.float64).reshape(-1, 1)

    # 4. Chronological 80/20 split
    X_train, X_test, y_train, y_test = chronological_split(
        X, y, train_ratio
    )

    # 5. MinMax normalization (fit only on train!)
    X_train, X_test, _, _ = minmax_normalize(X_train, X_test)

    return X_train, X_test, y_train, y_test


# ── Quick module test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    test_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "REMX.csv"
    )

    print("Test preprocessing.py")
    print(f"  File: {test_path}")

    X_train, X_test, y_train, y_test = load_and_preprocess(test_path)

    print(f"  X_train: {X_train.shape}  |  X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape}  |  y_test: {y_test.shape}")
    print(f"  X_train min={X_train.min():.4f}, max={X_train.max():.4f}")
    print(f"  y_train mean={y_train.mean():.4f}, std={y_train.std():.4f}")
    print("  OK!")
