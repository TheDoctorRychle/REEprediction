import numpy as np
import pandas as pd
import os


FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Volume", "Close_mean_5", "Close_q75_5"]


def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df

def add_historical_features(df, window=5):
    df = df.copy()
    df["Close_mean_5"] = df["Close"].rolling(window).mean()
    df["Close_q75_5"] = df["Close"].rolling(window).quantile(0.75)
    return df
def compute_target(df):
    changes = df["Close"].diff(1).shift(-1)
    df = df.copy()
    df["price_change"] = changes
    df = df.dropna()
    return df


def minmax_normalize(X_train, X_test):
    min_val = X_train.min(axis=0)
    max_val = X_train.max(axis=0)
    range_ = max_val - min_val

    range_[range_ == 0] = 1.0

    X_train_norm = (X_train - min_val) / range_
    X_test_norm  = (X_test  - min_val) / range_

    return X_train_norm, X_test_norm, min_val, max_val


def chronological_split(X, y, train_ratio=0.8):
    n = len(X)
    n_train = int(n * train_ratio)

    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    return X_train, X_test, y_train, y_test


def load_and_preprocess(path, train_ratio=0.8, feature_columns=None):
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS

    df = load_csv(path)
    df = add_historical_features(df)
    df = compute_target(df)
    df = df.dropna()

    X = df[feature_columns].values.astype(np.float64)
    y = df["price_change"].values.astype(np.float64).reshape(-1, 1)

    X_train, X_test, y_train, y_test = chronological_split(X, y, train_ratio)
    X_train, X_test, _, _ = minmax_normalize(X_train, X_test)

    return X_train, X_test, y_train, y_test


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
