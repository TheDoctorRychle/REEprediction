"""
Script for downloading/generating stock market data for REE instruments.

Attempts to download data from Yahoo Finance (yfinance). If there is no
connection, generates realistic synthetic data using Geometric Brownian
Motion with parameters based on the historical characteristics of the
instruments.

Instruments:
  - REMX   : VanEck Rare Earth ETF (USD)
  - AMG.AS : AMG Critical Materials, Amsterdam (EUR)
  - KGH.WA : KGHM Polska Miedź, Warsaw (PLN)
"""

import numpy as np
import pandas as pd
import os
import sys

# Date range
START_DATE = "2020-01-01"
END_DATE   = "2025-04-01"

# Output directory
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Historical parameters for synthetic data generation
# (daily drift, daily sigma, starting price)
PARAMETERS = {
    "REMX":   {"mu": 0.0003,  "sigma": 0.020, "S0": 52.0,  "vol_base": 500_000},
    "AMG_AS": {"mu": 0.0002,  "sigma": 0.022, "S0": 28.5,  "vol_base": 80_000},
    "KGH_WA": {"mu": 0.00015, "sigma": 0.018, "S0": 112.0, "vol_base": 600_000},
}

TICKERS_YF = {
    "REMX":   "REMX",
    "AMG_AS": "AMG.AS",
    "KGH_WA": "KGH.WA",
}


def generate_gbm_data(params, dates):
    """
    Generates OHLCV via Geometric Brownian Motion.
    OHLC are created as perturbations of each day's closing price.
    """
    np.random.seed(42)
    n = len(dates)
    mu    = params["mu"]
    sigma = params["sigma"]
    S0    = params["S0"]

    # Generate daily returns
    returns = np.random.normal(mu - 0.5 * sigma**2, sigma, n)
    close_prices = S0 * np.exp(np.cumsum(returns))

    # Simulate OHLC
    noise = sigma * 0.5
    open_ = close_prices * np.exp(np.random.normal(0, noise * 0.3, n))
    high  = np.maximum(close_prices, open_) * (1 + np.abs(np.random.normal(0, noise, n)))
    low   = np.minimum(close_prices, open_) * (1 - np.abs(np.random.normal(0, noise, n)))

    # Ensure High >= Open,Close and Low <= Open,Close
    high = np.maximum(high, np.maximum(close_prices, open_))
    low  = np.minimum(low,  np.minimum(close_prices, open_))

    # Volume with trend and noise
    vol_base   = params["vol_base"]
    vol_trend  = np.linspace(1.0, 1.5, n)
    vol_noise  = np.abs(np.random.normal(1.0, 0.3, n))
    volume     = (vol_base * vol_trend * vol_noise).astype(int)

    df = pd.DataFrame({
        "Open":   np.round(open_, 4),
        "High":   np.round(high, 4),
        "Low":    np.round(low, 4),
        "Close":  np.round(close_prices, 4),
        "Volume": volume,
    }, index=dates)
    df.index.name = "Date"
    return df


def download_yfinance(ticker_yf, start, end):
    """Attempts to download data from Yahoo Finance. Returns None if no connection."""
    try:
        import yfinance as yf
        df = yf.download(ticker_yf, start=start, end=end,
                         auto_adjust=True, progress=False)
        if df.empty:
            return None
        # Flatten MultiIndex (yfinance >= 0.2.x)
        if hasattr(df.columns, 'levels'):
            df.columns = df.columns.get_level_values(0)
        columns = [k for k in ["Open", "High", "Low", "Close", "Volume"]
                   if k in df.columns]
        return df[columns].copy()
    except Exception:
        return None


def download_and_save(name, ticker_yf, params, start, end):
    """Downloads data (yfinance or GBM) and saves CSV."""
    print(f"--- {name} ({ticker_yf}) ---")

    # Try Yahoo Finance
    df = download_yfinance(ticker_yf, start, end)

    if df is not None:
        print(f"  [yfinance] {len(df)} records: "
              f"{df.index[0].date()} – {df.index[-1].date()}")
        source = "Yahoo Finance"
    else:
        print("  [yfinance] No connection — generating synthetic data (GBM)")
        # Business days (trading days)
        dates = pd.bdate_range(start=start, end=end)
        df = generate_gbm_data(params, dates)
        print(f"  [GBM] {len(df)} records: "
              f"{df.index[0].date()} – {df.index[-1].date()}")
        source = "GBM (synthetic)"

    path = os.path.join(DATA_DIR, f"{name}.csv")
    df.to_csv(path)
    size = os.path.getsize(path)
    print(f"  Saved: {path}  ({size:,} B, source: {source})\n")
    return df


def main():
    print("=" * 55)
    print("  Downloading REE data — 2020-01-01 → 2025-04-01")
    print("=" * 55 + "\n")

    for name, ticker_yf in TICKERS_YF.items():
        download_and_save(name, ticker_yf, PARAMETERS[name], START_DATE, END_DATE)

    print("=== Done ===")


if __name__ == "__main__":
    main()
