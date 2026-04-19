"""
Porównanie modeli MLP, Random Forest i SVM na danych REE.

Dla każdego z 3 instrumentów (REMX, AMG_AS, KGH_WA) testowane są:
  - MLP       : hidden=[10], lr=0.01, epochs=300
  - RandomForest: n_estimators w [50, 100, 200]
  - SVM (SVR) : kernel='rbf', C w [0.1, 1.0, 10.0]

Wyniki:
  - Zapisane do results/comparison_results.csv
  - Tabela porównawcza w konsoli
  - Wykres grouped bar chart RMSE → results/plots/model_comparison.png
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # backend bez okna graficznego
import matplotlib.pyplot as plt

# Dodaj katalog główny projektu do ścieżki importów
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model.mlp import MLP
from train.train import train_model
from evaluate.metrics import evaluate_model, rmse as rmse_fn, mae as mae_fn, direction_accuracy as dir_acc_fn
from utils.preprocessing import load_and_preprocess
from models.random_forest import RandomForestModel
from models.svm_model import SVMModel

# ── Konfiguracja ─────────────────────────────────────────────────────────────

TICKERS = {
    "REMX":   os.path.join(ROOT, "data", "REMX.csv"),
    "AMG_AS": os.path.join(ROOT, "data", "AMG_AS.csv"),
    "KGH_WA": os.path.join(ROOT, "data", "KGH_WA.csv"),
}

# Najlepsza konfiguracja MLP z poprzednich eksperymentów
MLP_HIDDEN  = [10]
MLP_LR      = 0.01
MLP_EPOCHS  = 300
INPUT_SIZE  = 5
OUTPUT_SIZE = 1

# Siatka parametrów dla RF i SVM
RF_N_ESTIMATORS = [50, 100, 200]
SVM_C_VALUES    = [0.1, 1.0, 10.0]

# Pliki wyjściowe
WYNIKI_CSV   = os.path.join(ROOT, "results", "comparison_results.csv")
WYKRES_PNG   = os.path.join(ROOT, "results", "plots", "model_comparison.png")

# ── Pomocnicze ───────────────────────────────────────────────────────────────

def ewaluuj_sklearn(model_obj, X_test, y_test):
    """Ewaluuje model z interfejsem predict() (RF / SVM)."""
    y_pred = model_obj.predict(X_test)
    return {
        "mae":                mae_fn(y_test, y_pred),
        "rmse":               rmse_fn(y_test, y_pred),
        "direction_accuracy": dir_acc_fn(y_test, y_pred),
    }


def trenuj_mlp(X_train, y_train, X_test, y_test):
    """Trenuje MLP z ustalonymi parametrami i zwraca metryki oraz czas."""
    np.random.seed(42)
    model = MLP(input_size=INPUT_SIZE, hidden_layers=MLP_HIDDEN, output_size=OUTPUT_SIZE)
    t0 = time.time()
    train_model(model, X_train, y_train, epochs=MLP_EPOCHS, lr=MLP_LR, verbose=False)
    czas = time.time() - t0
    metryki = evaluate_model(model, X_test, y_test)
    return metryki, czas


def trenuj_rf(X_train, y_train, X_test, y_test, n_estimators):
    """Trenuje Random Forest z zadaną liczbą drzew i zwraca metryki."""
    model = RandomForestModel(n_estimators=n_estimators)
    t0 = time.time()
    model.train(X_train, y_train)
    czas = time.time() - t0
    metryki = ewaluuj_sklearn(model, X_test, y_test)
    return metryki, czas


def trenuj_svm(X_train, y_train, X_test, y_test, C):
    """Trenuje SVR z zadanym parametrem C i zwraca metryki."""
    model = SVMModel(kernel="rbf", C=C)
    t0 = time.time()
    model.train(X_train, y_train)
    czas = time.time() - t0
    metryki = ewaluuj_sklearn(model, X_test, y_test)
    return metryki, czas


# ── Tabelka w konsoli ────────────────────────────────────────────────────────

def drukuj_tabele(wiersze):
    """Drukuje czytelną tabelę porównawczą w konsoli."""
    szer = 100
    print("\n" + "=" * szer)
    print(f"{'TICKER':<10} {'MODEL':<15} {'PARAMETRY':<22} "
          f"{'MAE':>9} {'RMSE':>9} {'DIR%':>7} {'CZAS':>7}")
    print("=" * szer)

    poprzedni = None
    for w in wiersze:
        if w["ticker"] != poprzedni and poprzedni is not None:
            print("-" * szer)
        poprzedni = w["ticker"]
        print(f"{w['ticker']:<10} {w['model']:<15} {w['parametry']:<22} "
              f"{w['mae']:>9.4f} {w['rmse']:>9.4f} "
              f"{w['direction_accuracy']:>6.1f}% "
              f"{w['czas_s']:>6.2f}s")

    print("=" * szer)


# ── Wykres porównawczy ───────────────────────────────────────────────────────

def rysuj_wykres(najlepsze_rmse, tickers, modele):
    """
    Tworzy grouped bar chart RMSE: 3 grupy (tickery) × 3 słupki (modele).

    najlepsze_rmse -- słownik {ticker: {model: rmse_value}}
    """
    os.makedirs(os.path.dirname(WYKRES_PNG), exist_ok=True)

    x       = np.arange(len(tickers))
    szerokosc = 0.25
    kolory  = ["#4C72B0", "#DD8452", "#55A868"]   # niebieski, pomarańczowy, zielony

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, model_nazwa in enumerate(modele):
        wartosci = [najlepsze_rmse[t][model_nazwa] for t in tickers]
        bary = ax.bar(x + i * szerokosc, wartosci, szerokosc,
                      label=model_nazwa, color=kolory[i], alpha=0.85, edgecolor="white")
        # Etykiety wartości nad słupkami
        for bar, val in zip(bary, wartosci):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Instrument giełdowy", fontsize=12)
    ax.set_ylabel("RMSE (najlepsza konfiguracja)", fontsize=12)
    ax.set_title("Porównanie modeli — RMSE na zbiorze testowym\n"
                 "(MLP vs Random Forest vs SVM)", fontsize=13, fontweight="bold")
    ax.set_xticks(x + szerokosc)
    ax.set_xticklabels(tickers, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(WYKRES_PNG, dpi=150)
    plt.close()
    print(f"\nWykres zapisany: {WYKRES_PNG}")


# ── Główna funkcja ───────────────────────────────────────────────────────────

def porownaj_modele():
    os.makedirs(os.path.dirname(WYNIKI_CSV), exist_ok=True)

    wszystkie_wiersze = []   # wszystkie wyniki do CSV
    # najlepszy RMSE dla każdego (ticker, model) — potrzebne do wykresu
    najlepsze_rmse = {t: {} for t in TICKERS}

    for ticker, sciezka_csv in TICKERS.items():
        print(f"\n{'━' * 70}")
        print(f"  Instrument: {ticker}")
        print(f"{'━' * 70}")

        X_train, X_test, y_train, y_test = load_and_preprocess(sciezka_csv)
        print(f"  Train: {X_train.shape[0]} próbek | Test: {X_test.shape[0]} próbek")

        # ── MLP ────────────────────────────────────────────────────────────
        print(f"\n  [MLP] hidden={MLP_HIDDEN}, lr={MLP_LR}, epochs={MLP_EPOCHS} ...", end=" ", flush=True)
        metryki, czas = trenuj_mlp(X_train, y_train, X_test, y_test)
        print(f"RMSE={metryki['rmse']:.4f}  Dir={metryki['direction_accuracy']:.1f}%")

        wiersz_mlp = {
            "ticker":             ticker,
            "model":              "MLP",
            "parametry":          f"hidden={MLP_HIDDEN} lr={MLP_LR}",
            "mae":                metryki["mae"],
            "rmse":               metryki["rmse"],
            "direction_accuracy": metryki["direction_accuracy"],
            "czas_s":             round(czas, 3),
        }
        wszystkie_wiersze.append(wiersz_mlp)
        najlepsze_rmse[ticker]["MLP"] = metryki["rmse"]

        # ── Random Forest ──────────────────────────────────────────────────
        rf_rmse_min = float("inf")
        for n in RF_N_ESTIMATORS:
            print(f"  [RF]  n_estimators={n:<3} ...", end=" ", flush=True)
            metryki, czas = trenuj_rf(X_train, y_train, X_test, y_test, n)
            print(f"RMSE={metryki['rmse']:.4f}  Dir={metryki['direction_accuracy']:.1f}%")

            wszystkie_wiersze.append({
                "ticker":             ticker,
                "model":              "RandomForest",
                "parametry":          f"n_est={n}",
                "mae":                metryki["mae"],
                "rmse":               metryki["rmse"],
                "direction_accuracy": metryki["direction_accuracy"],
                "czas_s":             round(czas, 3),
            })
            if metryki["rmse"] < rf_rmse_min:
                rf_rmse_min = metryki["rmse"]

        najlepsze_rmse[ticker]["RandomForest"] = rf_rmse_min

        # ── SVM ────────────────────────────────────────────────────────────
        svm_rmse_min = float("inf")
        for c in SVM_C_VALUES:
            print(f"  [SVM] C={c:<5} ...", end=" ", flush=True)
            metryki, czas = trenuj_svm(X_train, y_train, X_test, y_test, c)
            print(f"RMSE={metryki['rmse']:.4f}  Dir={metryki['direction_accuracy']:.1f}%")

            wszystkie_wiersze.append({
                "ticker":             ticker,
                "model":              "SVM",
                "parametry":          f"rbf C={c}",
                "mae":                metryki["mae"],
                "rmse":               metryki["rmse"],
                "direction_accuracy": metryki["direction_accuracy"],
                "czas_s":             round(czas, 3),
            })
            if metryki["rmse"] < svm_rmse_min:
                svm_rmse_min = metryki["rmse"]

        najlepsze_rmse[ticker]["SVM"] = svm_rmse_min

    # ── Zapis do CSV ──────────────────────────────────────────────────────────
    df_wyniki = pd.DataFrame(wszystkie_wiersze)
    df_wyniki.to_csv(WYNIKI_CSV, index=False, encoding="utf-8")
    print(f"\nWyniki zapisane: {WYNIKI_CSV}")

    # ── Tabela w konsoli ───────────────────────────────────────────────────
    drukuj_tabele(wszystkie_wiersze)

    # ── Podsumowanie najlepszych RMSE ─────────────────────────────────────
    print("\n" + "=" * 55)
    print("  PODSUMOWANIE — Najlepsze RMSE per instrument/model")
    print("=" * 55)
    for ticker in TICKERS:
        print(f"\n  {ticker}:")
        for model_nazwa, rmse_val in najlepsze_rmse[ticker].items():
            print(f"    {model_nazwa:<15} RMSE = {rmse_val:.4f}")

    # ── Wykres ─────────────────────────────────────────────────────────────
    rysuj_wykres(najlepsze_rmse, list(TICKERS.keys()), ["MLP", "RandomForest", "SVM"])

    return wszystkie_wiersze, najlepsze_rmse


if __name__ == "__main__":
    porownaj_modele()
