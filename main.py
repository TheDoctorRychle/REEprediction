"""
Główny skrypt projektu REEprediction.
Przewiduje zmianę ceny akcji spółek powiązanych z metalami ziem rzadkich (REE).
Dostępne modele: MLP (NumPy), Random Forest, SVM (scikit-learn).

Użycie:
  python main.py --ticker REMX --model mlp --epochs 300 --lr 0.01 --hidden 10 5
  python main.py --ticker KGH_WA --model rf --n-estimators 100
  python main.py --ticker AMG_AS --model svm --C 1.0
  python main.py --ticker REMX   (domyślne: mlp, parametry z config.py)
"""

import argparse
import os
import sys
import numpy as np

# Dodaj katalog projektu do ścieżki (umożliwia uruchomienie z dowolnego katalogu)
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from model.mlp import MLP
from models.random_forest import RandomForestModel
from models.svm_model import SVMModel
from train.train import train_model
from evaluate.metrics import evaluate_model, mae as mae_fn, rmse as rmse_fn, direction_accuracy as dir_acc_fn
from utils.preprocessing import load_and_preprocess
from utils.visualization import zapisz_wszystkie_wykresy
import config as cfg


# Dostępne tickery i ich pliki CSV
DOSTEPNE_TICKERY = {
    "REMX":   os.path.join(ROOT, "data", "REMX.csv"),
    "AMG_AS": os.path.join(ROOT, "data", "AMG_AS.csv"),
    "KGH_WA": os.path.join(ROOT, "data", "KGH_WA.csv"),
}


def parsuj_argumenty():
    """Parsuje argumenty linii poleceń."""
    parser = argparse.ArgumentParser(
        description="REEprediction — predykcja cen akcji spółek REE (MLP / RF / SVM)"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="REMX",
        choices=list(DOSTEPNE_TICKERY.keys()),
        help="Instrument giełdowy: REMX, AMG_AS, KGH_WA (domyślnie: REMX)"
    )
    # ── Wybór modelu ────────────────────────────────────────────────────────
    parser.add_argument(
        "--model",
        type=str,
        default="mlp",
        choices=["mlp", "rf", "svm"],
        help="Model do treningu: mlp | rf (Random Forest) | svm (domyślnie: mlp)"
    )
    # ── Parametry MLP ───────────────────────────────────────────────────────
    parser.add_argument(
        "--epochs",
        type=int,
        default=cfg.EPOCHS,
        help=f"[MLP] Liczba epok treningowych (domyślnie: {cfg.EPOCHS})"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=cfg.LEARNING_RATE,
        help=f"[MLP] Współczynnik uczenia (domyślnie: {cfg.LEARNING_RATE})"
    )
    parser.add_argument(
        "--hidden",
        type=int,
        nargs="+",
        default=cfg.HIDDEN_LAYERS,
        help=f"[MLP] Rozmiary warstw ukrytych (domyślnie: {cfg.HIDDEN_LAYERS}). "
             f"Przykład: --hidden 10 5  →  [10, 5]"
    )
    # ── Parametry Random Forest ─────────────────────────────────────────────
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="[RF] Liczba drzew w lesie (domyślnie: 100)"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="[RF] Maksymalna głębokość drzewa (domyślnie: None = bez limitu)"
    )
    # ── Parametry SVM ───────────────────────────────────────────────────────
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="[SVM] Parametr regularyzacji C (domyślnie: 1.0)"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="[SVM] Epsilon — szerokość tunelu bez kary (domyślnie: 0.1)"
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="rbf",
        choices=["rbf", "linear", "poly"],
        help="[SVM] Typ jądra (domyślnie: rbf)"
    )
    # ── Inne ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--no-plot",
        action="store_true",
        default=False,
        help="Nie generuj wykresów PNG"
    )
    return parser.parse_args()


def main():
    args = parsuj_argumenty()

    print("=" * 60)
    print("  REEprediction — Predykcja cen akcji REE")
    print("=" * 60)
    print(f"  Ticker  : {args.ticker}")
    print(f"  Model   : {args.model.upper()}")

    # 1. Wczytaj i przetwórz dane
    sciezka_csv = DOSTEPNE_TICKERY[args.ticker]
    if not os.path.exists(sciezka_csv):
        print(f"BŁĄD: Brak pliku danych: {sciezka_csv}")
        print("Uruchom najpierw: python data/download_data.py")
        sys.exit(1)

    print("\nWczytywanie i preprocessing danych...")
    X_train, X_test, y_train, y_test = load_and_preprocess(sciezka_csv)
    print(f"  Train: {X_train.shape[0]} próbek  |  Test: {X_test.shape[0]} próbek\n")

    historia = None   # tylko MLP generuje historię strat

    # 2. Zainicjuj i wytrenuj model
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
        print(f"\nModel MLP: {cfg.INPUT_SIZE} → {' → '.join(str(h) for h in args.hidden)} → {cfg.OUTPUT_SIZE}")
        print(f"Trening ({args.epochs} epok, lr={args.lr})...")
        historia = train_model(model, X_train, y_train,
                               epochs=args.epochs, lr=args.lr, verbose=True)
        print(f"  Końcowa strata MSE (train): {historia[-1]:.6f}\n")
        metryki = evaluate_model(model, X_test, y_test)

    elif args.model == "rf":
        print(f"  n_estimators: {args.n_estimators}")
        print(f"  max_depth   : {args.max_depth}")
        model = RandomForestModel(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth
        )
        print(f"\nTrening Random Forest ({args.n_estimators} drzew)...")
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)
        metryki = {
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
        print(f"\nTrening SVM (kernel={args.kernel}, C={args.C})...")
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)
        metryki = {
            "mae":                mae_fn(y_test, y_pred),
            "mse":                float(np.mean((y_test - y_pred) ** 2)),
            "rmse":               rmse_fn(y_test, y_pred),
            "direction_accuracy": dir_acc_fn(y_test, y_pred),
        }

    # 3. Wyniki
    print("Wyniki na zbiorze TESTOWYM:")
    print(f"  MAE              : {metryki['mae']:.4f}")
    print(f"  MSE              : {metryki['mse']:.4f}")
    print(f"  RMSE             : {metryki['rmse']:.4f}")
    print(f"  Dokładność kier. : {metryki['direction_accuracy']:.1f}%")

    # 4. Wizualizacje (tylko dla MLP — używają historii strat)
    if not args.no_plot and args.model == "mlp" and historia is not None:
        print("\nGenerowanie wykresów...")
        pliki = zapisz_wszystkie_wykresy(
            model, X_test, y_test, historia,
            ticker=args.ticker,
            hidden=args.hidden,
            lr=args.lr,
            epochs=args.epochs
        )
        print(f"  Wykresy zapisane ({len(pliki)} pliki PNG)")

    print("\nZakończono.")


if __name__ == "__main__":
    main()
