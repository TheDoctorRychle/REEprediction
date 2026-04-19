"""
Moduł implementujący model Support Vector Regression (SVR).

Wrapper wokół sklearn.svm.SVR dopasowany do interfejsu projektu
REEprediction (metody train / predict / get_params).
"""

import numpy as np
from sklearn.svm import SVR


class SVMModel:
    """
    Model Support Vector Regression do predykcji zmian cen akcji.

    SVR szuka hiperpłaszczyzny, która mieści jak najwięcej punktów
    w tunelu szerokości epsilon, jednocześnie minimalizując błędy
    dla punktów poza tunelem (sterowane parametrem C).

    Parametry:
      kernel  -- typ jądra: 'rbf', 'linear', 'poly' itp. (domyślnie 'rbf')
      C       -- parametr regularyzacji — większe C = mniej regularyzacji (domyślnie 1.0)
      epsilon -- szerokość tunelu bez kary (domyślnie 0.1)
    """

    def __init__(self, kernel="rbf", C=1.0, epsilon=0.1):
        # Typ funkcji jądra (kernel trick) — 'rbf' sprawdza się dobrze dla danych nieliniowych
        self.kernel = kernel
        # Parametr C kontroluje kompromis między gładkością a dopasowaniem do danych
        self.C = C
        # Epsilon — strefa tolerancji, gdzie nie naliczamy kary za błąd
        self.epsilon = epsilon

        # Wewnętrzny model sklearn
        self._model = SVR(
            kernel=self.kernel,
            C=self.C,
            epsilon=self.epsilon,
        )

    def train(self, X_train, y_train):
        """
        Trenuje model SVR na zbiorze treningowym.

        Argumenty:
          X_train -- macierz cech treningowych (numpy array, shape [n, 5])
          y_train -- wektor wartości docelowych (numpy array, shape [n, 1] lub [n])
        """
        # sklearn SVR oczekuje wektora 1D — spłaszczamy jeśli trzeba
        y_flat = y_train.ravel()
        self._model.fit(X_train, y_flat)

    def predict(self, X_test):
        """
        Generuje predykcje dla zbioru testowego.

        Argumenty:
          X_test -- macierz cech testowych (numpy array, shape [m, 5])

        Zwraca:
          y_pred -- wektor predykcji (numpy array, shape [m, 1])
        """
        # Zwracamy kolumnowy wektor zgodny z resztą projektu
        return self._model.predict(X_test).reshape(-1, 1)

    def forward_propagation(self, X):
        """
        Alias metody predict — zgodność z interfejsem MLP używanym w evaluate_model().
        """
        return self.predict(X)

    def get_params(self):
        """
        Zwraca słownik z parametrami modelu.

        Przydatne do logowania wyników i porównywania konfiguracji.
        """
        return {
            "model":   "SVM",
            "kernel":  self.kernel,
            "C":       self.C,
            "epsilon": self.epsilon,
        }
