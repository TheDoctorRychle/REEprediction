"""
Moduł implementujący model Random Forest Regressor.

Wrapper wokół sklearn.ensemble.RandomForestRegressor dopasowany do
interfejsu projektu REEprediction (metody train / predict / get_params).
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor


class RandomForestModel:
    """
    Model Random Forest do regresji zmian cen akcji.

    Parametry:
      n_estimators -- liczba drzew w lesie (domyślnie 100)
      max_depth    -- maksymalna głębokość każdego drzewa (domyślnie None = bez limitu)
      random_state -- ziarno losowości dla powtarzalności wyników (domyślnie 42)
    """

    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        # Liczba drzew decyzyjnych tworzących las
        self.n_estimators = n_estimators
        # Maksymalna głębokość drzewa (None oznacza pełny rozwój)
        self.max_depth = max_depth
        # Ziarno generatora liczb losowych — gwarantuje powtarzalność
        self.random_state = random_state

        # Wewnętrzny model sklearn — inicjalizowany przy pierwszym treningu
        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,          # wykorzystaj wszystkie rdzenie procesora
        )

    def train(self, X_train, y_train):
        """
        Trenuje model Random Forest na zbiorze treningowym.

        Argumenty:
          X_train -- macierz cech treningowych (numpy array, shape [n, 5])
          y_train -- wektor wartości docelowych (numpy array, shape [n, 1] lub [n])
        """
        # sklearn oczekuje wektora 1D — spłaszczamy jeśli trzeba
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
        # Zwracamy kolumnowy wektor, żeby zachować zgodność z resztą projektu
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
            "model":        "RandomForest",
            "n_estimators": self.n_estimators,
            "max_depth":    self.max_depth,
            "random_state": self.random_state,
        }
