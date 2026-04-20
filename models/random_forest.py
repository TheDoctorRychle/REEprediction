"""
Random Forest Regressor model module.

Wrapper around sklearn.ensemble.RandomForestRegressor adapted to the
REEprediction project interface (train / predict / get_params methods).
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor


class RandomForestModel:
    """
    Random Forest model for regression of stock price changes.

    Parameters:
      n_estimators -- number of trees in the forest (default 100)
      max_depth    -- maximum depth of each tree (default None = no limit)
      random_state -- random seed for reproducibility (default 42)
    """

    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        # Number of decision trees in the forest
        self.n_estimators = n_estimators
        # Maximum tree depth (None means full growth)
        self.max_depth = max_depth
        # Random generator seed — guarantees reproducibility
        self.random_state = random_state

        # Internal sklearn model — initialized on first training
        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,          # use all CPU cores
        )

    def train(self, X_train, y_train):
        """
        Trains the Random Forest model on the training set.

        Arguments:
          X_train -- training feature matrix (numpy array, shape [n, 5])
          y_train -- target vector (numpy array, shape [n, 1] or [n])
        """
        # sklearn expects a 1D vector — flatten if needed
        y_flat = y_train.ravel()
        self._model.fit(X_train, y_flat)

    def predict(self, X_test):
        """
        Generates predictions for the test set.

        Arguments:
          X_test -- test feature matrix (numpy array, shape [m, 5])

        Returns:
          y_pred -- prediction vector (numpy array, shape [m, 1])
        """
        # Return a column vector for compatibility with the rest of the project
        return self._model.predict(X_test).reshape(-1, 1)

    def forward_propagation(self, X):
        """
        Alias for predict — compatibility with the MLP interface used in evaluate_model().
        """
        return self.predict(X)

    def get_params(self):
        """
        Returns a dictionary of model parameters.

        Useful for logging results and comparing configurations.
        """
        return {
            "model":        "RandomForest",
            "n_estimators": self.n_estimators,
            "max_depth":    self.max_depth,
            "random_state": self.random_state,
        }
