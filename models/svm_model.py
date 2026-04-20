"""
Support Vector Regression (SVR) model module.

Wrapper around sklearn.svm.SVR adapted to the REEprediction project
interface (train / predict / get_params methods).
"""

import numpy as np
from sklearn.svm import SVR


class SVMModel:
    """
    Support Vector Regression model for predicting stock price changes.

    SVR searches for a hyperplane that fits as many points as possible
    within a tube of width epsilon, while minimizing errors for points
    outside the tube (controlled by the C parameter).

    Parameters:
      kernel  -- kernel type: 'rbf', 'linear', 'poly', etc. (default 'rbf')
      C       -- regularization parameter — larger C = less regularization (default 1.0)
      epsilon -- width of the no-penalty tube (default 0.1)
    """

    def __init__(self, kernel="rbf", C=1.0, epsilon=0.1):
        # Kernel function type (kernel trick) — 'rbf' works well for nonlinear data
        self.kernel = kernel
        # C parameter controls the trade-off between smoothness and data fit
        self.C = C
        # Epsilon — tolerance zone where no penalty is applied
        self.epsilon = epsilon

        # Internal sklearn model
        self._model = SVR(
            kernel=self.kernel,
            C=self.C,
            epsilon=self.epsilon,
        )

    def train(self, X_train, y_train):
        """
        Trains the SVR model on the training set.

        Arguments:
          X_train -- training feature matrix (numpy array, shape [n, 5])
          y_train -- target vector (numpy array, shape [n, 1] or [n])
        """
        # sklearn SVR expects a 1D vector — flatten if needed
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
        # Return a column vector compatible with the rest of the project
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
            "model":   "SVM",
            "kernel":  self.kernel,
            "C":       self.C,
            "epsilon": self.epsilon,
        }
