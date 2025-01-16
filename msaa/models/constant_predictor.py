import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor

class ConstantPredictor(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.name = 'ConstantPredictor'

    def __str__(self):
        return 'ConstantPredictor'

    def fit(self, X, y):
        pass

    def predict(self, X):
        best_so_far = X.iloc[:, -1]

        return best_so_far
