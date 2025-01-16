import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor


class CustomRandomForestRegressor2(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.model = RandomForestRegressor()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):

        result = self.model.predict(X)
        best_so_far = X.iloc[:, -1]

        for i in range(len(result)):
            if best_so_far.iloc[i] < result[i]:
                result[i] = best_so_far.iloc[i]

        return result
