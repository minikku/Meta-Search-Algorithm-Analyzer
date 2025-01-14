import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, RegressorMixin


class CustomRandomForestRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, max_depth=None, max_features="sqrt", bootstrap=True, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []
        self.feature_indices = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.trees = []
        self.feature_indices = []

        n_samples, n_features = X.shape
        for _ in range(self.n_estimators):
            if isinstance(self.max_features, str) and self.max_features == "sqrt":
                n_selected_features = int(np.sqrt(n_features))
            elif isinstance(self.max_features, str) and self.max_features == "log2":
                n_selected_features = int(np.log2(n_features))
            elif isinstance(self.max_features, int):
                n_selected_features = self.max_features
            else:
                n_selected_features = n_features

            feature_idx = np.random.choice(n_features, n_selected_features, replace=False)
            self.feature_indices.append(feature_idx)

            if self.bootstrap:
                sample_idx = np.random.choice(n_samples, n_samples, replace=True)
            else:
                sample_idx = np.arange(n_samples)

            X_sample = X.iloc[sample_idx, feature_idx]
            y_sample = y.iloc[sample_idx]

            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = []
        for tree, features in zip(self.trees, self.feature_indices):
            tree_preds.append(tree.predict(X.iloc[:, features]))
        return np.mean(tree_preds, axis=0)
