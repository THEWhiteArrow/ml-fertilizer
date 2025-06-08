import numpy as np
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.utils.validation import check_is_fitted


class MinibatchSGDClassifier(SGDClassifier):

    def __init__(
        self,
        minibatch_size=32,
        alpha=0.0001,
        l1_ratio=0.15,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-3,
        shuffle=True,
        verbose=0,
        epsilon=0.1,
        random_state=None,
        learning_rate="invscaling",
        eta0=0.01,
        power_t=0.25,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        warm_start=False,
        average=False,
    ):
        super().__init__(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            shuffle=shuffle,
            verbose=verbose,
            epsilon=epsilon,
            random_state=random_state,
            learning_rate=learning_rate,
            eta0=eta0,
            power_t=power_t,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            warm_start=warm_start,
            average=average,
        )
        self.minibatch_size = minibatch_size

    def _get_minibatch_size(self, n_samples):
        if isinstance(self.minibatch_size, float):
            if 0 < self.minibatch_size < 1:
                return max(1, int(n_samples * self.minibatch_size))
            else:
                raise ValueError("minibatch_size as a float must be between 0 and 1.")
        elif isinstance(self.minibatch_size, int):
            if self.minibatch_size > 0:
                return self.minibatch_size
            else:
                raise ValueError("minibatch_size as an integer must be greater than 0.")
        else:
            raise TypeError("minibatch_size must be an integer or a float.")


class MinibatchSGDRegressor(SGDRegressor):
    def __init__(
        self,
        minibatch_size=32,
        alpha=0.0001,
        l1_ratio=0.15,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-3,
        shuffle=True,
        verbose=0,
        epsilon=0.1,
        random_state=None,
        learning_rate="invscaling",
        eta0=0.01,
        power_t=0.25,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        warm_start=False,
        average=False,
    ):
        super().__init__(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            shuffle=shuffle,
            verbose=verbose,
            epsilon=epsilon,
            random_state=random_state,
            learning_rate=learning_rate,
            eta0=eta0,
            power_t=power_t,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            warm_start=warm_start,
            average=average,
        )
        self.minibatch_size = minibatch_size

    def _get_minibatch_size(self, n_samples):
        if isinstance(self.minibatch_size, float):
            if 0 < self.minibatch_size < 1:
                return max(1, int(n_samples * self.minibatch_size))
            else:
                raise ValueError("minibatch_size as a float must be between 0 and 1.")
        elif isinstance(self.minibatch_size, int):
            if self.minibatch_size > 0:
                return self.minibatch_size
            else:
                raise ValueError("minibatch_size as an integer must be greater than 0.")
        else:
            raise TypeError("minibatch_size must be an integer or a float.")

    def fit(self, X, y):
        n_samples = X.shape[0]
        minibatch_size = self._get_minibatch_size(n_samples)
        for _ in range(self.max_iter):
            indices = np.random.permutation(n_samples)
            for start in range(0, n_samples, minibatch_size):
                end = start + minibatch_size
                minibatch_indices = indices[start:end]
                X_minibatch = (
                    X.iloc[minibatch_indices]
                    if hasattr(X, "iloc")
                    else X[minibatch_indices]
                )
                y_minibatch = (
                    y.iloc[minibatch_indices]
                    if hasattr(y, "iloc")
                    else y[minibatch_indices]
                )
                super().partial_fit(X_minibatch, y_minibatch)
        return self

    def partial_fit(self, X, y):
        n_samples = X.shape[0]
        minibatch_size = self._get_minibatch_size(n_samples)
        indices = np.random.permutation(n_samples)
        for start in range(0, n_samples, minibatch_size):
            end = start + minibatch_size
            minibatch_indices = indices[start:end]
            X_minibatch = (
                X.iloc[minibatch_indices]
                if hasattr(X, "iloc")
                else X[minibatch_indices]
            )
            y_minibatch = (
                y.iloc[minibatch_indices]
                if hasattr(y, "iloc")
                else y[minibatch_indices]
            )
            super().partial_fit(X_minibatch, y_minibatch)
        return self

    def predict(self, X):
        check_is_fitted(self, attributes=["coef_", "intercept_"])
        return super().predict(X)
