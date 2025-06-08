import warnings
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import (
    PassiveAggressiveClassifier,
    PassiveAggressiveRegressor,
    Ridge,
    RidgeClassifier,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from ml_fertilizers.lib.models.GpuModels import XGBClassifierGPU, XGBRegressorGPU

# NOTE: This line casues issue with LGBM and UserWarning: X does not have valid feature names, but LGBMClassifier was fitted with feature names
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
)
warnings.filterwarnings(
    "ignore", message='Parameters: { "extended_epsilon" } are not used.'
)


class ExtendedRidgeClassifier(RidgeClassifier):
    def __init__(
        self,
        alpha=1.0,
        fit_intercept=True,
        copy_X=True,
        max_iter=None,
        tol=1e-4,
        class_weight=None,
        solver="auto",
        positive=False,
        random_state=None,
        **kwargs
    ):
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            max_iter=max_iter,
            tol=tol,
            class_weight=class_weight,
            solver=solver,  # type: ignore
            positive=positive,
            random_state=random_state,
            **kwargs,
        )

    def fit(self, X, y, **kwargs):  # type: ignore
        return super().fit(X, (y > 0).astype(int), **kwargs)  # type: ignore


class ExtendedRandomForestClassifier(RandomForestClassifier):
    def __init__(
        self,
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        monotonic_cst=None,
        **kwargs
    ):
        self._extended_proba = kwargs.pop("extended_proba", None)
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
            monotonic_cst=monotonic_cst,  # type: ignore
            **kwargs,
        )

    def fit(self, X, y, **kwargs):  # type: ignore
        return super().fit(X, (y > 0).astype(int), **kwargs)  # type: ignore

    def predict(self, X):  # type: ignore
        if self._extended_proba is None:
            return super().predict(X)
        else:
            if not 0.0 <= self._extended_proba <= 1.0:
                raise ValueError(" proba threshold must be between 0 and 1.")
            y_proba = super().predict_proba(X)
            return (y_proba[:, 1] >= self._extended_proba).astype(int)  # type: ignore

    def get_params(self, deep: bool = True) -> dict:
        return {
            **super().get_params(deep),
            "extended_proba": self._extended_proba,
        }

    def set_params(self, **params) -> "RandomForestClassifier":
        extended_proba = params.pop("extended_proba", "")

        if extended_proba != "":
            self._extended_proba = extended_proba

        super().set_params(**params)
        return self


class ExtendedLGBMClassifier(LGBMClassifier):
    def __init__(self, **kwargs):
        extended_proba = kwargs.pop("extended_proba", None)
        self._extended_proba = extended_proba
        super().__init__(**kwargs)

    def get_params(self, deep: bool = True) -> dict:
        return {
            **super().get_params(deep),
            "extended_proba": self._extended_proba,
        }

    def set_params(self, **params) -> "LGBMClassifier":
        extended_proba = params.pop("extended_proba", "")

        if extended_proba != "":
            self._extended_proba = extended_proba

        super().set_params(**params)
        return self

    def predict(self, X):  # type: ignore
        if self._extended_proba is None:
            return super().predict(X)
        else:
            if not 0.0 <= self._extended_proba <= 1.0:
                raise ValueError(" proba threshold must be between 0 and 1.")
            y_proba = super().predict_proba(X)
            return (y_proba[:, 1] >= self._extended_proba).astype(int)  # type: ignore

    def fit(self, X, y, **kwargs):  # type: ignore
        return super().fit(X, (y > 0).astype(int), **kwargs)  # type: ignore


class ExtendedSGDClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        loss="hinge",
        penalty="l2",
        alpha=0.0001,
        l1_ratio=0.15,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-3,
        shuffle=True,
        verbose=0,
        epsilon=0.1,
        n_jobs=None,
        random_state=None,
        learning_rate="optimal",
        eta0=0.0,
        power_t=0.5,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        class_weight=None,
        warm_start=False,
        average=False,
        extended_proba=None,
    ):

        self.extended_proba = extended_proba
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.shuffle = shuffle
        self.verbose = verbose
        self.epsilon = epsilon
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.class_weight = class_weight
        self.warm_start = warm_start
        self.average = average

    def fit(self, X, y, **kwargs):
        self.base_clf_ = SGDClassifier(
            loss=self.loss,  # type: ignore
            penalty=self.penalty,  # type: ignore
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            shuffle=self.shuffle,
            verbose=self.verbose,
            epsilon=self.epsilon,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            learning_rate=self.learning_rate,
            eta0=self.eta0,
            power_t=self.power_t,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            class_weight=self.class_weight,
            warm_start=self.warm_start,
            average=self.average,
        )

        if self.extended_proba is None:
            self.base_clf_.fit(X, y, **kwargs)
        else:
            if not 0.0 <= self.extended_proba <= 1.0:
                raise ValueError(" proba threshold must be between 0 and 1.")
            y_binary = (y > 0).astype(int)
            self.calibrated_clf_ = CalibratedClassifierCV(
                self.base_clf_, method="sigmoid", cv=5, n_jobs=1
            )
            self.calibrated_clf_.fit(X, y_binary, **kwargs)  # type: ignore

        return self

    def predict(self, X):
        if self.extended_proba is None:
            return self.base_clf_.predict(X)
        else:
            if not 0.0 <= self.extended_proba <= 1.0:
                raise ValueError(" proba threshold must be between 0 and 1.")
            elif self.calibrated_clf_ is None:
                raise ValueError(
                    "You need to fit the model before using the extended proba."
                )
            y_proba = self.calibrated_clf_.predict_proba(X)
            return (y_proba[:, 1] >= self.extended_proba).astype(int)  # type: ignore


class ExtendedPassiveAggressiveClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        C=1.0,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-3,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        shuffle=True,
        verbose=0,
        loss="hinge",
        n_jobs=None,
        random_state=None,
        warm_start=False,
        class_weight=None,
        average=False,
        extended_proba=None,
    ):

        self.extended_proba = extended_proba
        self.C = C
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.shuffle = shuffle
        self.verbose = verbose
        self.loss = loss
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.average = average

    def fit(self, X, y, **kwargs):
        self.base_clf_ = PassiveAggressiveClassifier(
            C=self.C,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            shuffle=self.shuffle,
            verbose=self.verbose,
            loss=self.loss,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            warm_start=self.warm_start,
            class_weight=self.class_weight,
            average=self.average,
        )

        if self.extended_proba is None:
            self.base_clf_.fit(X, y, **kwargs)
        else:
            if not 0.0 <= self.extended_proba <= 1.0:
                raise ValueError(" proba threshold must be between 0 and 1.")
            y_binary = (y > 0).astype(int)
            self.calibrated_clf_ = CalibratedClassifierCV(
                self.base_clf_, method="sigmoid", cv=5, n_jobs=1
            )
            self.calibrated_clf_.fit(X, y_binary, **kwargs)  # type: ignore

        return self

    def predict(self, X):
        if self.extended_proba is None:
            return self.base_clf_.predict(X)
        else:
            if not 0.0 <= self.extended_proba <= 1.0:
                raise ValueError(" proba threshold must be between 0 and 1.")
            elif self.calibrated_clf_ is None:
                raise ValueError(
                    "You need to fit the model before using the extended proba."
                )
            y_proba = self.calibrated_clf_.predict_proba(X)
            return (y_proba[:, 1] >= self.extended_proba).astype(int)  # type: ignore


class ExtendedXGBClassifierGPU(XGBClassifierGPU):
    def __init__(self, **kwargs):
        extended_proba = kwargs.pop("extended_proba", None)
        self._extended_proba = extended_proba
        super().__init__(**kwargs)

    def get_params(self, deep: bool = True) -> dict:
        return {
            **super().get_params(deep),
            "extended_proba": self._extended_proba,
        }

    def set_params(self, **params) -> "XGBClassifierGPU":
        extended_proba = params.pop("extended_proba", "")

        if extended_proba != "":
            self._extended_proba = extended_proba

        super().set_params(**params)
        return self

    def predict(self, X, **kwargs):
        if self._extended_proba is None:
            return super().predict(X, **kwargs)
        else:
            if not 0.0 <= self._extended_proba <= 1.0:
                raise ValueError(" proba threshold must be between 0 and 1.")
            y_proba = super().predict_proba(X, **kwargs)  # type: ignore
            return (y_proba[:, 1] >= self._extended_proba).astype(int)

    def fit(self, X, y, **kwargs):  # type: ignore
        return super().fit(X, (y > 0).astype(int), **kwargs)  # type: ignore


class ExtendedKNeighborsClassifier(KNeighborsClassifier):
    def __init__(
        self,
        n_neighbors=5,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
        **kwargs
    ):
        extended_proba = kwargs.pop("extended_proba", None)

        self._extended_proba = extended_proba
        super().__init__(
            n_neighbors=n_neighbors,
            weights=weights,  # type: ignore
            algorithm=algorithm,  # type: ignore
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs,
            **kwargs,
        )

    def get_params(self, deep: bool = True) -> dict:
        return {
            **super().get_params(deep),
            "extended_proba": self._extended_proba,
        }

    def set_params(self, **params) -> "KNeighborsClassifier":
        extended_proba = params.pop("extended_proba", "")

        if extended_proba != "":
            self._extended_proba = extended_proba

        super().set_params(**params)
        return self

    def predict(self, X):
        if self._extended_proba is None:
            return super().predict(X)
        else:
            if not 0.0 <= self._extended_proba <= 1.0:
                raise ValueError(" proba threshold must be between 0 and 1.")
            y_proba = super().predict_proba(X)  # type: ignore
            return (y_proba[:, 1] >= self._extended_proba).astype(int)  # type: ignore

    def fit(self, X, y, **kwargs):  # type: ignore
        return super().fit(X, (y > 0).astype(int), **kwargs)  # type: ignore


class ExtendedRidgeRegressor(Ridge):
    def __init__(
        self,
        alpha=1.0,
        fit_intercept=True,
        copy_X=True,
        max_iter=None,
        tol=1e-4,
        solver="auto",
        positive=False,
        random_state=None,
        **kwargs
    ):
        extended_epsilon = kwargs.pop("extended_epsilon", None)
        self._extended_epsilon = extended_epsilon
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            max_iter=max_iter,
            tol=tol,
            solver=solver,  # type: ignore
            positive=positive,
            random_state=random_state,
            **kwargs,
        )

    def get_params(self, deep: bool = True) -> dict:
        return {
            **super().get_params(deep),
            "extended_epsilon": self._extended_epsilon,
        }

    def set_params(self, **params) -> "ExtendedRidgeRegressor":
        extended_epsilon = params.pop("extended_epsilon", "")
        if extended_epsilon != "":
            self._extended_epsilon = extended_epsilon
        super().set_params(**params)
        return self

    def predict(self, X):
        if self._extended_epsilon is None:
            return super().predict(X)
        else:
            y_pred = super().predict(X)
            return (y_pred >= self._extended_epsilon).astype(int)  # type: ignore


class ExtendedXGBRegressorGPU(XGBRegressorGPU):
    def __init__(self, **kwargs):
        extended_epsilon = kwargs.pop("extended_epsilon", None)
        self._extended_epsilon = extended_epsilon
        super().__init__(**kwargs)

    def get_params(self, deep: bool = True) -> dict:
        return {
            **super().get_params(deep),
            "extended_epsilon": self._extended_epsilon,
        }

    def set_params(self, **params) -> "XGBRegressorGPU":
        extended_epsilon = params.pop("extended_epsilon", "")
        if extended_epsilon != "":
            self._extended_epsilon = extended_epsilon
        super().set_params(**params)
        return self

    def predict(self, X, **kwargs):
        if self._extended_epsilon is None:
            return super().predict(X, **kwargs)
        else:
            y_pred = super().predict(X, **kwargs)
            return (y_pred >= self._extended_epsilon).astype(int)  # type: ignore


class ExtendedLGBMRegressor(LGBMRegressor):
    def __init__(self, **kwargs):
        extended_epsilon = kwargs.pop("extended_epsilon", None)
        self._extended_epsilon = extended_epsilon
        super().__init__(**kwargs)

    def get_params(self, deep: bool = True) -> dict:
        return {
            **super().get_params(deep),
            "extended_epsilon": self._extended_epsilon,
        }

    def set_params(self, **params) -> "LGBMRegressor":
        extended_epsilon = params.pop("extended_epsilon", "")
        if extended_epsilon != "":
            self._extended_epsilon = extended_epsilon
        super().set_params(**params)
        return self

    def predict(self, X):  # type: ignore
        if self._extended_epsilon is None:
            return super().predict(X)
        else:
            y_pred = super().predict(X)
            return (y_pred >= self._extended_epsilon).astype(int)  # type: ignore


class ExtendedSGDRegressor(SGDRegressor):
    def __init__(
        self,
        loss="squared_error",
        penalty="l2",
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
        **kwargs
    ):
        extended_epsilon = kwargs.pop("extended_epsilon", None)
        self._extended_epsilon = extended_epsilon
        super().__init__(
            loss=loss,
            penalty=penalty,  # type: ignore
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
            **kwargs,
        )

    def get_params(self, deep: bool = True) -> dict:
        return {
            **super().get_params(deep),
            "extended_epsilon": self._extended_epsilon,
        }

    def set_params(self, **params) -> "SGDRegressor":
        extended_epsilon = params.pop("extended_epsilon", "")
        if extended_epsilon != "":
            self._extended_epsilon = extended_epsilon
        super().set_params(**params)
        return self

    def predict(self, X):
        if self._extended_epsilon is None:
            return super().predict(X)
        else:
            y_pred = super().predict(X)
            return (y_pred >= self._extended_epsilon).astype(int)  # type: ignore


class ExtendedPassiveAggressiveRegressor(PassiveAggressiveRegressor):
    def __init__(
        self,
        C=1.0,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-3,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        shuffle=True,
        verbose=0,
        loss="epsilon_insensitive",
        epsilon=0.1,
        random_state=None,
        warm_start=False,
        average=False,
        **kwargs
    ):
        extended_epsilon = kwargs.pop("extended_epsilon", None)
        self._extended_epsilon = extended_epsilon
        super().__init__(
            C=C,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            shuffle=shuffle,
            verbose=verbose,
            loss=loss,
            epsilon=epsilon,
            random_state=random_state,
            warm_start=warm_start,
            average=average,
            **kwargs,
        )

    def get_params(self, deep: bool = True) -> dict:
        return {
            **super().get_params(deep),
            "extended_epsilon": self._extended_epsilon,
        }

    def set_params(self, **params) -> "PassiveAggressiveRegressor":
        extended_epsilon = params.pop("extended_epsilon", "")
        if extended_epsilon != "":
            self._extended_epsilon = extended_epsilon
        super().set_params(**params)
        return self

    def predict(self, X):
        if self._extended_epsilon is None:
            return super().predict(X)
        else:
            y_pred = super().predict(X)
            return (y_pred >= self._extended_epsilon).astype(int)  # type: ignore


class ExtendedKNeighborsRegressor(KNeighborsRegressor):
    def __init__(
        self,
        n_neighbors=5,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
        **kwargs
    ):
        extended_epsilon = kwargs.pop("extended_epsilon", None)
        self._extended_epsilon = extended_epsilon
        super().__init__(
            n_neighbors=n_neighbors,
            weights=weights,  # type: ignore
            algorithm=algorithm,  # type: ignore
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs,
            **kwargs,
        )

    def get_params(self, deep: bool = True) -> dict:
        return {
            **super().get_params(deep),
            "extended_epsilon": self._extended_epsilon,
        }

    def set_params(self, **params) -> "KNeighborsRegressor":
        extended_epsilon = params.pop("extended_epsilon", "")
        if extended_epsilon != "":
            self._extended_epsilon = extended_epsilon
        super().set_params(**params)
        return self

    def predict(self, X):
        if self._extended_epsilon is None:
            return super().predict(X)
        else:
            y_pred = super().predict(X)
            return (y_pred >= self._extended_epsilon).astype(int)  # type: ignore


class RidgeRegressor(Ridge):
    def __init__(
        self,
        alpha=1.0,
        fit_intercept=True,
        copy_X=True,
        max_iter=None,
        tol=1e-4,
        solver="auto",
        positive=False,
        random_state=None,
        **kwargs
    ):
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            max_iter=max_iter,
            tol=tol,
            solver=solver,  # type: ignore
            positive=positive,
            random_state=random_state,
            **kwargs,
        )
