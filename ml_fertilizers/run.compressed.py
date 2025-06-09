from typing import List
import multiprocessing as mp

import optuna
import pandas as pd
from sklearn import clone
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from dataclasses import dataclass, field
from typing import List


@dataclass
class FeatureCombination:
    features: List[str]
    name: str = field(default="")

    def __post_init__(self):
        if "-" in self.name:
            raise ValueError("Name cannot contain '-'")
import logging
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = "logs.log",
):
    """
    Set up a logger with the specified name, log file, level, and format.
    Logs to console by default; optionally logs to a file if log_file is provided.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): File path for file logging. If None, file logging is skipped.
        level (int, optional): Logging level (e.g., logging.INFO, logging.DEBUG).
        fmt (str, optional): Log message format.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    logger.setLevel(
        logging.DEBUG
    )  # Set the logger to the lowest level to capture all logs
    # Console handler
    stream_formatter = logging.Formatter(
        "%(levelname)s %(name)s %(asctime)s | %(message)s", datefmt="%H:%M:%S"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    # Optional file handler
    if log_file:
        file_formatter = logging.Formatter(
            "%(levelname)s %(name)s %(asctime)s [%(filename)s:%(funcName)s:%(lineno)d] | %(message)s",
            datefmt="%d/%m/%Y %H:%M:%S",
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

    # Prevent duplicate logs if function is called multiple times
    logger.propagate = False

    return logger


logger = setup_logger("Init")

logger.info("Logger setup complete")
logger.debug("This is a debug message")
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
import numpy as np
import pandas as pd
import scipy.sparse as sp
from xgboost import XGBClassifier, XGBRegressor


try:
    from numba import cuda  # type: ignore
except Exception:
    cuda = None

try:
    import cupy as cp  # type: ignore
except Exception:
    cp = None

try:
    import cudf  # type: ignore
except Exception:
    cudf = None


logger = setup_logger(__name__)


class XGBClassifierGPU(XGBClassifier):
    """SHEESH! This is a GPU compatible version of XGBClassifier. It uses the GPU for training and prediction if possible."""

    def __init__(self, **kwargs):

        kwargs.setdefault("device", "cuda")
        kwargs.setdefault("tree_method", "hist")
        super().__init__(**kwargs)

    def _use_gpu(self):
        # Decide based on the actual XGBClassifier parameters
        return (
            getattr(self, "tree_method", None) == "hist"
            and getattr(self, "device", None) == "cuda"
        )

    def _set_gpu(self, use_gpu: bool) -> "XGBClassifierGPU":
        """Toggle GPU usage for XGBoost."""
        if use_gpu:
            logger.info("Using GPU for XGBoost.")
            self.set_params(tree_method="hist", device="cuda")
        else:
            logger.warning("Using CPU for XGBoost. This may be slower than using GPU.")
            self.set_params(tree_method=None, device=None)

        return self

    def _to_gpu_array(self, X):

        if not self._use_gpu():
            arr = X
            if isinstance(X, pd.DataFrame) and any(
                [isinstance(col, pd.SparseDtype) for col in X.dtypes]
            ):
                arr = X.astype(pd.SparseDtype("float32", 0)).sparse.to_coo().tocsr()
            return arr
        # Pandas DataFrame -> cuDF if available, else CuPy, else NumPy

        elif isinstance(X, pd.DataFrame):
            if cudf is not None:
                return cudf.DataFrame.from_pandas(X)
            elif cp is not None:
                return cp.array(X.values)
            elif cuda is not None:
                # If using Numba's CUDA, convert to a CuPy array
                return cuda.to_device(X.values)
            else:
                return X.values
        elif isinstance(X, np.ndarray):
            if cp is not None:
                return cp.array(X)
            else:
                return X

        elif sp.issparse(X):
            X_dense = X.toarray()
            if cp is not None:
                return cp.array(X_dense)
            else:
                return X_dense
        elif cudf is not None and isinstance(X, cudf.DataFrame):
            return X
        elif cp is not None and isinstance(X, cp.ndarray):
            return X
        else:
            raise ValueError(f"Unsupported input type: {type(X)}")

    def _to_gpu_label(self, y):
        if not self._use_gpu():
            return y
        elif isinstance(y, pd.Series):
            if cudf is not None:
                return cudf.Series(y)
            elif cp is not None:
                return cp.array(y.values)
            else:
                return y.values
        elif isinstance(y, np.ndarray):
            if cp is not None:
                return cp.array(y)
            else:
                return y
        elif cudf is not None and isinstance(y, cudf.Series):
            return y
        elif cp is not None and isinstance(y, cp.ndarray):
            return y
        else:
            return y

    def fit(self, X, y, **kwargs):
        # X_fin = self._to_gpu_array(X)
        X_fin = X
        y_fin = y
        # y_fin = self._to_gpu_label(y)
        self_fitted = super().fit(X_fin, y_fin, **kwargs)
        del X_fin
        del y_fin
        return self_fitted

    def predict(self, X, **kwargs):
        # X_fin = self._to_gpu_array(X)
        X_fin = X
        y_pred = super().predict(X_fin, **kwargs)
        del X_fin
        return y_pred

    def predict_proba(self, X, **kwargs):
        # X_fin = self._to_gpu_array(X)
        X_fin = X
        return super().predict_proba(X_fin, **kwargs)


class XGBRegressorGPU(XGBRegressor):
    """SHEESH! This is a GPU compatible version of XGBRegressor. It uses the GPU for training and prediction if possible."""

    def __init__(self, **kwargs):

        kwargs.setdefault("device", "cuda")
        kwargs.setdefault("tree_method", "hist")
        super().__init__(**kwargs)

    def _use_gpu(self):
        # Decide based on the actual XGBClassifier parameters
        return (
            getattr(self, "tree_method", None) == "hist"
            and getattr(self, "device", None) == "cuda"
        )

    def _set_gpu(self, use_gpu: bool) -> "XGBClassifierGPU":
        """Toggle GPU usage for XGBoost."""
        if use_gpu:
            logger.info("Using GPU for XGBoost.")
            self.set_params(tree_method="hist", device="cuda")
        else:
            logger.warning("Using CPU for XGBoost. This may be slower than using GPU.")
            self.set_params(tree_method=None, device=None)

        return self

    def _to_gpu_array(self, X):

        if not self._use_gpu():
            arr = X
            if isinstance(X, pd.DataFrame) and any(
                [isinstance(col, pd.SparseDtype) for col in X.dtypes]
            ):
                arr = X.astype(pd.SparseDtype("float32", 0)).sparse.to_coo().tocsr()
            return arr
        # Pandas DataFrame -> cuDF if available, else CuPy, else NumPy
        elif isinstance(X, pd.DataFrame):
            if cudf is not None:
                return cudf.DataFrame.from_pandas(X)
            elif cp is not None:
                return cp.array(X.values)
            else:
                return X.values
        elif isinstance(X, np.ndarray):
            if cp is not None:
                return cp.array(X)
            else:
                return X
        elif sp.issparse(X):
            X_dense = X.toarray()
            if cp is not None:
                return cp.array(X_dense)
            else:
                return X_dense
        elif cudf is not None and isinstance(X, cudf.DataFrame):
            return X
        elif cp is not None and isinstance(X, cp.ndarray):
            return X
        else:
            raise ValueError(f"Unsupported input type: {type(X)}")

    def _to_gpu_label(self, y):
        if not self._use_gpu():
            return y
        elif isinstance(y, pd.Series):
            if cudf is not None:
                return cudf.Series(y)
            elif cp is not None:
                return cp.array(y.values)
            else:
                return y.values
        elif isinstance(y, np.ndarray):
            if cp is not None:
                return cp.array(y)
            else:
                return y
        elif cudf is not None and isinstance(y, cudf.Series):
            return y
        elif cp is not None and isinstance(y, cp.ndarray):
            return y
        else:
            return y

    def fit(self, X, y, **kwargs):
        X_fin = self._to_gpu_array(X)
        y_fin = self._to_gpu_label(y)
        self_fitted = super().fit(X_fin, y_fin, **kwargs)
        del X_fin
        del y_fin
        return self_fitted

    def predict(self, X, **kwargs):
        X_fin = self._to_gpu_array(X)
        y_pred = super().predict(X_fin, **kwargs)
        del X_fin
        return y_pred


class LGBMClassifierGPU(LGBMClassifier):
    """GPU-compatible version of LGBMClassifier. Uses GPU for training and prediction if possible."""

    def __init__(self, **kwargs):
        kwargs.setdefault("device", "gpu")
        # You can set other GPU-related params here if needed
        super().__init__(**kwargs)

    def _set_gpu(self, use_gpu: bool) -> "LGBMClassifierGPU":
        """Toggle GPU usage for LightGBM."""
        if use_gpu:
            logger.info("Using GPU for LightGBM.")
            self.set_params(device="gpu")
        else:
            logger.warning("Using CPU for LightGBM. This may be slower than using GPU.")
            self.set_params(device=None)

        return self


class LGBMRegressprGPU(LGBMRegressor):
    """GPU-compatible version of LGBMRegressor. Uses GPU for training and prediction if possible."""

    def __init__(self, **kwargs):
        kwargs.setdefault("device", "gpu")
        # You can set other GPU-related params here if needed
        super().__init__(**kwargs)

    def _set_gpu(self, use_gpu: bool) -> "LGBMRegressprGPU":
        """Toggle GPU usage for LightGBM."""
        if use_gpu:
            logger.info("Using GPU for LightGBM.")
            self.set_params(device="gpu")
        else:
            logger.warning("Using CPU for LightGBM. This may be slower than using GPU.")
            self.set_params(device=None)

        return self


class CatBoostClassifierGPU(CatBoostClassifier):
    """GPU-compatible version of CatBoostClassifier. Uses GPU for training and prediction if possible."""

    def __init__(self, **kwargs):
        kwargs.setdefault("task_type", "GPU")
        # You can set other GPU-related params here if needed
        super().__init__(**kwargs)

    def _set_gpu(self, use_gpu: bool) -> "CatBoostClassifierGPU":
        """Toggle GPU usage for CatBoost."""
        if use_gpu:
            logger.info("Using GPU for CatBoost.")
            self.set_params(task_type="GPU")
        else:
            logger.warning("Using CPU for CatBoost. This may be slower than using GPU.")
            self.set_params(task_type="CPU")

        return self


class CatBoostRegressorGPU(CatBoostRegressor):
    """GPU-compatible version of CatBoostRegressor. Uses GPU for training and prediction if possible."""

    def __init__(self, **kwargs):
        kwargs.setdefault("task_type", "GPU")
        # You can set other GPU-related params here if needed
        super().__init__(**kwargs)

    def _set_gpu(self, use_gpu: bool) -> "CatBoostRegressorGPU":
        """Toggle GPU usage for CatBoost."""
        if use_gpu:
            logger.info("Using GPU for CatBoost.")
            self.set_params(task_type="GPU")
        else:
            logger.warning("Using CPU for CatBoost. This may be slower than using GPU.")
            self.set_params(task_type="CPU")

        return self
from dataclasses import dataclass

from sklearn.base import BaseEstimator



@dataclass
class HyperOptCombination:
    name: str
    model: BaseEstimator
    feature_combination: FeatureCombination
from dataclasses import dataclass
from typing import Any, Dict

import optuna


@dataclass
class TrialParamWrapper:
    """A class that is to help the creation of the trial parameters."""

    def _use_proba(self) -> bool:
        return (
            self.model_name.lower().startswith("extended")
            and "classifier" in self.model_name.lower()
        )

    def _use_epsilon(self) -> bool:
        return (
            self.model_name.lower().startswith("extended")
            and "clssifier" not in self.model_name.lower()
        )

    def _get_ridge_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = {
            "alpha": trial.suggest_float("alpha", 1e-3, 1000, log=True),
            "tol": trial.suggest_float("tol", 1e-5, 1e-1, log=True),
        }

        if self._use_epsilon():
            params["extended_epsilon"] = trial.suggest_float(
                "extended_epsilon", 1e-3, 5, log=True
            )

        return params

    def _get_random_forest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical(
                "max_features", [None, "sqrt", "log2"]
            ),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }
        if self._use_proba():
            params["extended_proba"] = trial.suggest_float(
                "extended_proba", 0.5, 0.7, log=True
            )

        if self._use_epsilon():
            params["extended_epsilon"] = trial.suggest_float(
                "extended_epsilon", 1e-3, 5, log=True
            )

        return params

    def _get_kneighbors_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 50),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "p": trial.suggest_int("p", 1, 8),
            "leaf_size": trial.suggest_int("leaf_size", 10, 60),
        }

        if self._use_proba():
            params["extended_proba"] = trial.suggest_float(
                "extended_proba", 0.5, 0.7, log=True
            )

        if self._use_epsilon():
            params["extended_epsilon"] = trial.suggest_float(
                "extended_epsilon", 1e-3, 5, log=True
            )

        return params

    def _get_svc_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "C": trial.suggest_float("C", 1e-3, 10, log=True),
            "kernel": trial.suggest_categorical(
                "kernel", ["linear", "poly", "rbf", "sigmoid"]
            ),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            "tol": trial.suggest_float("tol", 1e-5, 1e-1, log=True),
        }

    def _get_lgbm_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 3000),
            "max_depth": trial.suggest_int("max_depth", 3, 22),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 50),
            "subsample": trial.suggest_float("subsample", 0.1, 1),
            # "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 100, log=True),
            # "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1),
            # "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100, log=True),
            # "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100, log=True),
        }

        if self._use_proba():
            params["extended_proba"] = trial.suggest_float(
                "extended_proba", 0.5, 0.7, log=True
            )

        if self._use_epsilon():
            params["extended_epsilon"] = trial.suggest_float(
                "extended_epsilon", 1e-3, 5, log=True
            )

        return params

    def _get_xgb_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 3000),
            "max_depth": trial.suggest_int("max_depth", 5, 22),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100, log=True),
            # "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
            # "tree_method": trial.suggest_categorical("tree_method", ["approx", "hist"]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            # "gamma": trial.suggest_float("gamma", 0.01, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True),
            # "min_child_weight": trial.suggest_float("min_child_weight", 1, 100, log=True),
        }

        if self._use_proba():
            params["extended_proba"] = trial.suggest_float(
                "extended_proba", 0.5, 0.7, log=True
            )

        if self._use_epsilon():
            params["extended_epsilon"] = trial.suggest_float(
                "extended_epsilon", 1e-3, 5, log=True
            )

        return params

    def _get_catboost_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "iterations": trial.suggest_int("iterations", 200, 3000),
            "depth": trial.suggest_int("depth", 4, 16),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            # "border_count": trial.suggest_int("border_count", 32, 255),
            # "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.0, 10.0),
            # "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
            # "one_hot_max_size": trial.suggest_int("one_hot_max_size", 2, 10),
            # "rsm": trial.suggest_float("rsm", 0.5, 1.0),
            # "od_type": trial.suggest_categorical("od_type", ["IncToDec", "Iter"]),
            # "od_wait": trial.suggest_int("od_wait", 10, 50),
        }

    def _get_adaboost_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1, log=True),
        }

    def _get_sgd_classifier_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        lr = trial.suggest_categorical("learning_rate", ["invscaling", "adaptive"])
        params = {
            "loss": trial.suggest_categorical(
                "loss",
                ["modified_huber", "squared_hinge"],
            ),
            "learning_rate": lr,
            "penalty": "elasticnet",
            "alpha": trial.suggest_float("alpha", 1e-4, 1.0, log=True),
            "max_iter": trial.suggest_int("max_iter", int(5e2), int(5e3)),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
            "eta0": trial.suggest_float("eta0", 1e-5, 1e-1, log=True),
            # "power_t": trial.suggest_float("power_t", 0.1, 0.9) if lr == "invscaling" else 0.5,
            # "tol": trial.suggest_float("tol", 1e-5, 1e-1, log=True),
        }

        if self._use_proba():
            params["extended_proba"] = trial.suggest_float(
                "extended_proba", 0.5, 0.7, log=True
            )
        return params

    def _get_sgd_regressor_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = {
            "loss": trial.suggest_categorical(
                "loss", ["squared_error", "huber", "epsilon_insensitive"]
            ),
            "penalty": "elasticnet",  # Fixed to L2 for simplicity
            "alpha": trial.suggest_float("alpha", 1e-5, 1.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
            "learning_rate": "invscaling",  # Fixed to a common schedule
            "max_iter": trial.suggest_int("max_iter", int(52), int(5e3)),
        }

        if self._use_epsilon():
            params["extended_epsilon"] = trial.suggest_float(
                "extended_epsilon", 1e-3, 5, log=True
            )

        return params

    def _get_passive_agressive_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = {
            "C": trial.suggest_float("C", 1e-3, 1.0, log=True),
            "tol": trial.suggest_float("tol", 1e-5, 1e-1, log=True),
            "validation_fraction": trial.suggest_float("validation_fraction", 0.1, 0.5),
            "n_iter_no_change": trial.suggest_int("n_iter_no_change", 5, 50),
            "max_iter": trial.suggest_int("max_iter", int(5e2), int(5e3)),
        }

        if self._use_proba():
            params["extended_proba"] = trial.suggest_float(
                "extended_proba", 0.5, 0.7, log=True
            )
        if self._use_epsilon():
            params["extended_epsilon"] = trial.suggest_float(
                "extended_epsilon", 1e-3, 5, log=True
            )

        return params

    def _get_logistic_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "C": trial.suggest_float("C", 1e-3, 1.0, log=True),
            "tol": trial.suggest_float("tol", 1e-5, 1e-1, log=True),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "saga"]),
            "max_iter": trial.suggest_int("max_iter", int(5e2), int(5e3)),
        }

    def _get_hist_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1, log=True),
            "max_iter": trial.suggest_int("max_iter", int(5e2), int(5e3)),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 31, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 1.0),
            # "max_bins": trial.suggest_int("max_bins", 2, 255),
            # "tol": trial.suggest_float("tol", 1e-5, 1e-1, log=True),
        }

    def _get_nncm_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "n_layers": trial.suggest_int("n_layers", 1, 10),
            "window": trial.suggest_int("window", 1, 20),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1, log=True),
            "n_epochs": trial.suggest_int("n_epochs", 10, 100),
            "batch_size": trial.suggest_int("batch_size", 1, 100),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "activation": trial.suggest_categorical(
                "activation", ["relu", "tanh", "sigmoid"]
            ),
            "positive_class_threshold": trial.suggest_float(
                "positive_class_threshold", 0.5, 0.7
            ),
        }

    def get_params(
        self,
        model_name: str,
        trial: optuna.Trial,
    ) -> Dict[str, Any]:
        self.model_name = model_name.lower()

        param_pairs = [
            ("ridge", self._get_ridge_params),
            ("randomforest", self._get_random_forest_params),
            ("kneighbors", self._get_kneighbors_params),
            ("svc", self._get_svc_params),
            ("lgbm", self._get_lgbm_params),
            ("xgb", self._get_xgb_params),
            ("catboost", self._get_catboost_params),
            ("adaboost", self._get_adaboost_params),
            ("sgdclassifier", self._get_sgd_classifier_params),
            ("sgdregressor", self._get_sgd_regressor_params),
            ("passive", self._get_passive_agressive_params),
            ("logistic", self._get_logistic_params),
            ("histgradient", self._get_hist_params),
            ("neuralnetworkcustommodel", self._get_nncm_params),
        ]

        for model_prefix, param_func in param_pairs:
            if model_name.lower().replace("extended", "").startswith(model_prefix):
                return param_func(trial)

        raise ValueError(
            f"Model {model_name} is not supported or does not have parameters defined."
        )
import os
from pathlib import Path

import optuna
import pandas as pd
from pathlib import Path
import pickle
from typing import List, Optional, cast

import optuna

from typing import Any, Dict, List, Optional, TypedDict

from sklearn.base import BaseEstimator


class HyperOptResultDict(TypedDict):
    name: str
    model: Optional[BaseEstimator]
    features: Optional[List[str]]
    params: Optional[Dict[str, Any]]
    score: Optional[float]
    n_trials: Optional[int]
    metadata: Optional[Dict[str, Any]]

logger = setup_logger(__name__)


def load_hyper_opt_results(
    model_run: str,
    output_dir_path: Path,
    hyper_opt_prefix: str,
    selection: Optional[List[str]] = None,
) -> List[HyperOptResultDict]:
    logger.info(f"Loading hyper opt results for run {model_run}...")

    hyper_opt_results_dir_path = output_dir_path / f"{hyper_opt_prefix}{model_run}"

    if not hyper_opt_results_dir_path.exists():
        logger.error(f"Directory {hyper_opt_results_dir_path} does not exist.")
        return []

    logger.info(f"Directory {hyper_opt_results_dir_path} exists.")

    hyper_opt_results = []

    for file_path in hyper_opt_results_dir_path.iterdir():
        if (
            file_path.is_file()
            and file_path.suffix == ".pkl"
            and (selection is None or file_path.stem in selection)
        ):
            logger.info(f"Loading {file_path.stem}...")
            result = cast(HyperOptResultDict, pickle.load(open(file_path, "rb")))
            result["name"] = file_path.stem
            hyper_opt_results.append(result)

    logger.info(f"Loaded {len(hyper_opt_results)} hyper opt results.")

    if selection is not None and len(hyper_opt_results) != len(selection):
        logger.error(
            f"Models not found: {list(set(selection) - set([model['name'] for model in hyper_opt_results]))}"
        )
        raise ValueError("Not all models were found in the specified path.")

    return hyper_opt_results


def load_hyper_opt_studies(
    model_run: str, output_dir_path: Path, study_prefix: str
) -> List[optuna.Study]:
    logger.info(f"Loading hyper opt studies for run {model_run}...")

    hyper_opt_studies_dir_path = output_dir_path / f"{study_prefix}{model_run}"

    if not hyper_opt_studies_dir_path.exists():
        logger.error(f"Directory {hyper_opt_studies_dir_path} does not exist.")
        return []

    logger.info(f"Directory {hyper_opt_studies_dir_path} exists.")

    hyper_opt_studies = []

    for file_path in hyper_opt_studies_dir_path.iterdir():
        if file_path.is_file() and file_path.suffix == ".db":
            logger.info(f"Loading {file_path}...")

            storage = f"sqlite:///{file_path}"

            frozen_studies = optuna.storages.RDBStorage(storage).get_all_studies()

            hyper_opt_studies.extend(
                [
                    optuna.study.load_study(
                        study_name=study.study_name,
                        storage=storage,
                    )
                    for study in frozen_studies
                ]
            )

    logger.info(f"Loaded {len(hyper_opt_studies)} hyper opt studies.")

    return hyper_opt_studies


logger = setup_logger(__name__)


def setup_analysis(
    model_run: str,
    output_dir_path: Path,
    hyper_opt_prefix: str,
    study_prefix: str,
    display_plots: bool,
    save_results: bool = True,
) -> pd.DataFrame:
    logger.info(f"Starting analysis for run {model_run}...")

    logger.info("Veryfing existing models...")
    hyper_opt_results = load_hyper_opt_results(
        model_run=model_run,
        output_dir_path=output_dir_path,
        hyper_opt_prefix=hyper_opt_prefix,
    )

    results_dict_list = []

    for result in hyper_opt_results:
        res_dict = {**result}
        if result["metadata"] is not None:
            res_dict.update(result["metadata"])

        results_dict_list.append(res_dict)

    results_df = pd.DataFrame([result for result in results_dict_list])
    results_df = results_df.sort_values("score", ascending=False)
    logger.info(f"Results data frame shape: {results_df.shape}")

    if save_results:
        results_df.to_csv(
            output_dir_path
            / f"{hyper_opt_prefix}{model_run}"
            / f"analysis_results_{model_run}.csv",
            index=False,
        )

    if display_plots:
        hyper_opt_studies = load_hyper_opt_studies(
            model_run=model_run,
            output_dir_path=output_dir_path,
            study_prefix=study_prefix,
        )
        for study in hyper_opt_studies:

            logger.info(
                f"Study {study.study_name}" + f" has {len(study.trials)} trials."
            )

            # optuna.visualization.plot_optimization_history(study).show()
            optuna.visualization.plot_slice(study).show()
            # optuna.visualization.plot_param_importances(study).show()

    logger.info("Analysis complete.")

    return results_df


import datetime as dt


import json
import signal
from pathlib import Path
import multiprocessing as mp
from typing import Callable, Dict, List, Optional, TypedDict


import pandas as pd

import os
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union


import optuna
import pandas as pd
import pickle

from sklearn import clone

from dataclasses import dataclass, field
from typing import List, Optional, Union

import optuna



logger = setup_logger(__name__)


@dataclass
class EarlyStoppingCallback:
    """When I wrote the initial version of this class, I knew what was up. Now, rewritten to accommodate multiple objectives, I just hope it works"""

    name: str
    patience: int
    min_percentage_improvement: float = 0.0
    best_value: Optional[Union[float, List[float]]] = None
    no_improvement_count: Optional[Union[int, List[int]]] = None
    directions: Optional[Union[str, List[str]]] = None

    def __post_init__(self):

        # NOTE: Normalize inputs to lists for consistency

        if self.best_value is not None and not isinstance(
            self.best_value, (list, tuple)
        ):
            self.best_value = [self.best_value]

        if self.no_improvement_count is not None and not isinstance(
            self.no_improvement_count, (list, tuple)
        ):
            self.no_improvement_count = [self.no_improvement_count]

        if self.directions is not None and not isinstance(
            self.directions, (list, tuple)
        ):
            self.directions = [self.directions]

        self.is_single_objective = len(self.directions) == 1 and self.directions[0] in [  # type: ignore
            "minimize",
            "maximize",
        ]

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:

        if trial.state in (
            optuna.trial.TrialState.PRUNED,
            optuna.trial.TrialState.FAIL,
        ):
            for i in range(len(self.no_improvement_count)):  # type: ignore
                self.no_improvement_count[i] += 1  # type: ignore

        elif trial.state == optuna.trial.TrialState.COMPLETE:

            if self.is_single_objective:
                current_best_values = [study.best_value]
            else:
                current_best_values = list(study.best_trials[0].values)

            if self.best_value is None or all(v is None for v in self.best_value):  # type: ignore
                self.best_value = current_best_values.copy()
                self.no_improvement_count = [0 for _ in current_best_values]

            # Track improvement per objective
            for i, (direction, cur, best) in enumerate(  # type: ignore
                zip(self.directions, current_best_values, self.best_value)  # type: ignore
            ):

                if (
                    (
                        best >= 0
                        and direction == "maximize"
                        and cur > best * (1.0 + self.min_percentage_improvement)
                    )
                    or (
                        best >= 0
                        and direction == "minimize"
                        and cur < best * (1.0 - self.min_percentage_improvement)
                    )
                    or (
                        best < 0
                        and direction == "maximize"
                        and cur > best * (1.0 - self.min_percentage_improvement)
                    )
                    or (
                        best < 0
                        and direction == "minimize"
                        and cur < best * (1.0 + self.min_percentage_improvement)
                    )
                ):
                    self.best_value[i] = cur  # type: ignore
                    self.no_improvement_count[i] = 0  # type: ignore
                else:
                    self.no_improvement_count[i] += 1  # type: ignore

        # Stop if any objective exceeds patience
        if any(c >= self.patience for c in self.no_improvement_count):  # type: ignore
            logger.warning(
                f"Early stopping the study: {self.name} due to "
                + f"no {self.min_percentage_improvement * 100}% improvement for "
                + f"{self.patience} trials | on trial: {trial.number}"
                + f" | best values: {self.best_value} | no improvement counts: {self.no_improvement_count}"
            )
            study.stop()
from dataclasses import dataclass
import gc
import datetime as dt
from typing import TypedDict



logger = setup_logger(__name__)


class ThresholdType(TypedDict):
    seconds: int
    milliseconds: int
    microseconds: int


@dataclass
class GarbageManagerClass:

    def clean(self, threshold: ThresholdType = {"seconds": 1}):
        # return
        start_time = dt.datetime.now()
        gb = gc.collect()
        end_time = dt.datetime.now()

        if end_time - start_time > dt.timedelta(**threshold):
            logger.warning("Garbage collector was slow")
            logger.warning(
                f"Garbage collector: {gb} objects collected ant took {end_time - start_time} seconds"
            )


garbage_manager = GarbageManagerClass()


logger = setup_logger(__name__)

OPTUNA_DIRECTION_TYPE = Union[
    List[Literal["maximize", "minimize"]], Literal["maximize", "minimize"]
]

CREATE_OBJECTIVE_TYPE = Callable[
    [pd.DataFrame, HyperOptCombination],
    Callable[[optuna.Trial], Union[Tuple[float], float]],
]
OBJECTIVE_RETURN_TYPE = Union[Tuple[float], float]
OBJECTIVE_FUNC_TYPE = Callable[[optuna.Trial], OBJECTIVE_RETURN_TYPE]
"""data class and model combination to objective function"""


def get_existing_trials_info_multiobj(
    trials: List[optuna.trial.FrozenTrial],
    min_percentage_improvement: float,
    directions: OPTUNA_DIRECTION_TYPE,
) -> Tuple[List[int], List[Optional[float]]]:

    temp_directions = directions if isinstance(directions, list) else [directions]

    n_objectives = len(temp_directions)
    no_improvement_counts = [0 for _ in range(n_objectives)]
    best_values = [None for _ in range(n_objectives)]

    for trial in trials:

        if trial.state in (
            optuna.trial.TrialState.PRUNED,
            optuna.trial.TrialState.FAIL,
        ):
            for i in range(n_objectives):
                no_improvement_counts[i] += 1
        elif trial.state == optuna.trial.TrialState.COMPLETE and trial.values is None:
            logger.error(
                f"Trial {trial.number} has no values, but is marked as COMPLETE"
            )
            continue
        elif (
            trial.state == optuna.trial.TrialState.COMPLETE and trial.values is not None
        ):
            for i, temp_direction in enumerate(temp_directions):
                value = trial.values[i]
                best_value = best_values[i]
                if (
                    best_value is None
                    or (
                        best_value >= 0
                        and temp_direction == "maximize"
                        and value > best_value * (1.0 + min_percentage_improvement)
                    )
                    or (
                        best_value >= 0
                        and temp_direction == "minimize"
                        and value < best_value * (1.0 - min_percentage_improvement)
                    )
                    or (
                        best_value < 0
                        and temp_direction == "maximize"
                        and value > best_value * (1.0 - min_percentage_improvement)
                    )
                    or (
                        best_value < 0
                        and temp_direction == "minimize"
                        and value < best_value * (1.0 + min_percentage_improvement)
                    )
                ):
                    best_values[i] = value
                    no_improvement_counts[i] = 0
                else:
                    no_improvement_counts[i] += 1

    return no_improvement_counts, best_values  # type: ignore


def save_hyper_result(
    trail: Optional[optuna.trial.FrozenTrial],
    is_single_objective: bool,
    model_combination: HyperOptCombination,
    output_dir_path: Path,
    hyper_opt_prefix: str,
    model_run: str,
    study: optuna.study.Study,
    metadata: Optional[Dict] = None,
    suffix: str = "",
) -> None:

    if metadata is None:
        metadata = {}

    if trail is not None:

        best_params = trail.params

        # optuna_result = trail.value if isinstance(trail.value, float) else trail.values

        if not is_single_objective:
            optuna_result = trail.values
            best_value = optuna_result[0]
        else:
            optuna_result = trail.value
            best_value = optuna_result

        metadata.update({"objective_result": optuna_result})

        best_model = clone(model_combination.model).set_params(**best_params)

        result = HyperOptResultDict(
            name=model_combination.name,
            score=best_value,  # type: ignore
            params=best_params,
            model=best_model,
            features=model_combination.feature_combination.features,
            n_trials=len(study.trials),
            metadata=metadata,
        )

    else:
        result = HyperOptResultDict(
            name=model_combination.name,
            score=None,  # type: ignore
            params={},
            model=None,  # type: ignore
            features=model_combination.feature_combination.features,
            n_trials=len(study.trials),
            metadata=metadata,
        )

    os.makedirs(output_dir_path / f"{hyper_opt_prefix}{model_run}", exist_ok=True)

    try:
        results_path = Path(
            f"{output_dir_path}/{hyper_opt_prefix}{model_run}/{model_combination.name}{suffix}.pkl"
        )

        pickle.dump(result, open(results_path, "wb"))
    except Exception as e:
        logger.error(
            f"Error saving model combination {model_combination.name}{suffix}: {e}"
        )
        raise e


def optimize_model_and_save(
    model_run: str,
    direction: OPTUNA_DIRECTION_TYPE,
    model_combination: HyperOptCombination,
    data: pd.DataFrame,
    n_optimization_trials: int,
    optimization_timeout: Optional[int],
    n_patience: int,
    min_percentage_improvement: float,
    i: int,
    output_dir_path: Path,
    hyper_opt_prefix: str,
    study_prefix: str,
    create_objective_func: CREATE_OBJECTIVE_TYPE,
    metadata: Optional[Dict] = None,
    evaluate_hyperopted_model_func: Optional[Callable] = None,
) -> None:
    combination_name = model_combination.name
    logger.info(f"Optimizing model combination {i}: {combination_name}")

    final_direction = None
    if isinstance(direction, list):
        if len(direction) > 1:
            final_direction = direction
        else:
            final_direction = direction[0]
    else:
        final_direction = direction

    is_single_objective = isinstance(final_direction, str)
    os.makedirs(output_dir_path / f"{study_prefix}{model_run}", exist_ok=True)

    sql_path = Path(
        f"{output_dir_path}/{study_prefix}{model_run}/{combination_name}.db"
        # f"{output_dir_path}/{study_prefix}{model_run}/studies.db"
    )

    # Create an Optuna study for hyperparameter optimization
    if not is_single_objective:
        study = optuna.create_study(
            directions=final_direction,
            study_name=f"{model_combination.name}",
            load_if_exists=True,
            storage=f"sqlite:///{sql_path}",
        )
    else:
        study = optuna.create_study(
            direction=final_direction,
            study_name=f"{model_combination.name}",
            load_if_exists=True,
            storage=f"sqlite:///{sql_path}",
        )

    no_improvement_count, best_value = get_existing_trials_info_multiobj(
        study.get_trials(),
        min_percentage_improvement,
        directions=final_direction,
    )

    early_stopping = EarlyStoppingCallback(
        name=model_combination.name,
        patience=n_patience,
        min_percentage_improvement=min_percentage_improvement,
        best_value=best_value,  # type: ignore
        no_improvement_count=no_improvement_count,  # type: ignore
        directions=final_direction,  # type: ignore
    )

    if n_patience <= 0 or n_patience < max(no_improvement_count):
        logger.warning(
            f"Skipping optuna optimization because of n_patience: {n_patience} was exceeded"
        )
    else:
        study.optimize(
            func=create_objective_func(data, model_combination),
            n_trials=n_optimization_trials,
            callbacks=[early_stopping],  # type: ignore
            timeout=optimization_timeout,
        )

    study_time_sum = sum(
        trial.duration.total_seconds() if trial.duration is not None else 0
        for trial in study.trials
    )
    best_trial = None
    completed_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if len(completed_trials) == 0:
        logger.error(
            f"No completed trials found for model combination {model_combination.name}"
        )

        save_hyper_result(
            trail=None,
            is_single_objective=is_single_objective,
            model_combination=model_combination,
            output_dir_path=output_dir_path,
            hyper_opt_prefix=hyper_opt_prefix,
            model_run=model_run,
            study=study,
            metadata=metadata,
        )
        return

    if not is_single_objective:
        # NOTE: ROBUST
        best_trial = max(study.best_trials, key=lambda t: min(t.values))
    else:
        best_trial = study.best_trial

    evaluation_metadata = (
        evaluate_hyperopted_model_func(
            model=model_combination.model,
            params=best_trial.params,
            features=model_combination.feature_combination.features,
            data=data,
        )
        if evaluate_hyperopted_model_func is not None
        else dict()
    )

    # top_trials = sorted_trials[:3]
    # for i, trial in enumerate(top_trials):
    #     save_hyper_result(
    #         trail=trial,
    #         model_combination=model_combination,
    #         output_dir_path=output_dir_path,
    #         hyper_opt_prefix=hyper_opt_prefix,
    #         model_run=model_run,
    #         study=study,
    #         metadata=metadata,
    #         suffix=f"top_{i}",
    #     )

    final_metadata = {}
    if metadata is not None:
        final_metadata.update(metadata)
    final_metadata.update(
        {
            "study_time_sum": study_time_sum,
        }
    )
    if evaluation_metadata is not None:
        final_metadata.update(evaluation_metadata)

    save_hyper_result(
        trail=best_trial,
        is_single_objective=is_single_objective,
        model_combination=model_combination,
        output_dir_path=output_dir_path,
        hyper_opt_prefix=hyper_opt_prefix,
        model_run=model_run,
        study=study,
        metadata=final_metadata,
    )

    del data
    garbage_manager.clean()


def aggregate_studies(
    study_dir_path: Path,
) -> Optional[str]:
    """
    Combine all studies into one.
    """
    study_summaries = None
    target_study = None
    source_study = None
    source_storage = None
    target_storage = None
    study_files = None
    logger.info(f"Combining studies in {study_dir_path}")
    AGGREGATED_STUDY_NAME = "aggregated_studies"
    # Get all the study files
    study_files = [
        file
        for file in list(study_dir_path.glob("*.db"))
        if file.stem != AGGREGATED_STUDY_NAME
    ]
    if not study_files:
        logger.warning("No study files found to combine.")
        return None

    logger.info(f"Found {len(study_files)} studies to combine.")

    target_storage_path = study_dir_path / f"{AGGREGATED_STUDY_NAME}.db"
    if target_storage_path.exists():
        logger.info(f"Deleting existing aggregated studies file {target_storage_path}")
        target_storage_path.unlink()
    target_storage = f"sqlite:///{target_storage_path}"

    for study_file in study_files:
        source_storage = f"sqlite:///{study_file}"
        study_summaries = optuna.get_all_study_summaries(storage=source_storage)
        if not study_summaries:
            logger.warning(f"No studies found in {study_file}")
            continue
        logger.info(f"Found {len(study_summaries)} studies in {study_file.stem}")

        for summary in study_summaries:
            # Optionally, make study names unique if needed
            new_study_name = f"{summary.study_name}"

            # Create the study in the target storage

            is_single_objective = not hasattr(summary, "_directions")

            if is_single_objective:
                target_study = optuna.create_study(
                    storage=target_storage,
                    study_name=new_study_name,
                    direction=summary.direction.name.lower(),
                    load_if_exists=False,
                )
            else:
                target_study = optuna.create_study(
                    storage=target_storage,
                    study_name=new_study_name,
                    directions=[
                        direction.name.lower() for direction in summary.directions
                    ],
                    load_if_exists=False,
                )

            # Load the source study and add its trials to the new study
            source_study = optuna.load_study(
                storage=source_storage,
                study_name=summary.study_name,
            )
            target_study.add_trials(source_study.get_trials())
            logger.info(
                f"Moved study {summary.study_name} from {target_storage_path.stem}"
            )

    del study_files
    del study_summaries
    del source_storage
    del target_study
    del source_study

    logger.info("Successfully combined studies.")
    logger.info(
        f"To view the aggregated studies, use:\noptuna-dashboard {target_storage}"
    )

    return target_storage

logger = setup_logger(__name__)


class HyperSetupDto(TypedDict):
    n_optimization_trials: int
    optimization_timeout: Optional[int]
    n_patience: int
    min_percentage_improvement: float
    model_run: Optional[str]
    limit_data_percentage: Optional[float]
    processes: Optional[int]
    max_concurrent_jobs: Optional[int]
    output_dir_path: Path
    hyper_opt_prefix: str
    study_prefix: str
    data: pd.DataFrame
    combinations: List[HyperOptCombination]
    hyper_direction: OPTUNA_DIRECTION_TYPE
    metadata: Optional[Dict]
    force_all_sequential: Optional[bool]
    omit_names: Optional[List[str]]


class HyperFunctionDto(TypedDict):
    create_objective_func: CREATE_OBJECTIVE_TYPE
    evaluate_hyperopted_model_func: Optional[Callable]


def init_worker() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def run_parallel_optimization(
    setup_dto: HyperSetupDto,
    function_dto: HyperFunctionDto,
) -> None:
    setup_dto = setup_dto.copy()
    if setup_dto["processes"] is None:
        setup_dto["processes"] = mp.cpu_count()
    if setup_dto["max_concurrent_jobs"] is None:
        setup_dto["max_concurrent_jobs"] = setup_dto["processes"]

    logger.info(f"Running optimization with {setup_dto['processes']} processes")

    # IMPORTANT NOTE: paralelism depends on the classification categories amount
    # for binary outputs it is not worth to run parallel optimization
    parallel_model_prefixes = [
        # "lgbm",
        "ridge",
        "sv",
        # "kn",
        "logistic",
        "passiveaggressive",
        "sgd",
        "minibatchsgd",
        "logistic",
        "neuralnetworkcustommodel",
    ]
    parallel_3rd_model_prefixes = []
    omit_mulit_sufixes = ["top_0", "top_1", "top_2", ""]
    omit_names = (
        setup_dto["omit_names"]
        if "omit_names" in setup_dto and setup_dto["omit_names"] is not None
        else []
    )
    all_model_combinations = setup_dto["combinations"]

    sequential_model_combinations = [
        model_combination
        for model_combination in setup_dto["combinations"]
        if all(  # type: ignore
            not model_combination.name.lower().startswith(prefix.lower())
            for prefix in parallel_model_prefixes + parallel_3rd_model_prefixes
        )
        and all(
            f"{model_combination.name}{omit_sufix}" not in omit_names
            for omit_sufix in omit_mulit_sufixes
        )
    ]

    parallel_model_combinations = [
        model_combination
        for model_combination in all_model_combinations
        if any(  # type: ignore
            model_combination.name.lower().startswith(prefix.lower())
            for prefix in parallel_model_prefixes
        )
        and all(
            f"{model_combination.name}{omit_sufix}" not in omit_names
            for omit_sufix in omit_mulit_sufixes
        )
    ]

    parallel_3rd_model_combinations = [
        model_combination
        for model_combination in all_model_combinations
        if any(  # type: ignore
            model_combination.name.lower().startswith(prefix.lower())
            for prefix in parallel_3rd_model_prefixes
        )
        and all(
            f"{model_combination.name}{omit_sufix}" not in omit_names
            for omit_sufix in omit_mulit_sufixes
        )
    ]

    if setup_dto["force_all_sequential"] is True:
        sequential_model_combinations.extend(
            parallel_model_combinations + parallel_3rd_model_combinations
        )
        parallel_model_combinations = []
        parallel_3rd_model_combinations = []

    logger.info(
        "Will be running parallel optimization for models: "
        + json.dumps([model.name for model in parallel_model_combinations], indent=4)
    )

    logger.info(
        "Will be running parallel 1/3rd optimization for models: "
        + json.dumps(
            [model.name for model in parallel_3rd_model_combinations], indent=4
        )
    )
    logger.info(
        "Will be running sequential optimization for models: "
        + json.dumps([model.name for model in sequential_model_combinations], indent=4)
    )

    if len(parallel_model_combinations) == 0:
        logger.info("No parallel models to optimize")
    else:
        # Set up multiprocessing pool
        with mp.Pool(
            processes=min(setup_dto["processes"], setup_dto["max_concurrent_jobs"]),
            initializer=init_worker,
        ) as pool:
            # Map each iteration of the loop to a process
            _ = pool.starmap(
                optimize_model_and_save,
                [
                    (
                        setup_dto["model_run"],
                        setup_dto["hyper_direction"],
                        model_combination,
                        setup_dto["data"],
                        setup_dto["n_optimization_trials"],
                        setup_dto["optimization_timeout"],
                        setup_dto["n_patience"],
                        setup_dto["min_percentage_improvement"],
                        i,
                        setup_dto["output_dir_path"],
                        setup_dto["hyper_opt_prefix"],
                        setup_dto["study_prefix"],
                        function_dto["create_objective_func"],
                        setup_dto["metadata"],
                        function_dto["evaluate_hyperopted_model_func"],
                    )
                    for i, model_combination in enumerate(parallel_model_combinations)
                    if model_combination.name not in omit_names
                ],
            )

    if len(parallel_3rd_model_combinations) == 0:
        logger.info("No parallel 1/3rd models to optimize")
    else:
        with mp.Pool(
            processes=min(
                setup_dto["processes"] // 3, setup_dto["max_concurrent_jobs"]
            ),
            initializer=init_worker,
        ) as pool:
            # Map each iteration of the loop to a process
            _ = pool.starmap(
                optimize_model_and_save,
                [
                    (
                        setup_dto["model_run"],
                        setup_dto["hyper_direction"],
                        model_combination,
                        setup_dto["data"],
                        setup_dto["n_optimization_trials"],
                        setup_dto["optimization_timeout"],
                        setup_dto["n_patience"],
                        setup_dto["min_percentage_improvement"],
                        i,
                        setup_dto["output_dir_path"],
                        setup_dto["hyper_opt_prefix"],
                        setup_dto["study_prefix"],
                        function_dto["create_objective_func"],
                        setup_dto["metadata"],
                        function_dto["evaluate_hyperopted_model_func"],
                    )
                    for i, model_combination in enumerate(
                        parallel_3rd_model_combinations
                    )
                    if model_combination.name not in omit_names
                ],
            )

    if len(sequential_model_combinations) == 0:
        logger.info("No sequential models to optimize")
    else:
        for i, model_combination in enumerate(sequential_model_combinations):
            if model_combination.name in omit_names:
                continue

            optimize_model_and_save(
                setup_dto["model_run"],  # type: ignore
                setup_dto["hyper_direction"],
                model_combination,
                setup_dto["data"],
                setup_dto["n_optimization_trials"],
                setup_dto["optimization_timeout"],
                setup_dto["n_patience"],
                setup_dto["min_percentage_improvement"],
                i,
                setup_dto["output_dir_path"],
                setup_dto["hyper_opt_prefix"],
                setup_dto["study_prefix"],
                function_dto["create_objective_func"],
                setup_dto["metadata"],
                function_dto["evaluate_hyperopted_model_func"],
            )

logger = setup_logger(__name__)


def setup_hyper(setup_dto: HyperSetupDto, function_dto: HyperFunctionDto) -> int:
    setup_dto = setup_dto.copy()

    if setup_dto["model_run"] is None:
        setup_dto["model_run"] = dt.datetime.now().strftime("%Y%m%d%H%M")

    if setup_dto["limit_data_percentage"] is not None and (
        setup_dto["limit_data_percentage"] < 0.0
        or setup_dto["limit_data_percentage"] > 1.0
    ):
        raise ValueError(
            "Invalid limit data percentage value: "
            + f"{setup_dto['limit_data_percentage']}"
        )

    logger.info(f"Starting hyper opt for run {setup_dto['model_run']}...")
    logger.info(f"Using {setup_dto['processes']} processes.")
    logger.info(f"Using {setup_dto['n_optimization_trials']} optimization trials.")
    logger.info(f"Using {setup_dto['n_patience']} patience.")
    logger.info(
        f"Using {setup_dto['min_percentage_improvement'] * 100}"
        + "% min percentage improvement."
    )
    logger.info(
        f"Using {setup_dto['limit_data_percentage'] * 100 if setup_dto['limit_data_percentage'] is not None else 'all '}% data"
    )

    logger.info("Loading data...")
    data = setup_dto["data"]

    logger.info("Engineering features...")

    if setup_dto["limit_data_percentage"] is not None:
        all_data_size = len(data)
        limited_data_size = int(all_data_size * setup_dto["limit_data_percentage"])
        logger.info(f"Limiting data from {all_data_size} to {limited_data_size}")
        setup_dto["data"] = data.sample(
            n=limited_data_size, random_state=42, replace=False, axis=0
        )

    logger.info(f"Created {len(setup_dto.get('combinations'))} combinations.")
    logger.info("Checking for existing models...")

    all_omit_names = [
        res["name"]
        for res in load_hyper_opt_results(
            model_run=setup_dto["model_run"],
            output_dir_path=setup_dto["output_dir_path"],
            hyper_opt_prefix=setup_dto["hyper_opt_prefix"],
        )
    ]
    omit_names = list(
        set(all_omit_names) & set([model.name for model in setup_dto["combinations"]])
    )
    setup_dto["omit_names"] = omit_names
    logger.info(f"Omitting {len(omit_names)} combinations.")
    logger.info(
        f"Running {len(setup_dto.get('combinations')) - len(omit_names)} combinations."
    )
    # --- NOTE ---
    # Metadata is a dictionary that can be used to store any additional information
    metadata = {
        "limit_data_percentage": setup_dto["limit_data_percentage"],
    }

    if setup_dto["metadata"] is not None:
        metadata.update(setup_dto["metadata"])

    logger.info("Starting parallel optimization...")
    _ = run_parallel_optimization(setup_dto=setup_dto, function_dto=function_dto)

    logger.info("Models has been pickled and saved to disk.")

    return len(setup_dto["combinations"]) - len(omit_names)
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, cast
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from autofeat import AutoFeatClassifier
from sklearn.pipeline import FunctionTransformer, Pipeline


class PathManager(Enum):
    cwd = Path('file').parent.parent.resolve()
    data = cwd / "data"
    output = cwd / "output"
    predictions = output / "predictions"
    trades = output / "trades"
    errors = output / "errors"


for path in PathManager:
    if not path.value.exists():
        path.value.mkdir(parents=True, exist_ok=True)


class PrefixManager(Enum):
    hyper = "hyper_opt_"
    ensemble = "ensemble_"
    study = "study_"


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(PathManager.data.value / "train.csv")
    test = pd.read_csv(PathManager.data.value / "test.csv")

    return train, test


def calc_mapk(y_true: pd.Series, y_probas: pd.DataFrame, k: int = 3) -> float:
    """
    Calculate Mean Average Precision at k (MAP@k) for a list of true labels and predicted probabilities.

    Parameters:
    y_true : pd.Series
        True labels for the samples.
    y_probas : pd.DataFrame
        Predicted probabilities for each class, where each
        row corresponds to a sample and each column corresponds to a class.
    k : int, optional
        The number of top predictions to consider for calculating MAP@k (default is 3).
    Returns:
    float
        The Mean Average Precision at k score.
    """
    if y_probas.shape[0] != len(y_true):
        raise ValueError("y_probas must have the same number of rows as y_true")
    if k <= 0:
        raise ValueError("k must be a positive integer")

    # Get top-k predicted class indices for each row
    topk = np.argpartition(y_probas.values, -k, axis=1)[:, -k:]
    # Sort top-k indices by probability descending
    row_indices = np.arange(y_probas.shape[0])[:, None]
    topk_sorted = topk[
        row_indices,
        np.argsort(y_probas.values[row_indices, topk], axis=1)[:, ::-1],
    ]
    # Map column indices to class labels
    class_labels = np.array(y_probas.columns)
    topk_labels = class_labels[topk_sorted]

    # Broadcast y_true for comparison
    y_true_arr = np.array(y_true).reshape(-1, 1)
    hits = topk_labels == y_true_arr

    # Compute reciprocal rank for each hit
    reciprocal_ranks = hits / (np.arange(1, k + 1))
    # Sum over k for each row (since only one hit per row is possible)
    scores = reciprocal_ranks.sum(axis=1)
    return scores.mean()


def mapk_scorer(estimator, X, y_true, k=3):
    """
    Uses estimator.predict_proba to compute MAP@k.
    y_val contains integer-encoded true labels.
    """
    probas = estimator.predict_proba(X)
    topk = np.argsort(probas, axis=1)[:, -k:][:, ::-1]  # shape: (n_samples, k)
    scores = []
    for i, true_label in enumerate(y_true):
        preds = topk[i]
        score = 0.0
        hits = 0
        seen = set()
        for rank, p in enumerate(preds):
            if p == true_label and p not in seen:
                hits += 1
                score += hits / (rank + 1)
                seen.add(p)
        scores.append(score / 1.0)  # each actual list has length=1
    return np.mean(scores)


def evaluate(estimator, X, y, cv=3) -> float:

    scores = cross_val_score(
        estimator, X, y.astype("category").cat.codes, cv=cv, scoring=mapk_scorer
    )
    return scores.mean()


def engineer_features(
    X: pd.DataFrame, autofeat_cls: Union[bool, Optional[AutoFeatClassifier]] = False
) -> Tuple[pd.DataFrame, Dict[str, List[str]], Union[AutoFeatClassifier, bool]]:
    raw_num_features = [
        "Temparature",
        "Humidity",
        "Moisture",
        "Nitrogen",
        "Phosphorous",
        "Potassium",
    ]
    raw_cat_features = ["Crop", "Soil"]

    X = X.copy()
    X = X.rename(
        columns={
            "Soil Type": "Soil",
            "Crop Type": "Crop",
        }
    )

    # X['Crop_x_Soil'] = X['Crop'] + '_' + X['Soil']
    X["Env_Stress_Index"] = (
        X["Temparature"] * 0.4 + X["Humidity"] * 0.3 + X["Moisture"] * 0.3
    )
    X["NPK_Index"] = X["Nitrogen"] * 0.5 + X["Phosphorous"] * 0.3 + X["Potassium"] * 0.2
    X["Temp_bin"] = pd.cut(
        X["Temparature"],
        bins=[-float("inf"), 15, 25, 35, float("inf")],
        labels=["low", "medium", "high", "very_high"],
    )
    X["Humidity_bin"] = pd.cut(
        X["Humidity"],
        bins=[-float("inf"), 30, 50, 70, float("inf")],
        labels=["low", "medium", "high", "very_high"],
    )
    X["Moisture_bin"] = pd.cut(
        X["Moisture"],
        bins=[-float("inf"), 20, 40, 60, float("inf")],
        labels=["low", "medium", "high", "very_high"],
    )
    X["PCA_Temparature"] = PCA(n_components=2).fit_transform(
        X[["Temparature", "Humidity", "Moisture"]]
    )[:, 0]

    print("Autofeating features...")

    if isinstance(autofeat_cls, bool):
        print("Skipping autofeat feature engineering.")
        X_autofeat = pd.DataFrame()
    elif autofeat_cls is None:
        autofeat_cls = AutoFeatClassifier(
            verbose=0, n_jobs=-1, feateng_steps=2, categorical_cols=raw_cat_features
        )
        X_autofeat = cast(pd.DataFrame, autofeat_cls.fit_transform(X[raw_num_features + raw_cat_features], X["Fertilizer Name"]))  # type: ignore
        print("Autofeat columns:", X_autofeat.columns.tolist())
    else:
        X_autofeat = cast(
            pd.DataFrame, autofeat_cls.transform(X[raw_num_features + raw_cat_features])
        )
        print("Autofeat columns:", X_autofeat.columns.tolist())

    X_final = pd.concat([X, X_autofeat], axis=1)
    X_final = X_final.loc[:, ~X_final.columns.duplicated()]

    final_dict = {
        "num_features": X_final.select_dtypes(include=["number"]).columns.tolist(),
        "cat_features": X_final.drop(
            columns=[col for col in ["Fertilizer Name"] if col in X_final.columns]
        )
        .select_dtypes(include=["object", "category"])
        .columns.tolist(),
        "autofeat_features": (
            X_autofeat.columns.tolist() if not isinstance(autofeat_cls, bool) else []
        ),
    }
    return X_final.set_index("id"), final_dict, autofeat_cls


def engineer_simple(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy().set_index("id")
    X = X.rename(
        columns={
            "Soil Type": "Soil",
            "Crop Type": "Crop",
        }
    )
    # NOTE: ENGINERRING
    # fmt: off
    X["Env_Stress_Index"] = X["Temparature"] * 0.4 + X["Humidity"] * 0.3 + X["Moisture"] * 0.3
    X["NPK_Index"] = X["Nitrogen"] * 0.5 + X["Phosphorous"] * 0.3 + X["Potassium"] * 0.2
    X["Temp_bin"] = pd.cut(X["Temparature"], bins=[-float("inf"), 15, 30, 45, float("inf")], labels=["low", "medium", "high", "very_high"])
    X["Humidity_bin"] = pd.cut(X["Humidity"], bins=[-float("inf"), 30, 50, 70, float("inf")], labels=["low", "medium", "high", "very_high"])
    X["Moisture_bin"] = pd.cut(X["Moisture"], bins=[-float("inf"), 20, 40, 60, float("inf")], labels=["low", "medium", "high", "very_high"])
    X['Soil_Nutrients'] = X['Nitrogen'] + X['Phosphorous'] + X['Potassium']
    X["Soil_Nutrient_Ratio"] = X["Nitrogen"] / (X["Potassium"] + X["Phosphorous"] + 1)
    X["Temp_Humidity"] = X["Temparature"] * X["Humidity"]
    X["Temp_Moisture"] = X["Temparature"] * X["Moisture"]
    # fmt: on
    cat_features = ["Crop", "Soil", "Temp_bin", "Humidity_bin", "Moisture_bin"]
    for col in cat_features:
        if col in X.columns:
            X[col] = X[col].astype("category")

    X_cat = pd.get_dummies(X[cat_features], drop_first=False, sparse=True)

    X_final = pd.concat([X, X_cat], axis=1)

    return X_final


def create_preprocessor() -> Pipeline:

    pipeline_steps = [
        ("simple_engineering", FunctionTransformer(engineer_simple)),
        (
            "ct",
            ColumnTransformer(
                transformers=[
                    (
                        "temp_pca",
                        PCA(n_components=2),
                        ["Temparature", "Humidity", "Moisture"],
                    ),
                    (
                        "temp_stuff",
                        "passthrough",
                        ["Temparature", "Humidity", "Moisture"],
                    ),
                ],
                remainder="passthrough",
            ),
        ),
        (
            "remove_prefixes",
            FunctionTransformer(
                lambda df: df.rename(
                    columns=lambda x: x.split("__")[-1] if "__" in x else x
                )
            ),
        ),
    ]

    preprocessor = Pipeline(steps=pipeline_steps).set_output(transform="pandas")
    return preprocessor  # type: ignore

model_run = "deepfear"
processes = None
gpu = True

logger = setup_logger(__name__)
train, test = load_data()
RANDOM_STATE = 69
job_count = mp.cpu_count() if processes is None else processes


xgb_model = XGBClassifierGPU(
    random_state=RANDOM_STATE,
    n_jobs=job_count,
    verbosity=1,
    objective="multi:softprob",
    eval_metric="mlogloss",
    enable_categorical=True,
    early_stopping_rounds=200,
)._set_gpu(use_gpu=gpu)

combinations: List[HyperOptCombination] = [
    HyperOptCombination(
        name="XGB_custom",
        model=clone(xgb_model),
        feature_combination=FeatureCombination(
            features=[
                "Temparature",
                "Humidity",
                "Moisture",
                "Nitrogen",
                "Potassium",
                "Phosphorous",
                "NPK_Index",
                "pca0",
                "Soil",
                "Crop",
                "Env_Stress_Index",
                "Soil_Nutrient_Ratio",
            ]
        ),
    ),
    HyperOptCombination(
        name="XGB_kaggle",
        model=clone(xgb_model),
        feature_combination=FeatureCombination(
            features=[
                "Temparature",
                "Humidity",
                "Moisture",
                "Nitrogen",
                "Potassium",
                "Phosphorous",
                "Temp_Humidity",
                "Temp_Moisture",
                "Soil_Nutrients",
                "Soil_Nutrient_Ratio",
                "Soil",
                "Crop",
                "Temp_bin",
            ]
        ),
    ),
    HyperOptCombination(
        name="XGB_sfs_20",
        model=clone(xgb_model),
        feature_combination=FeatureCombination(
            features=[
                "Temparature",
                "Moisture",
                "Nitrogen",
                "Potassium",
                "Phosphorous",
                "Soil",
                "Soil_Red",
                "Soil_Sandy",
                "Crop_Cotton",
                "Crop_Ground Nuts",
                "Crop_Maize",
                "Crop_Millets",
                "Crop_Oil seeds",
                "Crop_Paddy",
                "Crop_Pulses",
                "Crop_Sugarcane",
                "Crop_Wheat",
                "Temp_bin_high",
                "Humidity_bin_high",
                "Moisture_bin_high",
            ]
        ),
    ),
]


FOLDS = 3
lbe = LabelEncoder()
train["Fertilizer Name"] = lbe.fit_transform(train["Fertilizer Name"])
train["Fertilizer Name"] = train["Fertilizer Name"].astype("category")


def create_objective(data: pd.DataFrame, model_combination: HyperOptCombination):

    def objective(trial: optuna.Trial) -> OBJECTIVE_RETURN_TYPE:
        params = TrialParamWrapper().get_params(
            model_name=model_combination.name, trial=trial
        )
        model = clone(model_combination.model).set_params(**params)
        logger.info(
            f"Starting trial {trial.number} for model {model_combination.name} with params: {params}"
        )
        features = model_combination.feature_combination.features
        try:
            preprocessor = create_preprocessor()
            kfold = StratifiedKFold(
                n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE
            )
            oof_probas = pd.DataFrame(
                index=data.index, columns=lbe.transform(lbe.classes_)  # type: ignore
            )

            for fold, (train_index, val_index) in enumerate(
                kfold.split(data, data["Fertilizer Name"])
            ):
                logger.info(f"Fold {fold + 1}/{FOLDS} - {model_combination.name}")
                td = data.iloc[train_index]
                vd = data.iloc[val_index]

                X_train = preprocessor.fit_transform(
                    td.drop(columns=["Fertilizer Name"])
                )
                y_train = td["Fertilizer Name"]
                X_val = preprocessor.transform(vd.drop(columns=["Fertilizer Name"]))
                y_val = vd["Fertilizer Name"]

                model.fit(
                    X_train[features],
                    y_train,
                    eval_set=[(X_val[features], y_val)],
                    verbose=100,
                )

                oof_probas.iloc[val_index] = model.predict_proba(X_val[features])

            score = calc_mapk(y_true=data["Fertilizer Name"], y_probas=oof_probas, k=3)
            return score
        except optuna.TrialPruned as e:
            logger.warning(f"Trial {trial.number} was pruned: {e}")
            raise e
        except Exception as e:
            logger.error(f"Error during trial {trial.number}: {e}")
            raise e

    return objective


setup_dto = HyperSetupDto(
    n_optimization_trials=70,
    optimization_timeout=None,
    n_patience=30,
    min_percentage_improvement=0.005,
    model_run=model_run,
    limit_data_percentage=None,
    processes=processes,
    max_concurrent_jobs=None,
    output_dir_path=PathManager.output.value,
    hyper_opt_prefix=PrefixManager.hyper.value,
    study_prefix=PrefixManager.study.value,
    data=train,
    combinations=combinations,
    hyper_direction="maximize",
    metadata={},
    force_all_sequential=False,
    omit_names=None,
)


function_dto = HyperFunctionDto(
    create_objective_func=create_objective,
    evaluate_hyperopted_model_func=None,
)

n = setup_hyper(setup_dto=setup_dto, function_dto=function_dto)

if n > 0:
    df = setup_analysis(
        model_run=model_run,
        output_dir_path=PathManager.output.value,
        hyper_opt_prefix=PrefixManager.hyper.value,
        study_prefix=PrefixManager.study.value,
        display_plots=False,
    )

    studies_storage_path = aggregate_studies(
        study_dir_path=PathManager.output.value
        / f"{PrefixManager.study.value}{model_run}"
    )
