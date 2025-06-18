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
            "n_estimators": trial.suggest_int("n_estimators", 200, 2750),
            "max_depth": trial.suggest_int("max_depth", 6, 14),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.25, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.25, 1.0),
            "gamma": trial.suggest_float("gamma", 0.01, 10.0, log=True),
            # "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
            # "tree_method": trial.suggest_categorical("tree_method", ["approx", "hist"]),
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
            "iterations": trial.suggest_int("iterations", 200, 2000),
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
