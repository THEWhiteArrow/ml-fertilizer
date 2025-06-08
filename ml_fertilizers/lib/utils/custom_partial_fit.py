from catboost import CatBoostClassifier, CatBoostRegressor
import numpy as np
import pandas as pd
from sklearn.base import is_classifier
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import xgboost as xgb
from sklearn.utils.validation import check_is_fitted

from ml_fertilizers.lib.logger import setup_logger

logger = setup_logger(__name__)


def is_fitted(model):
    try:
        check_is_fitted(model)
        return True
    except NotFittedError:
        return False


def fit_custom_partial(
    pipeline: Pipeline,
    X_full: pd.DataFrame,
    y_full: pd.Series | pd.DataFrame,
    X_partial: pd.DataFrame,
    y_partial: pd.Series | pd.DataFrame,
) -> Pipeline:
    # Extract the model from the pipeline
    final_model = pipeline.steps[-1][1]  # Assuming the final model is the last step

    # Apply transformations
    for step_name, step in pipeline.named_steps.items():
        if step_name != "model":
            if hasattr(step, "fit_transform"):
                X_full = step.fit_transform(X_full, y_full)
                X_partial = step.transform(X_partial)
            else:
                X_full = step.transform(X_full)
                X_partial = step.transform(X_partial)

    # Check if the model is LGBM or XGB and perform partial fit
    if isinstance(final_model, lgb.LGBMClassifier) or isinstance(
        final_model, lgb.LGBMRegressor
    ):
        if not hasattr(final_model, "booster_"):
            final_model.fit(X_full, y_full)
        elif not is_fitted(final_model):
            raise ValueError("LGBM Model is not fitted but has a booster")
        else:
            # logger.info("Partial fit LGBM")
            lbg_train = lgb.Dataset(X_partial, label=y_partial)
            params = final_model.get_params()
            params.pop("n_estimators", None)
            final_model._Booster = lgb.train(  # type: ignore
                params,
                lbg_train,
                init_model=final_model.booster_,
                num_boost_round=final_model.get_params()["n_estimators"],
            )
    elif isinstance(final_model, xgb.XGBClassifier) or isinstance(
        final_model, xgb.XGBRegressor
    ):
        if not hasattr(final_model, "_Booster"):
            final_model.fit(X_full, y_full)
        elif not is_fitted(final_model):
            raise ValueError("XGB Model is not fitted but has a booster")
        else:
            # logger.info("Partial fit XGB")
            xgb_train = xgb.DMatrix(X_partial, label=y_partial)
            params = final_model.get_xgb_params()
            final_model._Booster = xgb.train(
                params,
                xgb_train,
                xgb_model=final_model._Booster,
            )
    elif isinstance(final_model, CatBoostClassifier) or isinstance(
        final_model, CatBoostRegressor
    ):
        if not final_model.is_fitted():
            final_model.fit(X_full, y_full)
        else:
            # logger.info("Partial fit CatBoost")
            final_model.fit(X_partial, y_partial, init_model=final_model)

    elif hasattr(final_model, "partial_fit"):
        # logger.info("Partial fit generic")

        # check if the model is regression or classification

        if not hasattr(final_model, "classes_") and is_classifier(final_model):
            for i in range(final_model.max_iter):
                final_model.partial_fit(X_full, y_full, classes=np.unique(y_full))
        elif not is_fitted(final_model):
            logger.info(
                "Fitting full data despite the model having a partial_fit method"
            )
            final_model.fit(X_full, y_full)
        else:
            for i in range(final_model.max_iter):
                # if fitted for the first time then it should be fitted with the full data
                final_model.partial_fit(X_partial, y_partial)
    else:
        raise ValueError("Model does not support partial fitting")

    # Update the pipeline with the partially fitted model
    pipeline.steps[-1] = ("model", final_model)

    return pipeline
