import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import datetime as dt
from typing import List, Optional

from ml_fertilizers.lib.models.EnsembleModel2 import EnsembleModel2
from ml_fertilizers.lib.utils.custom_partial_fit import fit_custom_partial
from ml_fertilizers.lib.logger import setup_logger
from ml_fertilizers.lib.utils.garbage_collector import garbage_manager

logger = setup_logger(__name__)


def get_tscv_predictions(
    X: pd.DataFrame,
    y: pd.DataFrame | pd.Series,
    estimator: Pipeline,
    train_start_datetime: dt.datetime,
    test_start_datetime: dt.datetime,
    test_end_datetime: dt.datetime,
    test_size: int = 5,
    max_train_size: Optional[int] = None,
    force_full_train: bool = True,
    proba_threshold: Optional[float] = None,
) -> pd.Series:
    # print("tscv")

    info_dict = dict()
    all_raw_predictions: List[float] = []

    # --- NOTE ---
    # Start works provided that the start datetime is at the midnight of the day
    if isinstance(estimator, Pipeline):
        is_nn_model = estimator[-1].__class__.__name__.lower() in (
            "neuralnetworkcustommodel"
        )
    else:
        is_nn_model = False

    test_start_row_number = len(X.loc[X.index < test_start_datetime])
    test_end_row_number = len(X.loc[X.index <= test_end_datetime])

    start = len(X.loc[X.index < train_start_datetime])
    end = test_start_row_number

    if end >= test_end_row_number:
        logger.error(
            f"Test start datetime {test_start_datetime} is greater than test end datetime {test_end_datetime}"
        )
        raise ValueError(
            f"Test start datetime {test_start_datetime} is greater than test end datetime {test_end_datetime}"
        )

    X_train = None
    X_test = None
    y_train = None
    raw_predictions = None

    while end < test_end_row_number:
        if max_train_size is not None:
            start = max(0, end - max_train_size)

        X_train = X.iloc[start:end]
        X_test = X.iloc[
            end - estimator[-1].window + 1 if is_nn_model else end : min(
                end + test_size, test_end_row_number
            )
        ]

        y_train = y.iloc[start:end]
        # y_test = y.iloc[end : end + test_size]
        if isinstance(estimator, EnsembleModel2) and not force_full_train:
            estimator = estimator.partial_fit(
                X_train, y_train, X_train[-test_size:], y_train[-test_size:]
            )  # type: ignore
        elif (
            not isinstance(estimator, EnsembleModel2)
            and (
                any(
                    name in estimator[-1].__class__.__name__.lower()
                    for name in ["lgbm", "xgb", "cat"]
                )
                or hasattr(estimator[-1], "partial_fit")
            )
            and not force_full_train
        ):
            estimator = fit_custom_partial(
                pipeline=estimator,
                X_full=X_train,
                y_full=y_train,
                X_partial=X_train[-test_size:],
                y_partial=y_train[-test_size:],
            )
        else:
            estimator = estimator.fit(X_train, y_train)

        if not is_nn_model:

            if proba_threshold is not None and hasattr(estimator[-1], "predict_proba"):
                if info_dict.get("predict_proba", 0) == 0:
                    info_dict["predict_proba"] = 1
                    logger.info(
                        "Using predict_proba. This message will be shown only once."
                    )
                raw_proba_predictions = estimator.predict_proba(X_test)[:, 1]
                raw_predictions = (raw_proba_predictions >= proba_threshold).astype(int)
            else:
                raw_predictions = estimator.predict(X_test)
        else:
            raise Exception("Neural network model prediction is supported.")
            X_test_transformed = X_test.copy()

            for name, transformer in estimator.named_steps.items():
                if hasattr(transformer, "transform"):
                    X_test_transformed = transformer.transform(X_test_transformed)
                if hasattr(transformer, "predict_proba"):
                    raw_predictions_arr = transformer.predict_proba(X_test_transformed)[
                        :, 1
                    ]
                elif hasattr(transformer, "predict"):
                    raw_predictions_arr = transformer.predict(X_test_transformed)

            raw_predictions = np.array(raw_predictions_arr).reshape(-1)

        all_raw_predictions.extend(raw_predictions)  # type: ignore

        end += test_size

    all_predictions = pd.Series(
        all_raw_predictions,
        index=y.loc[X.index[test_start_row_number:test_end_row_number]].index,
        name="prediction",
    )

    # if proba_threshold is not None:
    #     all_predictions = (all_predictions >= proba_threshold).astype(int)

    del X_train
    del X_test
    del y_train
    del raw_predictions
    del all_raw_predictions

    garbage_manager.clean()

    return all_predictions
