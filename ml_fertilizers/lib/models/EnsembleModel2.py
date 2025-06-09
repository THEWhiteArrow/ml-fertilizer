from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.base import BaseEstimator, check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from ml_fertilizers.lib.logger import setup_logger
from ml_fertilizers.lib.pipelines.ProcessingPipelineWrapper import (
    create_pipeline,
)
from ml_fertilizers.lib.utils.garbage_collector import garbage_manager

logger = setup_logger(__name__)


class EnsembleModel2(BaseEstimator):

    def __init__(
        self,
        models: List[BaseEstimator],
        combination_features: List[List[str]],
        combination_names: List[str],
        combination_scoring: Optional[List[float]] = None,
        combination_metadatas: Optional[List[Dict[str, Any]]] = None,
        scoring_direction: Optional[Literal["maximize", "minimize"]] = None,
        task: Literal["classification", "regression"] = "classification",
        metamodel: Optional[BaseEstimator] = None,
        metadata: Optional[dict] = None,
        prediction_method: Literal["predict", "predict_proba"] = "predict",
        metamodel_shuffle: bool = True,
        metamodel_kfold: int = 5,
        just_filtering: bool = False,
    ) -> None:
        # fmt: off
        self.estimators: List[Pipeline] = [create_pipeline(model=model, features_in=features, just_filtering=just_filtering) for model, features in zip(models, combination_features, strict=True)]
        self.combination_features : List[List[str]] = combination_features
        self.combination_names : List[str] = combination_names
        self.combination_metadatas: List[Dict[str, Any]] = combination_metadatas if combination_metadatas is not None else [dict() for _ in range(len(self.estimators))]

        self.task: Literal["classification", "regression"] = task
        self.metadata: Dict[str, Any] = metadata if metadata is not None else dict()
        self.metamodel_estimator : Optional[Pipeline] = create_pipeline(model=metamodel) if metamodel is not None else None
        self.metamodel_kfold: int = metamodel_kfold

        self.prediction_method: Literal["predict", "predict_proba"] = prediction_method
        self.metamodel_shuffle: bool = metamodel_shuffle
        self.just_filtering: bool = just_filtering

        if self.metamodel_estimator is None:
            self.combination_scoring : Optional[List[float]] = combination_scoring if combination_scoring is not None else [1.0] * len(self.estimators)
            self.scoring_direction: Optional[Literal["maximize", "minimize"]] = scoring_direction if scoring_direction is not None else "maximize"
            self.weights : Optional[List[float]] = self._calculate_weights(scoring=self.combination_scoring, scoring_direction=self.scoring_direction)  # type: ignore
        else:
            self.weights: Optional[List[float]] = None
            self.combination_scoring: Optional[List[float]] = None
            self.scoring_direction: Optional[Literal["maximize", "minimize"]] = None

        self._health_check()
        # fmt: on

    def _health_check(self) -> None:
        pass
        # if (
        #     len(self.combination_scoring) != len(self.estimators)
        #     or len(self.combination_scoring) != len(self.combination_names)
        #     or len(self.combination_scoring) != len(self.combination_features)
        #     or (
        #         self.predictions is not None
        #         and len(self.combination_scoring) != len(self.predictions)
        #     )
        #     or len(self.weights) != len(self.combination_scoring)
        # ):
        #     raise ValueError(
        #         "Lengths of the combination_scoring, estimators, combination_names, combination_features and predictions should be the same"
        #     )

        # if sum(self.weights) < 0.99 or sum(self.weights) > 1.01:
        #     raise ValueError("Weights should sum to 1")

        # if any(
        #     self.weights
        #     != self._calculate_weights(
        #         scoring=self.combination_scoring,
        #         scoring_direction=self.scoring_direction,
        #     )  # type: ignore
        # ):
        #     raise ValueError(
        #         "Weights should be calculated based on the scoring and scoring_direction"
        #     )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EnsembleModel2":

        if self.metamodel_estimator is None:
            logger.info("Fitting ensemble model...")
            for i, estimator in enumerate(self.estimators):
                target_name = self.combination_metadatas[i].get("target_name", None)

                logger.info(
                    f"Fitting model : {self.combination_names[i]} | target: {target_name}"
                )

                if target_name is not None and target_name != y.name:
                    estimator.fit(X, X[target_name])
                else:
                    estimator.fit(X, y)
        else:
            logger.info("Fitting stacked ensemble model...")

            kfold = KFold(
                n_splits=self.metamodel_kfold,
                shuffle=self.metamodel_shuffle,
                random_state=1000000007,
            )
            if self.prediction_method == "predict":
                all_predictions_out_of_fold = pd.DataFrame(
                    columns=self.combination_names, index=X.index, dtype=np.float64
                )
            elif self.prediction_method == "predict_proba":
                all_predictions_out_of_fold = pd.DataFrame(
                    index=X.index, dtype=np.float64
                )
            else:
                raise ValueError(
                    f"Prediction method {self.prediction_method} not recognized. Please use appropriate 'predict' or 'predict_proba'"
                )

            X_train = None
            X_test = None
            y_train = None

            for i, (train_idx, test_idx) in enumerate(kfold.split(X)):
                logger.info(f"Fold {i + 1}")

                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train = y.iloc[train_idx]

                for j, estimator in enumerate(self.estimators):
                    logger.info(f"Fitting model : {self.combination_names[j]}")
                    target_name = self.combination_metadatas[j].get("target_name", None)

                    if target_name is not None and target_name != y.name:
                        estimator.fit(X_train, X_train[target_name])
                    else:
                        estimator.fit(X_train, y_train)

                    logger.info(f"Predicting model : {self.combination_names[j]}")

                    if self.prediction_method == "predict":
                        y_pred = pd.Series(
                            estimator.predict(X_test), index=X_test.index
                        )
                        all_predictions_out_of_fold.loc[
                            X_test.index, self.combination_names[j]
                        ] = y_pred
                    elif self.prediction_method == "predict_proba":
                        y_pred = pd.DataFrame(
                            estimator.predict_proba(X_test),
                            columns=[
                                f"{self.combination_names[j]}__{class_name}"
                                for class_name in estimator.classes_
                            ],
                            index=X_test.index,
                        )
                        all_predictions_out_of_fold.loc[
                            X_test.index, y_pred.columns
                        ] = y_pred

            del X_train
            del X_test
            del y_train
            garbage_manager.clean()

            logger.info(
                "Completed fitting all models and making predictions on out of fold data"
            )
            logger.info("Fitting meta model...")
            # pkl.dump(
            #     all_predictions_out_of_fold,
            #     open("output/all_predictions_out_of_fold.pkl", "wb"),
            # )
            self.metamodel_estimator.fit(all_predictions_out_of_fold, y)

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        logger.info("Making predictions with ensemble model...")
        self.predictions: Optional[List[pd.Series | pd.DataFrame]] = None

        if self.prediction_method == "predict":
            self.predictions = [
                pd.Series(estimator.predict(X), name=name, index=X.index)
                for estimator, name in zip(
                    self.estimators, self.combination_names, strict=True
                )
            ]
        elif self.prediction_method == "predict_proba":
            self.predictions = [
                pd.DataFrame(
                    estimator.predict_proba(X),
                    columns=[f"{class_name}" for class_name in estimator.classes_],
                    index=X.index,
                )
                for estimator, _ in zip(self.estimators, self.combination_names)
            ]
        else:
            raise ValueError(
                f"Prediction method {self.prediction_method} not recognized. Please use appropriate 'predict' or 'predict_proba'"
            )

        final_pred = self._combine_predictions(self.predictions)

        return final_pred

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Making probability predictions with ensemble model...")
        self.predictions: Optional[List[pd.Series | pd.DataFrame]] = []
        if self.metamodel_estimator is not None:

            all_predictions_fold = pd.DataFrame(index=X.index, dtype=np.float64)
            for j, estimator in enumerate(self.estimators):
                y_pred = pd.DataFrame(
                    estimator.predict_proba(X),
                    columns=[
                        f"{self.combination_names[j]}__{class_name}"
                        for class_name in estimator.classes_
                    ],
                    index=X.index,
                )
                self.predictions.append(y_pred)

                all_predictions_fold.loc[:, y_pred.columns] = y_pred

            final_pred_raw = self.metamodel_estimator.predict_proba(all_predictions_fold)  # type: ignore
            final_pred = pd.DataFrame(
                final_pred_raw,
                index=X.index,
                columns=self.metamodel_estimator[-1].classes_,
            )

            return final_pred

        else:
            self.predictions = [
                pd.DataFrame(
                    estimator.predict_proba(X),
                    columns=[f"{class_name}" for class_name in estimator.classes_],
                    index=X.index,
                )
                for estimator, _ in zip(
                    self.estimators, self.combination_names, strict=True
                )
            ]

            final_probas = self._combine_classification_weighted_probablity(
                predictions=self.predictions, weights=self.weights  # type: ignore
            )

            return final_probas

    @classmethod
    def _combine_classification_weighted_probablity(
        cls,
        predictions: List[pd.Series | pd.DataFrame],
        weights: List[float],
    ) -> pd.DataFrame:

        final_proba = pd.DataFrame(
            index=predictions[0].index, columns=predictions[0].columns
        )

        for i, prediction in enumerate(predictions):
            final_proba = final_proba.add(prediction * weights[i], fill_value=0)

        final_proba = final_proba.div(final_proba.sum(axis=1), axis=0)
        final_proba = final_proba.fillna(0.0)
        return final_proba

    @classmethod
    def _combine_classification_weighted_voting2(
        cls, predictions: List[pd.Series | pd.DataFrame], weights: List[float]
    ) -> pd.Series:

        # Initialize a DataFrame to store the weighted votes
        if all(isinstance(prediction, pd.DataFrame) for prediction in predictions):
            unique_targets = sorted(
                set(
                    target
                    for prediction in predictions
                    if isinstance(prediction, pd.DataFrame)
                    for target in prediction.columns
                )
            )
        elif all(isinstance(prediction, pd.Series) for prediction in predictions):
            unique_targets = sorted(
                set(
                    target
                    for prediction in predictions
                    if isinstance(prediction, pd.Series)
                    for target in prediction.unique()
                )
            )
        else:
            raise ValueError(
                "All predictions should be either pd.Series or pd.DataFrame"
            )

        votes = pd.DataFrame(
            np.zeros((len(predictions[0]), len(unique_targets)), dtype=np.float32),
            columns=unique_targets,
            index=predictions[0].index,
        )

        # Accumulate weighted votes
        for i, prediction in enumerate(predictions):
            if isinstance(prediction, pd.Series):
                for target in unique_targets:
                    votes.loc[:, target] += (prediction == target).astype(
                        np.float32
                    ) * weights[i]
            elif isinstance(prediction, pd.DataFrame):
                for target in unique_targets:
                    if target in prediction.columns:
                        votes.loc[:, target] += prediction[target] * weights[i]
                    else:
                        logger.warning(
                            f"Target {target} not found in prediction DataFrame columns"
                        )
            else:
                raise ValueError(
                    "All predictions should be either pd.Series or pd.DataFrame"
                )

        final_vote = votes.idxmax(axis=1).astype(int)
        final_vote.name = "prediction"

        del votes
        del unique_targets

        return final_vote

    @classmethod
    def _combine_regression_weighted(
        cls, predictions: List[pd.Series], weights: List[float]
    ) -> pd.Series:
        final_pred = pd.Series(
            np.zeros(len(predictions[0])), index=predictions[0].index, name="prediction"
        )

        for i, prediction in enumerate(predictions):
            final_pred += prediction * weights[i]

        return final_pred

    def _combine_predictions(
        self, predictions: List[pd.Series | pd.DataFrame]
    ) -> pd.Series:

        final_pred: Optional[pd.Series] = None

        if self.metamodel_estimator is None:
            # classification task with mixed models
            if self.task == "classification":
                final_pred = self._combine_classification_weighted_voting2(
                    predictions=predictions, weights=self.weights  # type: ignore
                )
            else:
                if any(isinstance(p, pd.DataFrame) for p in predictions):
                    raise ValueError(
                        "All predictions should be pd.Series for regression task"
                    )
                final_pred = self._combine_regression_weighted(
                    predictions=predictions, weights=self.weights  # type: ignore
                )

        else:
            metamodel_X = pd.concat(predictions, axis=1, ignore_index=False)
            final_pred_raw = self.metamodel_estimator.predict(metamodel_X)  # type: ignore
            final_pred = pd.Series(
                final_pred_raw, index=metamodel_X.index, name="prediction"
            )

        if final_pred is None:
            raise ValueError("Final predictions could not be computed")

        return final_pred

    def __sklearn_clone__(self):
        """
        Support sklearn's clone functionality.
        """
        return self.clean_copy()

    def clean_copy(self) -> "EnsembleModel2":
        """
        Returns a fresh copy of the ensemble with cloned estimators and metamodel.
        """
        return EnsembleModel2(
            models=[clone(estimator[-1]) for estimator in self.estimators],
            combination_features=[
                features.copy() for features in self.combination_features
            ],
            combination_names=self.combination_names.copy(),
            combination_scoring=(
                self.combination_scoring.copy()
                if self.combination_scoring is not None
                else None
            ),
            combination_metadatas=[meta.copy() for meta in self.combination_metadatas],
            scoring_direction=self.scoring_direction,
            task=self.task,
            metamodel=(
                clone(self.metamodel_estimator[-1])
                if self.metamodel_estimator is not None
                else None
            ),
            metamodel_kfold=self.metamodel_kfold,
            prediction_method=self.prediction_method,
            metamodel_shuffle=self.metamodel_shuffle,
            metadata=self.metadata.copy() if self.metadata is not None else None,
            just_filtering=self.just_filtering,
        )

    def is_fitted(self) -> bool:
        def handle_is_fitted(model):
            try:
                check_is_fitted(model)
                return True
            except NotFittedError:
                return False

        return all([handle_is_fitted(estimator) for estimator in self.estimators])

    @classmethod
    def _get_list_by_bitmap(cls, lst: List, bitmap: int) -> List:
        return [lst[i] for i in range(len(lst)) if bitmap & (1 << i)]

    @classmethod
    def _calculate_weights(
        cls, scoring: List[float], scoring_direction: str
    ) -> List[float]:
        if scoring_direction == "maximize":
            return scoring / np.sum(scoring)
        elif scoring_direction == "minimize":
            return 1 - (scoring / np.sum(scoring))
        else:
            raise ValueError(
                "scoring_direction not recognized. Please use 'maximize' or 'minimize'"
            )
