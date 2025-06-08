import datetime as dt
import math
from multiprocessing.managers import ListProxy
import signal
from typing import Dict, Any, Callable, List, Literal, Optional, cast
import multiprocessing as mp

import pandas as pd
import numpy as np
from sklearn import clone
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from ml_fertilizers.lib.logger import setup_logger
from ml_fertilizers.lib.utils.garbage_collector import garbage_manager
from ml_fertilizers.lib.models.EnsembleModel2 import EnsembleModel2

logger = setup_logger(__name__)


def optimize_ensemble(
    ensemble_model: EnsembleModel2,
    X: pd.DataFrame,
    y: pd.Series,
    direction: Literal["maximize", "minimize", "equal"],
    task_type: Literal["classification", "regression"],
    processes: int,
    evaluation_column: str,
    calculate_predictions_func: Callable,
    calculate_score_func: Callable,
    evaluate_ensemble_func: Optional[Callable],
    ensemble_min_size: int,
) -> pd.DataFrame:

    logger.info("Optimizing ensemble model...")

    X = X.copy()
    y = y.copy()
    y_all_pred: List[pd.Series] = []

    # Here we are training the ensemble models separately and after all of it
    # we should retrain the final ensemble model and compare if the results are the same
    # NOTE: This confirms that there are no mistakes in the code
    for index, every_model in enumerate(ensemble_model.estimators):
        target_name = ensemble_model.combination_metadatas[index].get(
            "target_name", None
        )
        y_model = y.copy()
        if target_name is not None and target_name != y.name:
            y_model = X[target_name].copy()
        current_model_predictions = cast(
            pd.Series,
            calculate_predictions_func(estimator=clone(every_model), X=X, y=y_model),
        ).rename(ensemble_model.combination_names[index])

        y_all_pred.append(current_model_predictions)

    ensemble_model.predictions = y_all_pred
    test_indexes = y_all_pred[0].index
    results_list: List[Dict[str, Any]] = []

    logger.info("Multiprocessing bitmap v2")
    results_list.extend(
        run_parralel_bitmap_processing(
            ensemble_model,
            X=X.loc[test_indexes],
            y=X.loc[test_indexes, evaluation_column],
            processes=processes,
            version=2,
            direction=direction,
            task_type=task_type,
            calculate_score_func=calculate_score_func,
            evaluate_ensemble_func=evaluate_ensemble_func,
            ensemble_min_size=ensemble_min_size,
        )
    )

    results_df = pd.DataFrame(results_list)

    del X
    del y
    del y_all_pred
    del ensemble_model
    del results_list
    del test_indexes

    garbage_manager.clean()
    return results_df


def init_worker() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def calculate_n_combinations(num_models: int, ensemble_min_size: int) -> int:
    """
    Calculate the number of combinations of models in the ensemble.
    This is used to determine how many combinations we will evaluate.
    """
    total_combinations = 0
    for i in range(ensemble_min_size, num_models + 1):
        total_combinations += math.comb(num_models, i)
    return total_combinations


def run_parralel_bitmap_processing(
    ensemble_model: EnsembleModel2,
    X: pd.DataFrame,
    y: pd.Series,
    processes: int,
    version: Literal[1, 2, 3],
    direction: Literal["maximize", "minimize", "equal"],
    task_type: Literal["classification", "regression"],
    calculate_score_func: Callable,
    evaluate_ensemble_func: Optional[Callable],
    ensemble_min_size: int,
) -> List[Dict[str, Any]]:

    start_time = dt.datetime.now()
    if ensemble_model.predictions is None:
        raise ValueError("The ensemble model has not been predicted yet")

    # Number of models in the ensemble
    num_models = len(ensemble_model.predictions)
    n_samples = len(y)

    # --- Multiprocessing setup ---
    mp_predictions = mp.Array(
        "q", num_models * n_samples
    )  # Double type for predictions
    mp_predictions_np = np.frombuffer(mp_predictions.get_obj()).reshape(
        num_models, n_samples
    )
    # Ensure the shared array is filled with predictions
    for i in range(num_models):
        mp_predictions_np[i] = ensemble_model.predictions[i].to_numpy()

    # Setup for targets (use int64 type)
    mp_y_true = mp.Array("q", n_samples)  # Use 'q' for int64
    mp_y_true_np = np.frombuffer(mp_y_true.get_obj())
    mp_y_true_np[:] = y.to_numpy()  # Assign values to the shared array

    # Setup for scores
    mp_scores = mp.Array("d", len(ensemble_model.combination_scoring))
    mp_scores_np = np.frombuffer(mp_scores.get_obj())
    mp_scores_np[:] = (
        ensemble_model.combination_scoring
    )  # Assign values to the shared array

    # Setup for names
    mp_names_np = mp.Manager().list(ensemble_model.combination_names)

    combinations_to_evaluate = [
        c
        for c in range(1, 2**num_models)
        if (bin(c).count("1") >= ensemble_min_size or bin(c).count("1") == 1)
    ]
    logger.info(
        f"Starting multiprocessing of combinations: {len(combinations_to_evaluate)}"
    )

    # --- Multiprocessing ---
    with mp.Pool(processes=processes, initializer=init_worker) as pool:
        mp_results = pool.starmap(
            evaluate_combination,
            [
                (
                    i,
                    mp_predictions_np,
                    mp_scores_np,
                    mp_names_np,
                    mp_y_true_np,
                    version,
                    direction,
                    task_type,
                    calculate_score_func,
                    evaluate_ensemble_func,
                )
                for i in combinations_to_evaluate
            ],
        )

    end_time = dt.datetime.now()
    logger.info(f"Multiprocessing took: {end_time - start_time}")

    del mp_predictions
    del mp_y_true
    del mp_scores
    del mp_names_np
    del mp_predictions_np
    del mp_y_true_np
    del mp_scores_np

    garbage_manager.clean(threshold={"milliseconds": 500})

    return mp_results


def evaluate_combination(
    bitmap: int,
    predictions: np.ndarray,
    scores: np.ndarray,
    names: ListProxy,
    y_test: np.ndarray,
    version: Literal[1, 2, 3],
    direction: Literal["maximize", "minimize", "equal"],
    task_type: Literal["classification", "regression"],
    calculate_score_func: Callable,
    evaluate_ensemble_func: Optional[Callable],
) -> Dict[str, Any]:
    try:

        selected_indices = [i for i in range(len(names)) if bitmap & (1 << i)]
        y_sr = pd.Series(y_test)

        combination_names = [names[i] for i in selected_indices]
        combination_predictions = predictions[selected_indices]
        combination_weights = EnsembleModel2._calculate_weights(
            scores[selected_indices], direction
        )

        if version == 1:
            raise NotImplementedError("Not implemented")

        elif version == 2:

            # if task_type in ("classification", "mixed"):
            combination_y_pred = (
                EnsembleModel2._combine_classification_weighted_voting2(
                    predictions=[
                        pd.Series(prediction) for prediction in combination_predictions
                    ],
                    weights=combination_weights,
                )
            )
            # elif task_type == "regression":

        elif version == 3:
            combination_y_pred = np.average(
                predictions[selected_indices], axis=0, weights=combination_weights
            )
        else:
            raise ValueError("Invalid version")

        score = calculate_score_func(y_test, combination_y_pred)

        res = {
            "combination_names": combination_names,
            "bitmap": bitmap,
            "score": score,
            "models_cnt": len(combination_names),
            "predictions": combination_y_pred,
        }

        res.update(evaluate_ensemble_func(y_sr, combination_y_pred))

        del y_sr
        del combination_predictions
        del combination_weights

        if bitmap % 2**16 == 0:
            garbage_manager.clean(threshold={"milliseconds": 500})

        return res
    except Exception as e:
        logger.error(f"Error in evaluate_combination: {e}")
        raise e


def default_calculate_prediction(
    estimator: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    shuffle: bool = True,
    n_cv: int = 5,
    random_state: int = 42,
) -> pd.Series:
    """
    Default function to calculate predictions using cross-validation.
    This function is used when no specific prediction function is provided.
    """
    kf = KFold(n_splits=n_cv, shuffle=shuffle, random_state=random_state)
    predictions = pd.Series(index=X.index, dtype=np.float32)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train = y.iloc[train_index]

        estimator.fit(X_train, y_train)
        predictions.iloc[test_index] = estimator.predict(X_test)

    return predictions.rename("prediction")
