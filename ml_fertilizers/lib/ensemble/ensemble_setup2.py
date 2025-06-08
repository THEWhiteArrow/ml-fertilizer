from dataclasses import dataclass
import os
from pathlib import Path
import datetime as dt
from typing import Any, Dict, List, Literal, Optional, Callable, Tuple
import multiprocessing as mp

import pandas as pd
from sklearn.base import BaseEstimator

from ml_fertilizers.lib.logger import setup_logger
from ml_fertilizers.lib.models.EnsembleModel2 import EnsembleModel2
from ml_fertilizers.lib.utils.read_existing_models import read_hyper_results

# from ml_fertilizers.setup.objective_ensemble_task import (
#     create_ensemble_interpret_opt_results,
# )


logger = setup_logger(__name__)


@dataclass
class EnsembleFunctionDto2:
    engineer_features_func: Callable[[pd.DataFrame], pd.DataFrame]
    execute_optimization_process_func: Optional[
        Callable[
            [EnsembleModel2, pd.DataFrame, pd.Series],
            Tuple[pd.DataFrame, Dict[str, Any]],
        ]
    ]
    objective_ensemble_task_func: Optional[Callable]


@dataclass
class EnsembleSetupDto2:
    hyper_model_run: str
    model_run: Optional[str]
    meta_model: Optional[BaseEstimator]
    limit_data_percentage: float
    selected_model_names: Optional[List[str]]
    optimize: bool
    retrain: bool
    old_ensemble: Optional[EnsembleModel2]
    score_direction: Literal["maximize", "minimize"]
    target_column: str
    id_column: List[str] | str
    prediction_method: Literal["predict", "predict_proba"]
    prediction_proba_target: Optional[str]
    task: Literal["classification", "regression"]
    train_start_datetime: dt.datetime
    test_end_datetime: dt.datetime
    processes: int | None
    output_dir_path: Path
    hyper_opt_prefix: str
    ensemble_prefix: str
    data: pd.DataFrame
    ensemble_min_size: int


def setup_ensemble_v2(
    setup_dto: EnsembleSetupDto2, function_dto: EnsembleFunctionDto2
) -> EnsembleModel2:
    if setup_dto.model_run is None:
        setup_dto.model_run = dt.datetime.now().strftime("%Y%m%d%H%M")
    processes = setup_dto.processes
    if processes is None:
        processes = mp.cpu_count()

    logger.info(f"Starting ensemble setup v2 | model_run: {setup_dto.model_run}")
    logger.info("Loading data...")
    train = setup_dto.data
    logger.info(f"Limiting data to {setup_dto.limit_data_percentage * 100}%")
    train = train.head(int(len(train) * setup_dto.limit_data_percentage))
    logger.info("Engineering features...")
    data = function_dto.engineer_features_func(train)
    X = data.loc[setup_dto.train_start_datetime :]
    y = data[setup_dto.target_column].loc[setup_dto.train_start_datetime :]

    if setup_dto.optimize:

        logger.info("Retriving hyperopt models...")
        hyper_model_results = read_hyper_results(
            path=setup_dto.output_dir_path
            / f"{setup_dto.hyper_opt_prefix}{setup_dto.hyper_model_run}",
            selection=setup_dto.selected_model_names,
        )

        # scoring = [results["oracle"] for results in hyper_model_results]
        # scoring = [results["tscv_precision"] for results in hyper_model_results]
        scoring = [1 for results in hyper_model_results]

        logger.info("Creating ensemble model...")
        ensemble_model = EnsembleModel2(
            models=[results["model"] for results in hyper_model_results],
            combination_features=[
                results["features"] for results in hyper_model_results
            ],
            combination_metadatas=[
                results["metadata"] for results in hyper_model_results
            ],
            combination_names=[results["name"] for results in hyper_model_results],
            task=setup_dto.task,
            combination_scoring=scoring,
            scoring_direction=setup_dto.score_direction,
            prediction_method=setup_dto.prediction_method,
            meta_model=setup_dto.meta_model,
            prediction_proba_target=setup_dto.prediction_proba_target,
        )

    elif setup_dto.retrain:
        if setup_dto.old_ensemble is None:
            logger.info("Loading old ensemble model...")
            raise ValueError("Invalid setup_dto.")

        ensemble_model = setup_dto.old_ensemble

    if function_dto.execute_optimization_process_func is not None:
        opt_results, metadata = function_dto.execute_optimization_process_func(
            ensemble_model.clean_copy(), X, y
        )
    else:
        raise ValueError(
            "execute_optimization_process_func is None. Please provide a function."
        )

    os.makedirs(
        setup_dto.output_dir_path / f"{setup_dto.ensemble_prefix}{setup_dto.model_run}",
        exist_ok=True,
    )

    results_prefix = ""
    if setup_dto.optimize:
        results_prefix = "opt_"
    elif setup_dto.retrain:
        results_prefix = "retrain_"

    opt_results.to_csv(
        setup_dto.output_dir_path
        / f"{setup_dto.ensemble_prefix}{setup_dto.model_run}"
        / f"analysis_results_{results_prefix}{setup_dto.model_run}.csv",
        index=False,
    )

    if setup_dto.optimize:
        final_ensemble_model, final_metadata = create_ensemble_interpret_opt_results(
            ensemble_model=ensemble_model,
            hyper_model_results=hyper_model_results,
            opt_results=opt_results,
            metadata=metadata,
            ensemble_min_size=setup_dto.ensemble_min_size,
        )

        final_ensemble_model = final_ensemble_model.clean_copy()
        final_ensemble_model.metadata = final_metadata

    elif setup_dto.retrain:
        final_ensemble_model = setup_dto.old_ensemble.clean_copy()
        final_ensemble_model.metadata = metadata

    final_ensemble_model = final_ensemble_model.fit(
        X.loc[setup_dto.train_start_datetime : setup_dto.test_end_datetime],
        y.loc[setup_dto.train_start_datetime : setup_dto.test_end_datetime],
    )

    return final_ensemble_model
