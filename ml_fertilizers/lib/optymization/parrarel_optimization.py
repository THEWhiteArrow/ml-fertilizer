import json
import signal
from pathlib import Path
import multiprocessing as mp
from typing import Callable, Dict, List, Optional, TypedDict


import pandas as pd

from ml_fertilizers.lib.logger import setup_logger
from ml_fertilizers.lib.models.HyperOptCombination import HyperOptCombination
from ml_fertilizers.lib.optymization.optimization_study import (
    CREATE_OBJECTIVE_TYPE,
    OPTUNA_DIRECTION_TYPE,
    optimize_model_and_save,
)

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
