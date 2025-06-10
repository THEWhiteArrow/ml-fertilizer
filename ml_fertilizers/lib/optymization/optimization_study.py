import os
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

# import multiprocessing as mp

import optuna
import pandas as pd
import pickle

from sklearn import clone

from ml_fertilizers.lib.models.HyperOptCombination import HyperOptCombination
from ml_fertilizers.lib.models.HyperOptResultDict import HyperOptResultDict
from ml_fertilizers.lib.optymization.EarlyStoppingCallback import (
    EarlyStoppingCallback,
)
from ml_fertilizers.lib.logger import setup_logger
from ml_fertilizers.lib.utils.garbage_collector import garbage_manager


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
            gc_after_trial=True,
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
