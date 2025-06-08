from pathlib import Path
import pickle
from typing import List, Optional, cast

import optuna

from ml_fertilizers.lib.models.HyperOptResultDict import HyperOptResultDict
from ml_fertilizers.lib.logger import setup_logger

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
