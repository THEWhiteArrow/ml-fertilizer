import os
from pathlib import Path

import optuna
import pandas as pd
from ml_fertilizers.lib.logger import setup_logger
from ml_fertilizers.lib.utils.results import (
    load_hyper_opt_results,
    load_hyper_opt_studies,
)


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


if __name__ == "__main__":
    # Example usage
    model_run = "_classification_precision_maj_"
    output_dir_path = Path(os.getcwd()) / "output" / "ADI"
    hyper_opt_prefix = "hyper_opt"
    study_prefix = "study"
    display_plots = False

    setup_analysis(
        model_run=model_run,
        output_dir_path=output_dir_path,
        hyper_opt_prefix=hyper_opt_prefix,
        study_prefix=study_prefix,
        display_plots=display_plots,
    )
