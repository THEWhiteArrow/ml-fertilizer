import datetime as dt

from ml_fertilizers.lib.logger import setup_logger

from ml_fertilizers.lib.optymization.parrarel_optimization import (
    HyperFunctionDto,
    HyperSetupDto,
    run_parallel_optimization,
)
from ml_fertilizers.lib.utils.results import load_hyper_opt_results

logger = setup_logger(__name__)


def setup_hyper(setup_dto: HyperSetupDto, function_dto: HyperFunctionDto) -> int:
    setup_dto = setup_dto.copy()

    if setup_dto["model_run"] is None:
        setup_dto["model_run"] = dt.datetime.now().strftime("%Y%m%d%H%M")

    if setup_dto["limit_data_percentage"] is not None and (
        setup_dto["limit_data_percentage"] < 0.0
        or setup_dto["limit_data_percentage"] > 1.0
    ):
        raise ValueError(
            "Invalid limit data percentage value: "
            + f"{setup_dto['limit_data_percentage']}"
        )

    logger.info(f"Starting hyper opt for run {setup_dto['model_run']}...")
    logger.info(f"Using {setup_dto['processes']} processes.")
    logger.info(f"Using {setup_dto['n_optimization_trials']} optimization trials.")
    logger.info(f"Using {setup_dto['n_patience']} patience.")
    logger.info(
        f"Using {setup_dto['min_percentage_improvement'] * 100}"
        + "% min percentage improvement."
    )
    logger.info(
        f"Using {setup_dto['limit_data_percentage'] * 100 if setup_dto['limit_data_percentage'] is not None else 'all '}% data"
    )

    logger.info("Loading data...")
    data = setup_dto["data"]

    logger.info("Engineering features...")

    if setup_dto["limit_data_percentage"] is not None:
        all_data_size = len(data)
        limited_data_size = int(all_data_size * setup_dto["limit_data_percentage"])
        logger.info(f"Limiting data from {all_data_size} to {limited_data_size}")
        setup_dto["data"] = data.sample(
            n=limited_data_size, random_state=42, replace=False, axis=0
        )

    logger.info(f"Created {len(setup_dto["combinations"])} combinations.")
    logger.info("Checking for existing models...")

    all_omit_names = [
        res["name"]
        for res in load_hyper_opt_results(
            model_run=setup_dto["model_run"],
            output_dir_path=setup_dto["output_dir_path"],
            hyper_opt_prefix=setup_dto["hyper_opt_prefix"],
        )
    ]
    omit_names = list(
        set(all_omit_names) & set([model.name for model in setup_dto["combinations"]])
    )
    setup_dto["omit_names"] = omit_names
    logger.info(f"Omitting {len(omit_names)} combinations.")
    logger.info(
        f"Running {len(setup_dto["combinations"]) - len(omit_names)} combinations."
    )
    # --- NOTE ---
    # Metadata is a dictionary that can be used to store any additional information
    metadata = {
        "limit_data_percentage": setup_dto["limit_data_percentage"],
    }

    if setup_dto["metadata"] is not None:
        metadata.update(setup_dto["metadata"])

    logger.info("Starting parallel optimization...")
    _ = run_parallel_optimization(setup_dto=setup_dto, function_dto=function_dto)

    logger.info("Models has been pickled and saved to disk.")

    return len(setup_dto["combinations"]) - len(omit_names)
