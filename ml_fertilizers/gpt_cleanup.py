Okay i will give you main functions and i would like you to ask me for the subfunctions that you will see there.




def create_objective(data: pd.DataFrame, model_combination: HyperOptCombination):

    def objective(trial: optuna.Trial) -> OBJECTIVE_RETURN_TYPE:
        params = TrialParamWrapper().get_params(
            model_name=model_combination.name, trial=trial
        )
        model = clone(model_combination.model).set_params(**params)
        logger.info(
            f"Starting trial {trial.number} for model {model_combination.name} with params: {params}"
        )
        features = model_combination.feature_combination.features
        try:
            preprocessor = create_preprocessor()
            kfold = StratifiedKFold(
                n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE
            )
            oof_probas = pd.DataFrame(
                index=data.index, columns=lbe.transform(lbe.classes_)  # type: ignore
            )

            for fold, (train_index, val_index) in enumerate(
                kfold.split(data, data["Fertilizer Name"])
            ):
                logger.info(f"Fold {fold + 1}/{FOLDS} - {model_combination.name}")
                td = data.iloc[train_index]
                vd = data.iloc[val_index]

                X_train = preprocessor.fit_transform(
                    td.drop(columns=["Fertilizer Name"])
                )
                y_train = td["Fertilizer Name"]
                X_val = preprocessor.transform(vd.drop(columns=["Fertilizer Name"]))
                y_val = vd["Fertilizer Name"]

                model.fit(
                    X_train[features],
                    y_train,
                    eval_set=[(X_val[features], y_val)],
                    verbose=100,
                )

                oof_probas.iloc[val_index] = model.predict_proba(X_val[features])

            score = calc_mapk(y_true=data["Fertilizer Name"], y_probas=oof_probas, k=3)
            del oof_probas
            del preprocessor
            del td
            del vd
            del X_train
            del X_val
            del y_train
            del y_val
            del model
            return score
        except optuna.TrialPruned as e:
            logger.warning(f"Trial {trial.number} was pruned: {e}")
            raise e
        except Exception as e:
            logger.error(f"Error during trial {trial.number}: {e}")
            raise e

    return objective

xgb_model = XGBClassifierGPU(
    random_state=RANDOM_STATE,
    n_jobs=job_count,
    verbosity=1,
    objective="multi:softprob",
    eval_metric="mlogloss",
    enable_categorical=True,
    early_stopping_rounds=200,
)._set_gpu(use_gpu=gpu)



class XGBClassifierGPU(XGBClassifier):
    """SHEESH! This is a GPU compatible version of XGBClassifier. It uses the GPU for training and prediction if possible."""

    def __init__(self, **kwargs):

        kwargs.setdefault("device", "cuda")
        kwargs.setdefault("tree_method", "hist")
        kwargs.setdefault("predictor", "gpu_predictor")
        super().__init__(**kwargs)

    def _use_gpu(self):
        # Decide based on the actual XGBClassifier parameters
        return (
            getattr(self, "tree_method", None) == "hist"
            and getattr(self, "device", None) == "cuda"
        )

    def _set_gpu(self, use_gpu: bool) -> "XGBClassifierGPU":
        """Toggle GPU usage for XGBoost."""
        if use_gpu:
            logger.info("Using GPU for XGBoost.")
            self.set_params(
                tree_method="hist", device="cuda", predictor="gpu_predictor"
            )
        else:
            logger.warning("Using CPU for XGBoost. This may be slower than using GPU.")
            self.set_params(tree_method=None, device=None)

        return self

    def _to_gpu_array(self, X):

        if not self._use_gpu():
            arr = X
            if isinstance(X, pd.DataFrame) and any(
                [isinstance(col, pd.SparseDtype) for col in X.dtypes]
            ):
                arr = X.astype(pd.SparseDtype("float32", 0)).sparse.to_coo().tocsr()
            return arr
        # Pandas DataFrame -> cuDF if available, else CuPy, else NumPy

        elif isinstance(X, pd.DataFrame):
            if cudf is not None:
                return cudf.DataFrame.from_pandas(X)
            elif cp is not None:
                return cp.array(X.values)
            elif cuda is not None:
                # If using Numba's CUDA, convert to a CuPy array
                return cuda.to_device(X.values)
            else:
                return X.values
        elif isinstance(X, np.ndarray):
            if cp is not None:
                return cp.array(X)
            else:
                return X

        elif sp.issparse(X):
            X_dense = X.toarray()
            if cp is not None:
                return cp.array(X_dense)
            else:
                return X_dense
        elif cudf is not None and isinstance(X, cudf.DataFrame):
            return X
        elif cp is not None and isinstance(X, cp.ndarray):
            return X
        else:
            raise ValueError(f"Unsupported input type: {type(X)}")

    def _to_gpu_label(self, y):
        if not self._use_gpu():
            return y
        elif isinstance(y, pd.Series):
            if cudf is not None:
                return cudf.Series(y)
            elif cp is not None:
                return cp.array(y.values)
            else:
                return y.values
        elif isinstance(y, np.ndarray):
            if cp is not None:
                return cp.array(y)
            else:
                return y
        elif cudf is not None and isinstance(y, cudf.Series):
            return y
        elif cp is not None and isinstance(y, cp.ndarray):
            return y
        else:
            return y

    def fit(self, X, y, **kwargs):
        # X_fin = self._to_gpu_array(X)
        X_fin = X
        y_fin = y
        # y_fin = self._to_gpu_label(y)
        self_fitted = super().fit(X_fin, y_fin, **kwargs)
        del X_fin
        del y_fin
        return self_fitted

    def predict(self, X, **kwargs):
        # X_fin = self._to_gpu_array(X)
        X_fin = X
        y_pred = super().predict(X_fin, **kwargs)
        del X_fin
        return y_pred

    def predict_proba(self, X, **kwargs):
        # X_fin = self._to_gpu_array(X)
        X_fin = X
        return super().predict_proba(X_fin, **kwargs)


HyperOptCombination(
        name="XGB_custom",
        model=clone(xgb_model),
        feature_combination=FeatureCombination(
            features=[
                "Temparature",
                "Humidity",
                "Moisture",
                "Nitrogen",
                "Potassium",
                "Phosphorous",
                "NPK_Index",
                "pca0",
                "Soil",
                "Crop",
                "Env_Stress_Index",
                "Soil_Nutrient_Ratio",
            ]
        ),
    ),


@dataclass
class HyperOptCombination:
    name: str
    model: BaseEstimator
    feature_combination: FeatureCombination

@dataclass
class FeatureCombination:
    features: List[str]
    name: str = field(default="")

    def __post_init__(self):
        if "-" in self.name:
            raise ValueError("Name cannot contain '-'")


setup_dto = HyperSetupDto(
    n_optimization_trials=70,
    optimization_timeout=None,
    n_patience=30,
    min_percentage_improvement=0.005,
    model_run=model_run,
    limit_data_percentage=0.66,
    processes=processes,
    max_concurrent_jobs=None,
    output_dir_path=PathManager.output.value,
    hyper_opt_prefix=PrefixManager.hyper.value,
    study_prefix=PrefixManager.study.value,
    data=train,
    combinations=combinations,
    hyper_direction="maximize",
    metadata={},
    force_all_sequential=False,
    omit_names=None,
)


function_dto = HyperFunctionDto(
    create_objective_func=create_objective,
    evaluate_hyperopted_model_func=None,
)

n = setup_hyper(setup_dto=setup_dto, function_dto=function_dto)

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

    logger.info(f"Created {len(setup_dto.get('combinations'))} combinations.")
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
        f"Running {len(setup_dto.get('combinations')) - len(omit_names)} combinations."
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




















def engineer_simple(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy().set_index("id")
    X = X.rename(
        columns={
            "Soil Type": "Soil",
            "Crop Type": "Crop",
        }
    )
    # NOTE: ENGINERRING
    # fmt: off
    X["Env_Stress_Index"] = X["Temparature"] * 0.4 + X["Humidity"] * 0.3 + X["Moisture"] * 0.3
    X["NPK_Index"] = X["Nitrogen"] * 0.5 + X["Phosphorous"] * 0.3 + X["Potassium"] * 0.2
    X["Temp_bin"] = pd.cut(X["Temparature"], bins=[-float("inf"), 15, 30, 45, float("inf")], labels=["low", "medium", "high", "very_high"])
    X["Humidity_bin"] = pd.cut(X["Humidity"], bins=[-float("inf"), 30, 50, 70, float("inf")], labels=["low", "medium", "high", "very_high"])
    X["Moisture_bin"] = pd.cut(X["Moisture"], bins=[-float("inf"), 20, 40, 60, float("inf")], labels=["low", "medium", "high", "very_high"])
    X['Soil_Nutrients'] = X['Nitrogen'] + X['Phosphorous'] + X['Potassium']
    X["Soil_Nutrient_Ratio"] = X["Nitrogen"] / (X["Potassium"] + X["Phosphorous"] + 1)
    X["Temp_Humidity"] = X["Temparature"] * X["Humidity"]
    X["Temp_Moisture"] = X["Temparature"] * X["Moisture"]
    # fmt: on
    cat_features = ["Crop", "Soil", "Temp_bin", "Humidity_bin", "Moisture_bin"]
    for col in cat_features:
        if col in X.columns:
            X[col] = X[col].astype("category")

    X_cat = pd.get_dummies(X[cat_features], drop_first=False, sparse=True)

    X_final = pd.concat([X, X_cat], axis=1)

    return X_final


def create_preprocessor() -> Pipeline:

    pipeline_steps = [
        ("simple_engineering", FunctionTransformer(engineer_simple)),
        (
            "ct",
            ColumnTransformer(
                transformers=[
                    (
                        "temp_pca",
                        PCA(n_components=2),
                        ["Temparature", "Humidity", "Moisture"],
                    ),
                    (
                        "temp_stuff",
                        "passthrough",
                        ["Temparature", "Humidity", "Moisture"],
                    ),
                ],
                remainder="passthrough",
            ),
        ),
        (
            "remove_prefixes",
            FunctionTransformer(
                lambda df: df.rename(
                    columns=lambda x: x.split("__")[-1] if "__" in x else x
                )
            ),
        ),
    ]

    preprocessor = Pipeline(steps=pipeline_steps).set_output(transform="pandas")
    return preprocessor  # type: ignore



def calc_mapk(y_true: pd.Series, y_probas: pd.DataFrame, k: int = 3) -> float:
    """
    Calculate Mean Average Precision at k (MAP@k) for a list of true labels and predicted probabilities.

    Parameters:
    y_true : pd.Series
        True labels for the samples.
    y_probas : pd.DataFrame
        Predicted probabilities for each class, where each
        row corresponds to a sample and each column corresponds to a class.
    k : int, optional
        The number of top predictions to consider for calculating MAP@k (default is 3).
    Returns:
    float
        The Mean Average Precision at k score.
    """
    if y_probas.shape[0] != len(y_true):
        raise ValueError("y_probas must have the same number of rows as y_true")
    if k <= 0:
        raise ValueError("k must be a positive integer")

    # Get top-k predicted class indices for each row
    topk = np.argpartition(y_probas.values, -k, axis=1)[:, -k:]
    # Sort top-k indices by probability descending
    row_indices = np.arange(y_probas.shape[0])[:, None]
    topk_sorted = topk[
        row_indices,
        np.argsort(y_probas.values[row_indices, topk], axis=1)[:, ::-1],
    ]
    # Map column indices to class labels
    class_labels = np.array(y_probas.columns)
    topk_labels = class_labels[topk_sorted]

    # Broadcast y_true for comparison
    y_true_arr = np.array(y_true).reshape(-1, 1)
    hits = topk_labels == y_true_arr

    # Compute reciprocal rank for each hit
    reciprocal_ranks = hits / (np.arange(1, k + 1))
    # Sum over k for each row (since only one hit per row is possible)
    scores = reciprocal_ranks.sum(axis=1)
    return scores.mean()


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



PLEASE, go thru all of that and if you will need more time to think just tell me to ask you again to continue. this should be majoarity of my entire code. let me know what steps could be improved