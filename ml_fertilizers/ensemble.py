from copy import deepcopy
import json
import multiprocessing as mp
import os
import gc
from pathlib import Path
import pickle as pkl
from typing import List, Tuple, cast
import numpy as np
from collections import defaultdict

import optuna
import pandas as pd
from sklearn import clone
from sklearn.calibration import CalibratedClassifierCV, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from ml_fertilizers.lib.logger import setup_logger
from ml_fertilizers.lib.models.EnsembleModel2 import EnsembleModel2
from ml_fertilizers.lib.models.GpuModels import XGBClassifierGPU
from ml_fertilizers.lib.models.HyperOptResultDict import HyperOptResultDict
from ml_fertilizers.lib.pipelines.ProcessingPipelineWrapper import create_pipeline
from ml_fertilizers.lib.utils.results import load_hyper_opt_results
from ml_fertilizers.utils import (
    PathManager,
    PrefixManager,
    calc_mapk,
    engineer_features,
    evaluate,
    load_data,
    mapk_scorer,
)

model_run = "intial_run"
processes = 30
gpu = True

logger = setup_logger(__name__)


def setup() -> Tuple[
    List[HyperOptResultDict],
    pd.DataFrame,
    pd.Series,
    int,
    pd.DataFrame,
]:
    results = load_hyper_opt_results(
        model_run=model_run,
        output_dir_path=PathManager.output.value,
        hyper_opt_prefix=PrefixManager.hyper.value,
    )
    logger.info(f"Loaded hyperparameter optimization results for {model_run}.")
    train, test = load_data()
    eng_train, _, _ = engineer_features(train)
    eng_test, _, _ = engineer_features(test)
    X = pd.get_dummies(eng_train, drop_first=False, sparse=True)
    y = eng_train.loc[X.index, "Fertilizer Name"]
    X_test = pd.get_dummies(eng_test, drop_first=False, sparse=True)
    job_count = mp.cpu_count() if processes is None else processes
    os.environ["OMP_NUM_THREADS"] = str(job_count)
    os.environ["MKL_NUM_THREADS"] = str(job_count)

    return (results, X, y, job_count, X_test)


def full_evaluate_results(
    results: List[HyperOptResultDict],
    X: pd.DataFrame,
    y: pd.Series,
    job_count: int,
    gpu: bool,
) -> None:
    score_dict = dict()
    for result in results:
        res_features = result["features"]
        res_model = result["model"]
        res_params = result["params"]
        res_name = result["name"]

        if res_model is None or res_features is None or res_params is None:
            logger.warning(f"Skipping incomplete result: {result}")
            continue

        logger.info(f"Evaluating model: {res_name} with features: {res_features}")

        res_model = clone(res_model)

        if hasattr(res_model, "set_gpu"):
            res_model = res_model.set_gpu(gpu)

        if hasattr(res_model, "n_jobs"):
            res_model.set_params(n_jobs=job_count)
        elif hasattr(res_model, "thread_count"):
            res_model.set_params(thread_count=job_count)

        score = evaluate(
            estimator=res_model.set_params(**res_params),
            X=X[res_features],
            y=y,
        )
        logger.info(f"Score for {res_name}: {score}")
        score_dict[res_name] = score

        gc.collect()

    logger.info(f"Evaluation scores: {score_dict}")
    json.dump(
        score_dict,
        open(
            PathManager.output.value / f"{PrefixManager.ensemble.value}scores.json", "w"
        ),
        indent=4,
    )


def create_stacking(
    results: List[HyperOptResultDict],
    X: pd.DataFrame,
    y: pd.Series,
    job_count: int,
    gpu: bool,
) -> None:
    ensemble_scores_path = (
        PathManager.output.value / f"{PrefixManager.ensemble.value}scores.json"
    )
    if not ensemble_scores_path.exists():
        ensemble_scores = dict()
    else:
        ensemble_scores = json.load(open(ensemble_scores_path, "r"))

    stacking_estimators = []
    stacking_features = []
    stacking_names = []

    for result in results:
        res_features = result["features"]
        res_model = result["model"]
        res_params = result["params"]
        res_name = result["name"]
        res_score = result["score"]

        if (
            res_model is None
            or res_features is None
            or res_params is None
            or res_name is None
            or res_score is None
        ):
            logger.warning(f"Skipping incomplete result: {res_name}")
            continue

        if res_name in ensemble_scores and ensemble_scores[res_name] < 0.32:
            logger.info(
                f"Skipping {res_name} with score {ensemble_scores[res_name]} below threshold."
            )
            continue

        res_model = clone(res_model)

        if hasattr(res_model, "_set_gpu"):
            res_model = res_model._set_gpu(gpu)

        if hasattr(res_model, "n_jobs"):
            res_model.set_params(n_jobs=job_count)
        elif hasattr(res_model, "thread_count"):
            res_model.set_params(thread_count=job_count)

        stacking_estimators.append(res_model.set_params(**res_params))
        stacking_features.append(res_features)
        stacking_names.append(res_name)

    def objective(trial):
        """
        Objective function for hyperparameter optimization.
        """
        alpha = trial.suggest_float("alpha", 1e-5, 100.0, log=True)
        tol = trial.suggest_float("tol", 1e-5, 1e-2, log=True)

        metamodel = CalibratedClassifierCV(
            estimator=RidgeClassifier(alpha=alpha, tol=tol, random_state=42),
            method="sigmoid",
            cv=3,
            n_jobs=job_count,
        )

        stack_model = EnsembleModel2(
            models=stacking_estimators.copy(),
            combination_features=stacking_features,
            combination_names=stacking_names,
            just_filtering=True,
            prediction_method="predict_proba",
            metamodel=metamodel,
            metamodel_kfold=3,
        )

        score = evaluate(estimator=stack_model, X=X, y=y, cv=3)
        return score

    optuna_study_name = f"stacking_calibrated_ridge_limited_models_{model_run}"
    sql_path = Path(
        f"{PathManager.output.value}/{PrefixManager.study.value}{model_run}/{optuna_study_name}.db"
    )

    study = optuna.create_study(
        direction="maximize",
        study_name=optuna_study_name,
        load_if_exists=True,
        storage=f"sqlite:///{sql_path}",
    )

    study.optimize(
        objective,
        n_trials=40,
        show_progress_bar=True,
    )


def analyse_oof():

    oof_pred = cast(
        pd.DataFrame,
        pkl.load(
            open(PathManager.output.value / "all_predictions_out_of_fold.pkl", "rb"),
        ),
    )

    # NOTE: GROUP_STATS
    group_stats = dict()
    groups = list(set([col.split("__")[0] for col in oof_pred.columns.tolist()]))
    k = 3
    for g in groups:
        gdf = oof_pred.filter(like=g)
        group_stats[g] = defaultdict(dict)
        # Get top-k indices for each row
        topk = np.argsort(gdf.values, axis=1)[:, -k:][:, ::-1]
        col_labels = np.array(gdf.columns)
        # For each row, get the actual label names (not indices)
        for row_indices in topk:
            row_labels = col_labels[row_indices]
            for i, label in enumerate(row_labels):
                label = label.split("__")[-1]  # Get the actual label name
                other_labels = [l for j, l in enumerate(row_labels) if j != i]
                if len(other_labels) == 0:
                    continue
                if label not in group_stats[g]:
                    group_stats[g][label] = dict()
                for other_label in other_labels:
                    other_label = other_label.split("__")[-1]
                    group_stats[g][label][other_label] = (
                        group_stats[g][label].get(other_label, 0) + 1
                    )
                group_stats[g][label] = group_stats[g][label].get(label, 0) + 1

    json.dump(
        group_stats,
        open(
            PathManager.output.value
            / f"{PrefixManager.ensemble.value}group_stats.json",
            "w",
        ),
        indent=4,
    )

    # NOTE: MATRIX
    group_stats = json.load(
        open(
            PathManager.output.value
            / f"{PrefixManager.ensemble.value}group_stats.json",
            "r",
        )
    )
    group_stats_percent = deepcopy(group_stats)
    for g in group_stats:
        for label in group_stats[g]:
            total_count = sum(group_stats[g][label].values())
            if total_count > 0:
                for other_label in group_stats[g][label]:
                    if other_label == label:
                        continue
                    group_stats_percent[g][label][other_label] /= total_count
                    group_stats_percent[g][label][other_label] = round(
                        group_stats_percent[g][label][other_label] * 100, 2
                    )
    json.dump(
        group_stats_percent,
        open(
            PathManager.output.value
            / f"{PrefixManager.ensemble.value}group_stats_percent.json",
            "w",
        ),
        indent=4,
    )
    logger.info("Here")
    import math
    import matplotlib.pyplot as plt

    ncols = 3
    nrows = math.ceil(len(groups) / ncols)
    fig, axes = plt.subplots(
        figsize=(20, 5 * nrows), nrows=nrows, ncols=ncols, sharex=True, sharey=True
    )
    axes = axes.flatten() if len(groups) > 1 else [axes]
    for idx, g in enumerate(groups):
        stats = group_stats_percent[g]
        labels = sorted(stats.keys())
        matrix = np.zeros((len(labels), len(labels)))
        for i, label in enumerate(labels):
            for j, other_label in enumerate(labels):
                if label == other_label:
                    matrix[i, j] = 0
                else:
                    matrix[i, j] = stats[label].get(other_label, 0)
        ax = axes[idx]
        im = ax.imshow(matrix, cmap="viridis")
        ax.set_title(f"Top-k Label Co-occurrence Counts for Group: {g}")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Sibling Label")
        ax.set_ylabel("Reference Label")
        plt.colorbar(im, ax=ax)
        # Annotate each cell with the count
        for i in range(len(labels)):
            for j in range(len(labels)):
                if matrix[i, j] > 0:
                    ax.text(
                        j,
                        i,
                        matrix[i, j],
                        ha="center",
                        va="center",
                        color="w",
                        fontsize=8,
                    )

    # Hide unused axes if any
    for idx in range(len(groups), len(axes)):
        axes[idx].axis("off")

    # plt.tight_layout()
    plt.show()

    fig.savefig(
        PathManager.output.value
        / f"{PrefixManager.ensemble.value}group_stats_visualization.png",
        bbox_inches="tight",
    )


def predict(estimator, name) -> pd.Series:
    """
    Predicts using the provided estimator on the test set.
    """
    train, test = load_data()
    eng_train, _, _ = engineer_features(train)
    eng_test, _, _ = engineer_features(test)

    X_train = pd.get_dummies(eng_train, drop_first=False, sparse=True)
    X_test = pd.get_dummies(eng_test, drop_first=False, sparse=True)
    lbe = LabelEncoder()
    y_train = lbe.fit_transform(eng_train["Fertilizer Name"])

    estimator = clone(estimator)
    estimator.fit(X_train, y_train)

    y_pred_raw = estimator.predict_proba(X_test)
    y_topk = np.argsort(y_pred_raw, axis=1)[:, -3:][:, ::-1]
    y_pred = np.array([" ".join(lbe.inverse_transform(row)) for row in y_topk])
    y_sr = pd.Series(
        y_pred.tolist(),
        index=X_test.index,
        name="Fertilizer Name",
    )

    y_sr.to_csv(
        PathManager.output.value
        / f"{PrefixManager.ensemble.value}{name}_predictions.csv",
        index_label="id",
    )

    return y_sr


def predict_singular():
    sname = "XGBClassifier_20"
    sprefix = "depth10"
    sres = next((r for r in results if r["name"] == sname), None)
    spipeline = create_pipeline(
        features_in=sres["features"],  # type: ignore
        model=sres["model"].set_params(**sres["params"]),  # type: ignore
        just_filtering=True,
    )

    predict(
        estimator=spipeline,
        name=sname + sprefix,
    )


def create_ensemble():
    results, X, y, job_count, X_test = setup()

    ensemble_scores_path = (
        PathManager.output.value / f"{PrefixManager.ensemble.value}scores.json"
    )
    if not ensemble_scores_path.exists():
        ensemble_scores = dict()
    else:
        ensemble_scores = json.load(open(ensemble_scores_path, "r"))

    stacking_estimators = []
    stacking_features = []
    stacking_names = []
    stacking_scores = []
    for result in results:
        res_features = result["features"]
        res_model = result["model"]
        res_params = result["params"]
        res_name = result["name"]
        res_score = result["score"]

        if (
            res_model is None
            or res_features is None
            or res_params is None
            or res_name is None
            or res_score is None
        ):
            logger.warning(f"Skipping incomplete result: {res_name}")
            continue

        if res_name in ensemble_scores and ensemble_scores[res_name] < 0.32:
            logger.info(
                f"Skipping {res_name} with score {ensemble_scores[res_name]} below threshold."
            )
            continue

        res_model = clone(res_model)

        if hasattr(res_model, "_set_gpu"):
            res_model = res_model._set_gpu(gpu)

        if hasattr(res_model, "n_jobs"):
            res_model.set_params(n_jobs=job_count)
        elif hasattr(res_model, "thread_count"):
            res_model.set_params(thread_count=job_count)

        stacking_estimators.append(res_model.set_params(**res_params))
        stacking_features.append(res_features)
        stacking_names.append(res_name)
        stacking_scores.append(res_score**2)

    ens = EnsembleModel2(
        models=stacking_estimators.copy(),
        combination_features=stacking_features,
        combination_names=stacking_names,
        just_filtering=True,
        prediction_method="predict",
        scoring_direction="maximize",
        combination_scoring=stacking_scores,
    )

    score = evaluate(
        estimator=ens,
        X=X,
        y=y,
        cv=3,
    )

    logger.info(f"Ensemble score: {score}")
    ensemble_scores["ensemble_weighted_with_scores&2_limited_models"] = score

    json.dump(
        ensemble_scores,
        open(
            PathManager.output.value / f"{PrefixManager.ensemble.value}scores.json",
            "w",
        ),
        indent=4,
    )
    # predict(ens, "ensemble_weighted_with_scores^2_limited_models")


def test_xgb_kaggle():
    train, test = load_data()
    train = train.set_index("id")
    test = test.set_index("id")

    train["Temp_Humidity"] = train["Temparature"] * train["Humidity"]
    train["Temp_Moisture"] = train["Temparature"] * train["Moisture"]
    train["Soil_Nutrients"] = (
        train["Nitrogen"] + train["Potassium"] + train["Phosphorous"]
    )
    train["Soil_Nutrient_Ratio"] = train["Nitrogen"] / (
        train["Potassium"] + train["Phosphorous"] + 1
    )
    bins = [0, 15, 30, 45, 60]
    labels = ["Low", "Medium", "High", "Very High"]
    train["Temperature_Binned"] = pd.cut(train["Temparature"], bins=bins, labels=labels)

    test["Temp_Humidity"] = test["Temparature"] * test["Humidity"]
    test["Temp_Moisture"] = test["Temparature"] * test["Moisture"]
    test["Soil_Nutrients"] = test["Nitrogen"] + test["Potassium"] + test["Phosphorous"]
    test["Soil_Nutrient_Ratio"] = test["Nitrogen"] / (
        test["Potassium"] + test["Phosphorous"] + 1
    )
    test["Temperature_Binned"] = pd.cut(
        test["Temparature"],
        bins=bins,
        labels=labels,
    )

    le = LabelEncoder()
    train["Fertilizer Name"] = le.fit_transform(train["Fertilizer Name"])
    num_cols = [
        "Temparature",
        "Humidity",
        "Moisture",
        "Nitrogen",
        "Potassium",
        "Phosphorous",
        "Temp_Humidity",
        "Temp_Moisture",
        "Soil_Nutrients",
        "Soil_Nutrient_Ratio",
    ]
    cat_cols = ["Soil Type", "Crop Type", "Temperature_Binned"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OrdinalEncoder(), cat_cols),
        ]
    )

    # oof_probas = pd.DataFrame(
    #     index=train.index,
    #     columns=le.classes_,
    #     dtype=np.float32,
    # )
    # folds = 5
    # kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    # for fold, (train_idx, val_idx) in enumerate(
    #     kfold.split(train[num_cols + cat_cols], train["Fertilizer Name"])
    # ):
    #     logger.info(f"Processing fold {fold + 1}/{folds}")
    #     X_train, X_val = (
    #         train[num_cols + cat_cols].iloc[train_idx],
    #         train[num_cols + cat_cols].iloc[val_idx],
    #     )
    #     y_train, y_val = (
    #         train["Fertilizer Name"].iloc[train_idx],
    #         train["Fertilizer Name"].iloc[val_idx],
    #     )
    #     X_train_scaled = preprocessor.fit_transform(X_train)
    #     X_val_scaled = preprocessor.transform(X_val)
    #     model = XGBClassifierGPU(
    #         max_depth=18,
    #         colsample_bytree=0.2587327850345624,
    #         subsample=0.8276149323901826,
    #         n_estimators=4000,
    #         learning_rate=0.01,
    #         gamma=0.26,
    #         max_delta_step=6,
    #         reg_alpha=5.620898657099113,
    #         reg_lambda=0.05656209749983576,
    #         early_stopping_rounds=200,
    #         objective="multi:softprob",
    #         random_state=13,
    #         enable_categorical=True,
    #     )._set_gpu(True)
    #     model.fit(
    #         X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=100
    #     )
    #     y_val_proba = model.predict_proba(X_val_scaled)
    #     oof_probas.iloc[val_idx] = y_val_proba
    # oof_probas.to_csv(
    #     PathManager.output.value / "xgb_kaggle_oof_probas.csv",
    #     index_label="id",
    # )
    oof_probas = pd.read_csv(
        PathManager.output.value / "xgb_kaggle_oof_probas.csv",
        index_col="id",
    )

    score = calc_mapk(
        y_true=train["Fertilizer Name"].values,
        y_probas=oof_probas.values,
        k=3,
    )

    logger.info(f"OOF MAP@3 score: {score}")

    model = XGBClassifierGPU(
        max_depth=21,
        colsample_bytree=0.2587327850345624,
        subsample=0.8276149323901826,
        n_estimators=4000,
        learning_rate=0.01,
        gamma=0.26,
        max_delta_step=6,
        reg_alpha=5.620898657099113,
        reg_lambda=0.05656209749983576,
        # early_stopping_rounds=200,
        objective="multi:softprob",
        random_state=13,
        enable_categorical=True,
    )._set_gpu(True)

    model.fit(
        preprocessor.fit_transform(train[num_cols + cat_cols]),
        train["Fertilizer Name"],
    )

    y_pred_raw = model.predict_proba(preprocessor.transform(test[num_cols + cat_cols]))
    y_topk = np.argsort(y_pred_raw, axis=1)[:, -3:][:, ::-1]
    y_pred = np.array([" ".join(le.inverse_transform(row)) for row in y_topk])
    y_sr = pd.Series(
        y_pred.tolist(),
        index=test.index,
        name="Fertilizer Name",
    )

    y_sr.to_csv(
        PathManager.output.value / "xgb_kaggle_predictions.csv",
        index_label="id",
    )


if __name__ == "__main__":
    # results, X, y, job_count, X_test = setup()
    # full_evaluate_results(results, X, y, job_count, gpu)
    # create_stacking(results, X, y, job_count, gpu)
    # create_ensemble()
    # analyse_oof()
    # predict_singular()
    test_xgb_kaggle()

    logger.info("Completed")
