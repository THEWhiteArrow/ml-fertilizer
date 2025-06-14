import gc
import json
import datetime as dt
from typing import List, Literal, Optional, Tuple
from lightgbm import LGBMClassifier
import numpy as np
import optuna
import pandas as pd
from sklearn import clone
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV, LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from ml_fertilizers.lib.features.FeatureCombination import FeatureCombination
from ml_fertilizers.lib.logger import setup_logger
from ml_fertilizers.lib.models.GpuModels import (
    CatBoostClassifierGPU,
    LGBMClassifierGPU,
    XGBClassifierGPU,
    XGBClassifierLeanGPU,
)
from ml_fertilizers.lib.models.HyperOptCombination import HyperOptCombination
from ml_fertilizers.lib.optymization.TrialParamWrapper import TrialParamWrapper
from ml_fertilizers.lib.optymization.analysis_setup import setup_analysis
from ml_fertilizers.lib.optymization.optimization_study import (
    OBJECTIVE_RETURN_TYPE,
    aggregate_studies,
)
from ml_fertilizers.utils import (
    PathManager,
    calc_mapk,
    create_preprocessor2,
    evaluate,
    load_data,
    mapk_scorer,
)
from ml_fertilizers.lib.optymization.hyper_setup import setup_hyper

from ml_fertilizers.lib.optymization.parrarel_optimization import (
    HyperFunctionDto,
    HyperSetupDto,
)
from ml_fertilizers.utils import PrefixManager, PathManager

logger = setup_logger(__name__)

le = LabelEncoder()

train, test = load_data()


class CFG:
    random_state = 69
    n_jobs = -1
    cv = 5
    gpu = True
    sample_frac = 0.7


class LGBMClassifierCategoricalGPU(LGBMClassifierGPU):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X, y, **kwargs):  # type: ignore
        if isinstance(X, pd.DataFrame):
            cat_cols = X.select_dtypes(include=["category"]).columns.tolist()
            return super().fit(X, y, categorical_feature=cat_cols, **kwargs)
        else:
            return super().fit(X, y, **kwargs)


class CatBoostClassifierCategoricalGPU(BaseEstimator, ClassifierMixin):
    def __init__(self, gpu=True, verbose=0, **kwargs):
        self.base_clf_kwargs = kwargs
        for key, value in kwargs.items():
            self.__setattr__(key, value)

        self.base_clf_ = None
        self.gpu = gpu
        self.verbose = verbose

    def fit(self, X, y, **kwargs):
        self.base_clf_ = CatBoostClassifierGPU(
            **self.base_clf_kwargs,
            verbose=self.verbose,
            cat_features=X.select_dtypes(include=["category"]).columns.tolist(),
        )._set_gpu(self.gpu)
        self.base_clf_.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        if self.base_clf_ is None:
            raise ValueError("Model has not been fitted yet.")
        return self.base_clf_.predict(X)

    def predict_proba(self, X):
        if self.base_clf_ is None:
            raise ValueError("Model has not been fitted yet.")
        return self.base_clf_.predict_proba(X)

    def _set_gpu(self, gpu: bool):
        self.gpu = gpu
        return self

    def get_params(self, deep: bool = True) -> dict:
        return {
            "gpu": self.gpu,
            "verbose": self.verbose,
            **self.base_clf_kwargs,
        }

    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                self.__setattr__(key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        return self


def fertilize(
    estimator,
    X,
    y,
    cv,
    random_state,
    preprocessor=None,
    cv_type: Literal["oof", "averaged"] = "oof",
) -> float:
    kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    oof_proba = pd.DataFrame(index=X.index, columns=y.unique())
    scores = []
    for fold, (train_index, val_index) in enumerate(kfold.split(X, y)):
        curr_estimator = clone(estimator)
        # logger.info(f"Fold {fold + 1}/{cv}")
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        if preprocessor is not None:
            logger.warning(f"Fitting preprocessor for fold {fold + 1}/{cv}")
            X_train = preprocessor.fit_transform(X_train)
            X_val = preprocessor.transform(X_val)

        curr_estimator.fit(X_train, y_train)
        y_proba_raw = curr_estimator.predict_proba(X_val)
        oof_proba.iloc[val_index] = y_proba_raw

        y_proba = pd.DataFrame(
            y_proba_raw, index=X_val.index, columns=le.transform(le.classes_), dtype="float16"  # type: ignore,
        )
        score = calc_mapk(y_true=y_val, y_probas=y_proba, k=3)
        scores.append(score)

        curr_estimator = y_proba_raw = None
        del curr_estimator
        del y_proba_raw
        gc.collect()

    oof_score = calc_mapk(y_true=y, y_probas=oof_proba, k=3)
    averaged_score = float(np.mean(scores))
    kfold = oof_proba = X_train = X_val = y_train = y_val = estimator = y_proba = None
    del kfold, oof_proba, X_train, X_val, y_train, y_val, estimator, y_proba
    gc.collect()

    final_score = None
    if cv_type == "oof":
        final_score = oof_score
    elif cv_type == "averaged":
        final_score = averaged_score
    else:
        raise ValueError(f"Unknown cv_type: {cv_type}")

    return final_score


def dataing(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    data = data.set_index("id")

    # for col in ["Nitrogen", "Potassium", "Phosphorous"]:
    #     data[col] = data[col].clip(lower=1)

    data = data.rename(
        columns={
            "Soil Type": "Soil",
            "Crop Type": "Crop",
            "Potassium": "K",
            "Phosphorous": "P",
            "Nitrogen": "N",
            "Temparature": "T",
            "Humidity": "H",
            "Moisture": "M",
        }
    )
    cat_cols = ["Soil", "Crop", "Fertilizer Name"]
    for col in cat_cols:
        data[col] = data[col].astype("category")

    int_cols = ["N", "P", "K", "T", "H", "M"]
    for col in int_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce").astype("uint16")

    data["Soil_x_Crop"] = data["Soil"].astype(str) + "_" + data["Crop"].astype(str)
    data["Soil_x_Crop"] = data["Soil_x_Crop"].astype("category")
    data["Soil_x_Crop_limited"] = data["Soil_x_Crop"].replace(
        data["Soil_x_Crop"]
        .value_counts()[data["Soil_x_Crop"].value_counts() < 13000]
        .index.tolist(),
        "limited",
    )

    data["Fertilizer Name"] = le.fit_transform(data["Fertilizer Name"])
    data["N/K"] = data["N"] / (data["K"] + 1e-6)
    data["N/P"] = data["N"] / (data["P"] + 1e-6)
    data["K/P"] = data["P"] / (data["P"] + 1e-6)
    data["N+K+P"] = data["N"] + data["K"] + data["P"]
    data["N/(K+P)"] = data["N"] / (data["K"] + data["P"] + 1)
    data["Temp-Humidity"] = data["T"] * data["H"]
    data["Temp-Moisture"] = data["T"] * data["M"]

    data["1/T"] = 100.0 / (data["T"].clip(lower=1))

    data["T2"] = data["T"] ** 2
    data["H2"] = data["H"] ** 2
    data["M2"] = data["M"] ** 2
    data["N2"] = data["N"] ** 2
    data["P2"] = data["P"] ** 2
    data["K2"] = data["K"] ** 2
    data["1/D1"] = 100.0 / (data["P"].clip(lower=1) * data["M"].clip(lower=1))
    data["Temp_bin"] = pd.Categorical.from_codes(
        np.digitize(data["T"].values, bins=[-np.inf, 15, 30, 45, np.inf]) - 1,  # type: ignore
        categories=["low", "medium", "high", "very_high"],  # type: ignore
    )

    feat_data = data.drop(columns=["Fertilizer Name"])
    target_data = data["Fertilizer Name"]

    cat_features = ["Soil", "Crop", "Soil_x_Crop", "Temp_bin"]
    feat_cat_data = pd.get_dummies(feat_data[cat_features], drop_first=True)

    final_data = pd.concat([feat_data, feat_cat_data], axis=1)

    org_feat_list = feat_data.columns.tolist()
    ohe_feat_list = list(
        set(
            list(set(org_feat_list) - set(cat_features))
            + feat_cat_data.columns.tolist()
        )
    )
    return final_data, target_data, org_feat_list, ohe_feat_list


X_org, y_org, raw_feat, ohe_feat = dataing(train)

pre = create_preprocessor2()
X_org = X_org.sample(frac=CFG.sample_frac, random_state=CFG.random_state)
y_org = y_org[X_org.index]

# fmt: off
xgb_model = XGBClassifierGPU(enable_categorical=True, n_jobs=CFG.n_jobs, objective="multi:softprob", eval_metric="mlogloss", max_depth=10, n_estimators=500, allow_categorical_as_ordinal=False, verbosity=0)._set_gpu(CFG.gpu)
cat_model = CatBoostClassifierCategoricalGPU(gpu=CFG.gpu, thread_count=CFG.n_jobs, loss_function="MultiClass", eval_metric="MultiClass", verbose=0, max_depth=10)
lgbm_model = LGBMClassifierCategoricalGPU(n_jobs=CFG.n_jobs, verbosity=-1, objective="multiclass", eval_metric="multiclass_logloss", max_depth=10)._set_gpu(CFG.gpu)
combinations = [
    # ("cat_raw", cat_model, raw_feat),
    ("xgb_raw", xgb_model, raw_feat),
    ("lgbm_raw", lgbm_model, raw_feat),
    # ("log_ohe", LogisticRegression(), ohe_feat),
    # ("cal_ohe", CalibratedClassifierCV(method="sigmoid", cv=CFG.cv, n_jobs=CFG.n_jobs), ohe_feat),
    # ("xgb_ohe", XGBClassifierLeanGPU(enable_categorical=True, n_jobs=CFG.n_jobs, objective="multi:softprob", eval_metric="mlogloss")._set_gpu(CFG.gpu), ohe_feat),
    # ("cat_ohe", CatBoostClassifierCategoricalGPU(gpu=CFG.gpu, thread_count=CFG.n_jobs, loss_function="MultiClass", eval_metric="MultiClass"), ohe_feat),
    # ("lgbm_ohe", LGBMClassifierCategoricalGPU(n_jobs=CFG.n_jobs, verbosity=-1, objective="multiclass", eval_metric="multiclass_logloss")._set_gpu(CFG.gpu), ohe_feat),
]
# fmt: on


def fbfs(
    name: str,
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int,
    random_state: int,
    k: int,
    min_k: int = 5,
    min_k_to_remove: int = 3,
    starting_features: Optional[List[str]] = None,
):

    res_path = PathManager.output.value / f"fbfs_{name}_results.json"

    if res_path.exists():
        logger.info(f"Results already exist for {name}. Loading from {res_path}")
        res_progress = json.loads(res_path.read_text())
    else:
        logger.info(f"Results do not exist for {name}. Starting new run.")
        if starting_features is None:
            res_progress = [
                {
                    "selected_features": [],
                    "score": 0,
                    "added_feature": None,
                    "removed_feature": None,
                    "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            ]
        else:
            logger.warning(
                f"Starting features provided: {starting_features}. Will use them to start the feature selection."
            )
            starting_score = evaluate(
                estimator=clone(estimator),
                X=X[starting_features],
                y=y,
                cv=cv,
            )
            res_progress = [
                {
                    "selected_features": starting_features.copy(),
                    "score": starting_score,
                    "added_feature": None,
                    "removed_feature": None,
                    "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            ]

    logger.info(
        f"Starting Forward Backward Feature Selection (k={k}) with {estimator.__class__.__name__}"
    )
    selected_features = res_progress[-1]["selected_features"].copy()
    while len(selected_features) < k:
        was_added = False
        was_removed = False

        if len(selected_features) >= min_k_to_remove:
            logger.info(
                f"Current selected features: {selected_features}, total: {len(selected_features)}, score: {res_progress[-1]['score']}"
            )
            curr_score = res_progress[-1]["score"]

            for feature in selected_features:
                logger.info(f"Evaluating feature for removal: {feature}")
                if feature == res_progress[-1]["added_feature"]:
                    continue
                temp_features = selected_features.copy()
                temp_features.remove(feature)
                # score = fertilize(
                #     estimator=estimator,
                #     X=X[temp_features],
                #     y=y,
                #     cv=cv,
                #     random_state=random_state,
                # )
                score = evaluate(
                    estimator=clone(estimator),
                    X=X[temp_features],
                    y=y,
                    cv=cv,
                )
                if score > curr_score:
                    logger.info(
                        f"Removed feature: {feature} with score: {score} | old score: {curr_score}"
                    )
                    curr_score = score
                    selected_features.remove(feature)
                    res_progress.append(
                        {
                            "selected_features": selected_features.copy(),
                            "score": score,
                            "added_feature": None,
                            "removed_feature": feature,
                            "timestamp": dt.datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                        }
                    )
                    res_path.write_text(json.dumps(res_progress, indent=4))
                    was_removed = True
                    break
                # else:
                #     logger.info(
                #         f"Not removed feature: {feature} | new score {score} | old score {curr_score}"
                #     )

        if len(selected_features) < min_k:
            best_new_score = -1
        else:
            best_new_score = res_progress[-1]["score"]
        best_new_feature = None
        features_to_consider = [
            f
            for f in X.columns
            if f not in selected_features and f != res_progress[-1]["removed_feature"]
        ]
        for i, feature in enumerate(features_to_consider):
            logger.info(
                f"Evaluating feature {i + 1}/{len(features_to_consider)}: {feature}"
            )
            current_features = selected_features + [feature]
            # score = fertilize(
            #     estimator=estimator,
            #     X=X[current_features],
            #     y=y,
            #     cv=cv,
            #     random_state=random_state,
            # )
            score = evaluate(
                estimator=clone(estimator),
                X=X[current_features],
                y=y,
                cv=cv,
            )

            if score > best_new_score:
                logger.info(
                    f"Found better feature: {feature} with score: {score} | previous best: {best_new_score} | overall best: {max(p['score'] for p in res_progress)}"
                )
                best_new_score = score
                best_new_feature = feature

        if best_new_feature is not None:
            selected_features.append(best_new_feature)
            logger.info(
                f"Added feature: {best_new_feature} with score: {best_new_score}, total features: {len(selected_features)}"
            )
            res_progress.append(
                {
                    "selected_features": selected_features.copy(),
                    "score": best_new_score,
                    "added_feature": best_new_feature,
                    "removed_feature": None,
                    "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            res_path.write_text(json.dumps(res_progress, indent=4))
            was_added = True
        # else:
        #     logger.info(
        #         f"No more features to add. Current selected features: {selected_features}"
        #     )
        #     break

        if not was_added and not was_removed:
            logger.info(
                f"No more features to add or remove. Current selected features: {selected_features}"
            )
            break

    logger.info(f"Final selected features: {selected_features}")

    estimator = None
    del estimator
    gc.collect()

    return res_progress


def execute_fbfs(combinations: List[Tuple[str, BaseEstimator, List[str]]]):
    for name, clf, feat in combinations:
        logger.info(f"Running for {name} with {len(feat)} features")
        logger.info(
            f"Estimator: {clf.__class__.__name__} with {clf.get_params()} parameters"
        )
        ffs_results = fbfs(
            name=name,
            estimator=clf,
            X=X_org[[f for f in feat if f not in ["id", "Fertilizer Name"]]],
            y=y_org,
            cv=CFG.cv,
            random_state=CFG.random_state,
            k=int(len(feat) * 0.8),
            min_k=8,
            min_k_to_remove=5,
        )
        logger.info(f"FFS results for {name}:\n{json.dumps(ffs_results, indent=4)}")


def test_model():
    # fmt: off
    xgb = XGBClassifierGPU(enable_categorical=True, n_jobs=CFG.n_jobs, objective="multi:softprob", eval_metric="mlogloss", max_depth=10, n_estimators=500, allow_categorical_as_ordinal=False, verbosity=0)._set_gpu(CFG.gpu)
    # cat = CatBoostClassifierCategoricalGPU(gpu=CFG.gpu, thread_count=CFG.n_jobs, loss_function="MultiClass", eval_metric="MultiClass", verbose=100)
    # fmt: on
    feats = [
        "Crop",
        "Soil",
        # "Soil_x_Crop_limited",
        "M",
        "P",
        "N",
        "K",
        "H",
        "T",
        # "1/T",
        # "N+K+P",
        # "1/D1",
        # "Temp_bin",
    ]
    model = xgb
    f_score = fertilize(
        estimator=clone(model),
        X=X_org[feats],
        y=y_org,
        cv=CFG.cv,
        cv_type="averaged",
        random_state=CFG.random_state,
        preprocessor=pre,
    )
    logger.info(
        f"F_score for {model.__class__.__name__} with features {feats}: {f_score}"
    )
    # e_score = evaluate(
    #     estimator=clone(model),
    #     X=X_org[feats],
    #     y=y_org,
    #     cv=CFG.cv,
    # )
    # logger.info(
    #     f"Evaluation score for {model.__class__.__name__} with features {feats}: {e_score}"
    # )


def hyper():

    my_combinations = [
        HyperOptCombination(
            name="XGBClassifierGPU_default",
            # model=XGBClassifierGPU(
            #     enable_categorical=True,
            #     n_jobs=CFG.n_jobs,
            #     objective="multi:softprob",
            #     eval_metric="mlogloss",
            #     allow_categorical_as_ordinal=False,
            #     verbosity=0,
            # )._set_gpu(CFG.gpu),
            model=XGBClassifier(
                enable_categorical=True,
                n_jobs=CFG.n_jobs,
                objective="multi:softprob",
                eval_metric="mlogloss",
                verbosity=1,
                device="cuda",
                tree_method="hist",
            ),
            feature_combination=FeatureCombination(
                features=[
                    "Crop",
                    "Soil",
                    "M",
                    "P",
                    "N",
                    "K",
                    "H",
                    "T",
                ]
            ),
        )
    ]

    def create_objective(data: pd.DataFrame, model_combination: HyperOptCombination):
        X = data.drop(columns=["Fertilizer Name"])
        y = data["Fertilizer Name"]
        X_train = X[model_combination.feature_combination.features]
        model = model_combination.model
        model_name = model_combination.name

        def objective(trail: optuna.Trial) -> OBJECTIVE_RETURN_TYPE:
            params = TrialParamWrapper().get_params(
                model_name=model_name,
                trial=trail,
            )

            logger.info(
                f"Running trial {trail.number} for model {model_name} with parameters: {params}"
            )

            pipeline = clone(model).set_params(**params)

            try:
                # score = evaluate(pipeline, X_train, y, cv=CFG.cv)
                score = fertilize(
                    estimator=pipeline,
                    X=X_train,
                    y=y,
                    cv=CFG.cv,
                    cv_type="averaged",
                    random_state=CFG.random_state,
                    preprocessor=None,
                )

                return score

            except optuna.exceptions.TrialPruned as e:
                print(f"Trial {trail.number} was pruned: {e}")
                raise e
            except Exception as e:
                print(f"Error during evaluation of trial {trail.number}: {e}")
                raise e

        return objective

    model_run = "simplified"

    setup_dto = HyperSetupDto(
        n_optimization_trials=150,
        optimization_timeout=None,
        n_patience=50,
        min_percentage_improvement=0.005,
        model_run=model_run,
        limit_data_percentage=None,
        processes=CFG.n_jobs,
        max_concurrent_jobs=None,
        output_dir_path=PathManager.output.value,
        hyper_opt_prefix=PrefixManager.hyper.value,
        study_prefix=PrefixManager.study.value,
        data=X_org.join(y_org),
        combinations=my_combinations,
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

    if n > 0:
        df = setup_analysis(
            model_run=model_run,
            output_dir_path=PathManager.output.value,
            hyper_opt_prefix=PrefixManager.hyper.value,
            study_prefix=PrefixManager.study.value,
            display_plots=False,
        )

        studies_storage_path = aggregate_studies(
            study_dir_path=PathManager.output.value
            / f"{PrefixManager.study.value}{model_run}"
        )


if __name__ == "__main__":
    # execute_fbfs(combinations)
    # test_model()
    hyper()
