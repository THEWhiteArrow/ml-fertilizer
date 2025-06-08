from ml_fertilizers.lib.optymization.analysis_setup import setup_analysis
from ml_fertilizers.utils import load_data
from IPython.display import display

train, test = load_data()
train = train.set_index("id")
test = test.set_index("id")
display(train)

import pickle as pkl
from typing import Dict, List, Optional, Tuple, Union, cast
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from autofeat import AutoFeatClassifier


def engineer_features(
    X: pd.DataFrame, autofeat_cls: Union[bool, Optional[AutoFeatClassifier]] = False
) -> Tuple[pd.DataFrame, Dict[str, List[str]], Union[AutoFeatClassifier, bool]]:
    raw_num_features = [
        "Temparature",
        "Humidity",
        "Moisture",
        "Nitrogen",
        "Phosphorous",
        "Potassium",
    ]
    raw_cat_features = ["Crop", "Soil"]

    X = X.copy()
    X = X.rename(
        columns={
            "Soil Type": "Soil",
            "Crop Type": "Crop",
        }
    )

    # X['Crop_x_Soil'] = X['Crop'] + '_' + X['Soil']
    X["Env_Stress_Index"] = (
        X["Temparature"] * 0.4 + X["Humidity"] * 0.3 + X["Moisture"] * 0.3
    )
    X["NPK_Index"] = X["Nitrogen"] * 0.5 + X["Phosphorous"] * 0.3 + X["Potassium"] * 0.2
    X["Temp_bin"] = pd.cut(
        X["Temparature"],
        bins=[-float("inf"), 15, 25, 35, float("inf")],
        labels=["low", "medium", "high", "very_high"],
    )
    X["Humidity_bin"] = pd.cut(
        X["Humidity"],
        bins=[-float("inf"), 30, 50, 70, float("inf")],
        labels=["low", "medium", "high", "very_high"],
    )
    X["Moisture_bin"] = pd.cut(
        X["Moisture"],
        bins=[-float("inf"), 20, 40, 60, float("inf")],
        labels=["low", "medium", "high", "very_high"],
    )
    X["PCA_Temparature"] = PCA(n_components=2).fit_transform(
        X[["Temparature", "Humidity", "Moisture"]]
    )[:, 0]

    print("Autofeating features...")

    if isinstance(autofeat_cls, bool):
        print("Skipping autofeat feature engineering.")
        X_autofeat = pd.DataFrame()
    elif autofeat_cls is None:
        autofeat_cls = AutoFeatClassifier(
            verbose=0, n_jobs=-1, feateng_steps=2, categorical_cols=raw_cat_features
        )
        X_autofeat = cast(pd.DataFrame, autofeat_cls.fit_transform(X[raw_num_features + raw_cat_features], X["Fertilizer Name"]))  # type: ignore
        print("Autofeat columns:", X_autofeat.columns.tolist())
    else:
        X_autofeat = cast(
            pd.DataFrame, autofeat_cls.transform(X[raw_num_features + raw_cat_features])
        )
        print("Autofeat columns:", X_autofeat.columns.tolist())

    X_final = pd.concat([X, X_autofeat], axis=1)
    X_final = X_final.loc[:, ~X_final.columns.duplicated()]

    final_dict = {
        "num_features": X_final.select_dtypes(include=["number"]).columns.tolist(),
        "cat_features": X_final.drop(columns=["Fertilizer Name"])
        .select_dtypes(include=["object", "category"])
        .columns.tolist(),
        "autofeat_features": (
            X_autofeat.columns.tolist() if not isinstance(autofeat_cls, bool) else []
        ),
    }
    return X_final, final_dict, autofeat_cls


eng_train, feat_dict, auto_cls = engineer_features(train, autofeat_cls=False)

display(eng_train)


import numpy as np


def mapk_scorer(estimator, X, y_true, k=3):
    """
    Uses estimator.predict_proba to compute MAP@k.
    y_val contains integer-encoded true labels.
    """
    probas = estimator.predict_proba(X)
    topk = np.argsort(probas, axis=1)[:, -k:][:, ::-1]  # shape: (n_samples, k)
    scores = []
    for i, true_label in enumerate(y_true):
        preds = topk[i]
        score = 0.0
        hits = 0
        seen = set()
        for rank, p in enumerate(preds):
            if p == true_label and p not in seen:
                hits += 1
                score += hits / (rank + 1)
                seen.add(p)
        scores.append(score / 1.0)  # each actual list has length=1
    return np.mean(scores)


cat_set = [
    "Temparature",
    "Soil",
    "Potassium",
    "Phosphorous",
    "NPK_Index",
    "Temp_bin",
    "Humidity_bin",
    "Moisture_bin",
]

rfe_dict = {
    # "CatBoostClassifier_cat_features": cat_set,
    # "CatBoostClassifier_30": [
    #   "Temparature",
    #   "Humidity",
    #   "Moisture",
    #   "Nitrogen",
    #   "Potassium",
    #   "Phosphorous",
    #   "Env_Stress_Index",
    #   "NPK_Index",
    #   "PCA_Temparature",
    #   "Soil_Clayey",
    #   "Soil_Loamy",
    #   "Soil_Red",
    #   "Soil_Sandy",
    #   "Crop_Cotton",
    #   "Crop_Ground Nuts",
    #   "Crop_Maize",
    #   "Crop_Millets",
    #   "Crop_Oil seeds",
    #   "Crop_Paddy",
    #   "Crop_Pulses",
    #   "Crop_Sugarcane",
    #   "Crop_Tobacco",
    #   "Crop_Wheat",
    #   "Temp_bin_high",
    #   "Temp_bin_very_high",
    #   "Humidity_bin_medium",
    #   "Humidity_bin_high",
    #   "Humidity_bin_very_high",
    #   "Moisture_bin_high",
    #   "Moisture_bin_very_high"
    # ],
    "CatBoostClassifier_25": [
        "Temparature",
        "Humidity",
        "Moisture",
        "Nitrogen",
        "Potassium",
        "Phosphorous",
        "Env_Stress_Index",
        "NPK_Index",
        "PCA_Temparature",
        "Soil_Clayey",
        "Soil_Loamy",
        "Soil_Red",
        "Soil_Sandy",
        "Crop_Cotton",
        "Crop_Ground Nuts",
        "Crop_Maize",
        "Crop_Millets",
        "Crop_Oil seeds",
        "Crop_Paddy",
        "Crop_Pulses",
        "Crop_Sugarcane",
        "Crop_Tobacco",
        "Crop_Wheat",
        "Temp_bin_high",
        "Humidity_bin_high",
    ],
    "CatBoostClassifier_20": [
        "Temparature",
        "Humidity",
        "Moisture",
        "Nitrogen",
        "Potassium",
        "Phosphorous",
        "Env_Stress_Index",
        "NPK_Index",
        "PCA_Temparature",
        "Soil_Clayey",
        "Soil_Loamy",
        "Soil_Red",
        "Soil_Sandy",
        "Crop_Ground Nuts",
        "Crop_Oil seeds",
        "Crop_Paddy",
        "Crop_Pulses",
        "Crop_Sugarcane",
        "Crop_Tobacco",
        "Temp_bin_high",
    ],
    "CatBoostClassifier_15": [
        "Temparature",
        "Humidity",
        "Moisture",
        "Nitrogen",
        "Potassium",
        "Phosphorous",
        "Env_Stress_Index",
        "NPK_Index",
        "PCA_Temparature",
        "Soil_Loamy",
        "Soil_Red",
        "Soil_Sandy",
        "Crop_Paddy",
        "Crop_Pulses",
        "Crop_Sugarcane",
    ],
    "CatBoostClassifier_10": [
        "Temparature",
        "Humidity",
        "Moisture",
        "Nitrogen",
        "Potassium",
        "Phosphorous",
        "Env_Stress_Index",
        "NPK_Index",
        "PCA_Temparature",
        "Soil_Sandy",
    ],
    # "CatBoostClassifier_5": [
    #   "Humidity",
    #   "Nitrogen",
    #   "Potassium",
    #   "Phosphorous",
    #   "PCA_Temparature"
    # ],
    # "XGBClassifier_30": [
    #   "Temparature",
    #   "Humidity",
    #   "Moisture",
    #   "Nitrogen",
    #   "Potassium",
    #   "Phosphorous",
    #   "Env_Stress_Index",
    #   "NPK_Index",
    #   "PCA_Temparature",
    #   "Soil_Clayey",
    #   "Soil_Loamy",
    #   "Soil_Red",
    #   "Soil_Sandy",
    #   "Crop_Cotton",
    #   "Crop_Ground Nuts",
    #   "Crop_Maize",
    #   "Crop_Millets",
    #   "Crop_Oil seeds",
    #   "Crop_Paddy",
    #   "Crop_Pulses",
    #   "Crop_Sugarcane",
    #   "Crop_Tobacco",
    #   "Crop_Wheat",
    #   "Temp_bin_high",
    #   "Temp_bin_very_high",
    #   "Humidity_bin_medium",
    #   "Humidity_bin_high",
    #   "Humidity_bin_very_high",
    #   "Moisture_bin_medium",
    #   "Moisture_bin_high"
    # ],
    "XGBClassifier_25": [
        "Temparature",
        "Humidity",
        "Moisture",
        "Nitrogen",
        "Potassium",
        "Phosphorous",
        "NPK_Index",
        "PCA_Temparature",
        "Soil_Clayey",
        "Soil_Loamy",
        "Soil_Red",
        "Soil_Sandy",
        "Crop_Cotton",
        "Crop_Ground Nuts",
        "Crop_Maize",
        "Crop_Millets",
        "Crop_Oil seeds",
        "Crop_Paddy",
        "Crop_Pulses",
        "Crop_Sugarcane",
        "Crop_Tobacco",
        "Crop_Wheat",
        "Temp_bin_high",
        "Humidity_bin_high",
        "Moisture_bin_high",
    ],
    "XGBClassifier_20": [
        "Temparature",
        "Moisture",
        "Nitrogen",
        "Potassium",
        "Phosphorous",
        "Soil_Clayey",
        "Soil_Red",
        "Soil_Sandy",
        "Crop_Cotton",
        "Crop_Ground Nuts",
        "Crop_Maize",
        "Crop_Millets",
        "Crop_Oil seeds",
        "Crop_Paddy",
        "Crop_Pulses",
        "Crop_Sugarcane",
        "Crop_Wheat",
        "Temp_bin_high",
        "Humidity_bin_high",
        "Moisture_bin_high",
    ],
    "XGBClassifier_15": [
        "Moisture",
        "Potassium",
        "Phosphorous",
        "Soil_Clayey",
        "Soil_Red",
        "Soil_Sandy",
        "Crop_Ground Nuts",
        "Crop_Oil seeds",
        "Crop_Paddy",
        "Crop_Pulses",
        "Crop_Sugarcane",
        "Crop_Wheat",
        "Temp_bin_high",
        "Humidity_bin_high",
        "Moisture_bin_high",
    ],
    "XGBClassifier_10": [
        "Moisture",
        "Phosphorous",
        "Soil_Clayey",
        "Soil_Red",
        "Soil_Sandy",
        "Crop_Ground Nuts",
        "Crop_Paddy",
        "Crop_Pulses",
        "Crop_Sugarcane",
        "Temp_bin_high",
    ],
    # "XGBClassifier_5": [
    #   "Moisture",
    #   "Phosphorous",
    #   "Soil_Sandy",
    #   "Crop_Pulses",
    #   "Crop_Sugarcane"
    # ],
    # "LGBMClassifier_30": [
    #   "Temparature",
    #   "Humidity",
    #   "Moisture",
    #   "Nitrogen",
    #   "Potassium",
    #   "Phosphorous",
    #   "Env_Stress_Index",
    #   "NPK_Index",
    #   "PCA_Temparature",
    #   "Soil_Clayey",
    #   "Soil_Loamy",
    #   "Soil_Red",
    #   "Soil_Sandy",
    #   "Crop_Cotton",
    #   "Crop_Ground Nuts",
    #   "Crop_Maize",
    #   "Crop_Millets",
    #   "Crop_Oil seeds",
    #   "Crop_Paddy",
    #   "Crop_Pulses",
    #   "Crop_Sugarcane",
    #   "Crop_Tobacco",
    #   "Crop_Wheat",
    #   "Temp_bin_medium",
    #   "Temp_bin_high",
    #   "Temp_bin_very_high",
    #   "Humidity_bin_medium",
    #   "Humidity_bin_high",
    #   "Moisture_bin_medium",
    #   "Moisture_bin_high"
    # ],
    "LGBMClassifier_25": [
        "Temparature",
        "Humidity",
        "Moisture",
        "Nitrogen",
        "Potassium",
        "Phosphorous",
        "Env_Stress_Index",
        "NPK_Index",
        "PCA_Temparature",
        "Soil_Clayey",
        "Soil_Loamy",
        "Soil_Red",
        "Soil_Sandy",
        "Crop_Cotton",
        "Crop_Ground Nuts",
        "Crop_Maize",
        "Crop_Millets",
        "Crop_Oil seeds",
        "Crop_Paddy",
        "Crop_Pulses",
        "Crop_Sugarcane",
        "Crop_Tobacco",
        "Crop_Wheat",
        "Temp_bin_high",
        "Humidity_bin_high",
    ],
    "LGBMClassifier_20": [
        "Temparature",
        "Humidity",
        "Moisture",
        "Nitrogen",
        "Potassium",
        "Phosphorous",
        "Env_Stress_Index",
        "NPK_Index",
        "PCA_Temparature",
        "Soil_Clayey",
        "Soil_Loamy",
        "Soil_Red",
        "Soil_Sandy",
        "Crop_Maize",
        "Crop_Millets",
        "Crop_Oil seeds",
        "Crop_Paddy",
        "Crop_Pulses",
        "Crop_Tobacco",
        "Crop_Wheat",
    ],
    "LGBMClassifier_15": [
        "Temparature",
        "Humidity",
        "Moisture",
        "Nitrogen",
        "Potassium",
        "Phosphorous",
        "Env_Stress_Index",
        "NPK_Index",
        "PCA_Temparature",
        "Soil_Clayey",
        "Soil_Loamy",
        "Soil_Red",
        "Soil_Sandy",
        "Crop_Oil seeds",
        "Crop_Paddy",
    ],
    "LGBMClassifier_10": [
        "Temparature",
        "Humidity",
        "Moisture",
        "Nitrogen",
        "Potassium",
        "Phosphorous",
        "Env_Stress_Index",
        "NPK_Index",
        "PCA_Temparature",
        "Soil_Loamy",
    ],
    # "LGBMClassifier_5": [
    #   "Nitrogen",
    #   "Phosphorous",
    #   "Env_Stress_Index",
    #   "NPK_Index",
    #   "PCA_Temparature"
    # ],
    "XGBClassifier_SFS_Baseline_Kaggle": [
        "Moisture",
        "Phosphorous",
        "Potassium",
        "Soil_Black",
        "Nitrogen",
        "Soil_Sandy",
        "Crop_Sugarcane",
        "Temparature",
        "Crop_Oil seeds",
        "Crop_Cotton",
    ],
}

from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer


def evaluate(estimator, X, y, cv=3) -> float:

    scores = cross_val_score(
        estimator, X, y.astype("category").cat.codes, cv=cv, scoring=mapk_scorer
    )
    return scores.mean()


import os
from typing import List
import multiprocessing as mp

import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import optuna
from sklearn import clone

from ml_fertilizers.lib.features.FeatureCombination import FeatureCombination
from ml_fertilizers.lib.models.HyperOptCombination import HyperOptCombination
from ml_fertilizers.lib.models.GpuModels import CatBoostClassifierGPU, XGBClassifierGPU
from ml_fertilizers.lib.optymization.TrialParamWrapper import TrialParamWrapper
from ml_fertilizers.lib.optymization.optimization_study import (
    OBJECTIVE_RETURN_TYPE,
    aggregate_studies,
)
from ml_fertilizers.lib.utils.garbage_collector import garbage_manager


processes = 28
my_combinations: List[HyperOptCombination] = []
job_count = processes if processes is not None else mp.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(job_count)
os.environ["MKL_NUM_THREADS"] = str(job_count)
RANDOM_STATE = 42
gpu = True
models = [
    # LGBMClassifier(n_jobs=job_count, verbosity=-1, random_state=RANDOM_STATE, objective="multiclass", eval_metric="multi_logloss"),  # type: ignore
    # XGBClassifierGPU(
    #     random_state=RANDOM_STATE,
    #     n_jobs=job_count,
    #     verbosity=0,
    #     objective="multi:softmax",
    #     eval_metric="mlogloss",
    #     gpu_dtype=None,
    # )._set_gpu(use_gpu=True),
    CatBoostClassifierGPU(
        random_state=RANDOM_STATE,
        thread_count=job_count,
        verbose=False,
        allow_writing_files=False,
        loss_function="MultiClass",
        eval_metric="MultiClass",
    )._set_gpu(use_gpu=False),
]

for key, features in rfe_dict.items():
    for model in models:
        if model.__class__.__name__.startswith(key.split("_")[0]):
            my_combinations.append(
                HyperOptCombination(
                    name=key,
                    model=model,
                    feature_combination=FeatureCombination(
                        name="_".join(key.split("_")[1:]), features=features
                    ),
                )
            )


display(my_combinations)


def create_objective(data: pd.DataFrame, model_combination: HyperOptCombination):
    num_columns = data.select_dtypes(include=["number"]).columns.tolist()

    data[num_columns] = data[num_columns].astype(np.float16)

    X = pd.get_dummies(
        data.drop(columns=["Fertilizer Name"]), sparse=True, drop_first=False
    )

    y = (
        data["Fertilizer Name"].astype("category").cat.codes
    )  # Convert to integer codes for RFE

    model = model_combination.model
    model_name = model_combination.name

    def objective(trail: optuna.Trial) -> OBJECTIVE_RETURN_TYPE:
        params = TrialParamWrapper().get_params(
            model_name=model_name,
            trial=trail,
        )

        pipeline = clone(model).set_params(**params)

        try:
            X_train = X[model_combination.feature_combination.features]

            score = evaluate(pipeline, X_train, y, cv=3)

            return score

        except optuna.exceptions.TrialPruned as e:
            print(f"Trial {trail.number} was pruned: {e}")
            raise e
        except Exception as e:
            print(f"Error during evaluation of trial {trail.number}: {e}")
            raise e

    return objective


from ml_fertilizers.lib.optymization.hyper_setup import setup_hyper

from ml_fertilizers.lib.optymization.parrarel_optimization import (
    HyperFunctionDto,
    HyperSetupDto,
)
from ml_fertilizers.utils import PrefixManager, PathManager

model_run = "intial_run"

setup_dto = HyperSetupDto(
    n_optimization_trials=70,
    optimization_timeout=None,
    n_patience=30,
    min_percentage_improvement=0.005,
    model_run=model_run,
    limit_data_percentage=0.50,
    processes=processes,
    max_concurrent_jobs=None,
    output_dir_path=PathManager.output.value,
    hyper_opt_prefix=PrefixManager.hyper.value,
    study_prefix=PrefixManager.study.value,
    data=eng_train,
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

setup_hyper(setup_dto=setup_dto, function_dto=function_dto)


df = setup_analysis(
    model_run=model_run,
    output_dir_path=PathManager.output.value,
    hyper_opt_prefix=PrefixManager.hyper.value,
    study_prefix=PrefixManager.study.value,
    display_plots=False,
)

studies_storage_path = aggregate_studies(
    study_dir_path=PathManager.output.value / f"{PrefixManager.study.value}{model_run}"
)
