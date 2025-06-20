from typing import List
import multiprocessing as mp

import optuna
import pandas as pd
from sklearn import clone
from sklearn.calibration import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from ml_fertilizers.lib.features.FeatureCombination import FeatureCombination
from ml_fertilizers.lib.logger import setup_logger
from ml_fertilizers.lib.models.GpuModels import XGBClassifierGPU
from ml_fertilizers.lib.models.HyperOptCombination import HyperOptCombination
from ml_fertilizers.lib.optymization.TrialParamWrapper import TrialParamWrapper
from ml_fertilizers.lib.optymization.analysis_setup import setup_analysis
from ml_fertilizers.lib.optymization.hyper_setup import setup_hyper
from ml_fertilizers.lib.optymization.optimization_study import (
    OBJECTIVE_RETURN_TYPE,
    aggregate_studies,
)
from ml_fertilizers.lib.optymization.parrarel_optimization import (
    HyperFunctionDto,
    HyperSetupDto,
)
from ml_fertilizers.utils import (
    PathManager,
    PrefixManager,
    calc_mapk,
    create_preprocessor,
    load_data,
)
from ml_fertilizers.lib.utils.garbage_collector import garbage_manager

# CONFIGURATION
model_run = "deepfear"
processes = 20
gpu = False

logger = setup_logger(__name__)
train, test = load_data()
RANDOM_STATE = 69
job_count = mp.cpu_count() if processes is None else processes


xgb_model = XGBClassifierGPU(
    random_state=RANDOM_STATE,
    n_jobs=job_count,
    verbosity=0,
    objective="multi:softprob",
    eval_metric="mlogloss",
    enable_categorical=True,
    early_stopping_rounds=300,
)._set_gpu(use_gpu=gpu)

lr_model = LogisticRegression(
    random_state=RANDOM_STATE,
    n_jobs=job_count,
    max_iter=1000,
    verbose=0,
)

original_features = [
    "Temparature",
    "Humidity",
    "Moisture",
    "Soil",
    "Crop",
    "Nitrogen",
    "Potassium",
    "Phosphorous",
]
original_features_ohe = [
    "Temparature",
    "Humidity",
    "Moisture",
    "Nitrogen",
    "Potassium",
    "Phosphorous",
    "Soil_Black",
    "Soil_Clayey",
    "Soil_Loamy",
    "Soil_Red",
    "Soil_Sandy",
    "Crop_Barley",
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
]
original_features_ohe = [
    "Temparature",
    "Humidity",
    "Moisture",
    "Nitrogen",
    "Potassium",
    "Phosphorous",
    "Soil_Black",
    "Soil_Clayey",
    "Soil_Loamy",
    "Soil_Red",
    "Soil_Sandy",
    "Crop_Barley",
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
]
sfs_xgb = [
    "Moisture",
    "Phosphorous",
    "Nitrogen",
    "Crop",
    "Potassium",
    "Soil",
    "Temparature",
    "Humidity",
    "Moisture_bin",
    "Humidity_bin",
    "Temp_bin",
]
sfs_lr = [
    "Moisture",
    "Crop_Oil seeds",
    "Crop_Wheat",
    "Crop_Cotton",
    "Crop_Maize",
    "Crop_Tobacco",
    "Humidity",
    "Potassium",
    "Crop_Pulses",
    "Soil_Black",
    "Crop_Millets",
    "Nitrogen",
    "Crop_Barley",
    "Soil_Loamy",
    "Crop_Paddy",
]
xgb_kaggle = [
    "Temparature",
    "Humidity",
    "Moisture",
    "Nitrogen",
    "Potassium",
    "Phosphorous",
    "Temp_Humidity",
    "Temp_Moisture",
    "SoilNutrients",
    "SoilNutrient_Ratio",
    "Soil",
    "Crop",
    "Temp_bin",
]

combinations: List[HyperOptCombination] = [
    HyperOptCombination(
        name="XGB_sfs_10",
        model=clone(xgb_model),
        feature_combination=FeatureCombination(features=sfs_xgb),
    ),
    HyperOptCombination(
        name="XGB_kaggle",
        model=clone(xgb_model),
        feature_combination=FeatureCombination(features=xgb_kaggle),
    ),
    HyperOptCombination(
        name="XGB_original",
        model=clone(xgb_model),
        feature_combination=FeatureCombination(features=original_features),
    ),
    HyperOptCombination(
        name="Logistic_original",
        model=clone(lr_model),
        feature_combination=FeatureCombination(features=original_features_ohe),
    ),
    HyperOptCombination(
        name="Logistic_engineered",
        model=clone(lr_model),
        feature_combination=FeatureCombination(features=sfs_lr),
    ),
]


FOLDS = 3
lbe = LabelEncoder()
train["Fertilizer Name"] = lbe.fit_transform(train["Fertilizer Name"])
train["Fertilizer Name"] = train["Fertilizer Name"].astype("category")


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
            preprocessor = create_preprocessor(features_in=features)
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

                if model_combination.name.startswith("XGB"):
                    model.fit(
                        X_train[features],
                        y_train,
                        eval_set=[(X_val[features], y_val)],
                        verbose=100,
                    )
                else:
                    model.fit(X_train[features], y_train)

                oof_probas.iloc[val_index] = model.predict_proba(X_val[features])

            score = calc_mapk(y_true=data["Fertilizer Name"], y_probas=oof_probas, k=3)
            model = oof_probas = preprocessor = td = vd = X_train = X_val = y_train = (
                y_val
            ) = None
            del model
            del oof_probas
            del preprocessor
            del td
            del vd
            del X_train
            del X_val
            del y_train
            del y_val
            garbage_manager.clean()
            return score
        except optuna.TrialPruned as e:
            logger.warning(f"Trial {trial.number} was pruned: {e}")
            raise e
        except Exception as e:
            logger.error(f"Error during trial {trial.number}: {e}")
            raise e

    return objective


setup_dto = HyperSetupDto(
    n_optimization_trials=70,
    optimization_timeout=None,
    n_patience=30,
    min_percentage_improvement=0.005,
    model_run=model_run,
    limit_data_percentage=0.75,
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
