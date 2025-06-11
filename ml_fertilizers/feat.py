import gc
import json
import datetime as dt
from typing import List, Tuple
from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV, LabelEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedKFold

from ml_fertilizers.lib.logger import setup_logger
from ml_fertilizers.lib.models.GpuModels import (
    CatBoostClassifierGPU,
    LGBMClassifierGPU,
    XGBClassifierLeanGPU,
)
from ml_fertilizers.utils import PathManager, calc_mapk, load_data

logger = setup_logger(__name__)

le = LabelEncoder()

train, test = load_data()


class CFG:
    random_state = 69
    n_jobs = -1
    cv = 4
    gpu = True
    sample_frac = 0.75


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
    def __init__(self, gpu=True, **kwargs):
        self.kwargs = kwargs
        self.base_clf_ = None
        self.gpu = gpu

    def fit(self, X, y, **kwargs):
        self.base_clf_ = CatBoostClassifierGPU(
            **self.kwargs,
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


def fertilize(estimator, X, y, cv, random_state, preprocessor=None):
    kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    oof_proba = pd.DataFrame(index=X.index, columns=y.unique())

    for fold, (train_index, val_index) in enumerate(kfold.split(X, y)):
        # logger.info(f"Fold {fold + 1}/{cv}")
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        if preprocessor is not None:
            logger.warning(f"Fitting preprocessor for fold {fold + 1}/{cv}")
            X_train = preprocessor.fit_transform(X_train)
            X_val = preprocessor.transform(X_val)

        estimator.fit(X_train, y_train)
        oof_proba.iloc[val_index] = estimator.predict_proba(X_val)

    score = calc_mapk(y_true=y, y_probas=oof_proba, k=3)
    kfold = oof_proba = X_train = X_val = y_train = y_val = estimator = None
    del kfold, oof_proba, X_train, X_val, y_train, y_val, estimator
    gc.collect()
    return score


def dataing(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    data = data.set_index("id")

    data = data.rename(columns={"Soil Type": "Soil", "Crop Type": "Crop"})
    cat_cols = ["Soil", "Crop", "Fertilizer Name"]
    for col in cat_cols:
        data[col] = data[col].astype("category")

    data["Soil_x_Crop"] = data["Soil"].astype(str) + "_" + data["Crop"].astype(str)
    data["Soil_x_Crop"] = data["Soil_x_Crop"].astype("category")

    data["Fertilizer Name"] = le.fit_transform(data["Fertilizer Name"])
    data["N/Po"] = data["Nitrogen"] / (data["Potassium"] + 1e-6)
    data["N/Ph"] = data["Nitrogen"] / (data["Phosphorous"] + 1e-6)
    data["Po/Ph"] = data["Potassium"] / (data["Phosphorous"] + 1e-6)
    data["N+Po+Ph"] = data["Nitrogen"] + data["Potassium"] + data["Phosphorous"]
    data["Temp-Humidity"] = data["Temparature"] * data["Humidity"]

    feat_data = data.drop(columns=["Fertilizer Name"])
    target_data = data["Fertilizer Name"]

    cat_features = ["Soil", "Crop", "Soil_x_Crop"]
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

X_org = X_org.sample(frac=CFG.sample_frac, random_state=CFG.random_state)
y_org = y_org[X_org.index]

# fmt: off
combinations = [
    ("xgb_raw", XGBClassifierLeanGPU(enable_categorical=True, n_jobs=CFG.n_jobs, objective="multi:softprob", eval_metric="mlogloss", max_depth=10, n_estimators=1000)._set_gpu(CFG.gpu), raw_feat),
    ("cat_raw", CatBoostClassifierCategoricalGPU(gpu=CFG.gpu, thread_count=CFG.n_jobs, loss_function="MultiClass", eval_metric="MultiClass"), raw_feat),
    ("lgbm_raw", LGBMClassifierCategoricalGPU(n_jobs=CFG.n_jobs, verbosity=-1, objective="multiclass", eval_metric="multiclass_logloss")._set_gpu(CFG.gpu), raw_feat),
    ("log_ohe", LogisticRegression(), ohe_feat),
    ("cal_ohe", CalibratedClassifierCV(method="sigmoid", cv=CFG.cv, n_jobs=CFG.n_jobs), ohe_feat),
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
):
    res_path = PathManager.output.value / f"fbfs_{name}_results.json"

    if res_path.exists():
        logger.info(f"Results already exist for {name}. Loading from {res_path}")
        res_progress = json.loads(res_path.read_text())
    else:
        logger.info(f"Results do not exist for {name}. Starting new run.")
        res_progress = [
            {
                "selected_features": [],
                "score": 0,
                "added_feature": None,
                "removed_feature": None,
                "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        ]

    logger.info(
        f"Starting Forward Backward Feature Selection (k={k}) with {estimator.__class__.__name__}"
    )
    selected_features = res_progress[0]["selected_features"].copy()
    while len(selected_features) < k:

        if len(selected_features) > 2:
            logger.info(
                f"Current selected features: {selected_features}, total: {len(selected_features)}, score: {res_progress[-1]['score']}"
            )
            curr_score = res_progress[-1]["score"]

            for feature in selected_features:
                if feature == res_progress[-1]["added_feature"]:
                    continue
                temp_features = selected_features.copy()
                temp_features.remove(feature)
                score = fertilize(
                    estimator=estimator,
                    X=X[temp_features],
                    y=y,
                    cv=cv,
                    random_state=random_state,
                )
                if score > curr_score:
                    curr_score = score
                    selected_features.remove(feature)
                    logger.info(
                        f"Removed feature: {feature} with score: {score} | old score: {res_progress[-1]['score']}"
                    )
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
                    break

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
            score = fertilize(
                estimator=estimator,
                X=X[current_features],
                y=y,
                cv=cv,
                random_state=random_state,
            )

            if score > best_new_score:
                logger.info(
                    f"Found better feature: {feature} with score: {score} (previous best: {best_new_score})"
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
        else:
            logger.info(
                f"No more features to add. Current selected features: {selected_features}"
            )
            break

    logger.info(f"Final selected features: {selected_features}")

    estimator = None
    del estimator
    gc.collect()

    return res_progress


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
    )
    logger.info(f"FFS results for {name}:\n{json.dumps(ffs_results, indent=4)}")
