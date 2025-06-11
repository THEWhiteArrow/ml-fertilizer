import gc
from typing import List, Tuple
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import StratifiedKFold

from ml_fertilizers.lib.logger import setup_logger
from ml_fertilizers.utils import calc_mapk, load_data

logger = setup_logger(__name__)

le = LabelEncoder()

train, test = load_data()


def fertilize(estimator, X, y, cv=5, random_state=69, preprocessor=None):
    kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    oof_proba = pd.DataFrame(index=X.index, columns=y.unique())

    for fold, (train_index, val_index) in enumerate(kfold.split(X, y)):
        print(f"Fold {fold + 1}/{cv}")
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        if preprocessor is not None:
            logger.info(f"Fitting preprocessor for fold {fold + 1}/{cv}")
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

    final_data = pd.concat(
        [feat_data.drop(columns=cat_features), feat_cat_data], axis=1
    )
    org_feat_list = feat_data.columns.tolist()
    ohe_feat_list = list(
        set(
            list(set(org_feat_list) - set(cat_features))
            + feat_cat_data.columns.tolist()
        )
    )
    return final_data, target_data, org_feat_list, ohe_feat_list


X_org, y_org, raw_feat, ohe_feat = dataing(train)
