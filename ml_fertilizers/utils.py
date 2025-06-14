from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import cross_val_score
from autofeat import AutoFeatClassifier
from sklearn.pipeline import FunctionTransformer, Pipeline


class PathManager(Enum):
    cwd = Path(__file__).parent.parent.resolve()
    data = cwd / "data"
    output = cwd / "output"
    predictions = output / "predictions"
    trades = output / "trades"
    errors = output / "errors"


for path in PathManager:
    if not path.value.exists():
        path.value.mkdir(parents=True, exist_ok=True)


class PrefixManager(Enum):
    hyper = "hyper_opt_"
    ensemble = "ensemble_"
    study = "study_"


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(PathManager.data.value / "train.csv")
    test = pd.read_csv(PathManager.data.value / "test.csv")

    return train, test


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

    # NOTE: clean up memory
    topk = topk_sorted = class_labels = topk_labels = hits = reciprocal_ranks = None
    del topk, topk_sorted, class_labels, topk_labels, hits, reciprocal_ranks

    return scores.mean()


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


def evaluate(estimator, X, y, cv=3) -> float:

    scores = cross_val_score(
        estimator, X, y.astype("category").cat.codes, cv=cv, scoring=mapk_scorer
    )
    return scores.mean()


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
        "cat_features": X_final.drop(
            columns=[col for col in ["Fertilizer Name"] if col in X_final.columns]
        )
        .select_dtypes(include=["object", "category"])
        .columns.tolist(),
        "autofeat_features": (
            X_autofeat.columns.tolist() if not isinstance(autofeat_cls, bool) else []
        ),
    }
    return X_final.set_index("id"), final_dict, autofeat_cls


def engineer_simple(X: pd.DataFrame, features_in: Optional[List[str]]) -> pd.DataFrame:

    X = X.set_index("id")
    X = X.rename(
        columns={
            "Soil Type": "Soil",
            "Crop Type": "Crop",
        }
    )
    # NOTE: ENGINERRING
    # fmt: off
    if features_in is None or ('Env_Stress_Index' in features_in):
        X["Env_Stress_Index"] = X["Temparature"] * 0.4 + X["Humidity"] * 0.3 + X["Moisture"] * 0.3
    if features_in is None or ('NPK_Index' in features_in):
        X["NPK_Index"] = X["Nitrogen"] * 0.5 + X["Phosphorous"] * 0.3 + X["Potassium"] * 0.2
    if features_in is None or (any(pd.Series(features_in).str.contains(r'^Temp_bin\.*$'))):
        X["Temp_bin"] = pd.Categorical.from_codes(np.digitize(X["Temparature"].values, bins=[-np.inf, 15, 30, 45, np.inf]) - 1, categories=["low", "medium", "high", "very_high"])  # type: ignore
    if features_in is None or (any(pd.Series(features_in).str.contains(r'^Humidity_bin\.*$'))):
        X['Humidity_bin'] = pd.Categorical.from_codes(np.digitize(X["Humidity"].values, bins=[-np.inf, 30, 50, 70, np.inf]) - 1, categories=["low", "medium", "high", "very_high"])  # type: ignore
    if features_in is None or (any(pd.Series(features_in).str.contains(r'^Moisture_bin\.*$'))):
        X['Moisture_bin'] = pd.Categorical.from_codes(np.digitize(X["Moisture"].values, bins=[-np.inf, 20, 40, 60, np.inf]) - 1, categories=["low", "medium", "high", "very_high"])  # type: ignore

    if features_in is None or ('SoilNutrients' in features_in):
        X['SoilNutrients'] = X['Nitrogen'] + X['Phosphorous'] + X['Potassium']
    if features_in is None or ('SoilNutrient_Ratio' in features_in):
        X["SoilNutrient_Ratio"] = X["Nitrogen"] / (X["Potassium"] + X["Phosphorous"] + 1)
    if features_in is None or ('Temp_Humidity' in features_in):
        X["Temp_Humidity"] = X["Temparature"] * X["Humidity"]
    if features_in is None or ('Temp_Moisture' in features_in):
        X["Temp_Moisture"] = X["Temparature"] * X["Moisture"]
    # fmt: on
    cat_features = ["Crop", "Soil", "Temp_bin", "Humidity_bin", "Moisture_bin"]
    for col in cat_features:
        if col in X.columns:
            X[col] = X[col].astype("category")

    X_cat = pd.DataFrame()
    if features_in is None or (
        any(pd.Series(features_in).str.contains(r"^Soil_[A-Za-z]+$"))
    ):
        X_cat = pd.get_dummies(
            X["Soil"], prefix="Soil", drop_first=False, sparse=True, dtype="uint8"
        )
    if features_in is None or (
        any(pd.Series(features_in).str.contains(r"^Crop_[A-Za-z]+$"))
    ):
        X_cat = pd.concat(
            [
                X_cat,
                pd.get_dummies(
                    X["Crop"],
                    prefix="Crop",
                    drop_first=False,
                    sparse=True,
                    dtype="uint8",
                ),
            ],
            axis=1,
        )
    if features_in is None or (
        any(pd.Series(features_in).str.contains(r"^Temp_bin_[A-Za-z]+$"))
    ):
        X_cat = pd.concat(
            [
                X_cat,
                pd.get_dummies(
                    X["Temp_bin"],
                    prefix="Temp_bin",
                    drop_first=False,
                    sparse=True,
                    dtype="uint8",
                ),
            ],
            axis=1,
        )
    if features_in is None or (
        any(pd.Series(features_in).str.contains(r"^Humidity_bin_[A-Za-z]+$"))
    ):
        X_cat = pd.concat(
            [
                X_cat,
                pd.get_dummies(
                    X["Humidity_bin"],
                    prefix="Humidity_bin",
                    drop_first=False,
                    sparse=True,
                    dtype="uint8",
                ),
            ],
            axis=1,
        )
    if features_in is None or (
        any(pd.Series(features_in).str.contains(r"^Moisture_bin_[A-Za-z]+$"))
    ):
        X_cat = pd.concat(
            [
                X_cat,
                pd.get_dummies(
                    X["Moisture_bin"],
                    prefix="Moisture_bin",
                    drop_first=False,
                    sparse=True,
                    dtype="uint8",
                ),
            ],
            axis=1,
        )

    if not X_cat.empty:
        X_final = pd.concat([X, X_cat], axis=1)
    else:
        X_final = X.copy()

    X = X_cat = None  # type: ignore
    del X_cat, X

    # Change int64 to uint16
    for col in X_final.select_dtypes(include=["int64"]).columns:
        X_final[col] = X_final[col].astype("uint16")

    return X_final


def create_preprocessor(features_in: Optional[List[str]] = None) -> Pipeline:

    pipeline_steps: List[Tuple[str, Any]] = [
        (
            "simple_engineering",
            FunctionTransformer(
                lambda df: engineer_simple(df, features_in=features_in)
            ),
        )
    ]

    if features_in is None or (
        features_in is not None
        and any(pd.Series(features_in).str.contains(r"^pca\d+$"))
    ):

        ct_step = (
            "ct",
            ColumnTransformer(
                transformers=(
                    [
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
                    ]
                ),
                remainder="passthrough",
            ),
        )
        pipeline_steps.append(ct_step)  # type: ignore

    pipeline_steps.append(
        (
            "remove_prefixes",
            FunctionTransformer(
                lambda df: df.rename(
                    columns=lambda x: x.split("__")[-1] if "__" in x else x
                )
            ),
        )
    )

    preprocessor = Pipeline(steps=pipeline_steps).set_output(transform="pandas")
    return preprocessor  # type: ignore


def create_preprocessor2(features_in: Optional[List[str]] = None) -> Pipeline:

    pipeline_steps: List[Tuple[str, Any]] = [
        (
            "ct",
            ColumnTransformer(
                transformers=(
                    [
                        (
                            "standardize",
                            StandardScaler(),
                            make_column_selector(dtype_include=np.number),  # type: ignore
                        )
                    ]
                ),
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
