import json
from typing import List
from ml_fertilizers.lib.pipelines.ProcessingPipelineWrapper import (
    create_pipeline,
)
from ml_fertilizers.lib.logger import setup_logger

logger = setup_logger(__name__)


def reverse_transformed_features(
    original_features: List[str], current_features: List[str]
) -> List[str]:
    ans: List[str] = []
    for feature in current_features:
        if feature.startswith("numerical"):
            original_feature_suffix = feature.split("__")[-1]
            if original_feature_suffix not in original_features:
                logger.error(
                    f"Feature {original_feature_suffix} not found in original features"
                )
                raise ValueError(
                    f"Feature {original_feature_suffix} not found in original features"
                )
            ans.append(original_feature_suffix)
        elif feature.startswith("string"):
            categorical_combined_suffix = feature.split("__")[-1]
            original_feature_prefix = categorical_combined_suffix.split("_cat")[0]
            if original_feature_prefix not in original_features:
                logger.error(
                    f"Feature {original_feature_prefix} not found in original features"
                )
                raise ValueError(
                    f"Feature {original_feature_prefix} not found in original features"
                )
            ans.append(original_feature_prefix)

    # Remove duplicates
    ans = list(set(ans))

    return ans


def correlation_simplification(
    engineered_data, features_in: list[str], threshold: float = 0.8
) -> tuple[list[str], list[str]]:
    """
    Simplifies the correlation matrix by removing highly correlated features.

    Args:
        engineered_data (pd.DataFrame): DataFrame containing engineered features.

    Returns:
        tuple: A tuple containing a list of selected feature names and a list of removed feature names.
    """
    # Select features to be included in the correlation analysis
    X = engineered_data[features_in].copy()
    pipeline = create_pipeline(allow_strings=False, pandas_output=True)

    T = pipeline.fit_transform(X)
    Tc = T.corr()  # type: ignore

    removed = []
    removal_correlation = dict()
    for col in Tc.columns:
        if col in ["year"]:
            continue
        if col not in removed:
            tmp = list(
                set(Tc[col].loc[Tc[col].gt(threshold)].index.to_list()) - set([col])
            )
            if len(tmp) > 0:
                removal_correlation[col] = tmp
                removed.extend(tmp)
                removed = list(set(removed))

    Xu = T.drop(columns=removed)  # type: ignore

    saved = reverse_transformed_features(X.columns, Xu.columns)
    dropped = reverse_transformed_features(X.columns, removed)

    logger.warning(f"Removed {len(removed)} features")
    logger.warning(f"Removal correlation: {json.dumps(removal_correlation)}")

    return saved, dropped
