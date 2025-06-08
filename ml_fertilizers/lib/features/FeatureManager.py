from dataclasses import dataclass
from typing import List

import pandas as pd

from ml_fertilizers.lib.features.FeatureCombination import FeatureCombination
from ml_fertilizers.lib.features.FeatureSet import FeatureSet
from ml_fertilizers.lib.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class FeatureManager:
    feature_sets: List[FeatureSet]

    def verify_features_existence(self, X: pd.DataFrame) -> bool:
        columns: List[str] = X.columns.to_list()

        all_features: List[str] = [
            feature
            for feature_group in self.feature_sets
            for feature in feature_group.features
        ]

        diff_feats: List[str] = list(set(all_features) - set(columns))

        if len(diff_feats) > 0:
            logger.warning(f"Missing features in the data: {diff_feats}")
            return False

        return True

    def get_all_possible_feature_combinations(self) -> List[FeatureCombination]:

        mandatory_feature_sets: List[FeatureSet] = [
            feat_set
            for feat_set in self.feature_sets
            if (
                feat_set.is_optional is False
                and feat_set.is_exclusive is False
                and feat_set.is_exclusive_mandatory is False
            )
        ]

        optional_feature_set: List[FeatureSet] = [
            feat_set
            for feat_set in self.feature_sets
            if (
                feat_set.is_optional is True
                and feat_set.is_exclusive is False
                and feat_set.is_exclusive_mandatory is False
            )
        ]

        exclusive_mandatory_feature_sets: List[FeatureSet] = [
            feat_set
            for feat_set in self.feature_sets
            if feat_set.is_exclusive_mandatory is True
        ]
        exclusive_feature_sets: List[FeatureSet] = [
            feat_set for feat_set in self.feature_sets if feat_set.is_exclusive is True
        ]

        for feat_set in exclusive_mandatory_feature_sets:
            for exclusive_set in exclusive_feature_sets:
                exclusive_set.features = list(
                    set(exclusive_set.features) | set(feat_set.features)
                )

        logger.info(f"Detected {len(optional_feature_set)} optional feature sets.")
        logger.info(f"Detected {len(mandatory_feature_sets)} mandatory feature sets.")
        logger.info(f"Detected {len(exclusive_feature_sets)} exclusive feature sets.")

        if len(optional_feature_set) > 10:
            logger.warning(
                "The number of optional feature sets is high: "
                + f"{len(optional_feature_set)}"
            )

        bitmap = 2 ** len(optional_feature_set) - 1
        possible_combinations: List[FeatureCombination] = []

        for i in range(bitmap + 1):
            if len(mandatory_feature_sets) == 0 and i == 0:
                continue

            combination_name: str = ""
            combination_features: List[str] = []
            combination_standalone: List[bool] = []

            for mandatory_set in mandatory_feature_sets:
                combination_name += f"{mandatory_set.name}_"
                combination_standalone.append(mandatory_set.is_standalone)
                combination_features.extend(mandatory_set.features)

            for j, optional_set in enumerate(optional_feature_set):
                if i & (1 << j):

                    if not any(ban in combination_name for ban in optional_set.bans):
                        # Add the optional feature set to the combination
                        combination_name += f"{optional_set.name}_"
                        combination_features.extend(optional_set.features)
                        combination_standalone.append(optional_set.is_standalone)

            if (
                any(combination_standalone)
                or len(combination_standalone) > 1
                and not any(
                    combination_name in possible_set.name
                    for possible_set in possible_combinations
                )
            ):
                possible_combinations.append(
                    FeatureCombination(
                        name=combination_name, features=list(set(combination_features))
                    )
                )

        possible_combinations.extend(exclusive_feature_sets)

        logger.info(
            f"Generated {len(possible_combinations)} possible feature combinations."
        )

        return possible_combinations
