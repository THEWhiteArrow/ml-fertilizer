from dataclasses import dataclass
from typing import List

from sklearn import clone
from sklearn.base import BaseEstimator

from ml_fertilizers.lib.features.FeatureCombination import FeatureCombination
from ml_fertilizers.lib.features.FeatureManager import FeatureManager
from ml_fertilizers.lib.models.HyperOptCombination import HyperOptCombination


@dataclass
class HyperOptManager:
    feature_manager: FeatureManager
    models: List[BaseEstimator]

    def get_model_combinations(self) -> List[HyperOptCombination]:
        all_feature_combinations: List[FeatureCombination] = (
            self.feature_manager.get_all_possible_feature_combinations()
        )

        hyper_opt_model_combinations: List[HyperOptCombination] = [
            HyperOptCombination(
                name=f"{model.__class__.__name__}" + f"_{combination.name}",
                model=clone(model),
                feature_combination=combination,
            )
            for model in self.models
            for combination in all_feature_combinations
        ]

        return hyper_opt_model_combinations
