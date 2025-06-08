from dataclasses import dataclass, field
from typing import List

from ml_fertilizers.lib.features.FeatureCombination import FeatureCombination


@dataclass
class FeatureSet(FeatureCombination):
    is_optional: bool = True
    is_exclusive: bool = False
    is_exclusive_mandatory: bool = False
    is_standalone: bool = True
    bans: List[str] = field(default_factory=list)
