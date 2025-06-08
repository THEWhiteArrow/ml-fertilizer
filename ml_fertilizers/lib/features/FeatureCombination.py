from dataclasses import dataclass
from typing import List


@dataclass
class FeatureCombination:
    name: str
    features: List[str]

    def __post_init__(self):
        if "-" in self.name:
            raise ValueError("Name cannot contain '-'")
