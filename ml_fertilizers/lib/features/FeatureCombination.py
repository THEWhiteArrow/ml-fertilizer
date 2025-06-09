from dataclasses import dataclass, field
from typing import List


@dataclass
class FeatureCombination:
    features: List[str]
    name: str = field(default="")

    def __post_init__(self):
        if "-" in self.name:
            raise ValueError("Name cannot contain '-'")
