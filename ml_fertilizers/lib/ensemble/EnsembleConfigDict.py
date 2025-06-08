from typing import List, TypedDict


class EnsembleConfigDict(TypedDict):
    model_run: str
    model_combination_names: List[str]
