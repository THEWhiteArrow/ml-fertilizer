from typing import Any, Dict, List, Optional, TypedDict

from sklearn.base import BaseEstimator


class HyperOptResultDict(TypedDict):
    name: str
    model: Optional[BaseEstimator]
    features: Optional[List[str]]
    params: Optional[Dict[str, Any]]
    score: Optional[float]
    n_trials: Optional[int]
    metadata: Optional[Dict[str, Any]]
