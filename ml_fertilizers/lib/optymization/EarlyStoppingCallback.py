from dataclasses import dataclass, field
from typing import List, Optional, Union

import optuna

from ml_fertilizers.lib.logger import setup_logger


logger = setup_logger(__name__)


@dataclass
class EarlyStoppingCallback:
    """When I wrote the initial version of this class, I knew what was up. Now, rewritten to accommodate multiple objectives, I just hope it works"""

    name: str
    patience: int
    min_percentage_improvement: float = 0.0
    best_value: Optional[Union[float, List[float]]] = None
    no_improvement_count: Optional[Union[int, List[int]]] = None
    directions: Optional[Union[str, List[str]]] = None

    def __post_init__(self):

        # NOTE: Normalize inputs to lists for consistency

        if self.best_value is not None and not isinstance(
            self.best_value, (list, tuple)
        ):
            self.best_value = [self.best_value]

        if self.no_improvement_count is not None and not isinstance(
            self.no_improvement_count, (list, tuple)
        ):
            self.no_improvement_count = [self.no_improvement_count]

        if self.directions is not None and not isinstance(
            self.directions, (list, tuple)
        ):
            self.directions = [self.directions]

        self.is_single_objective = len(self.directions) == 1 and self.directions[0] in [  # type: ignore
            "minimize",
            "maximize",
        ]

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:

        if trial.state in (
            optuna.trial.TrialState.PRUNED,
            optuna.trial.TrialState.FAIL,
        ):
            for i in range(len(self.no_improvement_count)):  # type: ignore
                self.no_improvement_count[i] += 1  # type: ignore

        elif trial.state == optuna.trial.TrialState.COMPLETE:

            if self.is_single_objective:
                current_best_values = [study.best_value]
            else:
                current_best_values = list(study.best_trials[0].values)

            if self.best_value is None or all(v is None for v in self.best_value):  # type: ignore
                self.best_value = current_best_values.copy()
                self.no_improvement_count = [0 for _ in current_best_values]

            # Track improvement per objective
            for i, (direction, cur, best) in enumerate(  # type: ignore
                zip(self.directions, current_best_values, self.best_value)  # type: ignore
            ):

                if (
                    (
                        best >= 0
                        and direction == "maximize"
                        and cur > best * (1.0 + self.min_percentage_improvement)
                    )
                    or (
                        best >= 0
                        and direction == "minimize"
                        and cur < best * (1.0 - self.min_percentage_improvement)
                    )
                    or (
                        best < 0
                        and direction == "maximize"
                        and cur > best * (1.0 - self.min_percentage_improvement)
                    )
                    or (
                        best < 0
                        and direction == "minimize"
                        and cur < best * (1.0 + self.min_percentage_improvement)
                    )
                ):
                    self.best_value[i] = cur  # type: ignore
                    self.no_improvement_count[i] = 0  # type: ignore
                else:
                    self.no_improvement_count[i] += 1  # type: ignore

        # Stop if any objective exceeds patience
        if any(c >= self.patience for c in self.no_improvement_count):  # type: ignore
            logger.warning(
                f"Early stopping the study: {self.name} due to "
                + f"no {self.min_percentage_improvement * 100}% improvement for "
                + f"{self.patience} trials | on trial: {trial.number}"
                + f" | best values: {self.best_value} | no improvement counts: {self.no_improvement_count}"
            )
            study.stop()
