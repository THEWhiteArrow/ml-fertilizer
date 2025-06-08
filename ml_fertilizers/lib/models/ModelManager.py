from dataclasses import dataclass
import os
from typing import List, Optional
import multiprocessing as mp

from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    LogisticRegression,
    PassiveAggressiveClassifier,
    PassiveAggressiveRegressor,
    RidgeClassifier,
    SGDClassifier,
    SGDRegressor,
)
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from ml_fertilizers.lib.logger import setup_logger
from ml_fertilizers.lib.models.ExtendedModels import (
    ExtendedKNeighborsClassifier,
    ExtendedPassiveAggressiveClassifier,
    ExtendedPassiveAggressiveRegressor,
    ExtendedRidgeClassifier,
    ExtendedLGBMClassifier,
    ExtendedRidgeRegressor,
    ExtendedSGDClassifier,
    ExtendedSGDRegressor,
    ExtendedXGBClassifierGPU,
    ExtendedLGBMRegressor,
    ExtendedXGBRegressorGPU,
    ExtendedRandomForestClassifier,
    RidgeRegressor,
)
from ml_fertilizers.lib.models.GpuModels import XGBClassifierGPU, XGBRegressorGPU

logger = setup_logger(__name__)

# from ml_fertilizers.lib.models.NeuralNetworkCustomModel import NeuralNetworkCustomModel

RANDOM_STATE = 42


@dataclass
class ModelManager:

    def get_models(
        self,
        processes: Optional[int] = None,
        use_models: Optional[List[str]] = None,
        gpu: bool = True,
    ) -> List[BaseEstimator]:

        if use_models is None:
            raise ValueError(
                "You need to specify which models you want to use. Use the 'use_models' parameter."
            )

        job_count = processes if processes is not None else mp.cpu_count()

        os.environ["OMP_NUM_THREADS"] = str(job_count)
        os.environ["MKL_NUM_THREADS"] = str(job_count)

        logger.info(f"Using {job_count} jobs for model training. GPU enabled: {gpu}")
        models: List[BaseEstimator] = []
        # fmt: off
        supported_models = [
            # extended classification
            ExtendedRidgeClassifier(random_state=RANDOM_STATE),
            ExtendedLGBMClassifier(n_jobs=job_count, verbosity=-1, random_state=RANDOM_STATE),  # type: ignore
            ExtendedXGBClassifierGPU(random_state=RANDOM_STATE, n_jobs=job_count, verbosity=0)._set_gpu(use_gpu=gpu),
            ExtendedPassiveAggressiveClassifier(random_state=RANDOM_STATE, shuffle=False),
            ExtendedKNeighborsClassifier(n_jobs=job_count, metric="cosine"),
            ExtendedSGDClassifier(verbose=0, random_state=RANDOM_STATE, shuffle=False, penalty="elasticnet"),
            ExtendedRandomForestClassifier(n_jobs=job_count, random_state=RANDOM_STATE),

            # extended regression
            ExtendedRidgeRegressor(random_state=RANDOM_STATE),
            ExtendedLGBMRegressor(n_jobs=job_count, verbosity=-1, random_state=RANDOM_STATE),  # type: ignore
            ExtendedXGBRegressorGPU(n_jobs=job_count, random_state=RANDOM_STATE, verbosity=0)._set_gpu(use_gpu=gpu),
            ExtendedSGDRegressor(verbose=0, random_state=RANDOM_STATE, shuffle=False),
            ExtendedPassiveAggressiveRegressor(random_state=RANDOM_STATE, shuffle=False),

            # clean classification
            RidgeClassifier(random_state=RANDOM_STATE),
            LGBMClassifier(n_jobs=job_count, verbosity=-1, random_state=RANDOM_STATE),  # type: ignore
            XGBClassifierGPU(random_state=RANDOM_STATE, n_jobs=job_count, verbosity=0)._set_gpu(use_gpu=gpu),
            PassiveAggressiveClassifier(random_state=RANDOM_STATE, shuffle=False),
            KNeighborsClassifier(n_jobs=job_count, metric="cosine"),
            SGDClassifier(verbose=0, random_state=RANDOM_STATE, shuffle=False, penalty="elasticnet"),
            RandomForestClassifier(n_jobs=job_count, random_state=RANDOM_STATE),
            LogisticRegression(n_jobs=job_count, random_state=RANDOM_STATE, max_iter=1000, verbose=0),
            HistGradientBoostingClassifier(verbose=0, random_state=RANDOM_STATE),
            CatBoostClassifier(thread_count=job_count, random_state=RANDOM_STATE, verbose=False, allow_writing_files=False),

            # clean regression
            HistGradientBoostingRegressor(verbose=0, random_state=RANDOM_STATE),
            RidgeRegressor(random_state=RANDOM_STATE),
            LGBMRegressor(n_jobs=job_count, verbosity=-1, random_state=RANDOM_STATE),  # type: ignore
            XGBRegressorGPU(n_jobs=job_count, random_state=RANDOM_STATE, verbosity=0)._set_gpu(use_gpu=gpu),
            SGDRegressor(verbose=0, random_state=RANDOM_STATE, shuffle=False),
            PassiveAggressiveRegressor(random_state=RANDOM_STATE, shuffle=False),
            KNeighborsRegressor(n_jobs=job_count, metric="cosine"),
            RandomForestRegressor(n_jobs=job_count, random_state=RANDOM_STATE),
            CatBoostRegressor(thread_count=job_count, random_state=RANDOM_STATE, verbose=False, allow_writing_files=False)
        ]

        for model_name in use_models:
            for supported_model in supported_models:
                if supported_model.__class__.__name__.lower().startswith(model_name.lower()):
                    models.append(supported_model)

        if len(models) == 0:
            raise ValueError(
                "Bro. You had one job of selecting a proper task name and you failed..."
            )

        return models
