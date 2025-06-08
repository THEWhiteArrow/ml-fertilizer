import json
import multiprocessing as mp
import os
import gc

import pandas as pd
from sklearn import clone
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import RidgeClassifier
from ml_fertilizers.lib.logger import setup_logger
from ml_fertilizers.lib.models.EnsembleModel2 import EnsembleModel2
from ml_fertilizers.lib.pipelines.ProcessingPipelineWrapper import create_pipeline
from ml_fertilizers.lib.utils.results import load_hyper_opt_results
from ml_fertilizers.utils import (
    PathManager,
    PrefixManager,
    engineer_features,
    evaluate,
    load_data,
)

logger = setup_logger(__name__)

processes = 28
gpu = False

model_run = "intial_run"
results = load_hyper_opt_results(
    model_run=model_run,
    output_dir_path=PathManager.output.value,
    hyper_opt_prefix=PrefixManager.hyper.value,
)

logger.info(f"Loaded hyperparameter optimization results for {model_run}.")

train, test = load_data()

eng_train, _, _ = engineer_features(train)

X = pd.get_dummies(eng_train, drop_first=False, sparse=True)
y = eng_train.loc[X.index, "Fertilizer Name"]

job_count = mp.cpu_count() if processes is None else processes
score_dict = {}
os.environ["OMP_NUM_THREADS"] = str(job_count)
os.environ["MKL_NUM_THREADS"] = str(job_count)

# for result in results:
#     res_features = result["features"]
#     res_model = result["model"]
#     res_params = result["params"]
#     res_name = result["name"]

#     if res_model is None or res_features is None or res_params is None:
#         logger.warning(f"Skipping incomplete result: {result}")
#         continue

#     logger.info(f"Evaluating model: {res_name} with features: {res_features}")

#     res_model = clone(res_model)

#     if hasattr(res_model, "set_gpu"):
#         res_model = res_model.set_gpu(gpu)

#     if hasattr(res_model, "n_jobs"):
#         res_model.set_params(n_jobs=job_count)
#     elif hasattr(res_model, "thread_count"):
#         res_model.set_params(thread_count=job_count)

#     score = evaluate(
#         estimator=res_model.set_params(**res_params),
#         X=X[res_features],
#         y=y,
#     )
#     logger.info(f"Score for {res_name}: {score}")
#     score_dict[res_name] = score

#     gc.collect()

# logger.info(f"Evaluation scores: {score_dict}")
# json.dump(
#     score_dict,
#     open(PathManager.output.value / f"{PrefixManager.ensemble.value}scores.json", "w"),
#     indent=4,
# )

stacking_estimators = []
stacking_features = []
stacking_names = []

for result in results:
    res_features = result["features"]
    res_model = result["model"]
    res_params = result["params"]
    res_name = result["name"]

    if res_model is None or res_features is None or res_params is None:
        logger.warning(f"Skipping incomplete result: {result}")
        continue

    res_model = clone(res_model)

    if hasattr(res_model, "set_gpu"):
        res_model = res_model.set_gpu(gpu)

    if hasattr(res_model, "n_jobs"):
        res_model.set_params(n_jobs=job_count)
    elif hasattr(res_model, "thread_count"):
        res_model.set_params(thread_count=job_count)

    stacking_estimators.append(res_model.set_params(**res_params))
    stacking_features.append(res_features)
    stacking_names.append(res_name)

# stack_model = StackingClassifier(
#     estimators=stacking_estimators,
#     final_estimator=RidgeClassifier(),
#     n_jobs=job_count,
#     cv=3,
#     verbose=1,
#     stack_method="predict_proba",
#     passthrough=False,
# )

stack_model = EnsembleModel2(
    models=stacking_estimators,
    combination_features=stacking_features,
    combination_names=stacking_names,
    just_filtering=True,
    prediction_method="predict_proba",
    metamodel=RidgeClassifier(),
    metamodel_kfold=3,
)

stack_dict = dict()

stack_dict["basic_ridge"] = evaluate(
    estimator=stack_model,
    X=X,
    y=y,
    cv=3,
)
logger.info(f"Stacking model evaluation: {stack_dict['basic_ridge']}")
json.dump(
    stack_dict,
    open(
        PathManager.output.value / f"{PrefixManager.ensemble.value}stack_scores.json",
        "w",
    ),
    indent=4,
)
