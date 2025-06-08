import json
import multiprocessing as mp
import os
import gc
import pickle as pkl
from typing import cast
import numpy as np
from collections import defaultdict

import pandas as pd
from sklearn import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import GridSearchCV
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

ensemble_scores_path = (
    PathManager.output.value / f"{PrefixManager.ensemble.value}scores.json"
)
if not ensemble_scores_path.exists():
    ensemble_scores = dict()
else:
    ensemble_scores = json.load(open(ensemble_scores_path, "r"))

stacking_estimators = []
stacking_features = []
stacking_names = []

for result in results:
    res_features = result["features"]
    res_model = result["model"]
    res_params = result["params"]
    res_name = result["name"]
    res_score = result["score"]

    if (
        res_model is None
        or res_features is None
        or res_params is None
        or res_name is None
        or res_score is None
    ):
        logger.warning(f"Skipping incomplete result: {res_name}")
        continue

    if res_name in ensemble_scores and ensemble_scores[res_name] < 0.32:
        logger.info(
            f"Skipping {res_name} with score {ensemble_scores[res_name]} below threshold."
        )
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


alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
tols = [1e-3, 1e-2, 1e-1, 0.5, 1.0]


stack_score_path = (
    PathManager.output.value / f"{PrefixManager.ensemble.value}stack_scores.json"
)
if not stack_score_path.exists():
    stack_dict = dict()
else:
    stack_dict = json.load(open(stack_score_path, "r"))

for a in alphas:
    for t in tols:

        metamodel = CalibratedClassifierCV(
            estimator=RidgeClassifier(alpha=a, tol=t, random_state=42),
            method="sigmoid",
            cv=3,
        )

        stack_model = EnsembleModel2(
            models=stacking_estimators.copy(),
            combination_features=stacking_features,
            combination_names=stacking_names,
            just_filtering=True,
            prediction_method="predict_proba",
            metamodel=metamodel,
            metamodel_kfold=3,
        )

        stack_dict[f"calibrated_ridge_limited_models__alpha={a}__tol={t}"] = evaluate(
            estimator=stack_model,
            X=X,
            y=y,
            cv=3,
        )

logger.info(f"Stacking model evaluation: {stack_dict}")
json.dump(
    stack_dict,
    open(stack_score_path, "w"),
    indent=4,
)

oof_pred = cast(
    pd.DataFrame,
    pkl.load(
        open(PathManager.output.value / "all_predictions_out_of_fold.pkl", "rb"),
    ),
)

group_stats = dict()
groups = list(set([col.split("__")[0] for col in oof_pred.columns.tolist()]))
k = 3

# for g in groups:
#     gdf = oof_pred.filter(like=g)
#     # Use defaultdict for automatic initialization
#     stats_sum = defaultdict(lambda: defaultdict(float))
#     stats_count = defaultdict(lambda: defaultdict(int))
#     topk_indices = np.argsort(gdf.values, axis=1)[:, -k:][:, ::-1]
#     topk_labels = np.array(gdf.columns)[topk_indices]
#     topk_probs = np.take_along_axis(gdf.values, topk_indices, axis=1)
#     for row_labels, row_probs in zip(topk_labels, topk_probs):
#         for i, ref_label in enumerate(row_labels):
#             other_labels = tuple(l for j, l in enumerate(row_labels) if j != i)
#             other_probs = [p for j, p in enumerate(row_probs) if j != i]
#             if np.sum(other_probs) > 0:
#                 perc = row_probs[i] / np.sum(other_probs)
#             else:
#                 perc = 1.0
#             stats_sum[ref_label][other_labels] += perc
#             stats_count[ref_label][other_labels] += 1
#     # Compute mean
#     group_stats[g] = {
#         ref_label: {
#             other_labels: stats_sum[ref_label][other_labels]
#             / stats_count[ref_label][other_labels]
#             for other_labels in stats_sum[ref_label]
#         }
#         for ref_label in stats_sum
#     }

# for g in groups:
#     gdf = oof_pred.filter(like=g)
#     group_stats[g] = defaultdict(dict)
#     topk = np.argsort(gdf.values, axis=1)[:, -k:][:, ::-1]

#     for row in topk:
#         for label in row:
#             other_labels = [l for l in row if l != label]
#             if len(other_labels) == 0:
#                 continue

#             if label not in group_stats[g]:
#                 group_stats[g][label] = dict()

#             for other_label in other_labels:
#                 group_stats[g][label][other_label] = (
#                     group_stats[g][label].get(other_label, 0) + 1
#                 )

for g in groups:
    gdf = oof_pred.filter(like=g)
    group_stats[g] = defaultdict(dict)
    # Get top-k indices for each row
    topk = np.argsort(gdf.values, axis=1)[:, -k:][:, ::-1]
    col_labels = np.array(gdf.columns)
    # For each row, get the actual label names (not indices)
    for row_indices in topk:
        row_labels = col_labels[row_indices]
        for i, label in enumerate(row_labels):
            label = label.split("__")[-1]  # Get the actual label name
            other_labels = [l for j, l in enumerate(row_labels) if j != i]
            if len(other_labels) == 0:
                continue
            if label not in group_stats[g]:
                group_stats[g][label] = dict()
            for other_label in other_labels:
                other_label = other_label.split("__")[-1]
                group_stats[g][label][other_label] = (
                    group_stats[g][label].get(other_label, 0) + 1
                )

json.dump(
    group_stats,
    open(
        PathManager.output.value / f"{PrefixManager.ensemble.value}group_stats.json",
        "w",
    ),
    indent=4,
)

logger.info("Here")

import math
import matplotlib.pyplot as plt

ncols = 3
nrows = math.ceil(len(groups) / ncols)
fig, axes = plt.subplots(
    figsize=(14, 5 * nrows), nrows=nrows, ncols=ncols, sharex=True, sharey=True
)

axes = axes.flatten() if len(groups) > 1 else [axes]

for idx, g in enumerate(groups):
    stats = group_stats[g]
    labels = sorted(stats.keys())
    matrix = np.zeros((len(labels), len(labels)))
    for i, label in enumerate(labels):
        for j, other_label in enumerate(labels):
            if label == other_label:
                matrix[i, j] = 0
            else:
                matrix[i, j] = stats[label].get(other_label, 0)
    ax = axes[idx]
    im = ax.imshow(matrix, cmap="viridis")
    ax.set_title(f"Top-k Label Co-occurrence Counts for Group: {g}")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    plt.colorbar(im, ax=ax)
    # Annotate each cell with the count
    for i in range(len(labels)):
        for j in range(len(labels)):
            if matrix[i, j] > 0:
                ax.text(
                    j,
                    i,
                    int(matrix[i, j]),
                    ha="center",
                    va="center",
                    color="w",
                    fontsize=8,
                )

# Hide unused axes if any
for idx in range(len(groups), len(axes)):
    axes[idx].axis("off")

# plt.tight_layout()
plt.show()


fig.savefig(
    PathManager.output.value
    / f"{PrefixManager.ensemble.value}group_stats_visualization.png",
    bbox_inches="tight",
)
