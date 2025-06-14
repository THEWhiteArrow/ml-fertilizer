# KAGGLE Comapring Multiclass Ensmbleing TGechniques
### Source: https://www.kaggle.com/ravaghi/comparing-multiclass-ensembling-techniques

# Stacking Trick
```python
# This is just a trick to avoid refitting all base models from scratch, which is what StackingClassifier and VotingClassifier do by default

class PassThroughClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, idx_cols):
        self.idx_cols = idx_cols
        self.is_fitted_ = True

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        return check_array(X)[:, self.idx_cols]


estimators = [
    ('LightGBM (gbdt)', PassThroughClassifier(list(range(0, 7)))),
    ('LightGBM (goss)', PassThroughClassifier(list(range(7, 14)))),
    ('XGBoost', PassThroughClassifier(list(range(14, 21)))),
    ('AutoGluon', PassThroughClassifier(list(range(21, 28)))),
]
```

# Trainer trick
```python
class Trainer:
    def __init__(self, model, config=CFG):
        self.model = model
        self.config = config

    def fit_predict(self, X, y, X_test):
        print(f"Training {self.model.__class__.__name__}\n")
        
        scores = []        
        oof_pred_probs = np.zeros((X.shape[0], y.nunique()))
        test_pred_probs = np.zeros((X_test.shape[0], y.nunique()))
        
        skf = StratifiedKFold(n_splits=self.config.n_folds, random_state=self.config.seed, shuffle=True)
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = clone(self.model)
            model.fit(X_train, y_train)
            
            y_pred_probs = model.predict_proba(X_val)
            oof_pred_probs[val_idx] = y_pred_probs
            
            temp_test_pred_probs = model.predict_proba(X_test)
            test_pred_probs += temp_test_pred_probs / self.config.n_folds
            
            score = map3(y_val, y_pred_probs)
            scores.append(score)
            
            del model, X_train, y_train, X_val, y_val, y_pred_probs
            gc.collect()
        
            print(f"--- Fold {fold_idx + 1} - MAP@3: {score:.6f}")
                            
        overall_score = map3(y, oof_pred_probs)
            
        print(f"\n------ Overall MAP@3: {overall_score:.6f} | Average MAP@3: {np.mean(scores):.6f} ± {np.std(scores):.6f}")
        
        return oof_pred_probs, test_pred_probs, scores

    def tune(self, X, y):        
        scores = []        
        
        skf = StratifiedKFold(n_splits=self.config.n_folds, random_state=self.config.seed, shuffle=True)
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = clone(self.model)
            model.fit(X_train, y_train)
            
            y_pred_probs = model.predict_proba(X_val)            
            score = map3(y_val, y_pred_probs)
            scores.append(score)
            
            del model, X_train, y_train, X_val, y_val, y_pred_probs
            gc.collect()
            
        return np.mean(scores)
```

# Execution logs
```
INFO __main__ 21:18:08 | Fertilization score: 0.2624782222222222
INFO __main__ 21:27:02 | MAP@K score: 0.27474844444444446
---
INFO __main__ 21:40:13 | Fertilization score: 0.258728
INFO __main__ 21:47:06 | MAP@K score: 0.30092266666666667
---
I think that my own calc_mapk is not working properly, so I will use the one from the some kaggle notebook:
EDIT
It is working correclty:
INFO __main__ 22:10:08 | Fertilization score: 0.29888
INFO __main__ 22:10:09 | MAP@K score: 0.29888

So it must be different way cross-validation is handled - needs further investigation
---
I think that mine is calculating OOF and then MAP@K, while the one from the notebook is calculating MAP@K for each fold and then averaging it


# kaggle_feat = [
#     "Temparature",
#     "Humidity",
#     "Moisture",
#     "Nitrogen",
#     "Potassium",
#     "Phosphorous",
#     "Temp-Humidity",
#     "Temp-Moisture",
#     "N+Po+Ph",
#     "N/(Po+Ph)",
#     "Soil",
#     "Crop",
#     "Temp_bin",
# ]
# f_score = fertilize(
#     estimator=clone(xgb_model),
#     X=X_org[kaggle_feat],
#     y=y_org,
#     cv=CFG.cv,
#     random_state=CFG.random_state,
#     preprocessor=None,
# )
# logger.info(f"Fertilization score: {f_score}")


# m_score = evaluate(
#     estimator=clone(xgb_model),
#     X=X_org[kaggle_feat],
#     y=y_org,
#     cv=CFG.cv,
# )
# logger.info(f"MAP@K score: {m_score}")
# X_train = X_org.sample(frac=0.8, random_state=CFG.random_state)
# X_test = X_org.drop(X_train.index)
# y_train = y_org[X_train.index]
# y_test = y_org[X_test.index]

# fitted_model = clone(xgb_model).fit(X_train[kaggle_feat], y_train)
# y_proba_raw = fitted_model.predict_proba(X_test[kaggle_feat])
# y_proba = pd.DataFrame(
#     y_proba_raw, index=X_test.index, columns=le.transform(le.classes_)  # type: ignore
# )

# f_score = calc_mapk(y_true=y_test, y_probas=y_proba, k=3)
# logger.info(f"Fertilization score: {f_score}")
# m_score = mapk_scorer(estimator=fitted_model, X=X_test[kaggle_feat], y_true=y_test, k=3)
# logger.info(f"MAP@K score: {m_score}")

```