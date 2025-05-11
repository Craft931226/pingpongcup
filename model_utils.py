import joblib, lightgbm as lgb
from pathlib import Path
import numpy as np, pandas as pd
# from sklearn.calibration import label_binarize
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

TARGETS = {
    "gender":             {"type": "bin", "name": "gender"},
    "hold racket handed": {"type": "bin", "name": "hold"},
    "play years":         {"type": "multi", "num_class": 3, "name": "play_years"},
    "level":              {"type": "multi", "num_class": 4, "name": "level"},
}

def build_scaler(X: np.ndarray):
    sc = MinMaxScaler()
    return sc.fit(X)

def build_model(y, target_info):
    if target_info["type"] == "bin":
        params = dict(
            objective   = "binary" if target_info["type"]=="bin" else "multiclass",
            metric      = "auc"    if target_info["type"]=="bin" else "multi_logloss",
            num_class   = target_info.get("num_class", None),
            learning_rate = 0.01,             # ↓ 降低學習率
            num_leaves    = 31,               # ↓ 減少複雜度
            n_estimators  = 1500,             # ↑ 配合較小 LR
            max_depth     = -1,
            min_child_samples = 20,
            bagging_freq      = 5,
            colsample_bytree  = 0.8,
            reg_alpha         = 0.5,
            reg_lambda        = 0.5,
            bagging_fraction  = 0.9,
            feature_fraction  = 0.9,
            random_state      = 42,
            n_jobs            = -1,
            class_weight      = "balanced",
            boosting_type     = "dart",
            drop_rate         = 0.1,
            # learning_rate_drop  = 0.02,
            # num_iterations_drop = 2500,
            device_type       = 'gpu',   # ← 必加：改用 GPU
            gpu_platform_id   = 0,       # ← 視顯卡調整，預設 0
            gpu_device_id     = 0,       # ← 同上

        )

    else:
        params = dict(
            objective   = "binary" if target_info["type"]=="bin" else "multiclass",
            metric      = "auc"    if target_info["type"]=="bin" else "multi_logloss",
            num_class   = target_info.get("num_class", None),
            learning_rate = 0.01,             # ↓ 降低學習率
            num_leaves    = 31,               # ↓ 減少複雜度
            n_estimators  = 2500,             # ↑ 配合較小 LR
            max_depth     = -1,
            min_child_samples = 20,
            colsample_bytree  = 0.8,
            bagging_freq      = 5,
            reg_alpha         = 0.5,
            reg_lambda        = 0.5,
            bagging_fraction  = 0.9,
            feature_fraction  = 0.9,
            random_state      = 42,
            n_jobs            = -1,
            class_weight      = "balanced",
            boosting_type     = "dart",
            drop_rate         = 0.1,
            device_type       = 'gpu',   # ← 必加：改用 GPU
            gpu_platform_id   = 0,       # ← 視顯卡調整，預設 0
            gpu_device_id     = 0,       # ← 同上

        )
    return lgb.LGBMClassifier(**params)

def cv_evaluate(model, X, y, groups, target_info, early_stopping_rounds=50):
    gkf = GroupKFold(n_splits=5)
    scores = []

    for tr_idx, val_idx in gkf.split(X, y, groups):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        eval_set = [(X_val, y_val)]
        
        # Use different metrics for binary and multiclass
        eval_metric = 'auc' if target_info["type"] == "bin" else 'multi_logloss'
        
        model.fit(X_tr, y_tr,
                 eval_set=eval_set,
                 eval_metric=eval_metric,
                 callbacks=[lgb.early_stopping(early_stopping_rounds)])
                 
        proba = model.predict_proba(X_val)

        # ---------- Binary ----------
        if target_info["type"] == "bin":
            pos_prob = proba[:, 1] if proba.ndim == 2 else proba.ravel()
            scores.append(roc_auc_score(y_val, pos_prob))
            continue

        # ---------- Multi-class ----------
        present = np.unique(y_val)             
        if len(present) == 1:                       
            continue                                

        col_of = {c: i for i, c in enumerate(model.classes_)}
        
        if len(present) == 2:
            pos_cls   = present[1]
            pos_prob  = proba[:, col_of[pos_cls]]
            y_bin = (y_val == pos_cls).astype(int)   
            score = roc_auc_score(y_bin, pos_prob)
            scores.append(score)
            continue

        proba_use = proba[:, [col_of[c] for c in present]]

        if proba_use.ndim > 1:
            proba_use = proba_use / proba_use.sum(axis=1, keepdims=True)

        score = roc_auc_score(
            y_val, proba_use,
            labels=present, average="micro", multi_class="ovr"
        )
        scores.append(score)

    return np.mean(scores) if scores else 0.5

def save_model(model, scaler, name):
    Path("models").mkdir(exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler},
                f"models/{name}.pkl")

def load_model(name):
    return joblib.load(f"models/{name}.pkl")
