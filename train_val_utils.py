import logging
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from feature_utils import aggregate_group_prob
import lightgbm as lgb
logging.basicConfig(
    filename="log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
# －－ 在檔案開頭 import 區之後加 －－
from collections import defaultdict

def max_feasible_splits(y, groups):
    """
    回傳可用的最大 n_splits，使得『每折 training 都能看到所有類別』。
    原理：一個類別若僅存在 k 位 player，就不可能做 k+1 折。
    """
    cls2players = defaultdict(set)
    for label, gid in zip(y, groups):
        cls2players[label].add(gid)
    min_players = min(len(p) for p in cls2players.values())
    return max(1, min_players)           # 至少 1 折

def train_validate_split(X, y, groups, test_size=0.2, random_state=42):
    from sklearn.model_selection import GroupKFold
    # Log the length of groups before splitting
    # print(f"Total number of unique groups: {len(groups)}")

    # Debug: Print the first few values of groups
    # print(f"First 10 values in groups: {groups[:10]}")

    # Check unique values in groups
    unique_groups, group_counts = np.unique(groups, return_counts=True)
    # print(f"Number of unique groups: {len(unique_groups)}")
    logging.info(f"Number of unique groups: {len(unique_groups)}")
    # print(f"Group counts: {dict(zip(unique_groups, group_counts))}")
    logging.info(f"Group counts: {dict(zip(unique_groups, group_counts))}")


    gkf = GroupKFold(n_splits=5)
    # 只取第一個 fold 當驗證，其餘 4 folds 當訓練
    
    for train_idx, val_idx in gkf.split(X, y, groups):
        if len(np.unique(y[val_idx])) >= 2:
            break
    # Ensure unique_id is not repeated
    train_unique_ids = np.unique(groups[train_idx])
    val_unique_ids = np.unique(groups[val_idx])

    if set(train_unique_ids).intersection(set(val_unique_ids)):
        raise ValueError("Data leakage detected: Some unique_ids are in both training and validation sets.")

    # Save unique_ids for debugging
    np.savetxt("train_ids_from_split.txt", train_unique_ids, fmt="%d")
    np.savetxt("val_ids_from_split.txt", val_unique_ids, fmt="%d")

    return {
        'X_train': X[train_idx],
        'y_train': y[train_idx],
        'groups_train': groups[train_idx],
        'X_val': X[val_idx],
        'y_val': y[val_idx],
        'groups_val': groups[val_idx]
    }

def oof_training(X, y, groups, target_info, build_model_fn, n_splits=3,
                 early_stopping_rounds=50, random_state=42):
    """
    使用 StratifiedGroupKFold 產生 OOF 預測與平均最佳迭代數。
    回傳：
        oof_proba   : (n_sample, n_class) or (n_sample,)  預測機率
        best_iters  : List[int]  各折最佳迭代
    """
    if n_splits < 2:
        return np.zeros((len(y), target_info["num_class"]))*np.nan, [200]
    # ---- 1. 準備 stratify label ----
    if target_info["type"] == "bin":
        strat_y = y
    else:
        # 多分類場景：直接用原 y 當 stratify（sklearn 允許不平衡）
        strat_y = y

    all_classes = np.unique(y)                            # 期望出現的完整類別
    for seed in range(100):                               # 最多嘗試 100 次
        sgkf = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state + seed
        )
        splits = list(sgkf.split(X, strat_y, groups))
        
        # 檢查每折 training 是否含有所有類別
        valid_splits = True
        for tr_idx, _ in splits:
            if len(np.unique(y[tr_idx])) < len(all_classes):
                valid_splits = False
                break
        
        # 若每折有效，則跳出循環
        if valid_splits:
            break
    else:
        # 若經過 100 次還無法分割，報錯
        raise RuntimeError("無法在 100 次內找到含完整類別的分割")

    # ---- 2. 預留空間 ----
    n_class = target_info["num_class"] if "num_class" in target_info else len(np.unique(y))
    oof_proba = np.zeros((len(y), n_class), dtype=float)
    best_iters = []

    # ---- 3. 逐折訓練 ----
    for fold, (tr_idx, va_idx) in enumerate(splits):
        mdl = build_model_fn(y, target_info)
        mdl.fit(
            X[tr_idx], y[tr_idx],
            eval_set=[(X[va_idx], y[va_idx])],
            eval_metric='auc' if target_info["type"] == "bin" else 'multi_logloss',
            callbacks=[lgb.early_stopping(30, verbose=False)]
        )
        best_it = mdl.best_iteration_ or mdl.n_estimators_
        best_iters.append(best_it)

        proba = mdl.predict_proba(X[va_idx])
        if target_info["type"] != "bin" and proba.shape[1] < n_class:
            full_prob = np.zeros((proba.shape[0], n_class), dtype=float)
            present   = mdl.classes_
            for col_idx, cls in enumerate(present):
                full_prob[:, cls] = proba[:, col_idx]
            proba = full_prob

        # ↓ 二元時把 proba[:,1] 填到第二欄；第一欄照樣保留
        if target_info["type"] == "bin" and proba.ndim == 1:
            proba = np.vstack([1 - proba, proba]).T
        oof_proba[va_idx] = proba

    return oof_proba, best_iters

def check_data_leakage(train_ids, val_ids):
    """Check for data leakage between training and validation datasets."""
    train_set = set(train_ids)
    val_set = set(val_ids)

    # Find intersection
    leakage = train_set.intersection(val_set)
    if leakage:
        print("Data leakage detected! Overlapping IDs:", leakage)
        logging.error(f"Data leakage detected! Overlapping IDs: {leakage}")
        return True
    else:
        print("No data leakage detected.")
        logging.info("No data leakage detected.")
        return False

def evaluate_model(model, data_dict, target_info):
    """Evaluate model performance on validation set"""
    model.fit(
        data_dict['X_train'], 
        data_dict['y_train'],
        eval_set=[(data_dict['X_val'], data_dict['y_val'])],
        eval_metric='auc' if target_info["type"] == "bin" else 'multi_logloss',
        callbacks=[lgb.early_stopping(50)]
    )
    
    proba = model.predict_proba(data_dict['X_val'])
    
    if target_info["type"] == "bin":
        score = roc_auc_score(data_dict['y_val'], proba[:, 1])
    else:
        # Convert validation labels to one-hot encoding
        classes = np.unique(data_dict['y_train'])
        y_val_onehot = np.zeros((len(data_dict['y_val']), len(classes)))
        for i, cls in enumerate(classes):
            y_val_onehot[:, i] = (data_dict['y_val'] == cls).astype(int)
            
        score = roc_auc_score(
            y_val_onehot,
            proba,
            multi_class="ovr", 
            average="micro"
        )
    
    return score, model

def evaluate_validation_set(data_dict, models_dict, target_info):
    """Calculate ROC AUC scores for validation data using same logic as evaluate_predictions"""
    scores = {}
    
    for target_name, meta in target_info.items():
        model = models_dict[target_name]['model']
        scaler = models_dict[target_name]['scaler']
        X_val_scaled = scaler.transform(data_dict['X_val'])
        proba = model.predict_proba(X_val_scaled)
        
        if meta["type"] == "bin":
            # 二元分類 - 轉換為 0/1
            true_vals = (data_dict['y_val'][target_name] == 1).astype(int)
            pred_vals = proba[:, 1]  # 使用正類的概率
            score = roc_auc_score(true_vals, pred_vals)
        else:
            # 多分類 - 使用 one-hot 編碼
            if target_name == "play years":
                true_vals = pd.get_dummies(data_dict['y_val'][target_name]).reindex(columns=[0,1,2], fill_value=0)
                pred_vals = pd.DataFrame(proba, columns=range(3))
                score = roc_auc_score(true_vals, pred_vals,
                                    multi_class="ovr", average="micro")
            elif target_name == "level":
                # ---------- 階層式：先 gender 再 level ----------
                # 1. 取三個 bundle
                gender_bdl  = models_dict["gender"]
                male_bdl    = models_dict["level_male"]
                female_bdl  = models_dict["level_female"]

                # 2. gender 預測
                X_val_scaled = gender_bdl["scaler"].transform(data_dict['X_val'])
                proba_g = gender_bdl["model"].predict_proba(X_val_scaled)
                pos_idx = np.where(gender_bdl["model"].classes_ == 1)[0][0]  # 1=男
                pred_male = (proba_g[:, pos_idx] >= 0.5)

                # 3. 準備 level 預測矩陣 (n_val, 4)
                proba_lv = np.zeros((len(X_val_scaled), 4))

                # ---- 男生 ----
                if pred_male.any():
                    idx_m = np.where(pred_male)[0]
                    X_m = male_bdl["scaler"].transform(data_dict['X_val'][idx_m])
                    p_m = male_bdl["model"].predict_proba(X_m)
                    proba_lv[idx_m] = aggregate_group_prob(p_m)[0]   # shape=(n_m,4)

                # ---- 女生 ----
                if (~pred_male).any():
                    idx_f = np.where(~pred_male)[0]
                    X_f = female_bdl["scaler"].transform(data_dict['X_val'][idx_f])
                    p_f = female_bdl["model"].predict_proba(X_f)
                    proba_lv[idx_f] = aggregate_group_prob(p_f)[0]

                # 4. 計算多分類 AUC
                true_vals = pd.get_dummies(data_dict['y_val'][target_name]).reindex(columns=[2,3,4,5], fill_value=0)
                pred_vals = pd.DataFrame(proba_lv, columns=[2,3,4,5])
                score = roc_auc_score(true_vals, pred_vals,
                                    multi_class="ovr", average="micro")

            
        scores[target_name] = score
        print(f"{target_name} ROC AUC: {score:.4f}")
        logging.info(f"{target_name} ROC AUC: {score:.4f}")
    
    # avg_score = np.mean(list(scores.values()))
    valid_scores = [s for s in scores.values() if not np.isnan(s)]
    avg_score = np.nan if len(valid_scores)==0 else float(np.nanmean(valid_scores))
    print(f"\nAverage ROC AUC: {avg_score:.4f}")
    logging.info(f"Average ROC AUC: {avg_score:.4f}")
    return scores, avg_score

def check_unique_id_overlap(train_file, val_file):
    """Check if any unique_id exists in both train and validation files."""
    with open(train_file, 'r') as f:
        train_ids = set(map(int, f.readlines()))

    with open(val_file, 'r') as f:
        val_ids = set(map(int, f.readlines()))

    overlap = train_ids.intersection(val_ids)
    if overlap:
        print("Data leakage detected! Overlapping unique_ids:", overlap)
        logging.error(f"Data leakage detected! Overlapping unique_ids: {overlap}")
    else:
        print("No data leakage detected.")
        logging.info("No data leakage detected.")

# Example usage
check_unique_id_overlap("train_ids_from_split.txt", "val_ids_from_split.txt")