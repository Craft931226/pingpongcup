import argparse, pandas as pd, numpy as np, lightgbm as lgb
from pathlib import Path
from feature_utils import generate_features, aggregate_group_prob
from model_utils import TARGETS, build_scaler, build_model
from model_utils import cv_evaluate, save_model, load_model
from train_val_utils import train_validate_split, evaluate_model, evaluate_validation_set
import warnings
import logging
from sklearn.base import clone
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename="log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Log the start of the script
logging.info("Script started.")

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)

def load_features(feat_dir):
    X, uid_idx = [], []
    for p in Path(feat_dir).glob("*.csv"):
        df = pd.read_csv(p)
        X.append(df.values)         # (n_swing, feat_dim)
        uid_idx.extend([int(p.stem)] * len(df))
    return np.vstack(X), np.array(uid_idx)

def prepare_train():
    # 1. 產生特徵
    # generate_features("./train_data", "train_info.csv", "tabular_data_train")

    # 2. 讀取 info & 特徵
    info = pd.read_csv("train_info.csv")
    X, uid_idx = load_features("tabular_data_train")
    groups = info.set_index("unique_id").loc[uid_idx, "player_id"].values


    # 3. 數據標準化
    # scaler = build_scaler(X)
    # X_scaled = scaler.transform(X)
        # --- 重新分割：同時照顧分組 & 二元標籤平衡 -----------------
    from sklearn.model_selection import GroupShuffleSplit

    y_gender_full = info.set_index("unique_id").loc[uid_idx, "gender"].values
    y_hold_full   = info.set_index("unique_id").loc[uid_idx, "hold racket handed"].values

    for seed in range(50):         # 最多嘗試 50 次
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42 + seed)
        train_idx_glb, val_idx_glb = next(gss.split(X, groups=groups))

        if (len(np.unique(y_gender_full[val_idx_glb])) == 2 and
            len(np.unique(y_hold_full[val_idx_glb]))   == 2):
            break
    else:
        raise RuntimeError("Could not build a balanced group split.")

    val_mask_global = np.zeros(len(groups), dtype=bool)
    val_mask_global[val_idx_glb] = True
        # --- 只用訓練資料做 scaler，避免 data-leakage -----------
    scaler = build_scaler(X[train_idx_glb])
    X_scaled = scaler.transform(X)

    X_train_global      = X_scaled[train_idx_glb]
    X_val_global        = X_scaled[val_idx_glb]
    groups_train_global = groups[train_idx_glb]
    groups_val_global   = groups[val_idx_glb]


    # 儲存訓練/驗證集的分割
    all_targets = {}
    holdout_data = None
    
    # 4. 對每個 target 建模
    for col, meta in TARGETS.items():
        print(f"\nTraining for {col}:")
        y = info.set_index("unique_id").loc[uid_idx, col].values
        
        # 拆分訓練集和驗證集
        data_dict = {
            'X_train': X_train_global,
            'y_train': y[~val_mask_global],
            'groups_train': groups_train_global,
            'X_val':   X_val_global,
            'y_val':   y[val_mask_global],
            'groups_val': groups_val_global
        }
        if holdout_data is None:
            holdout_data = {
                'X_val': X_val_global,
                'X_train': X_train_global,
                'y_val': {},
                'y_train': {},
                'groups_val': groups_val_global
            }

        
        holdout_data['y_val'][col] = data_dict['y_val']
        holdout_data['y_train'][col] = data_dict['y_train']
        
        # # 訓練和驗證
        # mdl = build_model(y, meta)
        # val_score, trained_model = evaluate_model(mdl, data_dict, meta)
        # print(f"Training Score for {col}: {val_score:.4f}")
        # save_model(trained_model, scaler, col)
        # all_targets[col] = {'model': trained_model, 'scaler': scaler}
            # ⬇︎ 新增 ▼
        if col == "level":
            # ❶ 依性別分割索引（1: 男, 2: 女）
            male_idx   = np.where(y_gender_full == 1)[0]
            female_idx = np.where(y_gender_full != 1)[0]

            # ❷ 建立基底模型並 clone 出兩份
            base_mdl = build_model(y, meta)
            level_model_male   = clone(base_mdl)
            level_model_female = clone(base_mdl)

            # ❸ 分別訓練
            level_model_male.fit(X_scaled[male_idx],   y[male_idx])
            level_model_female.fit(X_scaled[female_idx], y[female_idx])

            # ❹ 儲存
            save_model(level_model_male,   scaler, "level_male")
            save_model(level_model_female, scaler, "level_female")
            all_targets["level_male"]   = {"model": level_model_male,   "scaler": scaler}
            all_targets["level_female"] = {"model": level_model_female, "scaler": scaler}

            # 為了不破壞舊的驗證流程，仍留一個占位
            all_targets["level"] = all_targets["level_male"]
        else:
            # 其他 target 沿用舊流程
            mdl = build_model(y, meta)
            val_score, trained_model = evaluate_model(mdl, data_dict, meta)
            print(f"Training Score for {col}: {val_score:.4f}")
            save_model(trained_model, scaler, col)
            all_targets[col] = {'model': trained_model, 'scaler': scaler}

    print("\nEvaluating validation set:")
    scores, avg_score = evaluate_validation_set(holdout_data, all_targets, TARGETS)
    np.save("split_uid_val.npy", np.unique(uid_idx[np.isin(groups, holdout_data['groups_val'])]))
    print("\n✅ Models saved to ./models/")

def predict_test():
    # 1. 產生特徵
    generate_features("./test_data", "test_info.csv", "tabular_data_test")

    # 2. 預先加載所有模型
    models = {col: load_model(col) for col in TARGETS.keys()}
    models["level_male"] = load_model("level_male")
    models["level_female"] = load_model("level_female")
    # 3. 批量讀取所有測試數據
    test_files = sorted(Path("tabular_data_test").glob("*.csv"))
    all_uids = []
    all_features = []
    
    for p in test_files:
        uid = int(p.stem)
        df = pd.read_csv(p)
        all_uids.append(uid)
        all_features.append(df.values)
    
    sub_rows = []
    for idx, uid in enumerate(all_uids):
        X = all_features[idx]
        pred_gender_cls = None # 0/1 尚未決定
        row = {"unique_id": uid}

        for col, meta in TARGETS.items():
            if col == "gender":
                bundle = models[col]
                scaler = bundle["scaler"]
                model  = bundle["model"]

                X_scaled = scaler.transform(X)
                proba = model.predict_proba(X_scaled)
                grp = aggregate_group_prob(proba)[0]

                pos_idx = np.where(model.classes_ == 1)[0][0]
                row[col] = grp[pos_idx]           # 機率
                pred_gender_cls = 1 if row[col] >= 0.5 else 0  # 1=男, 0=女
                continue
            elif col == "level":
                # 根據上一步性別選模型
                bundle = models["level_male"]   if pred_gender_cls == 1 else models["level_female"]
                scaler = bundle["scaler"]
                model  = bundle["model"]
                X_scaled = scaler.transform(X)
                proba = model.predict_proba(X_scaled)
                grp = aggregate_group_prob(proba)[0]
                needed_labels = [2, 3, 4, 5]
                for lbl in needed_labels:
                    row[f"level_{lbl}"] = 0.0
                for idx_lbl, lbl in enumerate(bundle["model"].classes_):
                    row[f"level_{lbl}"] = grp[idx_lbl]
                continue   # ← 直接跳到下一個 target，避免落入通用邏輯
                
            bundle = models[col]
            scaler = bundle["scaler"]
            model = bundle["model"]

            X_scaled = scaler.transform(X)
            proba = model.predict_proba(X_scaled)
            grp = aggregate_group_prob(proba)[0]

            if meta["type"] == "bin":
                pos_idx = np.where(model.classes_ == 1)[0][0]
                row[col] = grp[pos_idx]
                continue

            needed_labels = [0, 1, 2] if col == "play years" else [2, 3, 4, 5]
            for lbl in needed_labels:
                row[f"{col}_{lbl}"] = 0.0

            for idx, lbl in enumerate(model.classes_):
                row[f"{col}_{lbl}"] = grp[idx]

        sub_rows.append(row)

    sub_cols = ["unique_id", "gender", "hold racket handed",
                "play years_0","play years_1","play years_2",
                "level_2","level_3","level_4","level_5"]
    df_temp = pd.DataFrame(sub_rows)
    df_temp = df_temp.reindex(columns=sub_cols, fill_value=0.0)
    submission = df_temp[sub_cols]
    # 使用 DataFrame 批量處理
    submission.to_csv("submission.csv", index=False, float_format="%.8f")
    print("✅  submission.csv ready!")

if __name__ == "__main__": 
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train","predict"])
    args = ap.parse_args()

    try:
        if args.mode == "train":
            prepare_train()
        else:
            predict_test()
        logging.info("Script finished successfully.")
    except Exception as e:
        logging.error(f"Script encountered an error: {e}")
        raise
