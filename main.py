import argparse, pandas as pd, numpy as np, lightgbm as lgb
from pathlib import Path

from sklearn.metrics import roc_auc_score
from feature_utils import generate_features, aggregate_group_prob
from model_utils import TARGETS, build_scaler, build_model
from model_utils import cv_evaluate, save_model, load_model
from train_val_utils import max_feasible_splits, oof_training
from sklearn.model_selection import GroupShuffleSplit 
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
    generate_features("./train_data", "train_info.csv", "tabular_data_train")

    # 2. 讀取 info & 特徵
    info = pd.read_csv("train_info.csv")
    X, uid_idx = load_features("tabular_data_train")
    groups = info.set_index("unique_id").loc[uid_idx, "player_id"].values
    # === ① 先把 10 % player 做 hold-out =========================
    gss = GroupShuffleSplit(test_size=0.10, random_state=42)
    tr_idx, ho_idx = next(gss.split(X, groups=groups))

    X_tr, X_ho       = X[tr_idx],  X[ho_idx]
    uid_tr, uid_ho   = uid_idx[tr_idx], uid_idx[ho_idx]
    groups_tr        = groups[tr_idx]
    # ============================================================


    # 3. 數據標準化
    scaler = build_scaler(X_tr)          # 只 fit 90 % training
    X_tr_scaled = scaler.transform(X_tr)
    X_ho_scaled = scaler.transform(X_ho) # 之後評估 hold-out 用

    
    # 4. 逐 target 做 OOF 訓練 ─────────────────────────
    # ==== 取代舊的 for col, meta in TARGETS.items(): ====
    oof_dict, best_iter_dict = {}, {}
    for col, meta in TARGETS.items():
        print(f"\n▶ OOF training for {col}")
        logging.info(f"▶ OOF training for {col}")
        y_tr_raw = info.set_index("unique_id").loc[uid_tr, col].values
        if col == "level":
            # 把 4、5 併到 4，共 0~4 五類
            y_tr = y_tr_raw
            meta["num_class"] = 4            # ← 5 類 (0–4)
        else:
            y_tr = y_tr_raw

        n_fold = max_feasible_splits(y_tr, groups_tr)
        print(f"  → 使用 {n_fold}-fold OOF（level 稀有類別限制）")
        logging.info(f"  → 使用 {n_fold}-fold OOF（level 稀有類別限制）")
        if n_fold == 1:
            oof = np.zeros((len(X_tr), meta["num_class"]))*np.nan
            best_iters = [200]
        else:
            # 5. OOF 訓練 ───────────────────────────────
            oof, best_iters = oof_training(
                X_tr_scaled, y_tr, groups_tr,
                meta, build_model,
                n_splits=n_fold, early_stopping_rounds=30
            )

        best_iter = max(1, int(np.mean(best_iters)))
        best_iter_dict[col] = best_iter
        # -------- OOF AUC：只有 n_fold>=2 才計算 --------
        if np.isnan(oof).any():
            auc = np.nan
        else:
            if meta["type"] == "bin":
                auc = roc_auc_score(y_tr, oof[:, 1])
            else:
                y_onehot = (
                    pd.get_dummies(y_tr)
                      .reindex(columns=range(meta["num_class"]), fill_value=0)
                )
                auc = roc_auc_score(y_onehot, oof, multi_class="ovr", average="micro")
        print(f"OOF AUC = {auc:.4f}")
        logging.info(f"OOF AUC = {auc:.4f}")

        # 用平均迭代數重訓全資料並保存
        final_mdl = build_model(y_tr, meta)
        final_mdl.set_params(n_estimators=best_iter)
        final_mdl.fit(X_tr_scaled, y_tr)
        save_model(final_mdl, scaler, col)
        if col == "level":
            # 以 gender 分兩批再各自重訓
            gender_tr = info.set_index("unique_id").loc[uid_tr, "gender"].values
            male_mask = gender_tr == 1        # 1=male, 0=female

            for sex_label, sex_name in [(1, "male"), (0, "female")]:
                mask   = male_mask if sex_label == 1 else ~male_mask
                if mask.sum() == 0:
                    continue       # 若剛好沒樣本就跳過

                lvl_mdl = build_model(y_tr[mask], meta)
                lvl_mdl.set_params(n_estimators=best_iter)
                lvl_mdl.fit(X_tr_scaled[mask], y_tr[mask])
                save_model(lvl_mdl, scaler, f"level_{sex_name}")

        # --- hold-out 評分 -----------------------------
        y_ho_raw = info.set_index("unique_id").loc[uid_ho, col].values
        y_ho     = y_ho_raw
        proba_ho = final_mdl.predict_proba(X_ho_scaled)

        # ---------- 先依 27 段聚合到 unique_id ---------- 
        proba_uid = aggregate_group_prob(proba_ho)[0]      # (N_uid, n_class 或 2)

        # 對應的 unique_id ⇒ 只取前 N_uid 個即可
        uid_unique = np.unique(uid_ho)[: len(proba_uid)]

        # 取對應 label（level 需再把 4/5 併 4）
        y_uid = info.set_index("unique_id").loc[uid_unique, col].values
            
        # ---------- 計 AUC ----------
        if meta["type"] == "bin":
            # ---------- 二元 ----------
            if (proba_uid.ndim == 1) or (len(np.unique(y_uid)) < 2):
                auc_ho = np.nan           # 樣本或類別不足，跳過
            else:
                pos_prob = proba_uid[:, 1]
                auc_ho = roc_auc_score((y_uid == 1).astype(int), pos_prob)

        else:
            # ---------- 多類 ----------
            # 保證 2D
            if proba_uid.ndim == 1:
                proba_uid = proba_uid.reshape(1, -1)
                y_uid     = y_uid.reshape(-1)

            # 樣本或類別不足就略過
            if (proba_uid.shape[0] < 2) or (len(np.unique(y_uid)) < 2):
                auc_ho = np.nan
            else:
                y_one = (pd.get_dummies(y_uid)
                        .reindex(columns=range(meta["num_class"]), fill_value=0)
                        .to_numpy())
                auc_ho = roc_auc_score(
                    y_one, proba_uid, multi_class="ovr", average="micro"
                )
        print(f"[{col}] hold-out AUC = {auc_ho}")
        logging.info(f"[{col}] hold-out AUC = {auc_ho}")


    print("\n✅ OOF training finished. All models saved to ./models/")
    np.save("split_uid_val.npy", np.unique(uid_ho))

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
                grp = aggregate_group_prob(proba, strategy="max")[0]

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
                grp = aggregate_group_prob(proba, strategy="max")[0]
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
            grp = aggregate_group_prob(proba, strategy="max")[0]

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
