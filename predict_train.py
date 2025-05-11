import logging
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from feature_utils import aggregate_group_prob
from model_utils import TARGETS, load_model
from train_val_utils import train_validate_split
from sklearn.metrics import roc_auc_score

logging.basicConfig(
    filename="log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
# Log the start of the script

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)

def evaluate_predictions(pred_file: str, info: pd.DataFrame):
    pred = pd.read_csv(pred_file)
    data = pd.merge(info, pred, on="unique_id", suffixes=("_true", "_pred"))
    # 去除重覆預測，避免 merge 後行數膨脹
    data = data.drop_duplicates(subset="unique_id")

    # 二元任務
    gender_true = (data["gender_true"] == 1).astype(int)
    hold_true = (data["hold racket handed_true"] == 1).astype(int)

    gender_auc = roc_auc_score(gender_true, data["gender_pred"])
    hold_auc = roc_auc_score(hold_true, data["hold racket handed_pred"])

    # 多元任務
    play_year_true = pd.get_dummies(data["play years"])
    play_year_pred = data[[f"play years_{i}" for i in range(3)]]
    play_year_auc = roc_auc_score(
        play_year_true, play_year_pred,
        multi_class="ovr", average="micro"
    )

    level_true = (
    pd.get_dummies(data["level"])
      .reindex(columns=[2, 3, 4, 5], fill_value=0)
    )
    level_pred = data[[f"level_{i}" for i in [2, 3, 4, 5]]]
    level_auc = roc_auc_score(
        level_true, level_pred,
        multi_class="ovr", average="micro"
    )

    final_score = (gender_auc + hold_auc + play_year_auc + level_auc) / 4
    print(f"Gender ROC AUC       : {gender_auc:.4f}")
    print(f"Hold Racket ROC AUC  : {hold_auc:.4f}")
    print(f"Play Years ROC AUC   : {play_year_auc:.4f}")
    print(f"Level ROC AUC        : {level_auc:.4f}")
    print(f"Final Score          : {final_score:.4f}")
    return {
        'gender': gender_auc,
        'hold': hold_auc,
        'play_years': play_year_auc,
        'level': level_auc,
        'final': final_score
    }

def predict_train():
    # 讀取原始數據和特徵
    info = pd.read_csv("train_info.csv")
    # ─── 讀取 prepare_train() 儲存的固定驗證 UID ───
    VAL_UID_FILE = "split_uid_val.npy"
    if not Path(VAL_UID_FILE).exists():
        raise FileNotFoundError("請先執行 `python main.py train` 產生 split_uid_val.npy")

    val_uid_set = set(np.load(VAL_UID_FILE))

    all_features = []
    uid_idx = []
    
    for p in sorted(Path("tabular_data_train").glob("*.csv")):
        uid = int(p.stem)
        df = pd.read_csv(p)
        all_features.append(df.values)
        uid_idx.extend([uid] * len(df))
    
    X = np.vstack(all_features)
    uid_idx = np.array(uid_idx)
    groups = info.set_index("unique_id").loc[uid_idx, "player_id"].values
    
    # 分割驗證集
    val_mask = np.isin(uid_idx, list(val_uid_set))

    # 只預測驗證集數據
    val_uids = np.unique(uid_idx[val_mask])
    sub_rows = []
    
    for uid in val_uids:
        idx = np.where(uid_idx == uid)[0]
        X_current = X[idx]
        row = {"unique_id": uid}

        for col, meta in TARGETS.items():
            bundle = load_model(col)
            scaler = bundle["scaler"]
            model = bundle["model"]

            X_scaled = scaler.transform(X_current)
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
                "play years_0", "play years_1", "play years_2",
                "level_2", "level_3", "level_4", "level_5"]
    
    submission = pd.DataFrame(sub_rows)[sub_cols]
    submission.to_csv("val_pred.csv", index=False, float_format="%.8f")
    print("✅  val_pred.csv ready!")
    
    # 評估驗證集分數
    val_info = info[info['unique_id'].isin(val_uids)]
    scores = evaluate_predictions("val_pred.csv", val_info)
    return scores

if __name__ == '__main__':
    logging.info("Predict validation data started.")
    scores = predict_train()
    logging.info("Predict validation data finished successfully.")