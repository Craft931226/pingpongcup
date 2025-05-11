import csv
import logging
import math
from pathlib import Path
import numpy as np
from scipy.fft import rfft
import scipy.stats
import pandas as pd
from sklearn.metrics import roc_auc_score
logging.basicConfig(
    filename="log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
EXPECTED_DIM = 131  # Expected dimension of the feature vector

def aggregate_group_prob(proba_mat: np.ndarray, group_size: int = 27,
                         strategy: str = "mean") -> np.ndarray:
    if proba_mat.size == 0:
        return np.array([[0]])  # Return default value for empty input
        
    if len(proba_mat) < group_size:
        # If we have less samples than group_size, just average what we have
        return np.array([np.mean(proba_mat, axis=0)])
    
    num_groups = len(proba_mat) // group_size
    if num_groups == 0:
        # Handle case where we have some samples but less than group_size
        return np.array([np.mean(proba_mat, axis=0)])
        
    # Handle normal case
    proba_mat = proba_mat[:num_groups * group_size]  # Only keep complete groups
    proba_mat = proba_mat.reshape(num_groups, group_size, -1)
    
    if strategy == "mean":
        return proba_mat.mean(axis=1)  # (N_uid, n_class)
    else:
        return proba_mat.max(axis=1)
    
def FFT_data(input_data, swinging_times):   
    txtlength = swinging_times[-1] - swinging_times[0]
    a_mean = [0] * txtlength
    g_mean = [0] * txtlength
       
    for num in range(len(swinging_times)-1):
        a = []
        g = []
        for swing in range(swinging_times[num], swinging_times[num+1]):
            a.append(math.sqrt(math.pow((input_data[swing][0] + input_data[swing][1] + input_data[swing][2]), 2)))
            g.append(math.sqrt(math.pow((input_data[swing][3] + input_data[swing][4] + input_data[swing][5]), 2)))

        a_mean[num] = (sum(a) / len(a))
        g_mean[num] = (sum(g) / len(g))
    
    return a_mean, g_mean

def feature(input_data, swinging_now, swinging_times, n_fft, a_fft, g_fft, a_fft_imag, g_fft_imag, writer):
    # Convert input data to numpy array
    arr = np.array(input_data)
    
    if swinging_times == 0:  # Handle case where there are no swings
        swinging_times = 1
        
    # Calculate frequency domain features
    cut = int(n_fft / swinging_times)
    idx_start = cut * swinging_now
    idx_end = min(cut * (swinging_now + 1), len(a_fft))  # Ensure we don't go out of bounds
    
    # Rest of the function remains the same...
    # Calculate acceleration and gyroscope vectors
    a_vec = np.sqrt(np.sum(arr[:, :3]**2, axis=1))
    g_vec = np.sqrt(np.sum(arr[:, 3:6]**2, axis=1))
    
    a_stats = [a_vec.max(), a_vec.mean(), a_vec.min()]
    g_stats = [g_vec.max(), g_vec.mean(), g_vec.min()]
    
    # Calculate kurtosis and skewness
    a_centered = a_vec - a_vec.mean()
    g_centered = g_vec - g_vec.mean()
    
    a_moments = {
        'skew': np.mean(a_centered**3) / (np.std(a_centered)**3),
        'kurt': np.mean(a_centered**4) / (np.var(a_centered)**2)
    }
    g_moments = {
        'skew': np.mean(g_centered**3) / (np.std(g_centered)**3),
        'kurt': np.mean(g_centered**4) / (np.var(g_centered)**2)
    }
    
    # Get FFT slices safely
    a_fft_slice = a_fft[idx_start:idx_end]
    g_fft_slice = g_fft[idx_start:idx_end]
    a_fft_imag_slice = a_fft_imag[idx_start:idx_end]
    g_fft_imag_slice = g_fft_imag[idx_start:idx_end]
    
    # Handle empty slices
    if len(a_fft_slice) == 0:
        a_fft_slice = np.array([0])
        g_fft_slice = np.array([0])
        a_fft_imag_slice = np.array([0])
        g_fft_imag_slice = np.array([0])
    
    # Calculate PSD using vectorized operations
    a_psd = np.power(a_fft_slice, 2) + np.power(a_fft_imag_slice, 2)
    g_psd = np.power(g_fft_slice, 2) + np.power(g_fft_imag_slice, 2)
    
    # Calculate entropy with safety checks
    e1 = np.sqrt(a_psd)
    e3 = np.sqrt(g_psd)
    e2 = np.sum(e1) + 1e-10  # Avoid division by zero
    e4 = np.sum(e3) + 1e-10
    
    p_a = e1 / e2
    p_g = e3 / e4
    entropy_a = np.sum(p_a * np.log(p_a + 1e-10)) / max(cut, 1)
    entropy_g = np.sum(p_g * np.log(p_g + 1e-10)) / max(cut, 1)
    
    # Calculate basic statistics
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    rms = np.sqrt(np.mean(arr**2, axis=0))
    
    # Combine all features
    output = np.concatenate([
        mean, std, rms,
        a_stats, g_stats,
        [np.mean(a_fft_slice), np.mean(g_fft_slice)],
        [np.mean(a_psd), np.mean(g_psd)],
        [a_moments['kurt'], g_moments['kurt']],
        [a_moments['skew'], g_moments['skew']],
        [entropy_a, entropy_g]
    ]).tolist()
    
    writer.writerow(output)

def extract_features(df):
    features = []
    
    # 速度相關特徵
    speed_cols = ['Ax', 'Ay', 'Az']
    speeds = df[speed_cols].values
    
    # 基礎統計特徵
    features.extend([
        np.mean(speeds, axis=0),
        np.std(speeds, axis=0),
        np.max(speeds, axis=0),
        np.min(speeds, axis=0),
        np.percentile(speeds, 25, axis=0),
        np.percentile(speeds, 75, axis=0),
        scipy.stats.skew(speeds, axis=0),
        scipy.stats.kurtosis(speeds, axis=0)
    ])
    # --- Δ / Δ² 特徵：捕捉加速度變化趨勢 ---
    delta1 = np.diff(speeds, axis=0, n=1)   # 一階差分
    delta2 = np.diff(speeds, axis=0, n=2)   # 二階差分
    if delta1.size == 0:
        delta1 = np.zeros((1, 3))
    if delta2.size == 0:
        delta2 = np.zeros((1, 3))
    features.extend([
        np.mean(delta1, axis=0), np.std(delta1, axis=0),
        np.mean(delta2, axis=0), np.std(delta2, axis=0),
    ])
    # 速度交互特徵
    speed_means = np.mean(speeds, axis=0)
    features.extend([
        speed_means[0] * speed_means[1],  # x * y
        speed_means[1] * speed_means[2],  # y * z
        speed_means[0] * speed_means[2]   # x * z
    ])
    
    # 加速度特徵
    acc = np.diff(speeds, axis=0)
    features.extend([
        np.mean(acc, axis=0),
        np.std(acc, axis=0),
        np.max(acc, axis=0),
        np.min(acc, axis=0)
    ])
    
    # 位置相關特徵
    pos_cols   = ['Gx', 'Gy', 'Gz']
    positions = df[pos_cols].values
    
    # 位置統計特徵
    features.extend([
        np.mean(positions, axis=0),
        np.std(positions, axis=0),
        np.max(positions, axis=0),
        np.min(positions, axis=0),
        np.percentile(positions, 25, axis=0),
        np.percentile(positions, 75, axis=0)
    ])
    
    # 位置變化特徵
    pos_diff = np.diff(positions, axis=0)
    features.extend([
        np.mean(pos_diff, axis=0),
        np.std(pos_diff, axis=0)
    ])
    
    # 時序特徵
    for col in speed_cols + pos_cols:
        #
        ts = df[col].values
        # FFT特徵
        n = len(ts)
        window = np.hanning(n)                    # 1. 建立窗函數
        ts_windowed = ts * window                 # 2. 乘上窗
        n_fft = 1 << (n - 1).bit_length()         # 3. 計算下個 2 的冪次
        ts_padded = np.pad(ts_windowed,           # 4. 以零補齊
                        (0, n_fft - n),
                        mode='constant')
        fft_features = np.abs(np.fft.fft(ts_padded))[:5]
        features.extend(fft_features)
        
        # 自相關特徵
        acf = np.correlate(ts, ts, mode='full') / len(ts)
        features.extend(acf[len(acf)//2:len(acf)//2+3])  # 取中心點後3個值
    
    # 三維向量特徵
    speed_magnitudes = np.linalg.norm(speeds, axis=1)
    pos_magnitudes = np.linalg.norm(positions, axis=1)
    
    features.extend([
        np.mean(speed_magnitudes),
        np.std(speed_magnitudes),
        np.max(speed_magnitudes),
        np.min(speed_magnitudes),
        np.mean(pos_magnitudes),
        np.std(pos_magnitudes),
        np.max(pos_magnitudes),
        np.min(pos_magnitudes)
    ])
    
    # --- 把 features 轉成 1-D list ---
    flat = []
    for f in features:
        if isinstance(f, np.ndarray):
            flat.extend(f.ravel())        # array(3,) → 3 個值
        else:                             # Python float / int / scalar
            flat.append(float(f))

    flat = np.asarray(flat, dtype=np.float32)

    # ---- 保證長度一致：不足補 0，多餘截斷 ----
    if flat.size < EXPECTED_DIM:
        flat = np.pad(flat, (0, EXPECTED_DIM - flat.size))
    elif flat.size > EXPECTED_DIM:
        flat = flat[:EXPECTED_DIM]

    return flat



def generate_features(raw_dir: str, info_csv:str,  out_dir: str):
    # 讀取 cut_point；用 unique_id 當 index 方便隨查
    info_df = pd.read_csv(info_csv).set_index("unique_id")

    Path(out_dir).mkdir(exist_ok=True)
    pathlist_txt = Path(raw_dir).glob('*.txt')

    for file in pathlist_txt:
        # 讀取文件
        data = []
        with open(file, 'r') as f:
            for line in f.readlines()[1:]:  # Skip header
                if line.strip():  # Skip empty lines
                    values = line.strip().split()
                    if len(values) >= 6:
                        # 只取前6個值，並轉換為整數
                        row = [int(x) for x in values[:6]]
                        data.append(row)

        if not data:
            print(f"Warning: No valid data found in {file}")
            logging.warning(f"No valid data found in {file}")
            continue

        # 創建DataFrame並命名列
        df = pd.DataFrame(data, columns=[
            'Ax', 'Ay', 'Az',    # Accelerometer
            'Gx', 'Gy', 'Gz'     # Gyroscope
        ])
        
        try:
            # --- 依 cut_point 精準切 27 段；缺切點則回退等分 ---
            uid        = int(file.stem)
            try:
               cuts_raw = info_df.loc[uid, "cut_point"]          # "[0 35 ... 998]"
               cuts     = np.fromstring(cuts_raw.strip("[]"), sep=" ", dtype=int)
               assert len(cuts) == 28
               segments  = [df.iloc[cuts[i]:cuts[i+1]] for i in range(27)]
            except Exception as _:
               # 若該檔案沒有 cut_point 或解析失敗 → 回退均分
               idx_splits = np.array_split(np.arange(len(df)), 27)
               segments   = [df.iloc[idx] for idx in idx_splits]
            seg_feats = [extract_features(seg) for seg in segments]

            # 🟢 確認所有段落向量長度一致，否則捨棄異常段
            base_len = len(seg_feats[0])
            seg_feats = [v for v in seg_feats if len(v) == base_len]
            if len(seg_feats) == 0:                 # 全部失敗 → 填 0
                seg_feats = [np.zeros(base_len)]
            # 這裡示範取平均；想取最大值可改 np.max(seg_feats, axis=0)
            features  = np.mean(seg_feats, axis=0)
            
            # 保存特徵
            features_df = pd.DataFrame([features])
            output_path = Path(out_dir) / f"{file.stem}.csv"
            features_df.to_csv(output_path, index=False)
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            logging.error(f"Error processing {file}: {str(e)}")
            continue

def evaluate_predictions(pred_file: str, info_file: str):
    """Evaluate predictions against true values"""
    pred = pd.read_csv(pred_file)
    info = pd.read_csv(info_file)

    # Filter info to include only rows matching unique_id in pred
    info = info[info['unique_id'].isin(pred['unique_id'])]

    # Sort and align data by unique_id
    pred = pred.sort_values('unique_id').reset_index(drop=True)
    info = info.sort_values('unique_id').reset_index(drop=True)

    # Ensure alignment of unique_ids
    if not np.array_equal(pred['unique_id'].values, info['unique_id'].values):
        raise ValueError("Prediction and info DataFrames have mismatched unique_ids")

    # Binary tasks
    gender_true = (info["gender"] == 1).astype(int)
    hold_true = (info["hold racket handed"] == 1).astype(int)

    gender_auc = roc_auc_score(gender_true, pred["gender"])
    hold_auc = roc_auc_score(hold_true, pred["hold racket handed"])

    # Multiclass tasks - play years
    play_years_cols = [f"play years_{i}" for i in range(3)]
    play_year_true = pd.get_dummies(info["play years"]).reindex(columns=range(3), fill_value=0)
    play_year_pred = pred[play_years_cols]
    play_year_auc = roc_auc_score(
        play_year_true, play_year_pred,
        multi_class="ovr", average="micro"
    )

    # Multiclass tasks - level
    level_cols = [f"level_{i}" for i in [2, 3, 4, 5]]
    level_true = pd.get_dummies(info["level"]).reindex(columns=[2, 3, 4, 5], fill_value=0)
    level_pred = pred[level_cols]
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
    
if __name__ == '__main__':
    evaluate_predictions("val_pred.csv", "train_info.csv")
