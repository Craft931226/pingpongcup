# ──────────────────  smoke_gpu_check.py  ──────────────────
import time
import numpy as np
import lightgbm as lgb

def gpu_smoke_test():
    """回傳訓練耗時（秒）；低於 2 秒代表 GPU 有啟用。"""
    # 1. 隨機生成小型資料集
    X = np.random.rand(10_000, 10).astype(np.float32)
    y = (np.random.rand(10_000) > 0.5).astype(int)

    # 2. 指定 device_type='gpu'
    params = {
        "objective":      "binary",
        "metric":         "auc",
        "device_type":    "gpu",     # ← 關鍵：要求用 GPU
        "gpu_platform_id": 0,        # 視需要調整
        "gpu_device_id":   0,
        "verbosity":     -1,
    }

    dtrain = lgb.Dataset(X, label=y)

    t0 = time.time()
    lgb.train(params, dtrain, num_boost_round=20)   # 訓練 20 棵樹即可
    return time.time() - t0

if __name__ == "__main__":
    elapsed = gpu_smoke_test()
    print(f"⏱  Elapsed time : {elapsed:.3f} s")

    if elapsed < 2:
        print("✅  檢測通過：GPU 已啟用！")
    else:
        print("❌  檢測失敗：可能未使用 GPU，請檢查驅動或 LightGBM 安裝方式。")
# ───────────────────────────────────────────────────────────
