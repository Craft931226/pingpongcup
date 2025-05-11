import numpy as np, lightgbm as lgb
from lightgbm.basic import LightGBMError
try:
    lgb.train({"objective":"binary",
               "device_type":"gpu",
               "verbosity":-1,
               "num_iterations":1},
              lgb.Dataset(np.random.rand(50,3), label=np.random.randint(0,2,50)))
    print("✅ GPU OK")
except LightGBMError as e:
    print("❌ GPU still fails ->", e)
