import numpy as np
import os

def load_month(root, month, feats):
    arrs = []
    for f in feats:
        a = np.load(os.path.join(root, month, f"{f}.npy")).astype(np.float32)
        if a.ndim == 2:
            a = a[np.newaxis]
        arrs.append(a)
    T_min = min(a.shape[0] for a in arrs)
    return np.stack([a[:T_min] for a in arrs], axis=1)

def apply_log_transform(arr):
    PM_CHANNELS = [0, 9, 10, 11, 12, 13, 14, 15]
    arr[:, PM_CHANNELS] = np.log1p(arr[:, PM_CHANNELS])
    return arr
