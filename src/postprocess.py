import numpy as np
from scipy.ndimage import gaussian_filter

def smooth(pred):
    out = np.empty_like(pred)
    for i in range(pred.shape[0]):
        for t in range(pred.shape[1]):
            out[i,t] = gaussian_filter(pred[i,t], sigma=0.5)
    return out
