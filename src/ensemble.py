import numpy as np

def weighted_ensemble(a, b, c, wa=0.4, wb=0.3, wc=0.3):
    return (wa*a + wb*b + wc*c) / (wa+wb+wc)
