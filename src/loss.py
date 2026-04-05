import torch
import torch.nn.functional as F

def smape(pred, target):
    return ((pred-target).abs()/((pred.abs()+target.abs())/2+1)).mean()

def pearson_loss(p, t):
    p_ = p - p.mean()
    t_ = t - t.mean()
    r = (p_*t_).sum()/(p_.norm()*t_.norm()+1e-8)
    return 1-r

def compute_loss(pred, target):
    return smape(pred, target) + 0.3*pearson_loss(pred, target)
