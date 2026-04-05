def smape(pred, target):
    return ((pred - target).abs() / ((pred.abs()+target.abs())/2 + 1)).mean()
