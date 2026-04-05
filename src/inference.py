import torch
import numpy as np

def infer(model, data, lp):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    preds = []

    with torch.no_grad():
        for i in range(len(data)):
            x = torch.tensor(data[i]).unsqueeze(0).to(device)
            l = torch.tensor(lp[i]).unsqueeze(0).to(device)

            p = model(x, l).cpu().numpy()
            preds.append(p)

    return np.concatenate(preds)
