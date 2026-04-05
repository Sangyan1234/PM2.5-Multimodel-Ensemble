import torch
from torch.optim import AdamW
from tqdm import tqdm
from src.loss import compute_loss

def train_model(model, loader, epochs=10, lr=2e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    opt = AdamW(model.parameters(), lr=lr)

    for ep in range(epochs):
        model.train()
        total = 0

        for x, y, lp in tqdm(loader):
            x, y, lp = x.to(device), y.to(device), lp.to(device)

            opt.zero_grad()
            pred = model(x, lp)
            loss = compute_loss(pred, y)
            loss.backward()
            opt.step()

            total += loss.item()

        print(f"Epoch {ep+1} Loss {total/len(loader):.4f}")
