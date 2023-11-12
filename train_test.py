import torch
from utils import CoxLoss
from sksurv.metrics import concordance_index_censored

def train_step(model, optim, dataloader, device):
    model.train()
    C_tot = 0
    loss_tot = 0
    for X, e, t, in dataloader:
        X = X.to(device)
        e = e.to(device)
        t = t.to(device)

        h = model(X)

        loss = CoxLoss(t, e, h, device)
        loss_tot += loss
        C, _, _, _, _ = concordance_index_censored(e, t, h.detach().squeeze().numpy())
        C_tot += C

        optim.zero_grad()
        loss.backward()
        optim.step()

    return (loss_tot / len(dataloader)), (C_tot / len(dataloader))

def test_step(model, dataloader, device):
    model.eval()
    C_tot = 0
    loss_tot = 0
    for X, e, t in dataloader:
        X = X.to(device)
        e = e.to(device)
        t = t.to(device)

        h = model(X)

        loss = CoxLoss(t, e, h, device)
        loss_tot += loss
        C, _, _, _, _ = concordance_index_censored(e, t, h.detach().squeeze().numpy())
        C_tot += C

    return (loss_tot / len(dataloader)), (C_tot / len(dataloader))
    


            