import torch
import numpy as np
from utils import CoxLoss
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc
from lifelines.utils import concordance_index

def train_step(model, optim, dataloader, device):
    '''
    Args:
        cens_info - Structured array of survival times and censor info for c_indx calc
            Expects tuple: (train_info, test_info)
    '''
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

        optim.zero_grad()
        loss.backward()
        optim.step()

    return (loss_tot / len(dataloader)).item(), (C_tot / len(dataloader))

def test_step(model, dataloader, train_cens_info, device):
    '''

    '''
    model.eval()
    C_tot = 0
    loss_tot = 0
    hazards = []
    events = []
    times = []
    for X, e, t in dataloader:
        events += e.tolist()
        times += t.tolist()

        X = X.to(device)
        e = e.to(device)
        t = t.to(device)

        h = model(X)
        hazards += h.cpu().detach().squeeze().tolist()

        loss = CoxLoss(t, e, h, device)
        loss_tot += loss
    C, _, _, _, _ = concordance_index_censored(events, times, hazards)
        # C, _, _, _, _ = concordance_index_ipcw(train_cens_info, test_cens_info, h.cpu().detach().squeeze().numpy())
    # C = concordance_index(times, hazards, events)
    # C_tot += C

    return (loss_tot / len(dataloader)).item(), C #(C_tot / len(dataloader))


            