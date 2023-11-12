import pandas as pd
from data import *
from network import DeepSurv
from torch.utils.data import DataLoader
from sksurv.metrics import concordance_index_censored
from torch.optim import Adam, lr_scheduler
from train_test import *
from utils import update_optim


file = 'C:\\Users\\jthra\\Documents\\data\\normalized.csv'
df = pd.read_csv(file)
df = compress_data(df)
train_samples, test_samples, _ = get_train_test_samples(df, test_size=0.2)

train_dataset = ADNI(train_samples, timeframe=60, c_encode='none', filters=['PTMARRY', 'PTGENDER'], as_tensor=True)
test_dataset = ADNI(test_samples, timeframe=60, c_encode='none', filters=['PTMARRY', 'PTGENDER'], as_tensor=True)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader  = DataLoader(test_dataset, batch_size=32, shuffle=True)

model = DeepSurv(len(train_dataset[0][0]), n_hidden_layers=3, hidden_dim=15, activation_fn='selu', dropout=0.5, do_batchnorm=True)
optim = Adam(model.parameters(), lr=0.001)
lr_decay = 0.0005
do_lr_decay = True

epochs = 10
train_losses = []
test_losses = []
train_Cs = []
test_Cs = []
lrs = []
for epoch in range(epochs):
    train_loss, train_C = train_step(model, optim, train_dataloader, device='cpu')
    test_loss, test_C = test_step(model, optim, test_dataloader, device='cpu')

    if do_lr_decay:
        lrs.append(optim.param_groups[0]['lr'])
        update_optim(optim, epoch, lr_decay)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_Cs.append(train_C)
    test_Cs.append(test_Cs)

    print(f'Epoch {epoch+1}/{epochs}')
    print(f'\tTrain loss: {train_loss}\n\tTest loss : {test_loss}')
    print(f'\tTrain C: {train_C}\n\tTest C : {test_C}')



