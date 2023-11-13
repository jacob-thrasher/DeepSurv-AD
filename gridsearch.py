import itertools
import numpy as np
import pandas as pd
import torch
import json
from torch.optim import Adam
from torch.utils.data import DataLoader
from data import *
from network import DeepSurv
from train_test import *
from utils import update_optim

# TODO: Loss is not the best metric to evaluate performance
# Once I figure out what to do about C-index issue change to that
def execute_trail(config, epochs, train_dataloader, test_dataloader):
    '''
    Perform experiment with desired hyperparameter config.
    Saves model with best validation score

    Args:
     - config: (Dict) Hyperparameter configuration with the following keys
                Expects keys: activation_fn, num_hidden_layers, num_hidden_nodes, lr, lr_decay

    Returns:
     - Best validation score
     - Epoch of best score
    '''
    # Get number of inputs nodes
    X, _, _ = next(iter(train_dataloader))

    model = DeepSurv(len(X), 
                    n_hidden_layers=config['num_hidden_layers'], 
                    hidden_dim=config['num_hidden_nodes'], 
                    activation_fn=config['activation_fn'], 
                    dropout=0.5, 
                    do_batchnorm=True)
    
    optim = Adam(model.parameters(), lr=config['lr'])
    lr_decay = config['lr_decay']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    model.to(device)

    best_val_loss = 100000000000
    best_epoch = -1
    for epoch in range(epochs):
        train_loss, train_C = train_step(model, optim, train_dataloader, device=device)
        test_loss, test_C = test_step(model, test_dataloader, device=device)

        # lrs.append(optim.param_groups[0]['lr'])
        update_optim(optim, epoch, lr_decay)

        if test_loss < best_val_loss:
            best_val_loss = test_loss
            best_epoch = epoch

    return best_val_loss, best_epoch
        

def generate_experiment_settings(gridsearch_config):
    '''
    Generate a list of hyperparameter configuations based on inputted file
    '''
    keys, values = zip(*gridsearch_config.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return experiments

def gridsearch(gridsearch_config, epochs=20):
    torch.manual_seed(0)

    file = 'D:\\Big_Data\\ADNI\\normalized.csv'
    df = pd.read_csv(file)
    df = compress_data(df)
    train_samples, test_samples, _ = get_train_test_samples(df, test_size=0.2)

    train_dataset = ADNI(train_samples, timeframe=60, c_encode='none', filters=['PTMARRY', 'PTGENDER'], as_tensor=True)
    test_dataset  = ADNI(test_samples, timeframe=60, c_encode='none', filters=['PTMARRY', 'PTGENDER'], as_tensor=True)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader  = DataLoader(test_dataset, batch_size=32, shuffle=True)

    experiments = generate_experiment_settings(gridsearch_config)
    results = {}
    for exp_number, experiment_config in enumerate(experiments):
        val, best_epoch = execute_trail(experiment_config, epochs, train_dataloader, test_dataloader)
        results[str(exp_number)] = {
            "val": val,
            "best_epoch": best_epoch
        }

    with open('files\\results.txt', 'w') as f:
        f.write(json.dumps(results))
    
    vals = [x['val'] for x in results.values()]
    best = np.argmin(vals)

    return results[str(best)]

test = {
    'activation_fn': ['relu', 'selu'],
    'num_hidden_layers': [1, 2, 3, 4],
    'num_hidden_nodes': [20, 30, 40, 50],
    'lr': list(np.linspace(1e-4, 0.05, 8)),
    'lr_decay': [0, 1e-4, 5e-4, 1e-3]
}

# test = {
#     'activation_fn': ['relu', 'selu'],
#     'num_hidden_layers': [1],
#     'num_hidden_nodes': [20],
#     'lr': [0.001],
#     'lr_decay': [1e-4]
# }

experiments = generate_experiment_settings(test)
print(len(experiments))