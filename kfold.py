from torch.utils.data import DataLoader
from gridsearch import execute_trail
from data import ADNI
from data import compress_data
import torch
import pandas as pd

def get_fold_samples(all_samples, fold_number, num_folds=5):
    '''
    Get train/valid split for 
    '''
    # Number of valid samples in fold
    fold_size = int(len(all_samples) * (1 / num_folds))


    valid_samples = all_samples.iloc[fold_size*fold_number:fold_size*(fold_number+1)]
    # train_samples = [x for x in all_samples if x not in valid_samples]
    train_samples = all_samples[~all_samples.index.isin(valid_samples.index)]


    return train_samples, valid_samples

def kfold_validation(all_samples, num_folds=5, epochs=20):
    '''
    Args:
     - all_samples: (pd.DataFrame) All subjects and features
     - num_folds: (int) number of folders for validation
     - epochs: (int) number of epochs to train on per fold
    '''
    results = []
    for k in range(num_folds):
        print(f"Beginning fold: {k}/{num_folds}")
        train_samples, valid_samples = get_fold_samples(all_samples, fold_number=k, num_folds=num_folds)

        train_dataset = ADNI(train_samples, timeframe=60, c_encode='none', filters=['PTMARRY', 'PTGENDER'], as_tensor=True)
        test_dataset  = ADNI(valid_samples, timeframe=60, c_encode='none', filters=['PTMARRY', 'PTGENDER'], as_tensor=True)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        valid_dataloader  = DataLoader(test_dataset, batch_size=32, shuffle=True)

        trial_config = {
            'activation_fn': 'relu',
            'num_hidden_layers': 3,
            'num_hidden_nodes': 20,
            'lr': 0.014357142857142857,
            'lr_decay': 0
        }
        val, _ = execute_trail(trial_config, epochs, train_dataloader, valid_dataloader)

        results.append(val)

    for i, result, in enumerate(results):
        print(f"Best loss fold {i}: {result}")
    
    avg = sum(results) / len(results)
    print(f"Average: {avg}")

torch.manual_seed(0)

file = 'D:\\Big_Data\\ADNI\\normalized.csv'
df = pd.read_csv(file)
df = compress_data(df)
# train_samples, test_samples, _ = get_train_test_samples(df, test_size=0.2)
kfold_validation(df.sample(frac=1), num_folds=5, epochs=20)