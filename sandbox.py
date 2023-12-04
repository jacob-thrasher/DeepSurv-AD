import pandas as pd
from data import *
from network import DeepSurv
from torch.utils.data import DataLoader
from sksurv.metrics import concordance_index_censored
from torch.optim import Adam
from utils import update_optim
import random
import json


# Generate selected_features only csv
# df = pd.read_csv('D:\\Big_Data\\ADNI\\ADNIMERGE.csv')
# columns = list(df.columns)
# selected = ['RID', 'DX_bl', 'AGE', 'PTGENDER', 'PTEDUCAT', 'PTMARRY', 'ADAS11', 'ADAS13', 'CDRSB', 'FAQ', 'LDELTOTAL', 'MMSE', 'RAVLT_forgetting', 'RAVLT_immediate', 'RAVLT_learning', 'RAVLT_perc_forgetting', 'DX', 'M']
# to_drop = [x for x in columns if x not in selected]

# for c in to_drop:
#     df.drop(c, axis=1, inplace=True)

# df.to_csv('D:\\Big_Data\\ADNI\\selected_features.csv')

# normalize file
# df = pd.read_csv('D:\\Big_Data\\ADNI\\clinical_only.csv')

# df['AGE'] = (df['AGE'] - min(df['AGE'])) / (max(df['AGE'] - min(df['AGE'])))
# df['PTEDUCAT'] = (df['PTEDUCAT'] - min(df['PTEDUCAT'])) / (max(df['PTEDUCAT'] - min(df['PTEDUCAT'])))
# df['ADAS11'] = df['ADAS11'] / 70
# df['ADAS13'] = df['ADAS13'] / 85
# df['CDRSB'] = df['CDRSB'] / 18
# df['FAQ'] = df['FAQ'] / 30
# df['LDELTOTAL'] = df['LDELTOTAL'] / 25
# df['MMSE'] = df['MMSE'] / 30
# df['RAVLT_forgetting'] = df['RAVLT_forgetting'] / 15
# df['RAVLT_immediate'] = df['RAVLT_immediate'] / 75
# df['RAVLT_learning'] = df['RAVLT_learning'] / 14
# df['RAVLT_perc_forgetting'] = df['RAVLT_perc_forgetting'] / 100


# df.to_csv('D:\\Big_Data\\ADNI\\normalized.csv')


###################################################################w

train = '/home/jacob/Documents/ADNI/csvs/train_normalized.csv'
# test = 'D:\\Big_Data\\ADNI\\test_normalized.csv'
# train_df = pd.read_csv(train)
# test_df = pd.read_csv(test)

# train_df, test_df = split_df(df, test_size=0.2)
# train_df.to_csv('D:\\Big_Data\\ADNI\\train_normalized.csv')
# test_df.to_csv('D:\\Big_Data\\ADNI\\test_normalized.csv')

# df = compress_data(df)
# train_samples, test_samples, _ = get_train_test_samples(df, test_size=0.2)

train_dataset = ADNI(train, timeframe=60, c_encode='none', drop_cols=['PTMARRY', 'PTGENDER', 'DX_bl'], as_tensor=True, label_type='future')
test_dataset = ADNI(train, timeframe=-1, c_encode='none', drop_cols=['PTMARRY', 'PTGENDER', 'DX_bl'], as_tensor=True, label_type='future')
print(len(train_dataset), len(test_dataset))

print(train_dataset[0][0])
# test_dataset = ADNI(test_samples, timeframe=60, c_encode='none', filters=['PTMARRY', 'PTGENDER'], as_tensor=True)

# slabels = train_dataset.get_structured_labels()
# print(len(slabels))

# print(train_dataset.get_cens_distribution())