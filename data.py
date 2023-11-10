import torch
from torch.utils.data import Dataset
import pandas as pd
from sksurv.preprocessing import OneHotEncoder
import numpy as np
import os

def compress_data(data:pd.DataFrame) -> pd.DataFrame:
    '''
    Compresses data such that each patient has one row (X, y) such that
    X = data from first (baseline) visit
    y = label (or censor) from final visit

    Args:
        data (pandas.DataFrame) - Data to compress
    '''
    compressed_data = pd.DataFrame(columns=data.columns)
    patients = data['RID'].unique()
    for p in patients:
        history = data[data['RID'] == p].reset_index()
        history.sort_values(by=['M'], inplace=True)
        row = history.loc[0].copy() # get entire baseline row

        # Note: 
        # Theoretically, we can always set the DX to the DX of the last visit
        # but there is a small chance the DX was not recorded on the last visit, resulting in
        # a false UNKNOWN DX. Instead, we manually check if each of the three possible
        # outcomes are present in the patient's history and choose the worst one
        events = history['DX'].to_list()
        if 'Dementia' in events:
            row['DX'] = 'Dementia'
            row['M'] = history['M'].iloc[events.index('Dementia')]
        else:
            if 'MCI' in events: row['DX'] = 'MCI'
            else: row['DX'] = 'CN'
            row['M'] = history['M'].iloc[-1] # Date of last visit

        compressed_data.loc[len(compressed_data.index)] = row

    return compressed_data

def get_train_test_samples(df:pd.DataFrame, test_size:float=0.1, do_valid:bool=False, valid_size:float=0.2) -> pd.DataFrame:
    '''
    Divides master csv into train/test split. Optinal: valid split

    Args: 
        df (pandas.DataFrame): Dataframe for master csv
        test_size (float): proportion of df for test set
        do_valid (bool): Create valid set?
        valid_size
    '''
    df = df.sample(frac=1) # Shuffles df

    n_test_samples = int(len(df) * test_size)
    n_valid_samples = int(len(df) * valid_size)

    train_samples = df.iloc[n_test_samples:]
    test_samples = df.iloc[:n_test_samples]
    valid_samples = None

    if do_valid:
        train_samples = train_samples.iloc[n_valid_samples:]
        valid_samples = train_samples.iloc[:n_valid_samples]

    return train_samples, test_samples, valid_samples


class ADNI(Dataset):
    def __init__(self, 
                 df:pd.DataFrame, 
                 filters:list=None, 
                 timeframe:int=-1, 
                 c_encode:str='onehot', 
                 as_tensor:bool=False,
                 normalize:bool=False):
        '''
        Initialize dataframe from ADNIMERGE dataset

        Args:
            path: path to csv file containing data
            filters: List of columns to be dropped
            timeframe: Maximum number of months since baseline visit to keep
            c_encode: Method of encoding categorical variables.
                            'onehot' performs onehot encoding
                            'code' converts strings to int representation
                            'none' skips encoding entirely
        '''

        assert c_encode in ['onehot', 'code', 'none'], f'c_encode must be in [onehot, code, none], found {c_encode}'
        self.as_tensor = as_tensor
        self.data = df
        if filters is not None: self.filter_data(filters)


        self.impute_data()
        self.data['DX'].fillna("UNKNOWN", inplace=True)                # Unknown DX data can be considered censored, therefore it is a nonissue
        self.data.dropna(inplace=True)                                 # Drop remaining rows with missing categorical data

        if timeframe > -1: self.define_timeframe(timeframe)

        self.labels = self.create_label_array(self.data)
        self.filter_data(['DX', 'M', 'RID', 'DX_bl']) # Drop label data from input matrix

        if c_encode == 'onehot':
            self.data = self.data.astype({'PTGENDER':  'category',
                                         'PTMARRY':  'category'})
            
            self.onehot_data = OneHotEncoder().fit_transform(self.data)
        elif c_encode == 'code':
            # self.data.DX_bl = pd.Categorical(self.data.DX_bl)
            # self.data['DX_bl'] = self.data.DX_bl.cat.codes
            self.data.PTGENDER = pd.Categorical(self.data.PTGENDER)
            self.data['PTGENDER'] = self.data.PTGENDER.cat.codes
            self.data.PTMARRY = pd.Categorical(self.data.PTMARRY)
            self.data['PTMARRY'] = self.data.PTMARRY.cat.codes

        if as_tensor: 
            self.data = torch.tensor(self.data.values.astype(np.float32))

            # Cannot convert to tensor bc of structured array stuff I think?
            # self.labels = torch.tensor(self.labels.astype(float))

        if normalize: self.normalize()

    def impute_data(self):
        '''
        Impute missing data with column mean
        Inplace operation
        '''
        for col in self.data.columns:
            if self.data.dtypes[col] != 'object':
                self.data[col].fillna(self.data[col].mean(), inplace=True)


    def define_timeframe(self, timeframe):
        '''
        Selects only visits within a certain number of months since baseline visit. 
        For example, timeframe=24 with select all visits that occur within a 24 month
        period from baseline visit. All others are dropped

        Args:
            timeframe (int): Number of months since baseline visit
        '''
        self.data.loc[self.data['M'] > timeframe, 'DX'] = 'UNKNOWN'
        self.data.loc[self.data['M'] > timeframe, 'M'] = timeframe

    def filter_data(self, filters):
        '''
        Drop erroneous columns from dataset based on parameter filters

        Args:
            filters: list of columns to be dropped
        '''
        for item in filters:
            self.data.drop(item, axis=1, inplace=True)

    def create_exit_value(self):
        # Create exit value column
        self.data['EXIT'] = 0
        for patient in self.data['RID'].unique():
            history = self.data[self.data['RID'] == patient].sort_values(by='M')
            print(history)
            # When DX is not dementia, EXIT will be the date of the next visit
            # To create the initial EXIT list, we will shift all M values left 1
            # And duplicate the final value
            M = history['M'].tolist()
            M[:len(M)-1] = M[1:]
            history['EXIT'] = M
            
            # Determine visit of first AD diagnosis and set all future visits to that time
            init_AD_diag = -1
            for i, visit in history.iterrows():
                if visit['DX'] == 'Dementia' and init_AD_diag == -1:
                    init_AD_diag = visit['M']
                if init_AD_diag != -1:
                    history.at[i, 'EXIT'] = init_AD_diag

            self.data.update(history)

    def create_label_array(self, x):
        """
        Create structured label array from input matrix

        Args:
        x (dataframe) - Input dataframe containing DX and M columns
        """
        if not self.as_tensor:
            labels = np.zeros(len(x), np.dtype({'names': ['cens', 'time'], 
                                                                'formats': ['?', '<f8']}))
            labels['cens'] = x['DX'] == 'Dementia'
            # self.labels['time'] = self.data['EXIT']
            labels['time'] = x['M']
        else: 
            indicators = (x['DX'] == 'Dementia').tolist()
            times = x['M'].tolist()
            labels = list(zip(indicators, times))
        return labels # torch.tensor(labels)
    
    def normalize(self):
        assert type(self.data) is torch.Tensor, f'Normalization input data must be torch.Tensor, found: {type(self.data)}'

        min_, _ = torch.min(self.data, dim=0, keepdim=True)
        max_, _ = torch.max(self.data, dim=0, keepdim=True)
        self.data = (self.data - min_) / (max_ - min_)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
file = 'D:\\Big_Data\\ADNI\\normalized.csv'
df = pd.read_csv(file)
df = compress_data(df)
df.to_csv('D:\\Big_Data\\ADNI\\norm_compress.csv')
train_samples, test_samples, _ = get_train_test_samples(df, test_size=0.2)

train_dataset = ADNI(train_samples, timeframe=60, c_encode='none', filters=['PTMARRY', 'PTGENDER'], as_tensor=True)
test_dataset = ADNI(test_samples, timeframe=60, c_encode='none', filters=['PTMARRY', 'PTGENDER'], as_tensor=True)

print(len(train_dataset), len(test_dataset))

for i in range(0, 10):
    X, (e, t) = train_dataset[i]
    print(X)   
    print(e, t)
    print()