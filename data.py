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
                 drop_cols:list=None, 
                 timeframe:int=-1, 
                 c_encode:str='onehot',
                 label_type:str='stubby', 
                 as_tensor:bool=False,
                 normalize:bool=False):
        '''
        Initialize dataframe from ADNIMERGE dataset

        Args:
            path: path to csv file containing data
            filters: List of columns to be dropped
            timeframe: Censor all data after <timeframe> months
            c_encode: Method of encoding categorical variables.
                            'onehot' performs onehot encoding
                            'code' converts strings to int representation
                            'none' skips encoding entirely
        '''

        assert c_encode in ['onehot', 'code', 'none'], f'c_encode must be in [onehot, code, none], found {c_encode}'
        assert label_type in ['stubby', 'future', 'past'], f'label_type must be in [stubby, future, past], found {label_type}'
        self.as_tensor = as_tensor
        self.data = df
        if drop_cols is not None: self.filter_data(drop_cols)


        # Clean data
        self.impute_data()
        self.data['DX'].fillna("UNKNOWN", inplace=True)                # Unknown DX data can be considered censored, therefore it is a nonissue
        self.data.dropna(inplace=True)                                 # Drop remaining rows with missing categorical data


        # Prep labels


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
        # TODO: This function doesn't make sense. I can't grasp why but I know it doesn't
        # work like I want it do. I should fix it
        '''
        Censors all elements after selected timeframes
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
        indicators = (x['DX'] == 'Dementia').tolist()
        times = x['M'].tolist()
        labels = list(zip(indicators, times))
        return labels # torch.tensor(labels)
    
    def normalize(self):
        assert type(self.data) is torch.Tensor, f'Normalization input data must be torch.Tensor, found: {type(self.data)}'

        min_, _ = torch.min(self.data, dim=0, keepdim=True)
        max_, _ = torch.max(self.data, dim=0, keepdim=True)
        self.data = (self.data - min_) / (max_ - min_)

    def get_structured_labels(self):
        labels = np.zeros(len(self.labels), np.dtype({'names': ['cens', 'time'], 
                                                                'formats': ['?', '<f8']}))
        labels['cens'] = [x[0] for x in self.labels]
        labels['time'] = [x[1] for x in self.labels]
        return labels

    def get_cens_distribution(self):
        num_success = sum([x[0] for x in self.labels])
        num_cens = len(self.labels) - num_success
        return {
            'num_success': num_success,
            'num_cens': num_cens,
            'prop_success': num_success / len(self.labels),
            'prop_cens': num_cens / len(self.labels)
        }

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        '''
        Returns:
            X - Features
            e - Event indicator
            t - time of event/censor
        '''
        X = self.data[idx]
        e, t = self.labels[idx]
        return X, e, t # Cast bool to int
    

