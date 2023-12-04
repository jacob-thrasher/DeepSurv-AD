import torch
from torch.utils.data import Dataset
import pandas as pd
from sksurv.preprocessing import OneHotEncoder
import numpy as np
import os
import random



def split_df(df, test_size=0.2):
    '''
    Splits dataframe into train/test dataframes based on RID
    '''
    subjects = df.RID.unique()                    # Get all unique subject IDs
    num_test_subjects = int(len(subjects) * test_size)
    test_subjects = random.sample(subjects, num_test_subjects)
    train_subjects = [s for s in subjects if s not in test_subjects]

    test_df = df[df['RID'].isin(test_subjects)]
    train_df = df[df['RID'].isin(train_subjects)]

    return train_df, test_df

class ADNI(Dataset):
    def __init__(self, 
                 csvpath, 
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
        self.data = pd.read_csv(csvpath)



        # Clean data
        if drop_cols is not None: self.filter_data(drop_cols)
        self.impute_data()
        self.data['DX'].fillna("UNKNOWN", inplace=True)                # Unknown DX data can be considered censored, therefore it is a nonissue
        self.data.dropna(inplace=True)                                 # Drop remaining rows with missing categorical data
        if timeframe > -1: self.define_timeframe(timeframe)

        # Prep labels
        if label_type == 'stubby':
            self.data = self.compress_data(self.data)
            self.labels = self.create_label_array(self.data)
        elif label_type == 'future':
            pass

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





        if as_tensor: self.data = torch.tensor(self.data.values.astype(np.float32))
        if normalize: self.normalize()

    def compress_data(self, data):
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

    def create_label_columns(self):
        return

    def create_label_array(self, data, label_type):
        """
        Create structured label array from input matrix

        Args:
        x (dataframe) - Input dataframe containing DX and M columns
        """
        if label_type == 'stubby':
            indicators = (data['DX'] == 'Dementia').tolist()
            times = data['M'].tolist()
            labels = list(zip(indicators, times))
        elif label_type == 'future':
            subjects = data.RID.unique()
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
    

