import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class CMAPSSDataHandler:
    def __init__(self, config):
        self.config = config
        self.train_path = config['data']['train_path']
        self.test_path = config['data']['test_path']
        self.rul_path = config['data']['rul_path']
        self.window_size = config['data']['window_size']
        self.max_rul = config['data']['max_rul']
        
        # CMAPSS columns
        self.columns = ['unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3'] + \
                       [f's_{i}' for i in range(1, 22)]
        
        # 14 sensors that carry useful information for FD001
        self.features = [f's_{i}' for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]]
        
        self.std_scaler = None
        self.minmax_scaler = None

    def load_data(self):
        train_df = pd.read_csv(self.train_path, sep=r'\s+', header=None, names=self.columns)
        test_df = pd.read_csv(self.test_path, sep=r'\s+', header=None, names=self.columns)
        truth_df = pd.read_csv(self.rul_path, sep=r'\s+', header=None, names=['RUL'])
        return train_df, test_df, truth_df

    def calculate_rul(self, train_df):
        # Calculate maximum cycles for each unit
        rul = pd.DataFrame(train_df.groupby('unit_nr')['time_cycles'].max()).reset_index()
        rul.columns = ['unit_nr', 'max_cycles']
        train_df = train_df.merge(rul, on=['unit_nr'], how='left')
        
        # Linear RUL
        train_df['RUL'] = train_df['max_cycles'] - train_df['time_cycles']
        
        # Piecewise RUL
        train_df['RUL'] = train_df['RUL'].clip(upper=self.max_rul)
        train_df.drop('max_cycles', axis=1, inplace=True)
        return train_df

    def normalize(self, train_df, test_df):
        # Uses standard scaler followed by min-max scaler for robust feature normalization
        self.std_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        
        train_df[self.features] = self.std_scaler.fit_transform(train_df[self.features])
        train_df[self.features] = self.minmax_scaler.fit_transform(train_df[self.features])
        
        test_df[self.features] = self.std_scaler.transform(test_df[self.features])
        test_df[self.features] = self.minmax_scaler.transform(test_df[self.features])
        
        return train_df, test_df

    def windowing(self, df):
        # Generate 3D sequences (samples, window_size, features)
        units = df['unit_nr'].unique()
        X, y = [], []
        
        for unit in units:
            unit_df = df[df['unit_nr'] == unit]
            data_matrix = unit_df[self.features].values
            rul_matrix = unit_df['RUL'].values if 'RUL' in unit_df.columns else None
            
            for i in range(len(unit_df) - self.window_size + 1):
                X.append(data_matrix[i:i + self.window_size])
                if rul_matrix is not None:
                    y.append(rul_matrix[i + self.window_size - 1])
                    
        if len(y) > 0:
            return np.array(X), np.array(y)
        return np.array(X)

    def prepare_test_validation(self, test_df, truth_df):
        # Prepare test RUL by using max_cycles mapping
        rul = pd.DataFrame(test_df.groupby('unit_nr')['time_cycles'].max()).reset_index()
        rul.columns = ['unit_nr', 'max_cycles']
        truth_df['unit_nr'] = truth_df.index + 1
        truth_df['max_rul'] = truth_df['RUL']
        rul = rul.merge(truth_df, on=['unit_nr'], how='left')
        rul['max_cycles_true'] = rul['max_cycles'] + rul['max_rul']
        
        test_df = test_df.merge(rul[['unit_nr', 'max_cycles_true']], on=['unit_nr'], how='left')
        test_df['RUL'] = test_df['max_cycles_true'] - test_df['time_cycles']
        test_df['RUL'] = test_df['RUL'].clip(upper=self.max_rul)
        test_df.drop(['max_cycles_true'], axis=1, inplace=True)
        return test_df

class CMAPSSDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

def get_dataloaders(config):
    handler = CMAPSSDataHandler(config)
    train_df, test_df, truth_df = handler.load_data()
    
    train_df = handler.calculate_rul(train_df)
    test_df = handler.prepare_test_validation(test_df, truth_df)
    
    train_df, test_df = handler.normalize(train_df, test_df)
    
    X_train, y_train = handler.windowing(train_df)
    X_test, y_test = handler.windowing(test_df)
    
    # Train/Val split 80/20 from training data
    np.random.seed(42)
    val_split = int(0.8 * len(X_train))
    idx = np.random.permutation(len(X_train))
    train_idx, val_idx = idx[:val_split], idx[val_split:]
    
    X_train_split, y_train_split = X_train[train_idx], y_train[train_idx]
    X_val_split, y_val_split = X_train[val_idx], y_train[val_idx]
    
    train_dataset = CMAPSSDataset(X_train_split, y_train_split)
    val_dataset = CMAPSSDataset(X_val_split, y_val_split)
    test_dataset = CMAPSSDataset(X_test, y_test)
    
    batch_size = config['train']['batch_size']
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, handler
