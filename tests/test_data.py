import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data import CMAPSSDataHandler

@pytest.fixture
def mock_config():
    return {
        'data': {
            'train_path': 'dummy_train.txt',
            'test_path': 'dummy_test.txt',
            'rul_path': 'dummy_rul.txt',
            'window_size': 30,
            'max_rul': 125
        }
    }

@pytest.fixture
def sample_train_df(mock_config):
    handler = CMAPSSDataHandler(mock_config)
    data = []
    # Create a simple dataset for unit 1 with 50 cycles
    for i in range(1, 51):
        row = [1, i, 0.0, 0.0, 0.0] + [np.random.rand() for _ in range(21)]
        data.append(row)
    df = pd.DataFrame(data, columns=handler.columns)
    return df

def test_calculate_rul(mock_config, sample_train_df):
    handler = CMAPSSDataHandler(mock_config)
    df = handler.calculate_rul(sample_train_df)
    
    assert 'RUL' in df.columns
    # Last cycle RUL should be 0
    assert df.loc[df['time_cycles'] == 50, 'RUL'].values[0] == 0
    # First cycle RUL should be 49
    assert df.loc[df['time_cycles'] == 1, 'RUL'].values[0] == 49

def test_windowing(mock_config, sample_train_df):
    handler = CMAPSSDataHandler(mock_config)
    sample_train_df = handler.calculate_rul(sample_train_df)
    
    X, y = handler.windowing(sample_train_df)
    assert X.shape[0] == 50 - 30 + 1
    assert X.shape[1] == 30
    assert X.shape[2] == len(handler.features)
    assert y.shape[0] == 50 - 30 + 1
