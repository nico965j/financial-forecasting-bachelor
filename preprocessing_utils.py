import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from copy import deepcopy


def ThreeMonthReturn(closing_price, lags=63, log=False): 
    offset = 0.0001
    return closing_price.pct_change(lags) if log == False else closing_price.apply(lambda x: np.log(x + offset)).pct_change(lags)

def n_lag_return(price, lags=63):
    return price - price.shift(lags) #look lags back, meaning the return that is subtracted is shifted lags forward...


def CreateSequences(df, look_back=63, horizon=63, target_name='log_return_3m'):
    """
    outputs feature sequences of length look_back
    for each day in the df, starting from the look_back'th day (62nd day e.g.)
    
    target is the return at the horizon'th day (62 days later e.g.)
    """
    
    # check if any column has non-numerical values and if there are, 
    # print a warning but continue with dropping the offending columns
    if df.applymap(np.isreal).all().all() == False:
        print("WARNING: Non-numerical values detected in dataframe. Dropping columns with non-numerical values.")
        df = df.select_dtypes(exclude=['object'])
    
    features = []
    target = []

    for timestep in range(look_back, len(df) - horizon):
        features.append(df.iloc[timestep-look_back:timestep])
        target.append(df[target_name].iloc[timestep+horizon])
    
    return torch.tensor(np.array(features)).float(), torch.tensor(np.array(target)).float()


def SplitData(data, split_idx):
    """
    data: tensors of features and targets
    split_idx: index to split the data at (test set will be inclusive of this index)
    
    returns: train and test sets
    
    training set is as long as possible
    while the test is 3 months long
    """

    # using index, we split the tensor into train and test
    train, test = data.split([split_idx, data.shape[0]-split_idx], dim=0)

    return train, test[0:63]


def FindNearestDateIndex(date, df):
    """
    data: date-str of YYYY-MM-DD or datetime object
    df: dataframe with datetime index

    returns: index of the nearest date in the future
    """
    date = pd.Timestamp(date) if type(date) == str else date

    # If the date exists in the index, return its index
    if date in df.index:
        return df.index.get_loc(date)

    # If the date does not exist in the index, find the next closest date in the future
    mask = df.index > date
    if mask.any():
        return mask.argmax()
    
    # If no future date is found, raise an error
    raise ValueError("No available date found in the future.")


def DF2Tensors(df, split_date, look_back=63, horizon=63):
    """
    df: dataframe of features and targets
    look_back: length of each sequence
    horizon: how far in the future to predict
    
    returns: dictionary of train and test sets
    """
    
    # find the splitting index to apply to the tensors from the dataframe
    # we subtract the look_back since that portion of data dissapears
    # when we create sequences of look_back length

    split_index = FindNearestDateIndex(split_date, df) - look_back

    features, target = CreateSequences(df, look_back=look_back, horizon=horizon)
    
    train_features, test_features = SplitData(features, split_idx=split_index)
    train_targets, test_targets = SplitData(target, split_idx=split_index)
    
    return {'train_features': train_features, 'train_targets': train_targets,
            'test_features': test_features, 'test_targets': test_targets}


def ScaleData(df, split_date, scaler):
    """
    df: dataframe of features and targets
    split_date: datetime object to split the data at
    scaler: sklearn scaler object

    return: scaled concat dataframe and fitted scaler object
    """
    
    feature_scaler = deepcopy(scaler)
    target_scaler = deepcopy(scaler)

    ticker_rows_train = df.loc[df.index < split_date].copy()
    ticker_rows_test = df.loc[df.index >= split_date].copy()

    feature_columns = ticker_rows_train.drop(columns=['Ticker', 'Sector']).columns
    target_columns = ['log_return_3m']

    ticker_rows_train[feature_columns] = feature_scaler.fit_transform(ticker_rows_train[feature_columns])
    ticker_rows_test[feature_columns] = feature_scaler.transform(ticker_rows_test[feature_columns])

    ticker_rows_train[target_columns] = target_scaler.fit_transform(ticker_rows_train[target_columns])
    ticker_rows_test[target_columns] = target_scaler.transform(ticker_rows_test[target_columns])

    # concatenate the scaled rows horizontally
    scaled_stock_data = pd.concat([ticker_rows_train, ticker_rows_test], axis=0)

    return scaled_stock_data, feature_scaler, target_scaler


# Dataset sctructure
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def FetchDataLoader(ticker_sequences_dict, batch_size=16, num_workers=0):
    # Initialize the datasets
    train_dataset = TimeSeriesDataset(ticker_sequences_dict['train_features'], ticker_sequences_dict['train_targets'])
    # val_dataset = TimeSeriesDataset(ticker_sequences_dict['val_features'], ticker_sequences_dict['val_targets'])
    test_dataset = TimeSeriesDataset(ticker_sequences_dict['test_features'], ticker_sequences_dict['test_targets'])

    # Initialize the dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_dataloader, test_dataloader