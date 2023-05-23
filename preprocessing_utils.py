import pandas as pd
import numpy as np
import torch


def ThreeMonthReturn(closing_price, lags=63, log=False): 
    offset = 0.0001
    return closing_price.pct_change(lags) if log == False else closing_price.apply(lambda x: np.log(x + offset)).pct_change(lags)


def CreateSequences(data, look_back=62, horizon=62, target_name='log_return_3m'):
    """
    outputs feature sequences of length look_back
    for each day in the data, starting from the look_back'th day (62nd day e.g.)
    
    target is the return at the horizon'th day (62 days later e.g.)
    """
    
    # check if any column has non-numerical values and if there are, 
    # print a warning but continue with dropping the offending columns
    if data.applymap(np.isreal).all().all() == False:
        print("WARNING: Non-numerical values detected in dataframe. Dropping columns with non-numerical values.")
        data = data.select_dtypes(exclude=['object'])
    
    features = []
    target = []

    for timestep in range(look_back, len(data) - horizon):
        features.append(data.iloc[timestep-look_back:timestep])
        target.append(data[target_name].iloc[timestep+horizon])
    
    return torch.tensor(np.array(features)), torch.tensor(np.array(target))


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
    data: date-str of YYYY-MM-DD
    df: dataframe with datetime index

    returns: index of the nearest date in the future
    """
    date = pd.Timestamp(date)

    # If the date exists in the index, return its index
    if date in df.index:
        return df.index.get_loc(date)

    # If the date does not exist in the index, find the next closest date in the future
    mask = df.index > date
    if mask.any():
        return mask.argmax()
    
    # If no future date is found, raise an error
    raise ValueError("No available date found in the future.")


def DataFrame_to_Tensors(data, split_date, look_back=63, horizon=63):
    """
    data: dataframe of features and targets
    look_back: length of each sequence
    horizon: how far in the future to predict
    
    returns: dictionary of train and test sets
    """
    
    # find the splitting index to apply to the tensors from the dataframe
    # we subtract the look_back since that portion of data dissapears
    # when we create sequences of look_back length

    split_index = FindNearestDateIndex(split_date, data) - look_back

    features, target = CreateSequences(data, look_back=look_back, horizon=horizon)
    
    train_features, test_features = SplitData(features, split_idx=split_index)
    train_targets, test_targets = SplitData(target, split_idx=split_index)
    
    return {'train_features': train_features, 'train_targets': train_targets,
            'test_features': test_features, 'test_targets': test_targets}


