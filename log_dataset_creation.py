import pandas as pd
import numpy as np
import time
from datetime import datetime
from preprocessing_utils import n_lag_return, ThreeMonthReturn, CreateSequences, SplitData, FindNearestDateIndex


### Data loading and preprocessing
raw_stock_data = pd.read_csv('data/SP500_stock_prices.csv', index_col=0, parse_dates=True)
print(f'Dataset loaded')


start_time = time.time()
## simple preprocessing
raw_stock_data = raw_stock_data.dropna()

# make data log
log_stock_data = raw_stock_data[raw_stock_data.index >= datetime(2015, 10, 1)].copy()
# Apply the transformations
log_stock_data['Volume'] = log_stock_data['Volume'].apply(lambda x: 0 if x <= 0 else np.log(x)) # because of zero-vals in "volume"
numerical_columns = ['Open', 'High', 'Low', 'Close'] # volume taken care of previously
log_stock_data[numerical_columns] = np.log(log_stock_data[numerical_columns])

# some helper variables
stock_tickers = pd.Series(log_stock_data.Ticker.unique()).rename('Ticker')
period = log_stock_data.index.unique()
period_length = period.__len__()

# remove stock tickers if their length is shorter than the period length
tickers_to_remove = []

for t in stock_tickers:
    t_len = log_stock_data[log_stock_data['Ticker'] == t].__len__()
    if t_len < period_length:
        tickers_to_remove.append(t)
log_stock_data = log_stock_data[~log_stock_data['Ticker'].isin(tickers_to_remove)]

stop_time = time.time()
print(f'Simple preprocessing done in {stop_time-start_time:.2f} seconds, \nremoved {tickers_to_remove.__len__()} tickers with insufficient data.')

start_time = time.time()
## create log returns
log_returns_series = pd.concat([n_lag_return(log_stock_data[log_stock_data['Ticker'] == ticker]['Close'].copy()) for ticker in stock_tickers])
log_stock_data['log_return_3m'] = log_returns_series

stop_time = time.time()
print(f'Log returns created in {stop_time-start_time:.2f} seconds.')

# we cut the data down
log_stock_data = log_stock_data[log_stock_data.index >= datetime(2016, 1, 1)]

# convert to 32bit floats
float64_cols = list(log_stock_data.select_dtypes(include='float64'))
# The same code again calling the columns
log_stock_data[float64_cols] = log_stock_data[float64_cols].astype('float32')
log_stock_data.dtypes


### save the dataframe to csv
path = 'data/SP500_stock_prices_log_clean_3monthreturn.csv'
log_stock_data.to_csv(path) # keep date index
print(f'Dataframe saved to {path}.')

### save the stock tickers to csv
clean_stock_tickers = stock_tickers[~stock_tickers.isin(tickers_to_remove)]
path = 'data/SP500_tickers_clean.csv'
clean_stock_tickers.to_csv(path, index=False) # no important index
print(f'Suitable tickers saved to {path}.')