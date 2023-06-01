import pandas as pd
import numpy as np
import time
from preprocessing_utils import n_lag_return, ThreeMonthReturn, CreateSequences, SplitData, FindNearestDateIndex


### Data loading and preprocessing
raw_stock_data = pd.read_csv('data/SP500_stock_prices.csv', index_col=0, parse_dates=True)
print(f'Dataset loaded')

# some helper variables
stock_tickers = pd.Series(raw_stock_data.Ticker.unique()).rename('Ticker')
period = raw_stock_data.index.unique()
period_length = period.__len__()

start_time = time.time()
## simple preprocessing
raw_stock_data = raw_stock_data.dropna()

# make data log
log_stock_data = raw_stock_data.copy()
log_stock_data['Volume'].apply(lambda x: 0 if x <= 0 else np.log(x)) # because of volume-values of zero
numerical_columns = ['Open', 'High', 'Low', 'Close'] # volume taken care of previously
log_stock_data[numerical_columns] = np.log(raw_stock_data[numerical_columns])

# remove stock tickers if their length is shorter than the period length
tickers_to_remove = []

for t in stock_tickers:
    t_len = raw_stock_data[raw_stock_data['Ticker'] == t].__len__()
    if t_len < period_length:
        tickers_to_remove.append(t)
raw_stock_data = raw_stock_data[~raw_stock_data['Ticker'].isin(tickers_to_remove)]

stop_time = time.time()
print(f'Simple preprocessing done in {stop_time-start_time:.2f} seconds, \nremoved {tickers_to_remove.__len__()} tickers with insufficient data.')

start_time = time.time()
## create log returns
log_returns_series = pd.concat([n_lag_return(raw_stock_data[raw_stock_data['Ticker'] == ticker]['Close'].copy()) for ticker in stock_tickers])
raw_stock_data['log_return_3m'] = log_returns_series

stop_time = time.time()
print(f'Log returns created in {stop_time-start_time:.2f} seconds.')


# convert to 32bit floats
float64_cols = list(raw_stock_data.select_dtypes(include='float64'))
# The same code again calling the columns
raw_stock_data[float64_cols] = raw_stock_data[float64_cols].astype('float32')
raw_stock_data.dtypes


### save the dataframe to csv
path = 'data/SP500_stock_prices_log_clean_3monthreturn.csv'
raw_stock_data.to_csv(path)
print(f'Dataframe saved to {path}.')

### save the stock tickers to csv
stock_tickers = stock_tickers[~stock_tickers.isin(tickers_to_remove)]
path = 'data/SP500_tickers_clean.csv'
stock_tickers.to_csv(path)
print(f'Suitable tickers saved to {path}.')