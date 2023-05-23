import pandas as pd
import numpy as np

from preprocessing_utils import ThreeMonthReturn, CreateSequences, SplitData, FindNearestDateIndex


### Data loading and preprocessing
raw_stock_data = pd.read_csv('data/SP500_stock_prices_cleaned.csv', index_col=0, parse_dates=True)

# some helper variables
stock_tickers = pd.Series(raw_stock_data.Ticker.unique())
period = raw_stock_data.index.unique()
period_length = period.__len__()

## simple preprocessing
raw_stock_data = raw_stock_data.dropna()

# remove stock tickers if their length is shorter than the period length
tickers_to_remove = []

for t in stock_tickers:
    t_len = raw_stock_data[raw_stock_data['Ticker'] == t].__len__()
    if t_len < period_length:
        tickers_to_remove.append(t)
raw_stock_data = raw_stock_data[~raw_stock_data['Ticker'].isin(tickers_to_remove)]

## create log returns
log_returns_series = pd.concat([ThreeMonthReturn(raw_stock_data[raw_stock_data['Ticker'] == ticker]['Close'].copy(), log=True) for ticker in stock_tickers])
raw_stock_data['log_return_3m'] = log_returns_series


### save the dataframe to csv
raw_stock_data.to_csv('data/SP500_stock_prices_cleaned_with_3month_return.csv')