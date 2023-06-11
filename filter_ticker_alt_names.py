import pandas as pd
import numpy as np
import datetime as dt
import os


alt_names = pd.read_csv('data/tickers_alt_names_raw.csv')
tickers_clean = pd.read_csv('data/SP500_tickers_clean.csv')
print('Data loaded')

# filter our ticker_alt_names_raw to only include tickers in sp500_tickers_clean

alt_names_filtered = alt_names[alt_names.ticker.isin(tickers_clean.ticker)].copy()

# stats about number of removed tickers
print('Number of tickers removed: ', len(alt_names) - len(alt_names_filtered))

# save to csv
alt_names_filtered.to_csv('data/ticker_alt_names_filtered.csv', index=False)