# Implementing Support Vector machines model on dataset for stock price prediction
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import yfinance as yf
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle


# Importing the dataset
initial_df = pd.read_csv('SP500_stock_prices.csv', delimiter=',')

# Make date column the index
initial_df = initial_df.set_index('Date')

# only display year, month and day in index
initial_df.index = pd.to_datetime(initial_df.index)
data = initial_df

# make return column
data['Return'] = data['Close'].pct_change()

# assign each different Sector string its own number
data['Sector'] = data['Sector'].map({'Consumer Discretionary': 0, 'Consumer Staples': 1, 'Energy': 2, 'Financials': 3, 'Health Care': 4, 'Industrials': 5, 'Information Technology': 6, 'Materials': 7, 'Real Estate': 8, 'Communication Services': 9, 'Utilities': 10})

# make a new column for whether or not the price has increased or decreased from the previous 63 days
# 1 = increased, 0 = decreased, doing so by comparing openprice of today with openprice of 63 days ago
data['63-day Price Movement'] = np.where(data['Open'] > data['Open'].shift(-63), 1, 0)

# Make the tickers to categoricals, but keep the original order of the tickers
data['Ticker'] = pd.Categorical(data['Ticker'], categories=data['Ticker'].unique(), ordered=True)
data['Ticker'] = data['Ticker'].cat.codes

# select only the ticker index
indiviual_datas = []

for ticker in data['Ticker'].unique():
    #data_ticker = data.xs(0, level=1).copy()
    data_ticker = data[data['Ticker'] == ticker].copy()
    # make target column
    data_ticker['Target'] = data_ticker['63-day Price Movement'].shift(-63)
    data_ticker.dropna(inplace=True)

    indiviual_datas.append(data_ticker)

data_all_tickers = pd.concat(indiviual_datas)

# Splitting the dataset into the Training set and Test set according to date
# Define the specific date to split the DataFrame
# Create an offset of 63 Business days
bd = pd.tseries.offsets.BusinessDay(n = 63)
split_date = (pd.to_datetime('2019-09-30') - bd)

# set a start date for the training set to be 63 days after the first date in the dataset
start_date = (pd.to_datetime('2016-01-04') + bd)

# small test dataset creation:
print('Data Loaded:')

# bd = pd.tseries.offsets.BusinessDay(n = 63)
# split_date = pd.to_datetime('2019-11-30') - bd
# start_date = pd.to_datetime('2019-01-04') + bd

# Split the DataFrame into training and test sets based on the specific date
# for train we want all the data from the start date to the split date, this ensures that we have 62 days of data for each stock
# for test we want all the data from the split date to the end of the dataset
train = data_all_tickers.loc[(start_date < data_all_tickers.index) & (data_all_tickers.index < split_date)]
train_log = train
train_log['Low'] = np.log(train_log['Low'])
train_log['High'] = np.log(train_log['High'])
train_log['Open'] = np.log(train_log['Open'])
test = data_all_tickers.loc[data_all_tickers.index >= split_date]
test_log = test
test_log['Low'] = np.log(test_log['Low'])
test_log['High'] = np.log(test_log['High'])
test_log['Open'] = np.log(test_log['Open'])
X_train = train_log[['Ticker','Open', 'Low', 'High', 'Volume', 'Sector']]
y_train = train['Target']
X_test = test_log[['Ticker', 'Open', 'Low', 'High', 'Volume', 'Sector']]
y_test = test['Target']
print('Train and Test set created:')
# Standardize the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the SVM model
model = SVC(kernel='rbf', probability=True) # linear, rbf, poly, sigmoid
model.fit(X_train, y_train)

print('Model trained:')
# save model to file
pickle.dump(model, open('svm_modelFullData.sav', 'wb'))