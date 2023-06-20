print('Filename: text_cleanup_and_subset.py')
import pandas as pd
import numpy as np
import datetime as dt


ATN = pd.read_csv('data/all-the-news-2-1.csv')
print('Data loaded')

ATN.date = ATN.date.apply(lambda x: x[:10]) # only care about date
ATN.index = pd.to_datetime(ATN.date, format='%Y-%m-%d')

ATN_sorted = ATN.sort_index(ascending=False)

ATN_sorted_subset = ATN_sorted[['title', 'article', 'section', 'publication']].copy()

ATN_sorted_subset.dropna(subset=["title", "article", "publication"], inplace=True)

# we make a development subset
ATN_stripped = ATN_sorted_subset[ATN_sorted_subset.index >= dt.datetime(2020, 1, 1)].copy()
print('Processing done')

# Saving
p = 'data/ATN_cleaned.csv'
ATN_sorted_subset.to_csv(p, index=True)
print('Full all-the-news dataset saved')

pp = 'data/ATN_stripped2020.csv'
ATN_stripped.to_csv(pp, index=True)
print('Stripped all-the-news dataset saved')