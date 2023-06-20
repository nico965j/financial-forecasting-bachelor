import pandas as pd
from time import time

years = [2016, 2017, 2018, 2019, 2020]
dfs_to_join = []

for year in years:
    start_time = time()
    print(f'loading data for year {year}')
    df = pd.read_csv(f'data/ATN_split/ATN_{year}.csv')
    dfs_to_join.append(df)
    print(f'which took {time() - start_time:.2f} seconds')

ATN_joined = pd.concat(dfs_to_join, ignore_index=True, sort=False, axis=0)

ATN_joined.to_csv('data/all-the-news-original.csv', index=False)