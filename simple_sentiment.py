import pandas as pd
import numpy as np
import string
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from spacy import displacy
import datetime as dt
import re
from time import time
from transformers import AutoTokenizer

from transformers import pipeline

# uncomment for downloading spacy models
# !python -m spacy download en_core_web_sm
# !python -m spacy download en_core_web_lg
# give it 2 min


# ATN_raw = pd.read_csv('data/ATN_stripped2020.csv')
ATN_raw = pd.read_csv('data/all-the-news-2-1.csv')
print(f'dataload complete')

ATN_raw.date = ATN_raw.date.apply(lambda x: x[:10]) # only care about date
ATN_raw.index = pd.to_datetime(ATN_raw.date, format='%Y-%m-%d')
ATN = ATN_raw.drop(columns=['date'])

business_publications = ['Business Insider', 'Economist'] # removed reuters
business_sections = ['World News', 'Business News', 'us', 'Bonds News', 'Company News', 'Market News', 'Financials', 'business', 'Deals']

ATN_filtered = ATN[ATN['publication'].isin(business_publications) | ATN['section'].isin(business_sections)] # .iloc[:100] #! test
ATN_filtered.dropna(subset=['title', 'article'], inplace=True)

def SimpleProcessing(text):
    # remove all punctuation, but dont remove periods
    text = text.translate(str.maketrans('', '', string.punctuation.replace('.', '')))
    return text.lower()

string_columns = ['title', 'article']

ATN_max_dates = ATN_filtered.groupby('date').head(25)
ATN_max_dates[string_columns] = ATN_max_dates[string_columns].applymap(SimpleProcessing)

finbert = pipeline('sentiment-analysis', model='ProsusAI/finbert', device=0, truncation=True)
print('finbert loaded')
tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')

all_sentences = []
article_lengths = []
for article in ATN_max_dates['article']:
    sentences = article.split('.')[:-1]
    all_sentences.extend(sentences)
    article_lengths.append(len(sentences))

s = time()
sentiment_results = finbert(all_sentences)
sentiment_scores = [result['score'] if result['label'] == 'positive' else -result['score'] for result in sentiment_results]
print(f'finbert sentiment analysis took {time() - s} seconds')

start = 0
average_sentiment_scores = []
for length in article_lengths:
    article_scores = sentiment_scores[start:start+length]
    average_sentiment_scores.append(np.mean(article_scores))
    start += length
print('average sentiment scores calculated')

ATN_max_dates['sentiment'] = average_sentiment_scores

date_sentiment = ATN_max_dates.groupby('date')['sentiment'].sum()
print('date sentiment calculated')

print(ATN_max_dates.head(10))
ATN_max_dates['sentiment'].to_csv('data/ATN_sentiment_max_dates.csv', index=True) # only save sentiment scores for less disk use

date_sentiment.to_csv('data/ATN_date_sentiment_max_dates.csv', index=True)

# concatanate the sentiment scores to the articles
# atn_small = ATN_filtered.iloc[:10].copy()
# atn_small['sentiment'] = average_sentiment_scores
# # sum score up for each date
# atn_small['date'] = pd.to_datetime(atn_small['date'])
# atn_small['date'] = atn_small['date'].dt.date
# date_sentiment = atn_small.groupby('date')['sentiment'].sum()
# print('date sentiment calculated')
# print(atn_small.head(10))
# atn_small.to_csv('data/ATN_sentiment_small.csv', index=True)