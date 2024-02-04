# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 11:39:54 2023

@author: realmeid
"""

import modin.pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer
import numpy as np

df = pd.read_csv('fruits-360-original-size/train.tsv', sep='\t')
df.head()

#df.drop_duplicates(subset= ['SentenceId'], keep ='first')

df['Sentiment'].value_counts().plot(kind = 'bar')

seq_len = 512
num_len = len(df)

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

tokens = tokenizer(df["Phrase"].tolist(), max_length = seq_len, truncation = True, padding = 'max_length',
                   add_special_tokens= True, return_tensors='tf')

tokens.keys()

with open('movie-xids.npy', 'wb') as f:
    np.save(f, tokens['input_ids'])
with open('movie-xmasks.npy', 'wb') as f:
    np.save(f, tokens['attention_mask'])

sentiment_arr = df['Sentiment'].values

labels = np.zeros((num_len,sentiment_arr.max()+1))
labels[np.arange(num_len), sentiment_arr] = 1 

with open('movie-xlabelss.npy', 'wb') as f:
    np.save(f, labels)