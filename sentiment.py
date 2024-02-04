# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 13:57:21 2023

@author: realmeid
"""

import flair 
models = flair.models.TextClassifier.load('en-sentiment')

#tokenization

text = "Some Niggas are in Paris!"

sentence = flair.data.Sentence(text)

print(sentence.to_tokenized_string())

models.predict(sentence)

print(sentence)
label = sentence.get_labels()

from transformers import BertForSequenceClassification
model_name = 'ProsusAI/finbert'

model = BertForSequenceClassification.from_pretrained(model_name)