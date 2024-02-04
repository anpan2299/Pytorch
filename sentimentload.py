# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 18:04:22 2023

@author: realmeid
"""
import tensorflow as tf
from transformers import BertTokenizer
import numpy as np

model = tf.keras.models.load_model('sentiment_model_top')

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

def prep_data(text):
    tokens = tokenizer(text, max_length= 512, truncation = True, add_special_tokens=True,
                       padding = 'max_length', return_tensors='tf')
    return {'input_ids': tokens['input_ids'], 'attention_masks': tokens['attention_mask']}
In = prep_data('hello world')

prob = model.predict(In)

print(prob)
print(np.argmax(prob))