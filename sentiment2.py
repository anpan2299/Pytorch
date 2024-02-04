# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 14:43:17 2023

@author: realmeid
"""

from transformers import BertForSequenceClassification, BertTokenizer
import torch.nn.functional
import torch

model_name = 'ProsusAI/finbert'

model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

txt = ("Given the recent downturn in stocks especially in tech which is likely to persist as yields keep going up, I thought it would be prudent to share the risks of investing in ARK ETFs, written up very nicely by [The Bear Cave](https://...). The risks comes primarily from ARK's illiquid and very large holdings in small cap companies. ARK is forced to sell its holdings whenever its liquid ETF gets hit with outflows as is especially the case in market downturns. This could force very painful liquidations at unfavorable prices and the ensuing crash goes into a positive feedback loop leading into a death spiral enticing even more outflows and predatory shorts.")

tokens = tokenizer.encode_plus(txt, max_length = 512, truncation = True, padding = 'max_length',
                               add_special_tokens= True, return_tensors='pt') #pytorch, tf for tensorflow.
tokens2 = tokenizer.encode_plus(txt, 
                               add_special_tokens= False)
print(tokens)
Input_ids = tokens2['input_ids']
attention_mask = tokens2['attention_mask']

#in case of texts with more than 512 tokens
start = 0 
window_size = 510

total_len = len(Input_ids)
probs2 = []
loop = True
while loop:
    end = start + window_size
    if end >= total_len:
         end = total_len
         loop = False
    chunk = [101] + Input_ids[start:end] + [102]
    mask_chunk = [1] + attention_mask[start:end] + [1]
    
    chunk += [0]*(window_size - len(chunk) + 2)
    mask_chunk += [0]*(window_size - len(mask_chunk)  +2 )
    input_dict = {'input_ids': torch.Tensor([chunk]).long(), 'attention_mask': torch.Tensor([mask_chunk]).int()}
    
    print(input_dict)
    output2 = model(**input_dict)
    probs2.append(torch.nn.functional.softmax(output2[0], dim = -1))
    start = end
#Model output before softmax

output1 = model(**tokens)
print(output1)



#softmax
probs = torch.nn.functional.softmax(output1[0], dim = -1)

print(probs)

predict = torch.argmax(probs).item()
print(probs[0][predict])

