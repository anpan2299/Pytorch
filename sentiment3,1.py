# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 12:56:35 2023

@author: realmeid
"""

import numpy as np
import tensorflow as tf

with open('movie-xids.npy', 'rb') as f:
    Xids = np.load(f,allow_pickle= True)

with open('movie-xmasks.npy', 'rb') as f:
    Xmask = np.load(f,allow_pickle= True)
with open('movie-xlabelss.npy', 'rb') as f:
    Xlabels = np.load(f,allow_pickle= True)
    
dataset = tf.data.Dataset.from_tensor_slices((Xids, Xmask, Xlabels))

def map_function(input_ids, masks, labels):
    return {'input_ids': input_ids, 'attention_masks': masks}, labels

dataset = dataset.map(map_function)

batch_size = 16
dataset = dataset.shuffle(10000).batch(batch_size, drop_remainder=True)

split = 0.9

print(Xids.shape)

size = int(Xids.shape[0]/batch_size * split)

train_ds = dataset.take(size)
validation_ds = dataset.skip(size)

tf.data.experimental.save(train_ds, 'train')
tf.data.experimental.save(validation_ds, 'val')

ds = tf.data.experimental.load('train', element_spec = train_ds.element_spec)