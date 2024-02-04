# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 13:46:41 2023

@author: realmeid
"""

from transformers import TFAutoModel
import tensorflow as tf


model = TFAutoModel.from_pretrained('bert-base-cased')

model.summary()

input_ids = tf.keras.layers.Input(shape = (512), name = 'input_ids',dtype = 'int32')
mask = tf.keras.layers.Input(shape = (512), name = 'attention_masks',dtype = 'int32')

#transformers
embedding = model.bert(input_ids, attention_mask = mask)[1]

#classifier heads
x = tf.keras.layers.Dense(1024, activation = 'relu')(embedding)
y = tf.keras.layers.Dense(5, activation = 'softmax', name = 'outputs')(x)

models = tf.keras.Model(inputs = [input_ids, mask], outputs = y)
models.layers[2].trainable = False

optimizer = tf.keras.optimizers.Adam(learning_rate = 5e-5, decay = 1e-6)
loss = tf.keras.losses.CategoricalCrossentropy()
acc = tf.keras.metrics.CategoricalAccuracy('accuracy')

models.compile(optimizer=optimizer, loss = loss, metrics = [acc])

element_spec = ({'input_ids': tf.TensorSpec(shape=(16,512), dtype = tf.int32, name = None),
                'attention_masks': tf.TensorSpec(shape=(16,512), dtype = tf.int32, name = None)},
                tf.TensorSpec(shape=(16,5), dtype = tf.float64, name = None))

train_ds = tf.data.experimental.load('train', element_spec = element_spec)
val_ds = tf.data.experimental.load('val', element_spec = element_spec)

models.fit(train_ds, validation_data = val_ds, epochs = 3)

models.save("sentiment_model_top")