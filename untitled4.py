# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 13:16:39 2023

@author: realmeid
"""

import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sklearnex


def schedule(epoch, lr):
    if epoch >= 50:
        return 0.0001
    else:
        return 0.001
scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)

data = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size = 0.33)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
D = x_train.shape[1]
model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(D,)),
            tf.keras.layers.Dense(1, activation = "sigmoid")
    ])
model.compile(optimizer = "adam", loss = "binary_crossentropy",metrics = ['accuracy'])
r = model.fit(x_train,y_train, validation_data = (x_test,y_test), epochs = 100,callbacks = [scheduler])

print("Target Score", model.evaluate(x_train, y_train))
print("Test Score", model.evaluate(x_test, y_test))

plt.plot(r.history['loss'], label = 'loss')
plt.plot(r.history['val_loss'], label = 'val_loss')

plt.plot(r.history['accuracy'], label = 'accuracy')
plt.plot(r.history['val_accuracy'], label = 'val_accuracy')