# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 14:11:33 2023

@author: realmeid
"""

import tensorflow
import numpy as np
import matplotlib.pyplot as plt


from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, GlobalMaxPooling2D, BatchNormalization, MaxPooling2D
from tensorflow.keras.models import Model
data =  tensorflow.keras.datasets.cifar10.load_data()

(x_train, y_train),(x_test, y_test) = data
x_train, x_test = x_train/255, x_test/255 
y_train, y_test = y_train.flatten(), y_test.flatten()
labels = set(y_train) #set doesn`t allow repeated items
print(labels)
classes = len(labels)

input_shape = x_train[0].shape

i = Input(shape = input_shape)
#x = Conv2D(32, (3,3),strides = 2, activation = 'relu')(i) #32 features, 3x3 kernel, using stride instead of maxpooling
#Relu on hidden layers to avoid vanishing gradient 
#x = Conv2D(64, (3,3),strides = 2, activation = 'relu')(x)
#x = Conv2D(128, (3,3),strides = 2, activation = 'relu')(x)
x = Conv2D(32, (3,3), activation = 'relu', padding = 'same')(i)
x = BatchNormalization()(x)
x = Conv2D(32, (3,3), activation = 'relu', padding = 'same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)

x = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)

x = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)

x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation ='relu')(x)
x = Dropout(0.2)(x)
x = Dense(classes, activation ='softmax')(x)

model = Model(i,x)

batch_size = 32 
data_generator = tensorflow.keras.preprocessing.image.ImageDataGenerator(width_shift_range = 0.1, height_shift_range = 0.1, horizontal_flip=True)
train_generator = data_generator.flow(x_train,y_train, batch_size)
steps_per_epoch = x_train.shape[0]//batch_size
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics =['accuracy'])
#r = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 15)
r = model.fit_generator(train_generator, validation_data = (x_test, y_test),steps_per_epoch = steps_per_epoch, epochs = 15)
model.summary()

print("Target Score", model.evaluate(x_train, y_train))
print("Test Score", model.evaluate(x_test, y_test))

plt.plot(r.history['loss'], label = 'loss')
plt.plot(r.history['val_loss'], label = 'val_loss')

plt.plot(r.history['accuracy'], label = 'accuracy')
plt.plot(r.history['val_accuracy'], label = 'val_accuracy')
plt.legend()
