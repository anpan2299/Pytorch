# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 11:01:36 2023

@author: realmeid
"""
import tensorflow as tf2
from keras.layers import Input, Lambda, Dense, Flatten, LeakyReLU, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import sys, os
import matplotlib.pyplot as plt

from glob import glob

mnist = tf2.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255 *2 -1, x_test/255 *2 -1

print(x_train.shape)
N, H, W = x_train.shape
D = H*W
x_train = x_train.reshape(-1, D)
x_test = x_test.reshape(-1, D)
print(x_train.shape)
latent_dim = 100 

def build_generator(dim):
    i = Input(shape = (dim))
    x = Dense(256, activation = LeakyReLU(alpha = 0.2))(i)
    x = BatchNormalization(momentum = 0.8)(x)
    x = Dense(512, activation = LeakyReLU(alpha = 0.2))(x)
    x = BatchNormalization(momentum = 0.8)(x)
    x = Dense(1024, activation = LeakyReLU(alpha = 0.2))(x)
    x = BatchNormalization(momentum = 0.8)(x)
    x = Dense(D, activation = "tanh")(x) #tanh para dados de -1 a 1
    
    model = Model(i,x)
    return model

def build_discriminator(img_dim):
    i = Input(shape = (img_dim))
    x = Dense(512, activation = LeakyReLU(alpha = 0.2))(i)
    x = Dense(256, activation = LeakyReLU(alpha = 0.2))(x)
    x = Dense(1, activation = "sigmoid")(x) #binary activation
   
    model = Model(i,x)
    return model

def sample_images(epoch):
    row, cols, = 5,5
    noise = np.random.randn(row*cols, latent_dim)
    imgs = generator.predict(noise)
    
    #rescale
    imgs = 0.5*imgs + 0.5
    fig,axs = plt.subplots(row, cols)
    idx = 0
    for i in range(row):
        for j in range(cols):
            axs[i,j].imshow(imgs[idx].reshape(H, W), cmap = 'gray')
            axs[i,j].axis('off')
            idx += 1
    fig.savefig("gan_images/ %d.png" % epoch)
    plt.close()
    
discriminator = build_discriminator(D)
discriminator.compile(
    loss = "binary_crossentropy",
    optimizer = Adam(0.0002, 0.5),
    metrics = ['accuracy'])
generator = build_generator(latent_dim)
#z is noise
z = Input(shape = (latent_dim))

#img output
img = generator(z)
discriminator.trainable = False
fake_pred = discriminator(img)
combined_model = Model(z, fake_pred)

combined_model.compile(    
    loss = "binary_crossentropy",
    optimizer = Adam(0.0002, 0.5),
    )

#gradient descent

batch_size = 32
epochs = 30000
sample = 200

ones = np.ones(batch_size)
zeros = np.zeros(batch_size)

d_losses =[]
g_losses = []

if not os.path.exists("gan_images"):
    os.makedirs("gan_images")
    
for epoch in range(epochs):
    
    idx = np.random.randint(0, x_train.shape[0], batch_size)# 320 a 60000 e 32 imagens
    real_images = x_train[idx]
    
    noise= np.random.randn(batch_size,latent_dim)
    fake_imgs= generator.predict(noise)
    
    d_loss_real, d_acc_real =  discriminator.train_on_batch(real_images, ones)
    d_loss_fake, d_acc_fake = discriminator.train_on_batch(fake_imgs, zeros)
    d_loss = 0.5*(d_loss_real + d_loss_fake)
    d_acc = 0.5*(d_acc_real + d_acc_fake)
    
    noise= np.random.randn(batch_size,latent_dim)
    g_loss=  combined_model.train_on_batch(noise, ones)
    
    g_losses.append(g_loss)
    d_losses.append(d_loss)
    print(f"current epoch = {epoch}, d_loss = {d_loss}, d_acc = {d_acc}, g_loss = {g_loss}")
    if epoch % sample == 0:
        sample_images(epoch)