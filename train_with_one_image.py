# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 12:10:20 2018

@author: Sardor
"""

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import cv2

NONMITOSIS_PATH = 'NONMITOS_PATTERNS/'

MITOSIS_PATH = 'MITOS_PATTERNS/'

image_shape = (60, 60)

batch_size = 128
num_classes = 2
epochs = 100

#### train data
with open(csv,'r') as csv_file:
    print('\nOpened csv file',csv)
    mitosis_pixels = []
    for line in csv_file:
        splitted_line = line.split(',')
        pixel_data = []
        for x,y in zip(splitted_line[0::2],splitted_line[1::2]):
            pixel_data.append(( int(x), int(y) ))
        mitosis_pixels.append(pixel_data)
    
image = cv2.imread(image_name,0)
print('opened image',image_name)
image_mask = image.copy()
image_mask *= 0

for pixel_data in mitosis_pixels:
    for pixels in pixel_data:
        image_mask[pixels[1],pixels[0]] = 255



X_train = image.astype('float32')
X_train /= 255
print(X_train.shape)

   
Y_train = image_mask.astype('float32')
Y_train /= 255
Y_train = Y_train.flatten()
#Y_train = np_utils.to_categorical(Y_train, 2)
print(Y_train.shape)



#### test data
with open(csv_test,'r') as csv_file:
    print('\nOpened csv file',csv)
    mitosis_pixels_test = []
    for line in csv_file:
        splitted_line = line.split(',')
        pixel_data = []
        for x,y in zip(splitted_line[0::2],splitted_line[1::2]):
            pixel_data.append(( int(x), int(y) ))
        mitosis_pixels_test.append(pixel_data)
    
image = cv2.imread(image_name_test,0)
print('opened image',image_name)
image_mask = image.copy()
image_mask *= 0

for pixel_data in mitosis_pixels_test:
    for pixels in pixel_data:
        image_mask[pixels[1],pixels[0]] = 255
        
X_test = image.astype('float32')
X_test /= 255
#X_test = X_test.flatten()
print(X_test.shape)
   
Y_test = image_mask.astype('float32')
Y_test /= 255
#Y_test = Y_test.flatten()
Y_test = np_utils.to_categorical(Y_test, 2)
print(Y_test.shape)
     

#if K.image_data_format() == 'channels_first':
#    X_train = X_train.reshape(1, 1, 1360, 1360)
#    X_test = X_test.reshape(1, 1, 1360, 1360)
#    input_shape = (1, 1360, 1360)
#else:
#    X_train = X_train.reshape(1, 1360, 1360, 1)
#    X_test = X_test.reshape(1, 1360, 1360, 1)
#    input_shape = (1360, 1360,1)

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(1, 1360, 1360)
    X_test = X_test.reshape(1, 1360, 1360)
    input_shape = (1, 1360, 1360)
else:
    X_train = X_train.reshape(1360, 1360, 1)
    X_test = X_test.reshape(1360, 1360, 1)
    input_shape = (1360, 1360,1)


model = Sequential()
#model.add(Conv2D(32, kernel_size=(3, 3),
#                 activation='relu',
#                 input_shape=input_shape))
#model.add(Conv2D(64, (3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.15))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='tanh'))
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.08))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])