# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 12:07:28 2018

@author: Sardor
"""

import numpy as np
import glob
import cv2
from random import shuffle
import keras
from keras import backend as K
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers. normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
import matplotlib.pyplot as plt

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

if K.image_data_format() == 'channels_last':
    K.set_image_data_format('channels_first')
print(K.image_data_format())


cross_validation = 0.1

IMG_SIZE = 60

MITOSIS_PATH = 'MITOS_PATTERNS/'
NONMITOSIS_PATH = 'NONMITOS_PATTERNS/'

class_mitos = glob.glob(MITOSIS_PATH+'*.npy')
class_nonmitos = glob.glob(NONMITOSIS_PATH+'*.npy')

shuffle(class_mitos)
shuffle(class_nonmitos)


train_data = []
ii_step = int(len(class_mitos) / 10)
progress = 0
for ii,numpy_dump in enumerate(class_mitos):
    data = np.load(numpy_dump).astype('float32')/255.0
    train_data.append([data,np.array([1,0])])
    if ii>progress:
        print('mitosis',len(class_mitos),'/',progress)
        progress += ii_step
        
print('mitosis train data generated')

ii_step = int(len(class_nonmitos) / 10)
progress = 0
for ii,numpy_dump in enumerate(class_nonmitos):
    data = np.load(numpy_dump).astype('float32')/255.0
    train_data.append([data,np.array([0,1])])
    if ii>progress:
        print('nonmitosis',len(class_nonmitos),'/',progress)
        progress += ii_step
        
print('nonmitosis train data generated')



trainImages = np.array([i[0] for i in train_data])
trainLabels = np.array([i[1] for i in train_data])
print('data converted as numpy array')



inputShape = (10, IMG_SIZE, IMG_SIZE)
print(inputShape)

model = Sequential()
model.add(Convolution2D(32, kernel_size = (3, 3), activation='relu', input_shape=inputShape))
model.add(Convolution2D(128, kernel_size=(3,3), activation='relu'))
model.add(Convolution2D(128, kernel_size=(5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(256, kernel_size=(3,3), activation='relu'))
model.add(Convolution2D(256, kernel_size=(5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(2, activation = 'softmax'))

model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics = ['accuracy'])
train_model = model.fit(trainImages, trainLabels, batch_size = 50, epochs = 6, verbose = 1,
                        validation_split=0.2)



accuracy = train_model.history['acc']
val_accuracy = train_model.history['val_acc']
loss = train_model.history['loss']
val_loss = train_model.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'g', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


model2 = Sequential()
model2.add(Conv2D(96, kernel_size=(3, 3),activation='relu',input_shape=inputShape,padding='same'))
model2.add(LeakyReLU(alpha=0.1))
#model2.add(MaxPooling2D((2, 2),padding='same'))
model2.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
model2.add(LeakyReLU(alpha=0.1))
#model2.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model2.add(Conv2D(32, (3, 3), activation='relu',padding='same'))
model2.add(LeakyReLU(alpha=0.1))                  
#model2.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model2.add(Flatten())
model2.add(Dense(128, activation='relu'))
model2.add(LeakyReLU(alpha=0.1))
model2.add(Dense(64, activation='relu'))
model2.add(Dropout(0.2))
model2.add(Dense(2, activation='softmax'))
model2.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

checkpoint = ModelCheckpoint('weights', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# check 5 epochs
early_stop = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=5, mode='max') 

callbacks_list = [checkpoint,early_stop]

train_model = model2.fit(trainImages, trainLabels, batch_size = 50, epochs = 10, verbose = 1,
                         validation_data=(testImages, testLabels),callbacks=callbacks_list)






data = np.load(test_data_nonmitosis[2])
data_uint8 = data.copy().astype('uint8')
data = data.astype('float32')/255
print(data.shape)
plt.imshow(data_uint8[5,:,:], cmap = 'gist_gray')
data = np.reshape(data,[1,10,60,60])

prediction = model2.predict(data)
print(prediction)





#model2.save_weights('gdrive/My Drive/Colab Notebooks/aa.wgt')



#model = Sequential()
#model.add(Conv2D(128, kernel_size = (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 10)))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(BatchNormalization())
#model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(BatchNormalization())
#model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(BatchNormalization())
#model.add(Dropout(0.2))
#model.add(Flatten())
#model.add(Dense(256, activation='relu'))
#model.add(Dropout(0.2))
#model.add(Dense(128, activation='relu'))
##model.add(Dropout(0.3))
#model.add(Dense(2, activation = 'softmax'))
#
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
#model.fit(train_images, trainLabels, batch_size = 100, epochs = 500, verbose = 1)