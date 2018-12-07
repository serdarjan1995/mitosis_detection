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
from keras.layers import Conv2D, MaxPooling2D
from keras.layers. normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
import matplotlib.pyplot as plt


if K.image_data_format() == 'channels_last':
    K.set_image_data_format('channels_first')
print(K.image_data_format())


cross_validation = 0.3

IMG_SIZE = 60

MITOSIS_PATH = 'MITOS_PATTERNS/'
NONMITOSIS_PATH = 'NONMITOS_PATTERNS/'

class_mitos = glob.glob(MITOSIS_PATH+'*.npy')
class_nonmitos = glob.glob(NONMITOSIS_PATH+'*.npy')

shuffle(class_mitos)
shuffle(class_nonmitos)

split_p = int( len(class_mitos)*cross_validation )

train_data_mitosis = class_mitos[split_p:]
test_data_mitosis = class_mitos[:split_p]

split_p = int( len(class_nonmitos)*cross_validation )
train_data_nonmitosis = class_nonmitos[split_p:]
test_data_nonmitosis = class_nonmitos[:split_p]
print('split complete')



train_data = []
ii_step = int(len(train_data_mitosis) / 10)
progress = 0
for ii,numpy_dump in enumerate(train_data_mitosis):
    data = np.load(numpy_dump).astype('float32')/255.0
    train_data.append([data,np.array([1,0])])
    if ii>progress:
        print('mitosis',len(train_data_mitosis),'/',progress)
        progress += ii_step
        
print('mitosis train data generated')

ii_step = int(len(train_data_nonmitosis) / 10)
progress = 0
for ii,numpy_dump in enumerate(train_data_nonmitosis):
    data = np.load(numpy_dump).astype('float32')/255.0
    train_data.append([data,np.array([0,1])])
    if ii>progress:
        print('nonmitosis',len(train_data_nonmitosis),'/',progress)
        progress += ii_step
        
print('nonmitosis train data generated')



test_data = []
ii_step = int(len(test_data_mitosis) / 10)
progress = 0
for ii,numpy_dump in enumerate(test_data_mitosis):
    data = np.load(numpy_dump).astype('float32')/255.0
    #data = data.reshape(data.shape[1],data.shape[2],data.shape[0])
    test_data.append([data,np.array([1,0])])
    if ii>progress:
        print('test mitosis',len(test_data_mitosis),'/',progress)
        progress += ii_step

print('mitosis test data generated')

ii_step = int(len(test_data_nonmitosis) / 10)
progress = 0
for ii,numpy_dump in enumerate(test_data_nonmitosis):
    data = np.load(numpy_dump).astype('float32')/255.0
    #data = data.reshape(data.shape[1],data.shape[2],data.shape[0])
    test_data.append([data,np.array([0,1])])
    if ii>progress:
        print('test nonmitosis',len(test_data_nonmitosis),'/',progress)
        progress += ii_step
print('nonmitosis test data generated')





trainImages = np.array([i[0] for i in train_data])
trainLabels = np.array([i[1] for i in train_data])
testImages = np.array([i[0] for i in test_data])
testLabels = np.array([i[1] for i in test_data])
print('data converted as numpy array')



inputShape = (10, IMG_SIZE, IMG_SIZE)
print(inputShape)

model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=inputShape))
model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(128, kernel_size=(5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(256, kernel_size=(5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(2, activation = 'softmax'))

model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics = ['accuracy'])
train_model = model.fit(trainImages, trainLabels, batch_size = 50, epochs = 6, verbose = 1,
                        validation_data=(testImages, testLabels))



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
train_model = model2.fit(trainImages, trainLabels, batch_size = 50, epochs = 10, verbose = 1,
                         validation_data=(testImages, testLabels))






data = np.load(test_data_nonmitosis[2])
data_uint8 = data.copy().astype('uint8')
data = data.astype('float32')/255
print(data.shape)
plt.imshow(data_uint8[5,:,:], cmap = 'gist_gray')
data = np.reshape(data,[1,10,60,60])

prediction = model.predict(data)
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