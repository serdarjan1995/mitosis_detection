# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 14:43:16 2018

@author: Sardor
"""

import numpy as np
import glob
from random import shuffle
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
import matplotlib.pyplot as plt

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

class MITOSIS_CNN:
    def __init__(self,
                 img_size=80,
                 weights=None,
                 channels=1):
        #change channel format
        if K.image_data_format() == 'channels_last':
            K.set_image_data_format('channels_first')
                    
        self.weights = weights
        self.IMG_SIZE = img_size
        self.channels = channels
        self.train_data_generated = False

        self.model = Sequential()
        self.model.add(Convolution2D(32,
                                kernel_size=(3,3),
                                activation='relu',
                                input_shape=self.input_shape))
        self.model.add(Convolution2D(128,
                                kernel_size=(5,5),
                                activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Convolution2D(256,
                                kernel_size=(7,7),
                                activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(64, activation='relu'))
        #model.add(Dropout(0.2))
        self.model.add(Dense(2, activation = 'softmax'))
        
        if self.weights is not None:
            self.model.load_weights(self.weights)
            self.model_trained = True
        self.model.compile(loss='binary_crossentropy',
                           optimizer='adadelta',
                           metrics = ['accuracy'])

        
    
    def generate_train_data(self,
                            MITOSIS_PATH,
                            NONMITOSIS_PATH):
        self.class_mitos = glob.glob(MITOSIS_PATH+'*.npy')
        self.class_nonmitos = glob.glob(NONMITOSIS_PATH+'*.npy')
        
        if(self.model_trained==False):
            shuffle(self.class_mitos)
            shuffle(self.class_nonmitos)
            train_data = []
            len_class_mitos = len(self.class_mitos)
            ii_step = int(len_class_mitos/10)
            progress = 0
            for ii,numpy_dump in enumerate(self.class_mitos):
                data = np.load(numpy_dump).astype('float32')/255.0
                train_data.append([data,np.array([1,0])])
                if ii>progress:
                    print('mitosis',len_class_mitos,'/',progress)
                    progress += ii_step
            print('mitosis train data generated')
            
            len_class_nonmitos = len(self.class_nonmitos)
            ii_step = int(len(self.class_nonmitos) / 10)
            progress = 0
            for ii,numpy_dump in enumerate(self.class_nonmitos):
                data = np.load(numpy_dump).astype('float32')/255.0
                train_data.append([data,np.array([0,1])])
                if ii>progress:
                    print('nonmitosis',len_class_nonmitos,'/',progress)
                    progress += ii_step
            print('nonmitosis train data generated')
            
            self.trainImages = np.array([i[0] for i in train_data])
            self.trainLabels = np.array([i[1] for i in train_data])
            print('data converted as numpy array')
            
            self.trainImages = self.trainImages.reshape((self.trainImages.shape[0],
                                                         self.channels,
                                                         self.trainImages.shape[1],
                                                         self.trainImages.shape[2]))
            self.train_data_generated = True
            self.input_shape = inputShape = (self.channels, self.IMG_SIZE, self.IMG_SIZE)  
            print('Input shape:',inputShape)
            print('\nRun train_model()')
        
        else:
            print('Model is already trained. If you want to train again please run:')
            print('MITOSIS_CNN.model_trained = False')
    
    
    def train_model(self,cross_validation=0.3):
        if(self.train_data_generated==True):
            checkpoint = ModelCheckpoint('weights',
                                         monitor='acc',
                                         verbose=1,
                                         save_best_only=True,
                                         mode='max')
            early_stop = EarlyStopping(monitor='acc',
                                       min_delta=0.01,
                                       patience=5,
                                       mode='max') 
            
            callbacks_list = [checkpoint,early_stop]
            self.train_model = self.model.fit(self.trainImages,
                                         self.trainLabels,
                                         batch_size = 100,
                                         epochs = 10,
                                         verbose = 1,
                                         validation_split=cross_validation,
                                         callbacks=callbacks_list)
            self.model_trained = True
        else:
            print('Train data is not generated')
            
        
    def plot_accuracy(self):
        if(self.model_trained == True and self.weights is None):
            accuracy = self.train_model.history['acc']
            val_accuracy = self.train_model.history['val_acc']
            loss = self.train_model.history['loss']
            val_loss = self.train_model.history['val_loss']
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
        else:
            print('Model is not trained')
           
            
    def predict_class(self,data):
        if self.model_trained == True:
            data = data.astype('float32')/255
            data = np.reshape(data,[1,self.channels,self.IMG_SIZE,self.IMG_SIZE])
            prediction = self.model.predict(data)
            return prediction
        else:
            print('Weights is not defined or model is not trained')