# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 14:29:51 2018

@author: Sardor
"""

import glob
import cv2
from mitos_model import CNN_MODEL

IMG_SIZE = 80
image_shape = 1360,1360
kernel = IMG_SIZE,IMG_SIZE
step = 3,3

MITOSIS_PATH = 'MITOS_PATTERNS_80/'
NONMITOSIS_PATH = 'NONMITOS_PATTERNS_80/'

#image_regex = 'DATASET_RAW/M00_01/M00_01/M00_01a_0[0-9]06.bmp'
images_path = 'DI_IMAGES/'

model = CNN_MODEL(IMG_SIZE, weights=None, channels=1)
model.generate_train_data(MITOSIS_PATH,NONMITOSIS_PATH)
model.train_model(cross_validation=0.3)



for i,img in enumerate(glob.glob(images_path)):
    image = cv2.imread(img,0)
    for i in range(0,image_shape[0]-kernel[0], step[0]):
        for j in range(0,image_shape[1]-kernel[1], step[1]):
            area = image[i:i+kernel[0],j:j+kernel[1]]
            model.predict_class(area)
            break
        break
    break