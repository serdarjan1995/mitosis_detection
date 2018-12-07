# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 14:29:51 2018

@author: Sardor
"""

import glob
import cv2
import numpy as np

PATH_MAIN = 'DATASET_RAW/'

image_shape = 1360,1360
kernel = 60,60
step = 3,3

image_regex = 'DATASET_RAW/M00_01/M00_01/M00_01a_0[0-9]06.bmp'
image = np.empty([10,1360,1360])
for i,img in enumerate(glob.glob(image_regex)):
    image[i] = cv2.imread(img,0)

for i in range(0,image_shape[0]-kernel[0],step[0]):
    for j in range(0,image_shape[1]-kernel[1],step[1]):
        area = image[i:i+kernel[0],j:j+kernel[1]]
        break
    break