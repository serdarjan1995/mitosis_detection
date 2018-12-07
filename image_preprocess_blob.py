# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 13:14:19 2018

@author: Sardor
"""


import numpy as np
import cv2

image_name = 'DATASET_RAW\M00_01\M00_01\M00_01a_0607.bmp'

image_name_test = 'DATASET_RAW\M00_01\M00_01\M00_01b_0607.bmp'

csv= 'DATASET_RAW\M00_01\M00_01a_0607.csv'

csv_test= 'DATASET_RAW\M00_01\M00_01b_0607.csv'


#### train data
with open(csv,'r') as csv_file:
    print('\nOpened csv file',csv)
    mitosis_pixels_train = []
    for line in csv_file:
        splitted_line = line.split(',')
        pixel_data = []
        for x,y in zip(splitted_line[0::2],splitted_line[1::2]):
            pixel_data.append(( int(x), int(y) ))
        mitosis_pixels_train.append(pixel_data)
    
train_image = cv2.imread(image_name,0)
print('opened image',image_name)
train_image_mask = train_image.copy()*0

for pixel_data in mitosis_pixels_train:
    for pixels in pixel_data:
        train_image_mask[pixels[1],pixels[0]] = 255



# Set up the detector with default parameters.

params=cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea=250
params.maxArea=3500
params.filterByConvexity = False	
params.minConvexity=0.3	
#params.filterByCircularity = True
#params.minCircularity=0.1
detector=cv2.SimpleBlobDetector_create(params)
keypoints=detector.detect(train_image)
 
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_keypoint = cv2.drawKeypoints(train_image, keypoints, np.array([]),
                                      (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
# Show keypoints

resize_height = int(train_image.shape[0]/2)
resize_width = int(train_image.shape[1]/2)
train_image = cv2.resize( train_image, (resize_height, resize_width) )
#train_image_hist_eq = cv2.equalizeHist(train_image)

ret,th = cv2.threshold(train_image,np.amin(train_image)+20,255,cv2.THRESH_BINARY)
img_concat = np.concatenate((train_image, th), axis=1)

while True:
    cv2.imshow("Demo", img_concat)
    key_to_quit = cv2.waitKey(0)
    cv2.destroyAllWindows()
    if key_to_quit==27:    # Esc key to stop
        break
