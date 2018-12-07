# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 10:19:10 2018

@author: Sardor
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

file1 = 'image_mulM00_01b.bmp'
file2 = 'image_22M00_01a.bmp'
file3 = 'image_23M00_01a.bmp'
file4 = 'image_DI_div.bmp'

csv = 'DATASET_RAW\M00_01\M00_01b_0607.csv'


image = cv2.imread(file1,0)

def prob_pixel(array):
    bins, counts = np.unique(array, return_counts=True)
    counts = counts / array.size
    return dict(zip(bins, counts))

def posterior(mc_px_prob,img_px_prob,mc_prob):
    posterior = {}
    for i,prob in img_px_prob.items():
        prob_in_mc = mc_px_prob.get(i)
        if prob_in_mc is None:
            posterior[i] = 0
            continue
        else:
            posterior[i] = (prob_in_mc * mc_prob) / prob
    return posterior
        

with open(csv,'r') as csv_file:
    print('opened csv file',csv_file.name)
    mitosis_pixels = []
    mitosis_count = 0
    for line in csv_file:
        mitosis_count += 1
        splitted_line = line.split(',')
        pixel_data = []
        for x,y in zip(splitted_line[0::2],splitted_line[1::2]):
            pixel_data.append(( int(x), int(y) ))
        mitosis_pixels.append(pixel_data)

image_mask = image.copy()
image_mask *= 0
for pixel_data in mitosis_pixels:
    for pixels in pixel_data:
        image_mask[pixels[1],pixels[0]] = 255


pixels_array = image.copy().flatten()
mask_array = image_mask.copy().flatten()
mc_pixels = np.array([x for x,y in zip(pixels_array,mask_array) if y==255])
non_mc_pixels = np.array([x for x,y in zip(pixels_array,mask_array) if y==0])

p_mc = mc_pixels.size / pixels_array.size
#p_non_mc = non_mc_pixels.size / pixels_array.size

pixel_probabilities_image = prob_pixel(pixels_array)
pixel_probabilities_for_mc = prob_pixel(mc_pixels)

posterior = posterior(pixel_probabilities_for_mc,pixel_probabilities_image,p_mc)
xx = np.array([x for x in posterior.keys()])
yy = np.array([x for x in posterior.values()])

data = yy*255.0/yy.max()
data = data.astype('uint8')

dict_data = dict(zip(xx,data))

im = image.copy().flatten()

for i in range(0,im.size):
    if im[i] in dict_data:
        im[i] = dict_data.get(im[i])
    else:
        im[i] = 0

im = im.reshape(image.shape)
#cv2.imwrite('poster_map.bmp',im)
#kernel = np.ones((3,3),np.uint8)
#im = cv2.erode(im, kernel, iterations=1)

kernel = np.ones((7,7),np.float32)/25
dst = cv2.filter2D(im,-1,kernel)
resize_val = int(dst.shape[0]/2)
dst = cv2.resize( dst, (resize_val, resize_val) )
im = cv2.resize( im, (resize_val, resize_val) )


#dst = cv2.equalizeHist(dst)

_, thresh = cv2.threshold(im,100,255,cv2.THRESH_BINARY)

#kernel = np.ones((5,5),np.uint8)
#thresh = cv2.erode(dst, thresh, iterations=1)

while True:
    cv2.imshow('im',image_mask)
#    cv2.imshow('image',dst)
#    cv2.imshow('thresh',thresh)
    key_to_quit = cv2.waitKey(0)
    cv2.destroyAllWindows()
    if key_to_quit==27:    # Esc key to stop
        break