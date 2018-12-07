# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 10:13:24 2018

@author: Sardor
"""


import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

PATH_MAIN = 'DATASET_RAW/'

band_numbers = (0,1,2,3,4,5,6,7,8,9)

for main_directory in glob.glob(PATH_MAIN+'*/'):
    main_directory_name = main_directory[12:-1]
    HPF_NUMBER = []
    SPECTRAL_BAND = [[],[],[],[],[],[],[],[],[],[]]  # 10 spectral band
    
    for hpf_number_csv in glob.glob(main_directory+'*.csv'):
        hpf_number_csv_name = hpf_number_csv[19:-4]
        #parse csv file:
        with open(hpf_number_csv,'r') as csv_file:
            print('\nOpened csv file',hpf_number_csv)
            mitosis_pixels = []
            for line in csv_file:
                splitted_line = line.split(',')
                pixel_data = []
                for x,y in zip(splitted_line[0::2],splitted_line[1::2]):
                    pixel_data.append(( int(x), int(y) ))
                mitosis_pixels.append(pixel_data)
        
        # read images with mitosis 
        images_regex = main_directory+main_directory_name+'/' \
                        + hpf_number_csv_name[:-4] \
                        + '0[0-9]'+hpf_number_csv_name[-2:]+'.bmp'
        for image_name in glob.glob(images_regex):
            image = cv2.imread(image_name,0)
            print('  opened image',image_name)
            band_number = int(image_name[-7:-6])
            
            for pixel_data in mitosis_pixels:
                for pixels in pixel_data:
                    SPECTRAL_BAND[band_number].append(image[pixels[1],pixels[0]])

#        break
    HPF_NUMBER.append(SPECTRAL_BAND)
    break

for band_number in band_numbers:
    n, bins, patches = plt.hist(x=SPECTRAL_BAND[band_number], bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Band '+str(band_number))
    maxfreq = n.max()
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()


#transform data to hist

t = np.arange(0,256)
histogram = [[],[],[]]

for band_number in band_numbers:
    for xx in t:
        histogram[0].append(band_number)
        histogram[1].append(xx)
        histogram[2].append(SPECTRAL_BAND[band_number].count(xx))


# Make the plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(histogram[0], histogram[1], histogram[2], cmap=plt.cm.viridis, linewidth=0.2)
plt.show()
 
## to Add a color bar which maps values to colors.
#surf=ax.plot_trisurf(histogram[0], histogram[1], histogram[2], cmap=plt.cm.viridis, linewidth=0.2)
#fig.colorbar( surf, shrink=0.5, aspect=5)
#plt.show()
 
# Rotate it
ax.view_init(30, 45)
plt.show()
 
## Other palette
#ax.plot_trisurf(histogram[0], histogram[1], histogram[2], cmap=plt.cm.jet, linewidth=0.01)
#plt.show()

