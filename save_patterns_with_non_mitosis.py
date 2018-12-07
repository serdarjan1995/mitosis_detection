# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 00:28:36 2018

@author: Sardor
"""

import glob
import numpy as np
import cv2
import random

PATH_PATTERNS = 'NONMITOS_PATTERNS/'
PATH_MAIN = 'DATASET_RAW/'

IMAGE_SHAPE = (1360,1360)
NON_MITOSIS_AREA_SHAPE = (60,60)
NON_MITOSIS_RANDOM_PICK = 10

for main_directory in glob.glob(PATH_MAIN+'*/'):
    main_directory_name = main_directory[12:-1]
    for part in 'abcd':
        csv_name = main_directory + main_directory_name + part + '_0607.csv'
        image_mask = None
        #check whether there is csv annotation, if is then generate mitosis mask
        if csv_name in glob.glob(main_directory+'*'):
            with open(csv_name,'r') as csv_file:
                print('opened csv file',csv_name)
                mitosis_pixels = []
                for line in csv_file:
                    splitted_line = line.split(',')
                    pixel_data = []
                    for x,y in zip(splitted_line[0::2],splitted_line[1::2]):
                        pixel_data.append(( int(x), int(y) ))
                    mitosis_pixels.append(pixel_data)
            
            csv_name_prev = csv_name
            image_mask = np.zeros(IMAGE_SHAPE,dtype='uint8')
            for pixel_data in mitosis_pixels:
                for pixels in pixel_data:
                    image_mask[pixels[1],pixels[0]] = 255
        
        #iterate over 10 spectral band of image
        images_regex = main_directory + main_directory_name +'/' + main_directory_name \
                        + part + '_0[0-9]06.bmp'
        images = []
        for image_name in glob.glob(images_regex):
            image = cv2.imread(image_name,0)
            print('opened image',image_name)
            images.append(image)
        
        #random pick areas
        non_mitosis_area = np.empty((10,NON_MITOSIS_AREA_SHAPE[0],NON_MITOSIS_AREA_SHAPE[1]))
        for i in range(0,NON_MITOSIS_RANDOM_PICK):
            if image_mask is None:
                randomp_x = random.randint(0,IMAGE_SHAPE[0]-NON_MITOSIS_AREA_SHAPE[0])
                randomp_y = random.randint(0,IMAGE_SHAPE[1]-NON_MITOSIS_AREA_SHAPE[1])           
            else:
                is_area_non_mitosis = False
                while(is_area_non_mitosis == False):
                    randomp_x = random.randint(0,IMAGE_SHAPE[0]-NON_MITOSIS_AREA_SHAPE[0])
                    randomp_y = random.randint(0,IMAGE_SHAPE[1]-NON_MITOSIS_AREA_SHAPE[1])
                    image_mask_area = image_mask[randomp_x:randomp_x+NON_MITOSIS_AREA_SHAPE[0],
                                            randomp_y:randomp_y+NON_MITOSIS_AREA_SHAPE[1]]
                    if(np.mean(image_mask_area) == 0):
                        is_area_non_mitosis = True
                    
            for band,image in enumerate(images):
                non_mitosis_area[i] = image[randomp_x:randomp_x+NON_MITOSIS_AREA_SHAPE[0],
                                            randomp_y:randomp_y+NON_MITOSIS_AREA_SHAPE[1]].copy()
                cv2.imwrite(PATH_PATTERNS + main_directory_name + part \
                            + '_' + str(i) + '_' + str(band) + '.bmp',non_mitosis_area[i])
            np.save(PATH_PATTERNS + main_directory_name + part+'_'+str(i), non_mitosis_area)
                    
#        break
#    break




#            for (i,mitos) in enumerate(mitosis):
#                cv2.imshow(image_name,mitos)
#                key_to_quit = cv2.waitKey(0)
#                cv2.destroyAllWindows()
#                if key_to_quit==27:    # Esc key to stop
#                    break
        
#        break

