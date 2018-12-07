# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 11:34:01 2018

@author: Sardor
"""
import glob
import cv2
import numpy as np

RGB_DATASET_PATH = 'dataset_rgb/'
MULTISPECTRAL_DATASET_PATH = 'DATASET_RAW/'

for multispec_directory in glob.glob(MULTISPECTRAL_DATASET_PATH+'*/'):
    slide_number = multispec_directory[13:-4]
    hpf_number = multispec_directory[16:-1]

    if(slide_number != '00'):
        break
    if(hpf_number != '01'):
        break
    main_directory_name = multispec_directory[12:-1]

    for csv_file_in_dir in glob.glob(multispec_directory+'*.csv'):
        image_filename = csv_file_in_dir[19:-5]+'6'
        print(image_filename)
        #parse cv file:
        with open(csv_file_in_dir,'r') as csv_file:
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
                
        images_multispec_regex = multispec_directory + main_directory_name+'/' + image_filename[:-4] \
                        + '0[0-9]' + image_filename[-2:] + '.bmp'
        image_rgb_name = RGB_DATASET_PATH + 'A' + slide_number +'_v2/A' + slide_number \
                        + '_' + hpf_number + '.bmp'
                        
        image_rgb = cv2.imread(image_rgb_name)
        
        image_multispec_mask = None
        images_multispec = []
        for image_name in glob.glob(images_multispec_regex):
            image = cv2.imread(image_name,0)
            print('opened image',image_name)
            if( image_multispec_mask is None):
                image_multispec_mask = image.copy()
                image_multispec_mask *= 0
                for pixel_data in mitosis_pixels:
                    for pixels in pixel_data:
                        image_multispec_mask[pixels[1],pixels[0]] = 255
            
            
                # for optimizing contours
                kernel = np.ones((3,3),np.uint8)
                image_multispec_mask = cv2.dilate(image_multispec_mask, kernel, iterations=2)
                image_multispec_mask = cv2.erode(image_multispec_mask, kernel, iterations=2)
            
                # find contours in the binary image
                _, contours, _ = cv2.findContours(image_multispec_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                boundingBoxes = [cv2.boundingRect(c) for c in contours]
                if(len(boundingBoxes) > mitosis_count):
                    bbox_coors = []
                    removing_bbox = []
                    for boundingBox in boundingBoxes:
                        x,y,w,h = boundingBox
                        toggle = 0
                        for bbox in bbox_coors:
                            if bbox[0]<=x and (bbox[0]+bbox[2])>=(x+w) and \
                               bbox[1]<=y and (bbox[1]+bbox[3])>=(y+h) :
                                    toggle = 1
                                    removing_bbox.append(boundingBox)
                        if toggle == 0:
                            bbox_coors.append(boundingBoxes[0])
                    for rem in removing_bbox:
                        boundingBoxes.remove(rem)
            images_multispec.append(image)
        
        image_rgb_resize_val = image.shape[0]*2
        image_rgb = cv2.resize( image_rgb, (image_rgb_resize_val, image_rgb_resize_val) )
        
        if images_multispec_regex[-14:-13] == 'a':
            image_rgb_cropped = image_rgb[15:image.shape[0],:image.shape[0]-30].copy()
            image_rgb_cropped_reflect = cv2.copyMakeBorder(image_rgb_cropped,0,0,30,0,cv2.BORDER_REFLECT)
            image_rgb_cropped_reflect2 = image_rgb_cropped_reflect[30:,:-30]
        break


image_rgb_cropped = image_rgb[40:image.shape[0]+20,:image.shape[0]-40].copy()
image_rgb_cropped_reflect = cv2.copyMakeBorder(image_rgb_cropped,0,0,30,0,cv2.BORDER_REFLECT)
#image_rgb_cropped_reflect2 = image_rgb_cropped_reflect[30:,:-30]
cv2.imwrite('rgb_mul.bmp',image_rgb_cropped_reflect)
#cv2.imwrite('rgb_mul0.bmp',image_rgb_cropped_reflect2)
#cv2.imwrite('rgb_mul2.bmp',images_multispec[5])
#while True:
#    cv2.imshow('rgb',image_rgb_cropped)
#    cv2.imshow('band0',images_multispec[0])
#    key_to_quit = cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    if key_to_quit==27:    # Esc key to stop
#        break