# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 00:28:36 2018

@author: Sardor
"""

import glob
import numpy as np
import cv2

PATH_PATTERNS = 'MITOS_PATTERNS/'
PATH_MAIN = 'DATASET_RAW/'


for main_directory in glob.glob(PATH_MAIN+'*'):
    main_directory_name = main_directory[12:-1]
#    if(main_directory[16:-1] != '04'):
#        continue
    if(main_directory[14:-4] != '0'):
        break
    
    for hpf_number_csv in glob.glob(main_directory+'*.csv'):
#        if(hpf_number_csv[23:-9] != '04d'):
#            continue
        hpf_number_csv_name = hpf_number_csv[19:-4]
        #parse cv file:
        with open(hpf_number_csv,'r') as csv_file:
            print('opened csv file',hpf_number_csv)
            mitosis_pixels = []
            mitosis_count = 0
            for line in csv_file:
                mitosis_count += 1
                splitted_line = line.split(',')
                pixel_data = []
                for x,y in zip(splitted_line[0::2],splitted_line[1::2]):
                    pixel_data.append(( int(x), int(y) ))
                mitosis_pixels.append(pixel_data)
        
        # read images with mitosis 
        images_regex = main_directory+main_directory_name+'/' \
                        + hpf_number_csv_name[:-4] \
                        + '0[0-9]'+hpf_number_csv_name[-2:]+'.bmp'
        image_mask = None
        images = []
        contours = []
        for image_name in glob.glob(images_regex):
            image = cv2.imread(image_name,0)
            print('opened image',image_name)
            if( image_mask is None):
                image_mask = image.copy()
                image_mask *= 0
                for pixel_data in mitosis_pixels:
                    for pixels in pixel_data:
                        image_mask[pixels[1],pixels[0]] = 255
            
            
                # for optimizing contours
                kernel = np.ones((3,3),np.uint8)
                image_mask = cv2.dilate(image_mask, kernel, iterations=2)
                image_mask = cv2.erode(image_mask, kernel, iterations=2)
            
                # find contours in the binary image
                _, contours, _ = cv2.findContours(image_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
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
            images.append(image)
            
        
        for (num,bbox) in enumerate(boundingBoxes):            
            x,y,w,h = bbox
            mitosis = []
            for im in images:
                mitosis.append(im[y-5:y+h+5,x-5:x+w+5])
                
            mitosis_pattern_multispectral = np.empty((10,mitosis[0].shape[0],mitosis[0].shape[1]))
            for (i,mitos) in enumerate(mitosis):
                mitosis_pattern_multispectral[i] = mitos.copy()
                cv2.imwrite(PATH_PATTERNS+hpf_number_csv_name[:-4]+str(num)+'_'+str(i)+'.bmp',mitos)
            np.save(PATH_PATTERNS+hpf_number_csv_name[:-4]+str(num), mitosis_pattern_multispectral)
            

#            for (i,mitos) in enumerate(mitosis):
#                cv2.imshow(image_name,mitos)
#                key_to_quit = cv2.waitKey(0)
#                cv2.destroyAllWindows()
#                if key_to_quit==27:    # Esc key to stop
#                    break
        
#        break

