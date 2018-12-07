# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 11:46:53 2018

@author: Sardor
"""

import cv2
import glob
import numpy as np

PATH_MAIN = 'DATASET_RAW/'


for directory in glob.glob(PATH_MAIN+'*/'):
    slide_number = directory[13:-4]
    hpf_number = directory[16:-1]

    if(slide_number != '00'):
        break
    if(hpf_number != '01'):
        break
    main_directory_name = directory[12:-1]

    for csv_file_in_dir in glob.glob(directory+'*.csv'):
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
    
        images_regex = directory + main_directory_name+'/' + image_filename[:-4] \
                        + '0[0-9]' + image_filename[-2:] + '.bmp'
        image_mask = None
        images = []
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
        
        # mc and non_mc seperation
        mask_array = image_mask.copy().flatten()
        images_mc_variances = []
        images_non_mc_variances = []
        images_mc_means = []
        images_non_mc_means = []
        for image in images:
            pixels_array = image.copy().flatten()
            mc_pixels = np.array([x for x,y in zip(pixels_array,mask_array) if y==255])
            non_mc_pixels = np.array([x for x,y in zip(pixels_array,mask_array) if y==0])
            
            images_mc_means.append(np.mean(mc_pixels))
            images_mc_variances.append(np.var(mc_pixels))
            
            images_non_mc_means.append(np.mean(non_mc_pixels))
            images_non_mc_variances.append(np.var(non_mc_pixels))
            
        images_variances_np = np.array([ x for x in images_mc_variances ])
        images_variances_np += np.array([ x for x in images_non_mc_variances ])
        images_means_np = np.array([ x for x in images_non_mc_means ])
        images_means_np -= np.array([ x for x in images_mc_means ])
        
        discriminative_vector2 =  images_means_np * images_variances_np
        
        disc_image2 = None
        for coef,image in zip(discriminative_vector2,images):
            if(disc_image2 is None):
                disc_image2 = (image*coef)
            else:
                disc_image2 += (image*coef)
        
        norm_disc_image = disc_image2 * 255.0/disc_image2.max()
        norm_disc_image = norm_disc_image.astype('uint8')
        cv2.imwrite('image_mul'+image_filename[:-5]+'.bmp',norm_disc_image)
        resize_val = int(disc_image2.shape[0]/3)
        norm_disc_image = cv2.resize( norm_disc_image, (resize_val, resize_val) )
        
        
        while True:
            cv2.imshow(image_name,norm_disc_image)
            key_to_quit = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key_to_quit==27:    # Esc key to stop
                break
        
#        image_mc_variances = []
#        image_non_mc_variances = []
#        image_mc_means = []
#        image_non_mc_means = []
#        for i,image in enumerate(images):
#            #sum and mean calculation for mc and non_mc
#            image_mitosis_pixel_sum = 0
#            image_mitosis_pixel_cnt = 0
#            for pixel_data in mitosis_pixels:
#                for pixels in pixel_data:
#                    image_mitosis_pixel_sum += image[pixels[1],pixels[0]]
#                    image_mitosis_pixel_cnt += 1
#            image_non_mitosis_pixel_sum = np.sum(image) - image_mitosis_pixel_sum
#            image_non_mitosis_pixel_cnt = np.size(image) - image_mitosis_pixel_cnt
#            
#            image_non_mc_avg = image_non_mitosis_pixel_sum / image_non_mitosis_pixel_cnt
#            image_mc_avg = image_mitosis_pixel_sum / image_mitosis_pixel_cnt
#            image_mc_means.append(image_mc_avg)
#            image_non_mc_means.append(image_non_mc_avg)
#            
#            #variance calculation for mc and non_mc
#            mc_mean_subtracted_sum = 0
#            non_mc_mean_subtracted_sum = 0
#            for i in range(0,image.shape[0]):
#                for j in range(0,image.shape[1]):
#                    if(image_mask[i,j] == 0):
#                        non_mc_mean_subtracted_sum += (image[i,j]-image_non_mc_avg)
#                    else:
#                        mc_mean_subtracted_sum += (image[i,j]-image_mc_avg)
#            non_mc_variance = non_mc_mean_subtracted_sum / (image_non_mitosis_pixel_cnt-1)
#            mc_variance = mc_mean_subtracted_sum / (image_mitosis_pixel_cnt-1)
#            
#            image_mc_variances.append(mc_variance)
#            image_non_mc_variances.append(non_mc_variance)
#
#            
#        image_variances_np = np.array([ x for x in image_mc_variances ])
#        image_variances_np += np.array([ x for x in image_non_mc_variances ])
#        image_means_np = np.array([ x for x in image_non_mc_means ])
#        image_means_np -= np.array([ x for x in image_mc_means ])
#               
#        discriminative_vector =  image_means_np * image_variances_np
#        
#        disc_image = None
#        for coef,image in zip(discriminative_vector,images):
#            if(disc_image is None):
#                disc_image = (image*coef)
#            else:
#                disc_image += (image*coef)
#        norm_disc_image = disc_image * 255.0/disc_image.max()
#        norm_disc_image = norm_disc_image.astype('uint8')
#        #resize images
#        resize_val = int(norm_disc_image.shape[0]/3)
#        norm_disc_image = cv2.resize( norm_disc_image, (resize_val, resize_val) )
        
#        while True:
#            cv2.imshow(image_name,norm_disc_image)
#            key_to_quit = cv2.waitKey(0)
#            cv2.destroyAllWindows()
#            if key_to_quit==27:    # Esc key to stop
#                break
#            elif key_to_quit == ord('s'): # wait for 's' key to save and exit
#                cv2.imwrite('image_DI_div.bmp',norm_disc_image)
        
#        cv2.putText(norm_disc_image, 'DI image', (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
#                        0.5, 0, 2)
#        image_mask = cv2.resize( image_mask, (resize_val, resize_val) )
#        cv2.putText(image_mask, 'Mask', (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
#                        0.5, 255, 2)
#        for i in range(0,len(images)):
#            images[i] = cv2.resize( images[i], (resize_val, resize_val) )
#            cv2.putText(images[i], 'Band '+str(i), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
#                        0.5, (0, 255, 0), 2, cv2.LINE_AA)
#        
#        img_concat1 = np.concatenate((images[0], images[1], images[2], images[3]), axis=1)
#        img_concat2 = np.concatenate((images[4], images[5], images[6], images[7]), axis=1)
#        img_concat3 = np.concatenate((images[8], images[9], norm_disc_image, image_mask), axis=1)
#        img_concat_all = np.concatenate((img_concat1,img_concat2,img_concat3), axis=0)
#        cv2.imwrite('image_concat'+image_filename[:-5]+'.bmp',img_concat_all)
        print('done ',image_filename[:-5])
#        while True:
#            cv2.imshow(image_name,norm_disc_image)
#            key_to_quit = cv2.waitKey(0)
#            cv2.destroyAllWindows()
#            if key_to_quit==27:    # Esc key to stop
#                break
#            elif key_to_quit == ord('s'): # wait for 's' key to save and exit
#                cv2.imwrite('image_concat'+image_filename[:-5]+'.bmp',img_concat_all)
#                cv2.destroyAllWindows()
        
#        break   #break for one image on 10 different band
    break
