import glob
import numpy as np
import cv2

PATH_MAIN = 'DATASET_RAW/'

SLIDES = []


for main_directory in glob.glob(PATH_MAIN+'*/'):
    main_directory_name = main_directory[12:len(main_directory)-1]
    HPF_NUMBER = [main_directory_name]
    print(HPF_NUMBER)
    for hpf_number_csv in glob.glob(main_directory+'*.csv'):
        hpf_number_csv_name = hpf_number_csv[19:len(hpf_number_csv)-4]
        #parse cv file:
        with open(hpf_number_csv,'r') as csv_file:
            mitosis_pixels = []
            for line in csv_file:
                splitted_line = line.split(',')
                pixel_data = []
                for x,y in zip(splitted_line[0::2],splitted_line[1::2]):
                    pixel_data.append(( int(x), int(y) ))
                mitosis_pixels.append(pixel_data)
        
        # read images with mitosis 
        images_regex = main_directory+main_directory_name+'/' \
                        + hpf_number_csv_name[:len(hpf_number_csv_name)-4] \
                        + '0[0-9]'+hpf_number_csv_name[-2:]+'*.bmp'
        for image_name in glob.glob(images_regex):
            image = cv2.imread(image_name,0)
            image_mask = image.copy()
            image_mask *= 0
            for pixel_data in mitosis_pixels:
                for pixels in pixel_data:
                    image_mask[pixels[1],pixels[0]] = 255
            
            dst = image.copy()
            resize_height = int(dst.shape[0]/2)
            resize_width = int(dst.shape[1]/2)
            dst = cv2.resize( dst, (resize_height, resize_width) )
            
            image_mask = cv2.resize(image_mask, (resize_height, resize_width))
            
            # for optimizing contours
            kernel = np.ones((3,3),np.uint8)
            image_mask = cv2.dilate(image_mask, kernel, iterations=1)
            image_mask = cv2.erode(image_mask, kernel, iterations=1)
            
            
            # find contours in the binary image
            _, contours, _ = cv2.findContours(image_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            
            #change colorspace to bgr
            dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
            image_mask = cv2.cvtColor(image_mask, cv2.COLOR_GRAY2BGR)
            
            cv2.drawContours(dst, contours, -1, (255,0,255), 1)
            for c in contours:
                # calculate moments for each contour
                M = cv2.moments(c)

                # calculate x,y coordinate of center
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(dst,(x-3,y-3),(x+w+3,y+h+3),(100,255,150),1)
                
            
            dst = np.concatenate((dst, image_mask), axis=1)
            cv2.imshow(image_name,dst)
            key_to_quit = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key_to_quit==27:    # Esc key to stop
                break
            
        
#        break
    
    break
    