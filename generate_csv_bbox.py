# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 15:46:30 2018

@author: Sardor
"""
import numpy as np
import glob

PATH_PATTERNS = 'MITOS_PATTERNS/'

csv_data = ""
for numpy_file in glob.glob(PATH_PATTERNS+'*.npy'):
    print(numpy_file)
    data = np.load(numpy_file)
    csv_data += numpy_file
    csv_data += ',0,0,'
    csv_data += str(data.shape[1]) + ',' + str(data.shape[2])
    csv_data += ',mitosis\n'

with open('train_data.csv','w') as write_file:
    write_file.write(csv_data)