# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 21:37:40 2018

@author: Sardor
"""

import glob
from random import shuffle

cross_validation = 0.3

MITOSIS_PATH = 'MITOS_PATTERNS/'
NONMITOSIS_PATH = 'NONMITOS_PATTERNS/'

class_mitos = glob.glob(MITOSIS_PATH+'*.npy')
class_nonmitos = glob.glob(NONMITOSIS_PATH+'*.npy')

shuffle(class_mitos)
shuffle(class_nonmitos)


split_p = int( len(class_mitos)*cross_validation )

train_data = class_mitos[split_p:]
test_data = class_mitos[:split_p]
