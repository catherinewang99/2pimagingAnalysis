# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 12:31:52 2023

@author: Catherine Wang
"""

import sys
sys.path.append("C:\scripts\Imaging analysis")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import session
from matplotlib.pyplot import figure

path = r'F:\data\BAYLORCW021\python\2023_01_25'

total_n = 0
for i in range(1, 7):
    layer = scio.loadmat(r'{}\layer_{}.mat'.format(path, i))
    behavior = scio.loadmat(r'{}\behavior.mat'.format(path))
    l1 = session.Session(layer, i, behavior)
    # l1.crop_trials(245, end = 330)
    total_n += l1.num_neurons

print(total_n)