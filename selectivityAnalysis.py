# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 17:02:58 2023

@author: Catherine Wang
"""
import sys
sys.path.append("C:\scripts\Imaging analysis")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import session
from matplotlib.pyplot import figure

# from neuralFuncs import plot_average_PSTH
# path = r'F:\data\BAYLORCW021\python\2023_01_25'

# l1 = session.Session(path, 5)

# path = r'F:\data\BAYLORCW027\python\2023_05_05'

path = r'F:\data\BAYLORCW021\python\2023_04_25'
# path = r'F:\data\BAYLORCW021\python\2023_04_27'

# path = r'F:\data\BAYLORCW021\python\2023_05_03'


l1 = session.Session(path, 3)

l1.plot_number_of_sig_neurons()

        
    
    
    
    