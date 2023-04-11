# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 11:46:30 2023

@author: Catherine Wang
"""

import sys
sys.path.append("C:\scripts\Imaging analysis")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import session
from activityMode import Mode
from matplotlib.pyplot import figure

# from neuralFuncs import plot_average_PSTH
# path = r'F:\data\BAYLORCW021\python\2023_01_25'

# l1 = session.Session(path, 5)

path = r'F:\data\BAYLORCW021\python\2023_04_06'

l1 = mode(path, 6)