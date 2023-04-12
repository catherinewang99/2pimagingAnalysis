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
import numpy as np

# from neuralFuncs import plot_average_PSTH
path = r'F:\data\BAYLORCW021\python\2023_01_25'

l1 = session.Session(path, 5)

# path = r'F:\data\BAYLORCW021\python\2023_04_06'

# l1 = Mode(path, 3)


# orthonormal_basis, var_allDim = l1.func_compute_activity_modes_DRT(l1.PSTH_r_correct, 
#                                                                     l1.PSTH_l_correct, 
#                                                                     l1.PSTH_r_error, 
#                                                                     l1.PSTH_l_error)

# wt = (l1.PSTH_r_correct + l1.PSTH_r_error) / 2 - (l1.PSTH_l_correct + l1.PSTH_l_error) / 2

