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

# path = r'F:\data\BAYLORCW021\python\2023_04_06'

# l1 = Mode(path, 3)


# orthonormal_basis, var_allDim = l1.func_compute_activity_modes_DRT(l1.PSTH_r_correct, 
#                                                                    l1.PSTH_l_correct, 
#                                                                    l1.PSTH_r_error, 
#                                                                    l1.PSTH_l_error)

t_sample = time_epochs[0]
t_delay = time_epochs[1]
t_response = time_epochs[2]

activityRL = np.concatenate(l1.PSTH_r_correct, l1.PSTH_l_correct), axis=1)
activityRL = activityRL - np.mean(activityRL, axis=1, keepdims=True) # remove?
u, s, v = np.linalg.svd(activityRL.T)
proj_allDim = activityRL.T @ v

# Variance of each dimension normalized
var_s = np.square(np.diag(s[0:proj_allDim.shape[1], :]))
var_allDim = var_s / np.sum(var_s)