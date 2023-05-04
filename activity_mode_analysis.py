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
# path = r'F:\data\BAYLORCW021\python\2023_01_25'

# l1 = session.Session(path, 5)


path = r'F:\data\BAYLORCW021\python\2023_03_03'
path = r'F:\data\BAYLORCW021\python\2023_02_13'
path = r'F:\data\BAYLORCW021\python\2023_04_27'
# path = r'F:\data\BAYLORCW021\python\2023_04_06'


l1 = Mode(path, 6)


# a, b = l1.plot_activity_modes_err()


# a, b = l1.plot_activity_modes_ctl()
a, b = l1.plot_activity_modes_opto()

a, b = l1.plot_activity_modes_opto(error=True)

l1.plot_behaviorally_relevant_modes()
l1.plot_behaviorally_relevant_modes_opto(error=True)

### DEBUGGING MATERIAL ###

# activityRL = np.concatenate((l1.PSTH_r_correct, l1.PSTH_l_correct), axis=1)
# activityRL = activityRL - np.mean(activityRL, axis=1, keepdims=True) # remove?
# u, s, v = np.linalg.svd(activityRL.T)
# proj_allDim = activityRL.T @ v


# t_sample, t_delay = 7, 13
# wt = (l1.PSTH_r_correct + l1.PSTH_r_error) / 2 - (l1.PSTH_l_correct + l1.PSTH_l_error) / 2

# i_t = np.where((l1.T_cue_aligned_sel > t_sample) & (l1.T_cue_aligned_sel < t_delay))[0]
# CD_stim_mode = np.mean(wt[:, i_t], axis=1)

# CD_stim_mode = CD_stim_mode / np.linalg.norm(CD_stim_mode)
# CD_stim_mode = np.reshape(CD_stim_mode, (-1, 1)) 

# input_ = np.concatenate((CD_stim_mode, v), axis=1)
# orthonormal_basis = self.Gram_Schmidt_process(input_)

# orthonormal_basis, var_allDim = l1.func_compute_activity_modes_DRT(l1.PSTH_r_correct, 
#                                                                     l1.PSTH_l_correct, 
#                                                                     l1.PSTH_r_error, 
#                                                                     l1.PSTH_l_error)
