# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 15:29:46 2023

@author: Catherine Wang
"""

import sys
sys.path.append("C:\scripts\Imaging analysis")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import session
import behavior
# from neuralFuncs import plot_average_PSTH


# layer_1 = scio.loadmat(r'E:\data\BAYLORCW022\python\2022_12_15\layer_1.mat')
# layer_2 = scio.loadmat(r'E:\data\BAYLORCW022\python\2022_12_15\layer_2.mat')

# behavior = scio.loadmat(r'E:\data\BAYLORCW022\python\2022_12_15\behavior.mat')

### Plot over all imaging sessions

# b = behavior.Behavior('F:\data\BAYLORCW022\python')
# b.plot_LR_performance_over_sessions()
# b.plot_early_lick()

# b = behavior.Behavior('F:\data\BAYLORCW021\python')
# b.plot_LR_performance_over_sessions()
# b.plot_early_lick()


### Plot single session performance - diagnostic session
b = behavior.Behavior(r'F:\data\BAYLORCW022\python\2023_03_02', single=True)
b.plot_single_session()