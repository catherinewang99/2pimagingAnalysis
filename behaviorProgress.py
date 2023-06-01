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

### Plot learning progression

# b = behavior.Behavior('F:\data\Behavior data\BAYLORCW028\python_behavior', behavior_only=True)
# b.learning_progression()

# b = behavior.Behavior('F:\data\Behavior data\BAYLORCW029\python_behavior', behavior_only=True)
# b.learning_progression()

# b = behavior.Behavior('F:\data\Behavior data\BAYLORCW021\python_behavior', behavior_only=True)
# b.learning_progression(window = 100)

# b = behavior.Behavior('F:\data\Behavior data\BAYLORCW027\python_behavior', behavior_only=True)
# b.learning_progression(window = 50)

### Plot over all imaging sessions

# b = behavior.Behavior('F:\data\BAYLORCW022\python')
# b.plot_LR_performance_over_sessions()
# b.plot_early_lick()

# b = behavior.Behavior('F:\data\BAYLORCW021\python')
# b.plot_LR_performance_over_sessions()
# b.plot_early_lick()

# b = behavior.Behavior('F:\data\BAYLORCW024\python')
# b.plot_LR_performance_over_sessions()
# b.plot_early_lick()

### Plot single session performance - diagnostic session

# b = behavior.Behavior(r'F:\data\BAYLORCW022\python\2023_03_04', single=True)
# b.plot_single_session(save=True)

# b = behavior.Behavior(r'F:\data\BAYLORCW021\python\2023_05_17', single=True)
# b.plot_single_session_multidose(save=True)

# b = behavior.Behavior(r'F:\data\BAYLORCW027\python\2023_04_10', single=True)
# b.plot_single_session()

# b = behavior.Behavior(r'F:\data\BAYLORCW027\python\2023_04_11', single=True)
# b.plot_single_session()

# b = behavior.Behavior(r'F:\data\BAYLORCW021\python\2023_04_25', single=True)
# b.plot_single_session(save=True)

# b = behavior.Behavior(r'F:\data\BAYLORCW021\python\2023_05_23', single=True)
# b.plot_single_session_multidose(save=True)

# b = behavior.Behavior(r'F:\data\BAYLORCW021\python\2023_02_27', single=True)
# b.plot_single_session(save=True)

# b = behavior.Behavior(r'F:\data\BAYLORCW021\python\2023_03_03', single=True)
# b.plot_single_session(save=True)