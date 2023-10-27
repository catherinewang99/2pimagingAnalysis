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
# b.learning_progression(imaging=True)

b = behavior.Behavior('F:\data\Behavior data\BAYLORCW034\python_behavior', behavior_only=True)
b.learning_progression(imaging=True)
b.learning_progression()

# b = behavior.Behavior('F:\data\Behavior data\BAYLORCW034\python_behavior', behavior_only=True)
# b.learning_progression(imaging=True)
# b.learning_progression()

# b = behavior.Behavior('F:\data\Behavior data\BAYLORCW021\python_behavior', behavior_only=True)
# b.learning_progression(window = 100)
# b.learning_progression(window = 100, imaging=True)

# b = behavior.Behavior('F:\data\Behavior data\BAYLORCW027\python_behavior', behavior_only=True)
# b.learning_progression(window = 50)

### Compare learning curves ####

# b = behavior.Behavior('F:\data\Behavior data\BAYLORCW032\python_behavior', behavior_only=True)
# delays, acc_arr_32, numtrials_32 = b.learning_progression(return_results = True)


# b = behavior.Behavior('F:\data\Behavior data\BAYLORCW034\python_behavior', behavior_only=True)
# delays, acc_arr, numtrials = b.learning_progression(return_results = True)

# b = behavior.Behavior('F:\data\Behavior data\BAYLORCW036\python_behavior', behavior_only=True)
# delays, acc_arr_36, numtrials_36 = b.learning_progression(return_results = True)

# plt.plot(acc_arr_32[numtrials_32[0]:numtrials_32[4]] - acc_arr_32[numtrials_32[0]], color='r')
# plt.plot(acc_arr_32[numtrials_32[6]:numtrials_32[10]] - acc_arr_32[numtrials_32[6]], color='r')
# plt.plot(acc_arr_32[numtrials_32[4]:numtrials_32[6]] - acc_arr_32[numtrials_32[4]], color='g')
# plt.plot(acc_arr_32[numtrials_32[10]:numtrials_32[12]] - acc_arr_32[numtrials_32[10]], color='g')

# plt.plot(acc_arr_36[numtrials_36[0]:numtrials_36[2]] - acc_arr_36[numtrials_36[0]], color='r')
# plt.plot(acc_arr_36[numtrials_36[2]:numtrials_36[5]] - acc_arr_36[numtrials_36[2]], color='g')

# plt.plot(acc_arr[numtrials[0]:numtrials[2]] - acc_arr[numtrials[0]], color='r')
# plt.plot(acc_arr[numtrials[4]:numtrials[6]] - acc_arr[numtrials[4]], color='r', label='Full delay')
# plt.plot(acc_arr[numtrials[2]:numtrials[4]] - acc_arr[numtrials[2]], color='g', label='Varied delay')
# plt.plot(acc_arr[numtrials[6]:numtrials[8]] - acc_arr[numtrials[6]], color='g')

# plt.ylabel('Change in performance accuracy')
# plt.xlabel('Number of trials')

# plt.legend()
# plt.show()

### Plot session to match GLM HMM
# sessions = ['20230215', '20230322', '20230323',  '20230403', '20230406', '20230409', '20230411',
#             '20230413', '20230420', '20230421', '20230423', '20230424', '20230427',
#             '20230426', '20230503', '20230508', '20230509', '20230510', '20230511', '20230512',
#             '20230516', '20230517']

# b = behavior.Behavior('F:\data\Behavior data\BAYLORCW021\python_behavior', behavior_only=True, glmhmm=sessions)
# b.learning_progression()



### Plot over all imaging sessions

# b = behavior.Behavior('F:\data\BAYLORCW022\python')
# b.plot_LR_performance_over_sessions()
# b.plot_early_lick()

# b = behavior.Behavior('F:\data\BAYLORCW021\python')
# b.plot_LR_performance_over_sessions()
# b.plot_early_lick()

# b = behavior.Behavior('F:\data\BAYLORCW030\python')
# b.plot_LR_performance_over_sessions()
# b.plot_early_lick()

### Plot single session performance - diagnostic session

# b = behavior.Behavior(r'F:\data\BAYLORCW022\python\2023_03_04', single=True)
# b.plot_single_session(save=True)

# b = behavior.Behavior(r'F:\data\BAYLORCW021\python\2023_05_17', single=True)
# b.plot_single_session_multidose(save=True)

# b = behavior.Behavior(r'F:\data\BAYLORCW027\python\2023_04_10', single=True)
# b.plot_single_session()

# b = behavior.Behavior(r'F:\data\BAYLORCW034\python\2023_10_11', single=True)
# b.plot_single_session()

# b = behavior.Behavior(r'F:\data\BAYLORCW021\python\2023_04_25', single=True)
# b.plot_single_session(save=True)

# b = behavior.Behavior(r'F:\data\BAYLORCW021\python\2023_05_23', single=True)
# b.plot_single_session_multidose(save=True)

# b = behavior.Behavior(r'F:\data\BAYLORCW021\python\2023_02_27', single=True)
# b.plot_single_session(save=True)

# b = behavior.Behavior(r'F:\data\BAYLORCW021\python\2023_03_03', single=True)
# b.plot_single_session(save=True)