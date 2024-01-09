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
from sklearn.decomposition import PCA

# from neuralFuncs import plot_average_PSTH
# path = r'F:\data\BAYLORCW021\python\2023_01_25'

# l1 = session.Session(path, 5)


# path = r'F:\data\BAYLORCW021\python\2023_03_03'
# path = r'F:\data\BAYLORCW021\python\2023_02_13'
# path = r'F:\data\BAYLORCW021\python\2023_04_25'
# path = r'F:\data\BAYLORCW021\python\2023_04_27'
path = r'F:\data\BAYLORCW032\python\2023_10_24'
# path = r'F:\data\BAYLORCW036\python\2023_10_28'

# l1 = Mode(path, use_reg = True)
# l1.plot_CD()
# orthonormal_basis, mean, db, acc_within = l1.decision_boundary()
# print(np.mean(acc_within))

# path = r'F:\data\BAYLORCW021\python\2023_05_03'

#%%
# Choice decoder
# NAIVE to TRAINED
path = r'F:\data\BAYLORCW032\python\2023_10_08'
l1 = Mode(path, use_reg = True)
orthonormal_basis, mean = l1.plot_CD(save = 'F:\data\SFN 2023\CDdelay_naive_trainednaive.pdf')

path = r'F:\data\BAYLORCW032\python\2023_10_25'
l1 = Mode(path, use_reg = True)
l1.plot_appliedCD(orthonormal_basis, mean, save = 'F:\data\SFN 2023\CDdelay_expert_trainednaive.pdf')

#TRAINED to NAIVE
path = r'F:\data\BAYLORCW032\python\2023_10_25'
l1 = Mode(path, use_reg = True)
orthonormal_basis, mean = l1.plot_CD(save = 'F:\data\SFN 2023\CDdelay_expert_trainedexpert.pdf')

path = r'F:\data\BAYLORCW032\python\2023_10_08'
l1 = Mode(path, use_reg = True)
l1.plot_appliedCD(orthonormal_basis, mean,save = 'F:\data\SFN 2023\CDdelay_naive_trainedexpert.pdf')

#%%
# Stimulus decoder
# NAIVE to TRAINED
path = r'F:\data\BAYLORCW032\python\2023_10_08'
l1 = Mode(path, use_reg = True)
orthonormal_basis, mean = l1.plot_CD(epoch=range(l1.sample + 1, l1.delay + 1))

path = r'F:\data\BAYLORCW032\python\2023_10_25'
l1 = Mode(path, use_reg = True)
l1.plot_appliedCD(orthonormal_basis, mean)

#TRAINED to NAIVE
path = r'F:\data\BAYLORCW032\python\2023_10_25'
l1 = Mode(path, use_reg = True)
orthonormal_basis, mean = l1.plot_CD(epoch=range(l1.sample + 1, l1.delay + 1))

path = r'F:\data\BAYLORCW032\python\2023_10_08'
l1 = Mode(path, use_reg = True)
l1.plot_appliedCD(orthonormal_basis, mean)

# Different dates

# path = r'F:\data\BAYLORCW032\python\2023_10_05'
# l1 = Mode(path, use_reg = True)
# orthonormal_basis, mean = l1.plot_CD(epoch=range(l1.sample + 1, l1.delay + 1))

# path = r'F:\data\BAYLORCW032\python\2023_10_24'
# l1 = Mode(path, use_reg = True)
# l1.plot_appliedCD(orthonormal_basis, mean)

# #TRAINED to NAIVE
# path = r'F:\data\BAYLORCW032\python\2023_10_24'
# l1 = Mode(path, use_reg = True)
# orthonormal_basis, mean = l1.plot_CD(epoch=range(l1.sample + 1, l1.delay + 1))

# path = r'F:\data\BAYLORCW032\python\2023_10_05'
# l1 = Mode(path, use_reg = True)
# l1.plot_appliedCD(orthonormal_basis, mean)
#%%
# from activityMode import Mode
paths = [r'F:\data\BAYLORCW032\python\2023_10_08',
          r'F:\data\BAYLORCW032\python\2023_10_16',
          r'F:\data\BAYLORCW032\python\2023_10_25',]

# perf = []
# for path in paths:

#     l1 = Mode(path)
#     # l1.plot_CD()
#     # d = l1.plot_performance_distfromCD()
#     _,_,_,db = l1.decision_boundary(opto=True)
#     perf += [np.mean(db)]

# plt.bar(range(3), perf)
# plt.ylim([0.5,1])
# plt.show()

# paths = [r'F:\data\BAYLORCW032\python\2023_10_05',
#           r'F:\data\BAYLORCW032\python\2023_10_19',
#           r'F:\data\BAYLORCW032\python\2023_10_24',]

perf, perfctl = [], []
for path in paths:

    l1 = Mode(path)
    # l1.plot_CD()
    # d = l1.plot_performance_distfromCD()
    _,_,_,dbctl = l1.decision_boundary(opto=False)
    _,_,_,db = l1.decision_boundary(opto=True)
    perf += [np.mean(db)]
    perfctl += [np.mean(dbctl)]
    
plt.bar(np.arange(3)+0.2, perf, 0.4, label="Perturbation trials")

plt.bar(np.arange(3)-0.2, perfctl, 0.4, label="Control trials")

plt.xticks(range(3), ["Naive", "learning", "Expert"])
plt.ylim([0.4,1])
plt.legend()
plt.show()

#%% Opto effect on CD dimension
from activityMode import Mode

# path = r'F:\data\BAYLORCW032\python\2023_10_25'
# l1 = Mode(path, use_reg = True)
# l1.plot_CD_opto()

# path = r'F:\data\BAYLORCW032\python\2023_10_24'
# l1 = Mode(path, use_reg = True)
# l1.plot_CD_opto()

# path = r'F:\data\BAYLORCW034\python\2023_10_27'
# l1 = Mode(path)
# l1.plot_CD_opto()

# path = r'F:\data\BAYLORCW036\python\2023_10_09'
# l1 = Mode(path)
# l1.plot_CD_opto()

# path = r'F:\data\BAYLORCW036\python\2023_10_19'
# l1 = Mode(path)
# l1.plot_CD_opto()

# path = r'F:\data\BAYLORCW036\python\2023_10_30'
# l1 = Mode(path)
# l1.plot_CD_opto()

#IPSI INHIB DAYS
paths = [r'F:\data\BAYLORCW032\python\2023_10_08',
          r'F:\data\BAYLORCW032\python\2023_10_16',
          r'F:\data\BAYLORCW032\python\2023_10_25',]


paths =    [r'F:\data\BAYLORCW036\python\2023_10_07',
            r'F:\data\BAYLORCW036\python\2023_10_17',
            r'F:\data\BAYLORCW036\python\2023_10_30',]

# paths =          [ r'F:\data\BAYLORCW034\python\2023_10_10',
#             r'F:\data\BAYLORCW034\python\2023_10_18',
#             r'F:\data\BAYLORCW034\python\2023_10_28',]
for path in paths:
    # l1 = Mode(path, use_reg = True, triple=True)
    # l1.plot_CD_opto()
    
    l1 = Mode(path)
    l1.plot_CD_opto()
#%% Opto effect on Persistent mode
from activityMode import Mode

path = r'F:\data\BAYLORCW036\python\2023_10_30'
l1 = Mode(path)
l1.plot_persistent_mode_opto()
#%%
#Behavioral modes
# a, b = l1.plot_activity_modes_ctl()

# a, b = l1.plot_activity_modes_err()


# a, b = l1.plot_activity_modes_opto()

# a, b = l1.plot_activity_modes_opto(error=True)

orthonormal_basis, mean = l1.plot_behaviorally_relevant_modes()
# l1.plot_behaviorally_relevant_modes_opto(error=True)
# path = r'F:\data\BAYLORCW034\python\2023_10_27'
# l1 = Mode(path, use_reg = True)
# l1.plot_behaviorally_relevant_modes_appliedCD(orthonormal_basis, mean)
#%%
### Recovery for stim over sessions
# total_var = []
# alldates = ['06_16', '06_19', '06_21', '06_22', '06_23', '07_05', '07_06',
#             '07_07', '07_10', '07_11']

# for date in alldates:
    
    
#     path = r'F:\data\BAYLORCW030\python\2023_{}'.format(date)
    
    
#     l1 = Mode(path)
    
#     v = l1.get_all_recovery_vectors()
#     # pca = PCA()
#     # pca.fit(v)
#     # total_var += [sum(pca.explained_variance_)]
    
#     total_var += [np.mean([np.var(v,axis=1) / v.shape[1]]) / v.shape[0]]
    

# plt.bar(range(10), total_var)
# plt.xticks(range(10), alldates)
# plt.title('Variance of recovery over sessions')
# plt.xlabel('Sessions')
# plt.ylabel('Total variance')



# np.variance(v)
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
