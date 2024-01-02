# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 12:59:23 2023

@author: Catherine Wang
"""


import sys
sys.path.append("C:\scripts\Imaging analysis")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import session
from matplotlib.pyplot import figure
import decon
from scipy.stats import chisquare
import pandas as pd
from activityMode import Mode

# Plot choice decoding in control vs opto trials


#%%
paths = [[r'F:\data\BAYLORCW032\python\2023_10_08',
          r'F:\data\BAYLORCW032\python\2023_10_16',
          r'F:\data\BAYLORCW032\python\2023_10_25',
          r'F:\data\BAYLORCW032\python\cellreg\layer{}\1008_1016_1025pairs_proc.npy'],
         
         [ r'F:\data\BAYLORCW034\python\2023_10_12',
            r'F:\data\BAYLORCW034\python\2023_10_22',
            r'F:\data\BAYLORCW034\python\2023_10_27',
            r'F:\data\BAYLORCW034\python\cellreg\layer{}\1012_1022_1027pairs_proc.npy'],
         
         [r'F:\data\BAYLORCW036\python\2023_10_09',
            r'F:\data\BAYLORCW036\python\2023_10_19',
            r'F:\data\BAYLORCW036\python\2023_10_30',
            r'F:\data\BAYLORCW036\python\cellreg\layer{}\1009_1019_1030pairs_proc.npy'],
        ]

#%% Matched
#%% CW32 matched

# CONTRA
paths = [r'F:\data\BAYLORCW032\python\2023_10_05',
          r'F:\data\BAYLORCW032\python\2023_10_19',
          r'F:\data\BAYLORCW032\python\2023_10_24',]
for path in paths:
    
    l1 = Mode(path, use_reg=True, triple=True)
    

    orthonormal_basis, mean, db, acc_within = l1.decision_boundary()
    print(np.mean(acc_within))
    
    
    
    plt.bar([0], [np.mean(acc_within)])
    plt.scatter(np.zeros(len(acc_within)), np.mean(acc_within,axis=1), color = 'r')
    # plt.errorbar([0], [np.mean(acc_within)],
    #              [np.std(acc_within)/np.sqrt(len(acc_within))],
    #              color = 'r')
    plt.xticks([0], ['Left ALM -> behavior'])
    plt.ylim(bottom=0.4, top =1)
    plt.show()
#%% UNMATCHED ACROSS PERTURBATION/CONTROL TRIALS
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

#%% matched across sessions
# from activityMode import Mode
paths = [r'F:\data\BAYLORCW032\python\2023_10_08',
          r'F:\data\BAYLORCW032\python\2023_10_16',
          r'F:\data\BAYLORCW032\python\2023_10_25',]

paths =[r'F:\data\BAYLORCW037\python\2023_11_21',
            r'F:\data\BAYLORCW037\python\2023_12_08',
            r'F:\data\BAYLORCW037\python\2023_12_15',]
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
    _,_,_,dbctl = l1.decision_boundary()
    perfctl += [np.mean(dbctl)]
    
# plt.bar(np.arange(3)+0.2, perf, 0.4, label="Perturbation trials")

plt.bar(np.arange(3), perfctl, label="Control trials")

plt.xticks(range(3), ["Naive", "learning", "Expert"])
plt.ylim([0.4,1])
plt.legend()
plt.show()
