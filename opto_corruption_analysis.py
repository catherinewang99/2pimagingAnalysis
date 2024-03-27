# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 16:21:35 2024

@author: Catherine Wang
"""

import sys
sys.path.append("C:\scripts\Imaging analysis")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from session import Session
from matplotlib.pyplot import figure
from numpy import concatenate as cat
from sklearn.preprocessing import normalize
import quality
from scipy import stats
from scipy.stats import zscore
from activityMode import Mode
import behavior


plt.rcParams['pdf.fonttype'] = '42' 


#%% Selectivity recovery

paths = ['H:\\data\\BAYLORCW038\\python\\2024_02_05', 
        'H:\\data\\BAYLORCW038\\python\\2024_03_15']

for path in paths:
    
    l1 = Session(path, use_reg=True)
    l1.selectivity_optogenetics()
    
#%% CD recovery


intialpath, finalpath = ['H:\\data\\BAYLORCW038\\python\\2024_02_05', 
        'H:\\data\\BAYLORCW038\\python\\2024_03_15']

    
l1 = Mode(intialpath, use_reg=True)
orthonormal_basis, mean = l1.plot_CD()
l1.plot_CD_opto()

path = finalpath
l1 = Mode(path, use_reg = True)
l1.plot_appliedCD(orthonormal_basis, mean)
l1.plot_CD_opto()

                  
#%% Behavioral progress

b = behavior.Behavior(r'H:\data\Behavior data\BAYLORCW038\python_behavior', behavior_only=True)
b.learning_progression(window = 200)



#%% Behavioral recovery no L/R info

all_paths = [[r'H:\data\BAYLORCW038\python\2024_02_05',
          r'H:\data\BAYLORCW038\python\2024_02_15',
          r'H:\data\BAYLORCW038\python\2024_03_15',]]

performance_opto = []
performance_ctl = []
fig = plt.figure()
for paths in all_paths:
    counter = -1

    opt, ctl = [],[]
    for path in paths:
        counter += 1
        l1 = Session(path)
        stim_trials = np.where(l1.stim_ON)[0]
        control_trials = np.where(~l1.stim_ON)[0]
        
        perf_right, perf_left, perf_all = l1.performance_in_trials(stim_trials)
        opt += [perf_all]
        # plt.scatter(counter + 0.2, perf_right, c='b', marker='x')
        # plt.scatter(counter + 0.2, perf_left, c='r', marker='x')
       
        perf_rightctl, perf_left, perf_all_c = l1.performance_in_trials(control_trials)
        ctl += [perf_all_c]
        # plt.scatter(counter - 0.2, perf_rightctl, c='b', marker='o')
        # plt.scatter(counter - 0.2, perf_left, c='r', marker='o')
        plt.plot([counter - 0.2, counter + 0.2], [perf_all_c, perf_all], color='grey')
        
        
    performance_opto += [opt]
    performance_ctl += [ctl]


    plt.scatter(np.arange(3)+0.2, opt)
    plt.scatter(np.arange(3)-0.2, ctl)
    
    
plt.bar(np.arange(3)+0.2, np.mean(performance_opto, axis=0), 0.4, fill=False)

plt.bar(np.arange(3)-0.2, np.mean(performance_ctl, axis=0), 0.4, fill=False)

plt.xticks(range(3), ["Initial", "Day 7", "Day 30"])
# plt.ylim([0.4,1])
# plt.legend()
# plt.savefig(r'F:\data\Fig 1\beh_opto.pdf')
plt.show()

#%% Correlate modularity with behaavioral recovery (do with more data points)



