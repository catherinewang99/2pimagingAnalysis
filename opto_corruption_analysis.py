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

#%% Changes at single cell level - sankey SDR
agg_mice_paths = [[['H:\\data\\BAYLORCW038\\python\\2024_02_05', 
                    'H:\\data\\BAYLORCW038\\python\\2024_03_15'],
                   ]]
# p=0.0005
p=0.001

og_SDR = []
allstod = []
s1list, d1, r1, ns1 = np.zeros(4),np.zeros(4),np.zeros(4),np.zeros(4)
for paths in agg_mice_paths: # For each mouse
    stod = []
    s1 = Session(paths[0][0], use_reg=True, use_background_sub=False) # Naive
    # sample_epoch = range(s1.sample+2, s1.delay+2)
    sample_epoch = range(s1.sample, s1.delay+2)
    delay_epoch = range(s1.delay+9, s1.response)
    response_epoch = range(s1.response, s1.response + 12)
    
    naive_sample_sel = s1.get_epoch_selective(sample_epoch, p=p)
    
    naive_delay_sel = s1.get_epoch_selective(delay_epoch, p=p)
    naive_delay_sel = [n for n in naive_delay_sel if n not in naive_sample_sel]
    
    naive_response_sel = s1.get_epoch_selective(response_epoch, p=p)
    naive_response_sel = [n for n in naive_response_sel if n not in naive_sample_sel and n not in naive_delay_sel]

    naive_nonsel = [n for n in s1.good_neurons if n not in naive_sample_sel and n not in naive_delay_sel and n not in naive_response_sel]

    og_SDR += [[len(naive_sample_sel), len(naive_delay_sel), len(naive_response_sel), len(naive_nonsel)]]

    # s2 = session.Session(paths[0][1], use_reg=True, triple=True) # Learning
    s2 = Session(paths[0][1], use_reg=True) # Expert
    
    for n in naive_sample_sel:
        if s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], sample_epoch, p=p):
            s1list[0] += 1
        elif s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], delay_epoch, p=p):
            s1list[1] += 1
        elif s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], response_epoch, p=p):
            s1list[2] += 1
        else:
            s1list[3] += 1
    
    for n in naive_delay_sel:
        if s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], sample_epoch, p=p):
            d1[0] += 1
        elif s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], delay_epoch, p=p):
            d1[1] += 1
        elif s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], response_epoch, p=p):
            d1[2] += 1
        else:
            d1[3] += 1
    
    
    for n in naive_response_sel:
        if s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], sample_epoch, p=p):
            r1[0] += 1

        elif s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], delay_epoch, p=p):
            r1[1] += 1
            stod += [(n, s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]])]  #save delay to sample cells

        elif s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], response_epoch, p=p):
            r1[2] += 1
        else:
            r1[3] += 1
    
    
    for n in naive_nonsel:
        if s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], sample_epoch, p=p):
            ns1[0] += 1
        elif s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], delay_epoch, p=p):
            ns1[1] += 1
        elif s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], response_epoch, p=p):
            ns1[2] += 1
        else:
            ns1[3] += 1
    allstod += [[stod]]
    
og_SDR = np.sum(og_SDR, axis=0)


#%% Selectivity recovery

paths = ['H:\\data\\BAYLORCW038\\python\\2024_02_05', 
        'H:\\data\\BAYLORCW038\\python\\2024_03_15']

for path in paths:
    
    l1 = Session(path, use_reg=True)
    l1.selectivity_optogenetics()
    
#%% CD recovery


intialpath, middlepath, finalpath = ['H:\\data\\BAYLORCW038\\python\\2024_02_05', 
                         r'H:\data\BAYLORCW038\python\2024_02_15',
                         'H:\\data\\BAYLORCW038\\python\\2024_03_15']

    
l1 = Mode(intialpath, use_reg=True)
orthonormal_basis, mean = l1.plot_CD()
# l1.plot_CD_opto()
control_traces, opto_traces, error_bars, orthonormal_basis, mean, meantrain, meanstd = l1.plot_CD_opto(return_traces=True, return_applied=True)

l1 = Mode(middlepath)
l1.plot_CD_opto()

path = finalpath
l1 = Mode(path, use_reg = True)
l1.plot_appliedCD(orthonormal_basis, mean)
# l1.plot_CD_opto()
l1.plot_CD_opto_applied(orthonormal_basis, mean, meantrain, meanstd)
                  
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



