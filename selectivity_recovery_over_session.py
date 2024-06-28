# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 15:47:33 2024

@author: catherinewang

Make some plots showing the variable recovery to stim over the session,
possibly correlate with behavior state measurement

Key sessions:
    
    CW46 6/24, 6/25
"""
import sys
sys.path.append("C:\scripts\Imaging analysis")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import behavior
plt.rcParams['pdf.fonttype'] = '42' 

from session import Session
from matplotlib.pyplot import figure
from numpy import concatenate as cat
from sklearn.preprocessing import normalize
import quality
from scipy import stats
from scipy.stats import zscore
from activityMode import Mode


#%% Plot behavior, stim trials, correct/error on stim trials, recovery to stim as a measure of distance from control CD
path = 'H:\\data\\BAYLORCW046\\python\\2024_06_25'
        
l1 = Mode(path)

window = 25
correct = l1.L_correct + l1.R_correct
correct = np.convolve(correct, np.ones(window*2)/(window*2), mode = 'same')
# correctarr = correct[window:-window]
correctarr = correct


correct_stim = l1.L_correct + l1.R_correct
correct_stim = correct_stim[np.where(l1.stim_ON)]
wrong_stim = l1.L_wrong + l1.R_wrong
wrong_stim = wrong_stim[np.where(l1.stim_ON)]

# Instead of using only correct/error, use left right
right_stim = l1.L_wrong + l1.R_correct
right_stim = right_stim[np.where(l1.stim_ON)]
left_stim = l1.L_correct + l1.R_wrong
left_stim = left_stim[np.where(l1.stim_ON)]


period = range(l1.response-15, l1.response)
period = range(l1.delay, l1.delay + 15)

r_trials, l_trials, r_proj_delta, l_proj_delta = l1.modularity_proportion_by_CD(period = period, return_trials=True)
r_delta = np.mean(r_proj_delta, axis=1)
l_delta = np.mean(l_proj_delta, axis=1)

trials = cat((r_trials, l_trials))
deltas = np.take(cat((np.abs(r_delta), np.abs(l_delta))), np.argsort(trials))
trials = np.sort(trials)

f, axarr = plt.subplots(1, 1, sharex='col', figsize=(16,8))
axarr.plot(correctarr, 'g')        
axarr.set_ylabel('% correct')

axarr.scatter(np.where(l1.stim_ON)[0][np.where(correct_stim)[0]], np.ones(len(np.where(l1.stim_ON)[0][np.where(correct_stim)[0]])) * 0.99, color='g')
axarr.scatter(np.where(l1.stim_ON)[0][np.where(wrong_stim)[0]], np.ones(len(np.where(l1.stim_ON)[0][np.where(wrong_stim)[0]])) * 0.99, color ='r')

axarr.scatter(np.where(l1.stim_ON)[0][np.where(right_stim)[0]], np.ones(len(np.where(l1.stim_ON)[0][np.where(right_stim)[0]])) * 1.05, color='r')
axarr.scatter(np.where(l1.stim_ON)[0][np.where(left_stim)[0]], np.ones(len(np.where(l1.stim_ON)[0][np.where(left_stim)[0]])) * 1.05, color ='b')


ax2 = axarr.twinx()  # instantiate a second Axes that shares the same x-axis
# axarr.scatter(cat((r_trials, l_trials)), cat((r_delta, l_delta)), color='purple')        
ax2.plot(trials, deltas, color='purple', marker='o')        
ax2.set_ylabel('Distance from CD')

#%% Plot correlation between accuracy +- 20 trials and recovery to stim
allpaths = ['H:\\data\\BAYLORCW046\\python\\2024_06_24',
            'H:\\data\\BAYLORCW046\\python\\2024_06_25',
            'H:\\data\\BAYLORCW046\\python\\2024_06_19',
            'H:\\data\\BAYLORCW046\\python\\2024_06_11',
            'H:\\data\\BAYLORCW046\\python\\2024_06_10',
            'H:\\data\\BAYLORCW046\\python\\2024_06_07',
            ]

correctarr  = []
all_deltas = []
window = 25

for path in allpaths:
    l1 = Mode(path)
    
    correct = l1.L_correct + l1.R_correct
    correct = np.convolve(correct, np.ones(window*2)/(window*2), mode = 'same')
    # correctarr = correct[window:-window]
    
    r_trials, l_trials, r_proj_delta, l_proj_delta = l1.modularity_proportion_by_CD(period = range(l1.delay+15, l1.delay+45), return_trials=True)
    r_delta = np.mean(r_proj_delta, axis=1)
    l_delta = np.mean(l_proj_delta, axis=1)
    trials = cat((r_trials, l_trials))
    deltas = np.take(cat((np.abs(r_delta), np.abs(l_delta))), np.argsort(trials))
    trials = np.sort(trials)
    
    correctarr += [correct[trials]]
    all_deltas += [deltas]

plt.scatter(cat(correctarr), cat(all_deltas))
slope, intercept, r_value, p_value, std_err = stats.linregress(cat(correctarr), cat(all_deltas))
# plt.axline((0, intercept), slope=slope)
plt.plot(cat(correctarr), intercept + slope*cat(correctarr), 'r', label='fitted line')
plt.xlabel('Performance')
plt.ylabel('Delta from CD')

plt.show()








