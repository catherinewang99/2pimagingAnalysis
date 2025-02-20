# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:10:18 2025

Investigate if there is an early lick cd that predicts the final cd over learning

@author: catherinewang
"""

import sys
sys.path.append("C:\scripts\Imaging analysis")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from alm_2p import session
from matplotlib.pyplot import figure
# import decon
from scipy.stats import chisquare
import pandas as pd
from activityMode import Mode
from scipy import stats
cat=np.concatenate
from sklearn.metrics.pairwise import cosine_similarity

#%% paths

all_paths = [[    
            # r'F:\data\BAYLORCW032\python\2023_10_05',
            # r'F:\data\BAYLORCW034\python\2023_10_12',
            # r'F:\data\BAYLORCW036\python\2023_10_09',
            # r'F:\data\BAYLORCW035\python\2023_10_26',
            # r'F:\data\BAYLORCW037\python\2023_11_21',
            
            r'H:\data\BAYLORCW044\python\2024_05_22',
            r'H:\data\BAYLORCW044\python\2024_05_23',
            
            r'H:\data\BAYLORCW046\python\2024_05_29',
            r'H:\data\BAYLORCW046\python\2024_05_30',
            r'H:\data\BAYLORCW046\python\2024_05_31',
            ],

              [
             # r'F:\data\BAYLORCW032\python\2023_10_19',
            # r'F:\data\BAYLORCW034\python\2023_10_22',
            # r'F:\data\BAYLORCW036\python\2023_10_19',
            # r'F:\data\BAYLORCW035\python\2023_12_07',
            # r'F:\data\BAYLORCW037\python\2023_12_08',
            
            r'H:\data\BAYLORCW044\python\2024_06_06',
            r'H:\data\BAYLORCW044\python\2024_06_04',

            # r'H:\data\BAYLORCW046\python\2024_06_07', #sub out for below
            r'H:\data\BAYLORCW046\python\2024_06_24',
            r'H:\data\BAYLORCW046\python\2024_06_10',
            r'H:\data\BAYLORCW046\python\2024_06_11',
            ],


              [
             # r'F:\data\BAYLORCW032\python\2023_10_24',
            # r'F:\data\BAYLORCW034\python\2023_10_27',
            # r'F:\data\BAYLORCW036\python\2023_10_30',
            # r'F:\data\BAYLORCW035\python\2023_12_15',
            # r'F:\data\BAYLORCW037\python\2023_12_15',
            
            r'H:\data\BAYLORCW044\python\2024_06_19',
            r'H:\data\BAYLORCW044\python\2024_06_18',
            
            r'H:\data\BAYLORCW046\python\2024_06_28',
            r'H:\data\BAYLORCW046\python\2024_06_27',
            r'H:\data\BAYLORCW046\python\2024_06_26',
            
            ]]

#%% Work with early lick info

naivepath, learningpath, expertpath = [
            r'H:\data\BAYLORCW046\python\2024_05_29',
             r'H:\data\BAYLORCW046\python\2024_06_24',
             r'H:\data\BAYLORCW046\python\2024_06_28'
             ]

naivepath, learningpath, expertpath = [
            r'H:\data\BAYLORCW046\python\2024_05_30',
              r'H:\data\BAYLORCW046\python\2024_06_10',
              r'H:\data\BAYLORCW046\python\2024_06_27'
              ]



# naivepath, learningpath, expertpath = [
#             r'H:\data\BAYLORCW044\python\2024_05_22',
#             r'H:\data\BAYLORCW044\python\2024_06_06',
#             r'H:\data\BAYLORCW044\python\2024_06_19',
#              ]

naivepath, learningpath, expertpath = [
            r'F:\data\BAYLORCW032\python\2023_10_05',
            r'F:\data\BAYLORCW032\python\2023_10_19',
            r'F:\data\BAYLORCW032\python\2023_10_24',
              ]


s2 = Mode(learningpath, use_reg = True, triple=True, 
          baseline_normalization="median_zscore")

# calculate early lick CD in naive sessions - define here

early_lick_trials = np.where(s2.early_lick)[0]
early_lick_side = [i[0] for i in s2.early_lick_side] # use first lick as direction
early_lick_time = [i[0][0] for i in s2.early_lick_time]

# Filter out the early lick trials that happen too early (before end of 2.5 ITI)
drop_idx = [i for i in range(len(early_lick_time)) if early_lick_time[i] > 2.5]
early_lick_trials = np.array(early_lick_trials)[drop_idx]
early_lick_side = np.array(early_lick_side)[drop_idx]
early_lick_time = np.array(early_lick_time)[drop_idx]

# project early lick CD
r_trials = early_lick_trials[np.where(early_lick_side == 'r')[0]]
l_trials = early_lick_trials[np.where(early_lick_side == 'l')[0]]

r_trial_lick = early_lick_time[np.where(early_lick_side == 'r')[0]]
l_trial_lick = early_lick_time[np.where(early_lick_side == 'l')[0]]
# Get the frame where the early lick happened, per trial
r_trial_lick = [np.round(r/s2.fs) for r in r_trial_lick] 
l_trial_lick = [np.round(r/s2.fs) for r in l_trial_lick] 

# Align every trial to shortest early lick (new go cue time)
r_cue, l_cue = min(r_trial_lick), min(l_trial_lick)
r_end, l_end = max(r_trial_lick), max(l_trial_lick)


PSTH_yes_correct, PSTH_no_correct = [], []
for n in s2.good_neurons:
    
    r, l = s2.get_trace_matrix(n, rtrials=r_trials, ltrials=l_trials)
    
    for idx in range(len(r)): # adjust each trial to new go cue
        tmp = r[idx][int(r_trial_lick[idx] - r_cue) : ]
        r[idx] = tmp[ : int(s2.time_cutoff - (r_end-r_cue))]
    for idx in range(len(l)): # adjust each trial to new go cue
        tmp = l[idx][int(l_trial_lick[idx] - l_cue) : ]
        l[idx] = tmp[ : int(s2.time_cutoff - (l_end-l_cue))]
        
    r_train = np.mean(r, axis=0)
    l_train = np.mean(l, axis=0)
    
    if len(PSTH_yes_correct) == 0:
        PSTH_yes_correct = np.reshape(r_train, (1,-1))
        PSTH_no_correct = np.reshape(l_train, (1,-1))
    else: 
        PSTH_yes_correct = np.concatenate((PSTH_yes_correct, np.reshape(r_train, (1,-1))), axis = 0)
        PSTH_no_correct = np.concatenate((PSTH_no_correct, np.reshape(l_train, (1,-1))), axis = 0)

i_t_r = range(int(r_cue) - int(round(0.8*(1/s2.fs))), int(r_cue)+int(round(0.2*(1/s2.fs)))) # use 12 time steps before lick and 3 after lick
i_t_l = range(int(l_cue) - int(round(0.8*(1/s2.fs))), int(l_cue)+int(round(0.2*(1/s2.fs))))

wt = (PSTH_yes_correct[:, i_t_r] - PSTH_no_correct[:, i_t_l]) / 2
CD_choice_mode = np.mean(wt, axis=1)

#%% project onto other sessions
s2.plot_appliedCD(CD_choice_mode, 0, )

# project onto learning and expert sessions
s2 = Mode(expertpath, use_reg = True, triple=True, 
          baseline_normalization="median_zscore")
s2.plot_appliedCD(CD_choice_mode, 0)
_, mean, meantrain, meanstd = s2.plot_CD_opto(return_applied=True)
s2.plot_CD_opto_applied(CD_choice_mode, mean, meantrain, meanstd)
orthonormal_basis_learning, _ = s2.plot_CD(plot=False)


s2 = Mode(naivepath, use_reg = True, triple=True, 
          baseline_normalization="median_zscore")
s2.plot_appliedCD(CD_choice_mode, 0)
_, mean, meantrain, meanstd = s2.plot_CD_opto(return_applied=True)
s2.plot_CD_opto_applied(CD_choice_mode, mean, meantrain, meanstd)
orthonormal_basis_expert, _ = s2.plot_CD(plot=False)

# compare angle of early lick CD with final CD in expert session
np.cos(CD_choice_mode, orthonormal_basis_learning)
cosine_similarity(CD_choice_mode, orthonormal_basis_learning)

cos_sim = np.abs(cosine_similarity((CD_choice_mode, orthonormal_basis_learning)))
overall_similarity = np.mean(cos_sim[np.triu_indices_from(cos_sim, k=1)])  # Mean of upper triangle















