# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 14:01:59 2024

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
plt.rcParams['pdf.fonttype'] = '42' 
import random
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
cat = np.concatenate
#%% Paths

agg_matched_paths = [[    
            r'F:\data\BAYLORCW032\python\2023_10_05',
            # r'F:\data\BAYLORCW034\python\2023_10_12',
            r'F:\data\BAYLORCW036\python\2023_10_09',
            r'F:\data\BAYLORCW035\python\2023_10_26',
            r'F:\data\BAYLORCW037\python\2023_11_21',
            

        r'H:\data\BAYLORCW044\python\2024_05_22',
        r'H:\data\BAYLORCW044\python\2024_05_23',
        
        r'H:\data\BAYLORCW046\python\2024_05_29',
        r'H:\data\BAYLORCW046\python\2024_05_30',
        r'H:\data\BAYLORCW046\python\2024_05_31',
            ],
             [r'F:\data\BAYLORCW032\python\2023_10_19',
            # r'F:\data\BAYLORCW034\python\2023_10_22',
            r'F:\data\BAYLORCW036\python\2023_10_19',
            r'F:\data\BAYLORCW035\python\2023_12_07',
            r'F:\data\BAYLORCW037\python\2023_12_08',

            
        r'H:\data\BAYLORCW044\python\2024_06_06',
        r'H:\data\BAYLORCW044\python\2024_06_04',

        r'H:\data\BAYLORCW046\python\2024_06_24',
        r'H:\data\BAYLORCW046\python\2024_06_10',
        r'H:\data\BAYLORCW046\python\2024_06_11',

        ],
        [r'F:\data\BAYLORCW032\python\2023_10_24',
            # r'F:\data\BAYLORCW034\python\2023_10_27',
            r'F:\data\BAYLORCW036\python\2023_10_30',
            r'F:\data\BAYLORCW035\python\2023_12_15',
            r'F:\data\BAYLORCW037\python\2023_12_15',
            
            
            r'H:\data\BAYLORCW044\python\2024_06_19',
            r'H:\data\BAYLORCW044\python\2024_06_18',
            
            r'H:\data\BAYLORCW046\python\2024_06_28',
            r'H:\data\BAYLORCW046\python\2024_06_27',
            r'H:\data\BAYLORCW046\python\2024_06_26',

        ]]

#%% Get all selective neurons
by_FOV = False
all_sample_sel = []
all_delay_sel = []
all_response_sel = []
for paths in agg_matched_paths:
    sample_sel = []
    delay_sel = []
    response_sel = []
    all_pref_sample, all_nonpref_sample = np.zeros(61), np.zeros(61)
    all_pref_delay, all_nonpref_delay = np.zeros(61), np.zeros(61)
    all_pref_response, all_nonpref_response = np.zeros(61), np.zeros(61)

    for path in paths:
        s1 = session.Session(path, use_reg=True, triple=True)
                             # use_background_sub=False,
                             # baseline_normalization="median_zscore")   
        
        sample_epoch = range(s1.sample, s1.delay)
        delay_epoch = range(s1.delay+int(1.5 * 1/s1.fs), s1.response) 
        response_epoch = range(s1.response, s1.response + int(2*1/s1.fs))
        adj_p = 0.05 / len(s1.good_neurons)
        adj_p = 0.001
        naive_sample_sel_old = s1.get_epoch_selective(sample_epoch, p=adj_p)
        
        naive_delay_sel_old = s1.get_epoch_selective(delay_epoch, p=adj_p)
        
        naive_response_sel_old = s1.get_epoch_selective(response_epoch, p=adj_p)
        
        naive_delay_sel = [n for n in naive_delay_sel_old if n not in naive_sample_sel_old]# and n not in naive_response_sel_old]

        naive_response_sel = [n for n in naive_response_sel_old if n not in naive_sample_sel_old and n not in naive_delay_sel_old]
        
        # naive_sample_sel = [n for n in naive_sample_sel_old if n not in naive_delay_sel_old and n not in naive_response_sel_old]

        # naive_nonsel = [n for n in s1.good_neurons if n not in naive_sample_sel and n not in naive_delay_sel and n not in naive_response_sel]
        
        all_n_sample = []
        for n in naive_sample_sel_old:
            if by_FOV:
                all_n_sample += [s1.plot_selectivity(n, epoch=sample_epoch, plot=False, bootstrap = True, trialtype=True)]
            else:
                pref, nonpref = s1.plot_selectivity(n, epoch=sample_epoch, plot=False, 
                                                    bootstrap = True, trialtype=True, 
                                                    lickdir=False, return_pref_np = True)
                all_pref_sample = np.vstack((all_pref_sample, pref))
                all_nonpref_sample = np.vstack((all_nonpref_sample, nonpref))
                
        all_n_delay = []
        for n in naive_delay_sel:
            if by_FOV:
                all_n_delay += [s1.plot_selectivity(n, epoch=delay_epoch, plot=False, 
                                                    bootstrap = True, lickdir=True, trialtype=False)]
            else:
                pref, nonpref = s1.plot_selectivity(n, epoch=delay_epoch, plot=False, 
                                                    bootstrap = True, trialtype=False, 
                                                    lickdir=True, return_pref_np = True)
                all_pref_delay = np.vstack((all_pref_delay, pref))
                all_nonpref_delay = np.vstack((all_nonpref_delay, nonpref))
                
        all_n_response = []
        for n in naive_response_sel:
            if by_FOV:
                all_n_response += [s1.plot_selectivity(n, epoch=response_epoch, plot=False, 
                                                       bootstrap = True, lickdir=True, trialtype=False)]
            else:
                pref, nonpref = s1.plot_selectivity(n, epoch=response_epoch, plot=False, 
                                                    bootstrap = True, trialtype=False, 
                                                    lickdir=True, return_pref_np = True)
                all_pref_response = np.vstack((all_pref_response, pref))
                all_nonpref_response = np.vstack((all_nonpref_response, nonpref))
                
        if by_FOV:
            if len(all_n_sample) != 0:
                sample_sel += [np.mean(all_n_sample, axis=0)]
            if len(all_n_delay) != 0:
                delay_sel += [np.mean(all_n_delay, axis=0)]
            if len(all_n_response) != 0:
                response_sel += [np.mean(all_n_response, axis=0)]


    if by_FOV: 
        sample_sel = [cat(s) for s in sample_sel if s.shape[0]==1] # reshape each neuron/fov
        delay_sel = [cat(s) for s in delay_sel if s.shape[0]==1] # reshape each neuron
        response_sel = [cat(s) for s in response_sel if s.shape[0]==1] # reshape each neuron
        all_sample_sel += [sample_sel]
        all_delay_sel += [delay_sel]
        all_response_sel += [response_sel]
    else:
        all_pref_sample, all_nonpref_sample = all_pref_sample[1:], all_nonpref_sample[1:]
        all_pref_delay, all_nonpref_delay = all_pref_delay[1:], all_nonpref_delay[1:]
        all_pref_response, all_nonpref_response = all_pref_response[1:], all_nonpref_response[1:]
        all_sample_sel += [all_pref_sample - all_nonpref_sample]
        all_delay_sel += [all_pref_delay - all_nonpref_delay]
        all_response_sel += [all_pref_response - all_nonpref_response]
# if by_FOV:
#     all_sample_sel = [cat(s) for s in all_sample_sel if s.shape[0]==1] # reshape each fov
#     all_delay_sel = [cat(s) for s in all_delay_sel if s.shape[0]==1] # reshape each fov
#     all_response_sel = [cat(s) for s in all_response_sel if s.shape[0]==1] # reshape each fov
#%% FIX post
for i in range(3):
    # for j in range(len(all_sample_sel[i])):
    all_sample_sel[i] = [cat(s) for s in all_sample_sel[i] if len(s.shape)==2]
    all_delay_sel[i] = [cat(s) for s in all_sample_sel[i] if len(s.shape)==2]
    all_response_sel[i] = [cat(s) for s in all_sample_sel[i] if len(s.shape)==2]
    
    # all_sample_sel[i] = [s for s in all_sample_sel[i] if ~np.isnan(s)]

#%% Plot selectivity trace
x = np.arange(-6.97,4,1/6)[:61]

f=plt.figure()

sel = np.mean(all_sample_sel[0], axis=0)
err = np.std(all_sample_sel[0], axis=0) / np.sqrt(len(all_sample_sel[1]))
plt.plot(x, sel, color='green')
            
plt.fill_between(x, sel - err, 
              sel + err,
              color='lightgreen')
plt.axvline(-4.3, ls='--', color='grey')
plt.axvline(-3, ls='--', color='grey')
plt.axvline(0, ls='--', color='grey')
plt.axhline(0, ls='--', color='grey')
# plt.set_title(titles[0])
    
    
#%% PLOT ALL



f, axarr = plt.subplots(3,3, sharey='row', sharex = True, figsize=(18,18))
# plt.setp(axarr, ylim=(-0.2,1.2))

for j in range(3):
# for j in [2]:
    
    x = np.arange(-6.97,4,1/6)[:61]
    titles = ['Stimulus selective', 'Choice selective', 'Outcome selective', 'Action selective']

    
    sel = np.mean(all_sample_sel[j], axis=0)
    err = np.std(all_sample_sel[j], axis=0) / np.sqrt(len(all_sample_sel[j]))
    
    axarr[0,j].plot(x, sel, color='green')
            
    axarr[0,j].fill_between(x, sel - err, 
              sel + err,
              color='lightgreen')
    
    axarr[0,j].set_title(titles[0])

    sel = np.mean(all_delay_sel[j], axis=0)
    err = np.std(all_delay_sel[j], axis=0) / np.sqrt(len(all_delay_sel[j]))
    
    axarr[1,j].plot(x, sel, color='purple')
            
    axarr[1,j].fill_between(x, sel - err, 
              sel + err,
              color='violet')
    axarr[1,j].set_title(titles[1])
    
    sel = np.mean(all_response_sel[j], axis=0)
    err = np.std(all_response_sel[j], axis=0) / np.sqrt(len(all_response_sel[j]))
    
    axarr[2,j].plot(x, sel, color='goldenrod')
            
    axarr[2,j].fill_between(x, sel - err, 
              sel + err,
              color='wheat')
    
    axarr[2,j].set_title(titles[3])
    
    
for i in range(3):
    for j in range(3):
        axarr[j,i].axvline(0, color = 'grey', alpha=0.5, ls = '--')
        axarr[j,i].axvline(-4.3, color = 'grey', alpha=0.5, ls = '--')
        axarr[j,i].axvline(-3, color = 'grey', alpha=0.5, ls = '--')        
        axarr[j,i].axhline(0, color = 'grey', alpha=0.5, ls = '--')
        
        
axarr[0,0].set_ylabel('Selectivity')
axarr[2,1].set_xlabel('Time from Go cue (s)')



# plt.savefig(r'F:\data\Fig 2\SDR_by_bucket_sankeyp01.pdf')
plt.show()


        