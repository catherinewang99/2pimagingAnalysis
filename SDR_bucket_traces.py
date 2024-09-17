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

#%% Get sample selective neurons
by_FOV = True
for paths in [agg_matched_paths[0]]:
    sample_sel = []
    for path in paths:
        s1 = session.Session(path, use_reg=True, triple=True, use_background_sub=False) # Naive
        
        sample_epoch = range(s1.sample, s1.delay)
        delay_epoch = range(s1.delay+int(1.5 * 1/s1.fs), s1.response)
        response_epoch = range(s1.response, s1.response + int(2*1/s1.fs))
        adj_p = 0.01
        naive_sample_sel = s1.get_epoch_selective(sample_epoch, p=adj_p)
        
        # naive_delay_sel = s1.get_epoch_selective(delay_epoch, p=p)
        # naive_delay_sel = [n for n in naive_delay_sel if n not in naive_sample_sel]
        
        # naive_response_sel = s1.get_epoch_selective(response_epoch, p=p)
        # naive_response_sel = [n for n in naive_response_sel if n not in naive_sample_sel and n not in naive_delay_sel]
        
        # naive_nonsel = [n for n in s1.good_neurons if n not in naive_sample_sel and n not in naive_delay_sel and n not in naive_response_sel]
        
        all_n_sample = []
        for n in naive_sample_sel:
            if by_FOV:
                all_n_sample += [s1.plot_selectivity(n, plot=False)]
            else:
                sample_sel += [s1.plot_selectivity(n, plot=False)]
        if by_FOV:
            sample_sel += [np.mean(all_n_sample, axis=0)]
            
        
#%% Plot selectivity trace
x = np.arange(-6.97,4,1/6)[:61]

f=plt.figure()

sel = np.mean(sample_sel, axis=0)
err = np.std(sample_sel, axis=0) / np.sqrt(len(sample_sel))
plt.plot(x, sel, color='green')
            
plt.fill_between(x, sel - err, 
              sel + err,
              color='lightgreen')
plt.axvline(-4.3, ls='--', color='grey')
plt.axvline(-3, ls='--', color='grey')
plt.axvline(0, ls='--', color='grey')
plt.axhline(0, ls='--', color='grey')
# plt.set_title(titles[0])
    
    

        