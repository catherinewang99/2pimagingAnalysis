# -*- coding: utf-8 -*-
"""
Created on Fri May 24 10:12:42 2024

@author: catherinewang

Quantify the population level effect of perturbation, creating similar figs as 
in Chen et al., 2021 (Fig S3) and Yang et al 2022 (Fig 6)

to show the contralateral and ipsilateral to imaging side effect of stimulation

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
cat = np.concatenate

plt.rcParams['pdf.fonttype'] = '42' 

#%% Fraction change in dF/F0 by stimulation

ipsi_paths = [r'F:\data\BAYLORCW032\python\2023_10_23',
         r'F:\data\BAYLORCW036\python\2023_10_20',
         r'F:\data\BAYLORCW034\python\2023_10_24',
         r'F:\data\BAYLORCW035\python\2023_12_06',
         r'F:\data\BAYLORCW037\python\2023_11_22'
         ]

contra_paths = [
            r'F:\data\BAYLORCW032\python\2023_10_24',
            # r'F:\data\BAYLORCW034\python\2023_10_27',
            r'F:\data\BAYLORCW036\python\2023_10_30',
            r'F:\data\BAYLORCW035\python\2023_12_15',
            r'F:\data\BAYLORCW037\python\2023_12_15',
            ]


all_ipsi_fracs = []


for path in ipsi_paths:

    l1 = quality.QC(path, use_background_sub=False)
    
    frac, sig_n = l1.stim_effect_per_neuron()
    if frac < 5 and frac > -5:
        all_ipsi_fracs += [frac]
    
all_contra_fracs = []
for path in contra_paths:

    l1 = quality.QC(path, use_background_sub=False)
    
    frac, sig_n = l1.stim_effect_per_neuron()
    
    if frac < 5 and frac > -5:
        all_contra_fracs += [frac]
       
    
f, axarr = plt.subplots(1,1, sharex='col', figsize=(5,5))
axarr.bar([0, 1], [np.mean(all_ipsi_fracs), np.mean(all_contra_fracs)])
axarr.scatter(np.zeros(len(all_ipsi_fracs)), all_ipsi_fracs)
axarr.scatter(np.ones(len(all_contra_fracs)), all_contra_fracs)
axarr.axhline(1, ls = '--', color='lightgrey')
axarr.set_ylabel('Fraction change in dF/F0')
axarr.set_xticks([0,1], ['Ipsilateral to imaging', 'Contralateral'])
plt.show()

#%% Fraction of neurons affected by stimulation

ipsi_frac_sup, ipsi_frac_exc = [], []
for path in ipsi_paths:

    l1 = quality.QC(path, use_background_sub=True)
    
    _, sig_n = l1.stim_effect_per_neuron()
    
    all_ipsi_fracs += [frac]
    
    ipsi_frac_sup += [len(np.where(sig_n < 0)[0]) / len(sig_n)]
    ipsi_frac_exc += [len(np.where(sig_n > 0)[0]) / len(sig_n)]
 
contra_frac_sup, contra_frac_exc = [], []
for path in contra_paths:

    l1 = quality.QC(path, use_background_sub=True)
    
    _, sig_n = l1.stim_effect_per_neuron()
    
    all_ipsi_fracs += [frac]
    
    contra_frac_sup += [len(np.where(sig_n < 0)[0]) / len(sig_n)]
    contra_frac_exc += [len(np.where(sig_n > 0)[0]) / len(sig_n)]
    
    
plt.barh([0, 1], [np.mean(ipsi_frac_exc), np.mean(contra_frac_exc)], color = 'r', edgecolor = 'black', label = 'Excited')
plt.barh([0, 1], [-np.mean(ipsi_frac_sup), -np.mean(contra_frac_sup)], color = 'b', edgecolor = 'black', label = 'Inhibited')
plt.scatter(cat((ipsi_frac_exc, -1 * np.array(ipsi_frac_sup))), np.zeros(len(cat((ipsi_frac_exc, ipsi_frac_sup)))), facecolors='none', edgecolors='grey')
plt.scatter(cat((contra_frac_exc, -1 * np.array(contra_frac_sup))), np.ones(len(cat((contra_frac_exc, contra_frac_sup)))), facecolors='none', edgecolors='grey')

plt.axvline(0)
plt.yticks([0,1], ['Ipsilateral to imaging', 'Contralateral'])
plt.ylabel('Condition')
plt.xlabel('Fraction of neurons with significant dF/F0 change')
plt.legend()
plt.show()
    