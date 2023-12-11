# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 15:19:57 2023

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
cat=np.concatenate
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
# IPSI
paths = [r'F:\data\BAYLORCW032\python\2023_10_08',
          r'F:\data\BAYLORCW032\python\2023_10_16',
          r'F:\data\BAYLORCW032\python\2023_10_25',]
for path in paths:
    
    l1 = Mode(path, use_reg=True, triple=True)
    
    l1.selectivity_optogenetics(p=0.05)
    l1.plot_CD_opto()

#%% CW34 matched
paths = [ r'F:\data\BAYLORCW034\python\2023_10_12',
           r'F:\data\BAYLORCW034\python\2023_10_22',
           r'F:\data\BAYLORCW034\python\2023_10_27',]
for path in paths:
    
    l1 = Mode(path, use_reg=True, triple=True)
    
    l1.selectivity_optogenetics()
    l1.plot_CD_opto()

#%% CW36 matched
paths = [r'F:\data\BAYLORCW036\python\2023_10_09',
           r'F:\data\BAYLORCW036\python\2023_10_19',
           r'F:\data\BAYLORCW036\python\2023_10_30',]
for path in paths:
    
    l1 = Mode(path, use_reg=True, triple=True)
    
    l1.selectivity_optogenetics(p=0.01)
    l1.plot_CD_opto()

#%% CW32 Unmatched

#CONTRA
path = r'F:\data\BAYLORCW032\python\2023_10_24'
paths = [r'F:\data\BAYLORCW032\python\2023_10_05',
          r'F:\data\BAYLORCW032\python\2023_10_19',
          r'F:\data\BAYLORCW032\python\2023_10_24',]

# IPSI
# paths = [r'F:\data\BAYLORCW032\python\2023_10_08',
#           r'F:\data\BAYLORCW032\python\2023_10_16',
#            r'F:\data\BAYLORCW032\python\2023_10_25',]

for path in paths:
    l1 = session.Session(path)
    
    l1.selectivity_optogenetics(p=0.01)
#%% CW34 unmatched

paths = [ r'F:\data\BAYLORCW034\python\2023_10_12',
           r'F:\data\BAYLORCW034\python\2023_10_22',
           r'F:\data\BAYLORCW034\python\2023_10_27',]
for path in paths:
    
    l1 = session.Session(path)
    
    l1.selectivity_optogenetics()
#%% CW36 unmatched
paths = [r'F:\data\BAYLORCW036\python\2023_10_09',
           r'F:\data\BAYLORCW036\python\2023_10_19',
           r'F:\data\BAYLORCW036\python\2023_10_30',]
for path in paths:
    
    l1 = session.Session(path)
    
    l1.selectivity_optogenetics(p=0.01)
    
#%% CW35 unmatched
paths = [r'F:\data\BAYLORCW035\python\2023_10_12',
            r'F:\data\BAYLORCW035\python\2023_10_26',
           r'F:\data\BAYLORCW035\python\2023_11_29',
           r'F:\data\BAYLORCW035\python\2023_12_07',]
for path in paths:
    
    l1 = session.Session(path)
    
    l1.selectivity_optogenetics(p=0.01)
    
#%% CW37 unmatched
paths = [r'F:\data\BAYLORCW037\python\2023_11_21',
          r'F:\data\BAYLORCW037\python\2023_11_28',
            r'F:\data\BAYLORCW035\python\2023_12_08',]
for path in paths:
    
    l1 = session.Session(path)
    
    l1.selectivity_optogenetics(p=0.01)
    
#%% AGG all sesssions
pref, nonpref, optop, optonp = np.zeros(61), np.zeros(61), np.zeros(61), np.zeros(61)
num_neurons = 0
# CONTRA PATHS:
# paths = [r'F:\data\BAYLORCW032\python\2023_10_24',
#            r'F:\data\BAYLORCW034\python\2023_10_27',
#            r'F:\data\BAYLORCW036\python\2023_10_30',]
# paths = [r'F:\data\BAYLORCW032\python\2023_10_05',
#            r'F:\data\BAYLORCW034\python\2023_10_12',
#            r'F:\data\BAYLORCW036\python\2023_10_09',]
# paths = [r'F:\data\BAYLORCW032\python\2023_10_19',
#             r'F:\data\BAYLORCW034\python\2023_10_22',
#             r'F:\data\BAYLORCW036\python\2023_10_19',]

# IPSI PATHS:
paths = [r'F:\data\BAYLORCW032\python\2023_10_25',
            r'F:\data\BAYLORCW034\python\2023_10_28',
            r'F:\data\BAYLORCW036\python\2023_10_20',]
# paths = [r'F:\data\BAYLORCW032\python\2023_10_08',
#             r'F:\data\BAYLORCW035\python\2023_10_11',
#             r'F:\data\BAYLORCW036\python\2023_10_07',]
# paths = [r'F:\data\BAYLORCW032\python\2023_10_16',
#             r'F:\data\BAYLORCW035\python\2023_10_27',
#             r'F:\data\BAYLORCW036\python\2023_10_17',]

for path in paths:
    
    # l1 = session.Session(path, use_reg=True, triple=True)
    l1 = session.Session(path)
    
    pref_, nonpref_, optop_, optonp_ = l1.selectivity_optogenetics(p=0.01, return_traces=True)
    
    pref = np.vstack((pref, pref_))
    nonpref = np.vstack((nonpref, nonpref_))
    optop = np.vstack((optop, optop_))
    optonp = np.vstack((optonp, optonp_))
    
    num_neurons += len(l1.selective_neurons)
    
pref, nonpref, optop, optonp = pref[1:], nonpref[1:], optop[1:], optonp[1:]
f, axarr = plt.subplots(1,1, sharex='col', figsize=(5,5))

sel = np.mean(pref, axis = 0) - np.mean(nonpref, axis = 0)
err = np.std(pref, axis=0) / np.sqrt(len(pref)) 
err += np.std(nonpref, axis=0) / np.sqrt(len(nonpref))

selo = np.mean(optop, axis = 0) - np.mean(optonp, axis = 0)
erro = np.std(optop, axis=0) / np.sqrt(len(optop)) 
erro += np.std(optonp, axis=0) / np.sqrt(len(optonp))    
x = np.arange(-6.97,4,l1.fs)[:l1.time_cutoff]
axarr.plot(x, sel, 'black')
        
axarr.fill_between(x, sel - err, 
          sel + err,
          color=['darkgray'])

axarr.plot(x, selo, 'r-')
        
axarr.fill_between(x, selo - erro, 
          selo + erro,
          color=['#ffaeb1'])       

axarr.axvline(-4.3, color = 'grey', alpha=0.5, ls = '--')
axarr.axvline(-3, color = 'grey', alpha=0.5, ls = '--')
axarr.axvline(0, color = 'grey', alpha=0.5, ls = '--')
axarr.hlines(y=max(cat((selo, sel))), xmin=-3, xmax=-2, linewidth=10, color='red')

axarr.set_title('Optogenetic effect on selectivity (n = {} neurons)'.format(num_neurons))                  
axarr.set_xlabel('Time from Go cue (s)')
axarr.set_ylabel('Selectivity')
