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
from scipy import stats
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
#%% CW35 matched
paths = [r'F:\data\BAYLORCW035\python\2023_10_26',
            r'F:\data\BAYLORCW035\python\2023_12_07',
            r'F:\data\BAYLORCW035\python\2023_12_15',]
for path in paths:
    
    l1 = Mode(path, use_reg=True, triple=True)
    
    l1.selectivity_optogenetics(p=0.01)
    l1.plot_CD_opto()
#%% CW37 matched
paths = [r'F:\data\BAYLORCW037\python\2023_11_21',
            r'F:\data\BAYLORCW037\python\2023_12_08',
            r'F:\data\BAYLORCW037\python\2023_12_15',]
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

paths = [r'F:\data\BAYLORCW035\python\2023_12_16']
for path in paths:
    
    l1 = session.Session(path)
    l1.i_good_trials = l1.i_good_trials[80:]
    # l1.i_good_trials = l1.i_good_trials[:80]
    l1.selectivity_optogenetics(p=0.01)
    
#%% CW37 unmatched
paths = [r'F:\data\BAYLORCW037\python\2023_11_21',
          r'F:\data\BAYLORCW037\python\2023_11_28',
            r'F:\data\BAYLORCW035\python\2023_12_08',]


paths = [ 
    
    # r'H:\data\BAYLORCW044\python\2024_05_22',
    #       r'H:\data\BAYLORCW044\python\2024_05_23',
           # r'H:\data\BAYLORCW044\python\2024_05_24',
         
          r'H:\data\BAYLORCW046\python\2024_05_29',
          # r'H:\data\BAYLORCW046\python\2024_05_30',
          # r'H:\data\BAYLORCW046\python\2024_05_31',
    ]


# paths = [ r'H:\data\BAYLORCW044\python\2024_06_04',
#           r'H:\data\BAYLORCW044\python\2024_06_05',
#           r'H:\data\BAYLORCW044\python\2024_06_06',
         
#           r'H:\data\BAYLORCW046\python\2024_06_07',
#           r'H:\data\BAYLORCW046\python\2024_06_10',
#           r'H:\data\BAYLORCW046\python\2024_06_11',
#     ]

for path in paths:
    
    l1 = session.Session(path)
    
    l1.selectivity_optogenetics(p=0.05)
    
#%% AGG all sesssions matched neurons
pref, nonpref, optop, optonp = np.zeros(61), np.zeros(61), np.zeros(61), np.zeros(61)
num_neurons = 0
# NEW DATASET

paths = [ r'H:\data\BAYLORCW044\python\2024_05_22',
         r'H:\data\BAYLORCW044\python\2024_05_23',
         r'H:\data\BAYLORCW044\python\2024_05_24',
         
         r'H:\data\BAYLORCW046\python\2024_05_29',
         r'H:\data\BAYLORCW046\python\2024_05_30',
         r'H:\data\BAYLORCW046\python\2024_05_31',
    ]


# CONTRA PATHS:
paths = [    r'F:\data\BAYLORCW032\python\2023_10_05',
            # r'F:\data\BAYLORCW034\python\2023_10_12',
            r'F:\data\BAYLORCW036\python\2023_10_09',
            r'F:\data\BAYLORCW035\python\2023_10_26',
            r'F:\data\BAYLORCW037\python\2023_11_21',
            
            r'H:\data\BAYLORCW044\python\2024_05_22',
            r'H:\data\BAYLORCW044\python\2024_05_23',
            
            r'H:\data\BAYLORCW046\python\2024_05_29',
            r'H:\data\BAYLORCW046\python\2024_05_30',
            r'H:\data\BAYLORCW046\python\2024_05_31',
            ]

paths = [r'F:\data\BAYLORCW032\python\2023_10_19',
            # r'F:\data\BAYLORCW034\python\2023_10_22',
            r'F:\data\BAYLORCW036\python\2023_10_19',
            r'F:\data\BAYLORCW035\python\2023_12_07',
            r'F:\data\BAYLORCW037\python\2023_12_08',
            
            r'H:\data\BAYLORCW044\python\2024_06_06',
            r'H:\data\BAYLORCW044\python\2024_06_04',

            r'H:\data\BAYLORCW046\python\2024_06_07',
            r'H:\data\BAYLORCW046\python\2024_06_10',
            r'H:\data\BAYLORCW046\python\2024_06_11',
            ]


# paths = [r'F:\data\BAYLORCW032\python\2023_10_24',
#             # r'F:\data\BAYLORCW034\python\2023_10_27',
#             r'F:\data\BAYLORCW036\python\2023_10_30',
#             r'F:\data\BAYLORCW035\python\2023_12_15',
#             r'F:\data\BAYLORCW037\python\2023_12_15',
            
#             r'H:\data\BAYLORCW044\python\2024_06_19',
#             r'H:\data\BAYLORCW044\python\2024_06_18',
            
#             r'H:\data\BAYLORCW046\python\2024_06_24',
#             r'H:\data\BAYLORCW046\python\2024_06_27',
#             r'H:\data\BAYLORCW046\python\2024_06_26',
            
#             ]



# IPSI PATHS:
# paths = [r'F:\data\BAYLORCW032\python\2023_10_25',
#             r'F:\data\BAYLORCW034\python\2023_10_28',
#             r'F:\data\BAYLORCW036\python\2023_10_20',]
# paths = [r'F:\data\BAYLORCW032\python\2023_10_08',
#             r'F:\data\BAYLORCW035\python\2023_10_11',
#             r'F:\data\BAYLORCW036\python\2023_10_07',]
# paths = [r'F:\data\BAYLORCW032\python\2023_10_16',
#             r'F:\data\BAYLORCW035\python\2023_10_27',
#             r'F:\data\BAYLORCW036\python\2023_10_17',]
for path in paths:
    
    l1 = session.Session(path, use_reg=True, triple=True, remove_consec_opto=False)
    # l1 = session.Session(path)
    
    pref_, nonpref_, optop_, optonp_ = l1.selectivity_optogenetics(p=0.01, 
                                                                   lickdir=False, 
                                                                   return_traces=True,
                                                                   downsample='04' in path)
    
    pref = np.vstack((pref, np.mean(pref_,axis=0)))
    nonpref = np.vstack((nonpref, np.mean(nonpref_, axis=0)))
    optop = np.vstack((optop, np.mean(optop_,axis=0)))
    optonp = np.vstack((optonp, np.mean(optonp_,axis=0)))
    
    num_neurons += len(l1.selective_neurons)
    
pref, nonpref, optop, optonp = pref[1:], nonpref[1:], optop[1:], optonp[1:]

sel = np.mean(pref, axis = 0) - np.mean(nonpref, axis = 0)
err = np.std(pref, axis=0) / np.sqrt(num_neurons) 
err += np.std(nonpref, axis=0) / np.sqrt(num_neurons)

selo = np.mean(optop, axis = 0) - np.mean(optonp, axis = 0)
erro = np.std(optop, axis=0) / np.sqrt(num_neurons) 
erro += np.std(optonp, axis=0) / np.sqrt(num_neurons)  

f, axarr = plt.subplots(1,1, sharex='col', figsize=(5,5))  
x = np.arange(-6.97,4,1/6)[:pref.shape[1]]
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
axarr.set_ylim((-0.2, 0.7))

# plt.savefig(r'F:\data\Fig 3\nai_sel_recovery_updated.pdf')
plt.show()


#%% AGG all sesssions matched neurons only from expert pool

# NEW DATASET

paths = [ r'H:\data\BAYLORCW044\python\2024_05_22',
         r'H:\data\BAYLORCW044\python\2024_05_23',
         r'H:\data\BAYLORCW044\python\2024_05_24',
         
         r'H:\data\BAYLORCW046\python\2024_05_29',
         r'H:\data\BAYLORCW046\python\2024_05_30',
         r'H:\data\BAYLORCW046\python\2024_05_31',
    ]


# CONTRA PATHS:
allpaths = [[r'F:\data\BAYLORCW032\python\2023_10_24',
            # r'F:\data\BAYLORCW034\python\2023_10_27',
            r'F:\data\BAYLORCW036\python\2023_10_30',
            r'F:\data\BAYLORCW035\python\2023_12_15',
            r'F:\data\BAYLORCW037\python\2023_12_15',
            
            r'H:\data\BAYLORCW044\python\2024_06_19',
            r'H:\data\BAYLORCW044\python\2024_06_18',
            
            r'H:\data\BAYLORCW046\python\2024_06_24',
            r'H:\data\BAYLORCW046\python\2024_06_27',
            r'H:\data\BAYLORCW046\python\2024_06_26',
            
            ],
            [r'F:\data\BAYLORCW032\python\2023_10_19',
            # r'F:\data\BAYLORCW034\python\2023_10_22',
            r'F:\data\BAYLORCW036\python\2023_10_19',
            r'F:\data\BAYLORCW035\python\2023_12_07',
            r'F:\data\BAYLORCW037\python\2023_12_08',
            
            r'H:\data\BAYLORCW044\python\2024_06_06',
            r'H:\data\BAYLORCW044\python\2024_06_04',

            r'H:\data\BAYLORCW046\python\2024_06_07',
            r'H:\data\BAYLORCW046\python\2024_06_10',
            r'H:\data\BAYLORCW046\python\2024_06_11',
            ],
            [r'F:\data\BAYLORCW032\python\2023_10_05',
            # r'F:\data\BAYLORCW034\python\2023_10_12',
            r'F:\data\BAYLORCW036\python\2023_10_09',
            r'F:\data\BAYLORCW035\python\2023_10_26',
            r'F:\data\BAYLORCW037\python\2023_11_21',
            
            r'H:\data\BAYLORCW044\python\2024_05_22',
            r'H:\data\BAYLORCW044\python\2024_05_23',
            
            r'H:\data\BAYLORCW046\python\2024_05_29',
            r'H:\data\BAYLORCW046\python\2024_05_30',
            r'H:\data\BAYLORCW046\python\2024_05_31',
            ]]





# IPSI PATHS:
# paths = [r'F:\data\BAYLORCW032\python\2023_10_25',
#             r'F:\data\BAYLORCW034\python\2023_10_28',
#             r'F:\data\BAYLORCW036\python\2023_10_20',]
# paths = [r'F:\data\BAYLORCW032\python\2023_10_08',
#             r'F:\data\BAYLORCW035\python\2023_10_11',
#             r'F:\data\BAYLORCW036\python\2023_10_07',]
# paths = [r'F:\data\BAYLORCW032\python\2023_10_16',
#             r'F:\data\BAYLORCW035\python\2023_10_27',
#             r'F:\data\BAYLORCW036\python\2023_10_17',]

expert_neurons = []
for i in range(3):
    pref, nonpref, optop, optonp = np.zeros(61), np.zeros(61), np.zeros(61), np.zeros(61)
    num_neurons = 0
    paths = allpaths[i]
    for j in range(len(paths)):
        
        path = paths[j]
        l1 = session.Session(path, use_reg=True, triple=True, remove_consec_opto=True)
        # l1 = session.Session(path)
        if i == 0:
            pref_, nonpref_, optop_, optonp_ = l1.selectivity_optogenetics(p=0.01, 
                                                                           lickdir=False, 
                                                                           return_traces=True,
                                                                           downsample='04' in path)
        else: 
            sel_n = [l1.good_neurons[n] for n in expert_neurons[j]] 
            pref_, nonpref_, optop_, optonp_ = l1.selectivity_optogenetics(p=0.01, 
                                                                           lickdir=False, 
                                                                           return_traces=True,
                                                                           selective_neurons=sel_n,
                                                                           downsample='04' in path)
        pref = np.vstack((pref, np.mean(pref_,axis=0)))
        nonpref = np.vstack((nonpref, np.mean(nonpref_, axis=0)))
        optop = np.vstack((optop, np.mean(optop_,axis=0)))
        optonp = np.vstack((optonp, np.mean(optonp_,axis=0)))
        
        if i == 0:
            expert_neurons += [[np.where(l1.good_neurons == n)[0][0] for n in l1.selective_neurons]]
            num_neurons += len(expert_neurons[j])

        else:
            num_neurons += len(expert_neurons[j])

        
    pref, nonpref, optop, optonp = pref[1:], nonpref[1:], optop[1:], optonp[1:]
    
    sel = np.mean(pref, axis = 0) - np.mean(nonpref, axis = 0)
    err = np.std(pref, axis=0) / np.sqrt(num_neurons) 
    err += np.std(nonpref, axis=0) / np.sqrt(num_neurons)
    
    selo = np.mean(optop, axis = 0) - np.mean(optonp, axis = 0)
    erro = np.std(optop, axis=0) / np.sqrt(num_neurons) 
    erro += np.std(optonp, axis=0) / np.sqrt(num_neurons)  
    
    f, axarr = plt.subplots(1,1, sharex='col', figsize=(5,5))  
    x = np.arange(-6.97,4,1/6)[:pref.shape[1]]
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
    axarr.set_ylim((-0.2, 0.7))
    
    # plt.savefig(r'F:\data\Fig 3\nai_sel_recovery_updated.pdf')
    plt.show()



#%% AGG all sesssions ALL unmatched neurons MORE sessions
pref, nonpref, optop, optonp = np.zeros(61), np.zeros(61), np.zeros(61), np.zeros(61)
num_neurons = 0
# CONTRA PATHS:
paths = [    r'F:\data\BAYLORCW032\python\2023_10_05',
            # r'F:\data\BAYLORCW034\python\2023_10_12',
            r'F:\data\BAYLORCW036\python\2023_10_09',
            r'F:\data\BAYLORCW035\python\2023_10_26',
            r'F:\data\BAYLORCW037\python\2023_11_21',
            
            r'F:\data\BAYLORCW036\python\2023_10_16',
            r'F:\data\BAYLORCW035\python\2023_10_12',
        r'F:\data\BAYLORCW035\python\2023_11_02',

            ]

paths = [r'F:\data\BAYLORCW032\python\2023_10_19',
            # r'F:\data\BAYLORCW034\python\2023_10_22',
            r'F:\data\BAYLORCW036\python\2023_10_19',
            r'F:\data\BAYLORCW035\python\2023_12_07',
            r'F:\data\BAYLORCW037\python\2023_12_08',

        r'F:\data\BAYLORCW032\python\2023_10_18',
        r'F:\data\BAYLORCW035\python\2023_10_25',
            r'F:\data\BAYLORCW035\python\2023_11_27',
            r'F:\data\BAYLORCW035\python\2023_11_29',
            r'F:\data\BAYLORCW037\python\2023_11_28',

        ]


paths = [r'F:\data\BAYLORCW032\python\2023_10_24',
            # r'F:\data\BAYLORCW034\python\2023_10_27',
            r'F:\data\BAYLORCW036\python\2023_10_30',
            r'F:\data\BAYLORCW035\python\2023_12_15',
            r'F:\data\BAYLORCW037\python\2023_12_15',
            
            r'F:\data\BAYLORCW036\python\2023_10_28',
        r'F:\data\BAYLORCW035\python\2023_12_12',
            r'F:\data\BAYLORCW035\python\2023_12_14',
            r'F:\data\BAYLORCW035\python\2023_12_16',
            r'F:\data\BAYLORCW037\python\2023_12_13',

            ]



# IPSI PATHS:
# paths = [r'F:\data\BAYLORCW032\python\2023_10_25',
#             r'F:\data\BAYLORCW034\python\2023_10_28',
#             r'F:\data\BAYLORCW036\python\2023_10_20',]
# paths = [r'F:\data\BAYLORCW032\python\2023_10_08',
#             r'F:\data\BAYLORCW035\python\2023_10_11',
#             r'F:\data\BAYLORCW036\python\2023_10_07',]
# paths = [r'F:\data\BAYLORCW032\python\2023_10_16',
#             r'F:\data\BAYLORCW035\python\2023_10_27',
#             r'F:\data\BAYLORCW036\python\2023_10_17',]
for path in paths:
    
    l1 = session.Session(path)
    # l1 = session.Session(path)
    
    pref_, nonpref_, optop_, optonp_ = l1.selectivity_optogenetics(p=0.01, lickdir=False, return_traces=True)
    
    pref = np.vstack((pref, pref_))
    nonpref = np.vstack((nonpref, nonpref_))
    optop = np.vstack((optop, optop_))
    optonp = np.vstack((optonp, optonp_))
    
    num_neurons += len(l1.selective_neurons)
    
pref, nonpref, optop, optonp = pref[1:], nonpref[1:], optop[1:], optonp[1:]

sel = np.mean(pref, axis = 0) - np.mean(nonpref, axis = 0)
err = np.std(pref, axis=0) / np.sqrt(len(pref)) 
err += np.std(nonpref, axis=0) / np.sqrt(len(nonpref))

selo = np.mean(optop, axis = 0) - np.mean(optonp, axis = 0)
erro = np.std(optop, axis=0) / np.sqrt(len(optop)) 
erro += np.std(optonp, axis=0) / np.sqrt(len(optonp))  

f, axarr = plt.subplots(1,1, sharex='col', figsize=(5,5))  
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
axarr.set_ylim((-0.2, 0.7))

# plt.savefig(r'F:\data\Fig 3\lea_sel_recovery_ALL.pdf')
plt.show()

#%% Plot selectivity recovery as a bar graph only matched

# CONTRA PATHS:
all_paths = [[    r'F:\data\BAYLORCW032\python\2023_10_05',
            # r'F:\data\BAYLORCW034\python\2023_10_12',
            r'F:\data\BAYLORCW036\python\2023_10_09',
            r'F:\data\BAYLORCW035\python\2023_10_26',
            r'F:\data\BAYLORCW037\python\2023_11_21',],

        [r'F:\data\BAYLORCW032\python\2023_10_19',
            # r'F:\data\BAYLORCW034\python\2023_10_22',
            r'F:\data\BAYLORCW036\python\2023_10_19',
            r'F:\data\BAYLORCW035\python\2023_12_07',
            r'F:\data\BAYLORCW037\python\2023_12_08',],


        [r'F:\data\BAYLORCW032\python\2023_10_24',
            # r'F:\data\BAYLORCW034\python\2023_10_27',
            r'F:\data\BAYLORCW036\python\2023_10_30',
            r'F:\data\BAYLORCW035\python\2023_12_15',
            r'F:\data\BAYLORCW037\python\2023_12_15',]]

all_paths = [[    r'F:\data\BAYLORCW032\python\2023_10_05',
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

            r'H:\data\BAYLORCW046\python\2024_06_07',
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
            
            r'H:\data\BAYLORCW046\python\2024_06_24',
            r'H:\data\BAYLORCW046\python\2024_06_27',
            r'H:\data\BAYLORCW046\python\2024_06_26',
            
            ]]

naive_sel_recovery,learning_sel_recovery,expert_sel_recovery = [],[],[]
all_recovery = []
for paths in all_paths: # For each stage of training
    recovery = []
    for path in paths: # For each mouse
        
        l1 = session.Session(path, use_reg=True, triple=True, remove_consec_opto=True)
        # l1 = session.Session(path)
        temp, _ = l1.modularity_proportion(p=0.01, lickdir=True)
        # if temp > 0 and temp < 1: # Exclude values based on Chen et al method guidelines
        if True:
            recovery += [temp]
    
    all_recovery += [recovery]
        

plt.bar(range(3), [np.mean(a) for a in all_recovery])
plt.scatter(np.zeros(len(all_recovery[0])), all_recovery[0])
plt.scatter(np.ones(len(all_recovery[1])), all_recovery[1])
plt.scatter(np.ones(len(all_recovery[2]))+1, all_recovery[2])

plt.xticks(range(3), ['Naive', 'Learning', 'Expert'])
plt.ylabel('Modularity')
# plt.savefig(r'F:\data\Fig 3\updated_modularity_bargraph.pdf')

plt.show()

# Add t-test:

tstat, p_val = scipy.stats.ttest_ind(all_recovery[1], all_recovery[2], equal_var=False, permutations = np.inf, alternative='less')
tstat, p_val = scipy.stats.ttest_rel(all_recovery[1], all_recovery[2])
print("mod diff p-value: ", p_val)


#%% Plot selectivity recovery as a bar graph ALL sessions

# CONTRA PATHS:


all_paths = [[    r'F:\data\BAYLORCW032\python\2023_10_05',
            # r'F:\data\BAYLORCW034\python\2023_10_12',
            r'F:\data\BAYLORCW036\python\2023_10_09',
            r'F:\data\BAYLORCW035\python\2023_10_26',
            r'F:\data\BAYLORCW037\python\2023_11_21',
            
            r'F:\data\BAYLORCW036\python\2023_10_16',
            r'F:\data\BAYLORCW035\python\2023_10_12',
            r'F:\data\BAYLORCW035\python\2023_11_02',

            ],
             [r'F:\data\BAYLORCW032\python\2023_10_19',
            # r'F:\data\BAYLORCW034\python\2023_10_22',
            r'F:\data\BAYLORCW036\python\2023_10_19',
            r'F:\data\BAYLORCW035\python\2023_12_07',
            r'F:\data\BAYLORCW037\python\2023_12_08',

        r'F:\data\BAYLORCW032\python\2023_10_18',
        r'F:\data\BAYLORCW035\python\2023_10_25',
            r'F:\data\BAYLORCW035\python\2023_11_27',
            r'F:\data\BAYLORCW035\python\2023_11_29',
            r'F:\data\BAYLORCW037\python\2023_11_28',

        ],
        [r'F:\data\BAYLORCW032\python\2023_10_24',
            # r'F:\data\BAYLORCW034\python\2023_10_27',
            r'F:\data\BAYLORCW036\python\2023_10_30',
            r'F:\data\BAYLORCW035\python\2023_12_15',
            r'F:\data\BAYLORCW037\python\2023_12_15',
            
            r'F:\data\BAYLORCW036\python\2023_10_28',
        r'F:\data\BAYLORCW035\python\2023_12_12',
            r'F:\data\BAYLORCW035\python\2023_12_14',
            r'F:\data\BAYLORCW035\python\2023_12_16',
            r'F:\data\BAYLORCW037\python\2023_12_13',

            ]]

naive_sel_recovery,learning_sel_recovery,expert_sel_recovery = [],[],[]
all_recovery = []
for paths in all_paths: # For each stage of training
    recovery = []
    for path in paths: # For each mouse
        
        l1 = session.Session(path)
        # l1 = session.Session(path)
        temp = l1.modularity_proportion(p=0.01)
        if temp > 0 and temp < 1: # Exclude values based on Chen et al method guideliens
            recovery += [temp]
    
    all_recovery += [recovery]
        

plt.bar(range(3), [np.mean(a) for a in all_recovery])
plt.scatter(np.zeros(len(all_recovery[0])), all_recovery[0])
plt.scatter(np.ones(len(all_recovery[1])), all_recovery[1])
plt.scatter(np.ones(len(all_recovery[2]))+1, all_recovery[2])

plt.xticks(range(3), ['Naive', 'Learning', 'Expert'])
plt.ylabel('Modularity, unperturbed hemisphere')
# plt.savefig(r'F:\data\Fig 3\modularity_bargraph_ALL.pdf')

plt.show()

# Add t-test:

tstat, p_val = stats.ttest_ind(all_recovery[1], all_recovery[2], equal_var=False, permutations = np.inf, alternative='less')
print("mod diff p-value: ", p_val)


#%% Correlate behavior recovery with modularity only matched
    
all_paths = [[    r'F:\data\BAYLORCW032\python\2023_10_05',
            # r'F:\data\BAYLORCW034\python\2023_10_12',
            r'F:\data\BAYLORCW036\python\2023_10_09',
            r'F:\data\BAYLORCW035\python\2023_10_26',
            r'F:\data\BAYLORCW037\python\2023_11_21',],

             [r'F:\data\BAYLORCW032\python\2023_10_19',
            # r'F:\data\BAYLORCW034\python\2023_10_22',
            r'F:\data\BAYLORCW036\python\2023_10_19',
            r'F:\data\BAYLORCW035\python\2023_12_07',
            r'F:\data\BAYLORCW037\python\2023_12_08',],


             [r'F:\data\BAYLORCW032\python\2023_10_24',
            # r'F:\data\BAYLORCW034\python\2023_10_27',
            r'F:\data\BAYLORCW036\python\2023_10_30',
            r'F:\data\BAYLORCW035\python\2023_12_15',
            r'F:\data\BAYLORCW037\python\2023_12_15',]]

all_deltas = []
all_mod = []

for paths in all_paths:

    deltas = []
    modularity = []

    for path in paths:

        l1 = session.Session(path, use_reg=True, triple=True, remove_consec_opto=True)
        temp, _ = l1.modularity_proportion(p=0.01, lickdir=True)

        if temp > 0 and temp < 1: # Exclude values based on Chen et al method guidelines

            stim_trials = np.where(l1.stim_ON)[0]
            control_trials = np.where(~l1.stim_ON)[0]
            
            _, _, perf_all = l1.performance_in_trials(stim_trials)
            _, _, perf_all_c = l1.performance_in_trials(control_trials)
            
            if perf_all_c < 0.5: #Skip low performance sessions
                continue
            
            modularity += [temp]

            deltas += [perf_all / perf_all_c]
            # deltas += [perf_all_c - perf_all]

    all_deltas += [deltas]
    all_mod += [modularity]
    
    
fig = plt.figure(figsize=(7,6))

ls = ['Naive', 'Learning', 'Expert']

for i in range(3):
    plt.scatter(all_mod[i], all_deltas[i], label = ls[i])

plt.xlabel('Modularity, unperturbed hemisphere')
plt.ylabel('Behvioral recovery, (frac. correct, photoinhib./control)')
# plt.ylabel('Behvioral recovery, (delta correct, control-photoinhib.)')
plt.legend()
plt.title('Pearsons correlation: {}, p-val: {}'.format(stats.pearsonr(cat(all_mod), cat(all_deltas))[0], 
                                                       stats.pearsonr(cat(all_mod), cat(all_deltas))[1]))

plt.savefig(r'F:\data\Fig 3\corr_behaviordiff_modularity_matched.pdf')
plt.show()

#%% Correlate behavior recovery with modularity ALL SESS
    
all_paths = [[    r'F:\data\BAYLORCW032\python\2023_10_05',
            # r'F:\data\BAYLORCW034\python\2023_10_12',
            r'F:\data\BAYLORCW036\python\2023_10_09',
            r'F:\data\BAYLORCW035\python\2023_10_26',
            r'F:\data\BAYLORCW037\python\2023_11_21',
            
            r'F:\data\BAYLORCW036\python\2023_10_16',
            r'F:\data\BAYLORCW035\python\2023_10_12',
        r'F:\data\BAYLORCW035\python\2023_11_02',

        r'H:\data\BAYLORCW044\python\2024_05_22',
        r'H:\data\BAYLORCW044\python\2024_05_23',
        r'H:\data\BAYLORCW044\python\2024_05_24',
        
        r'H:\data\BAYLORCW046\python\2024_05_29',
        r'H:\data\BAYLORCW046\python\2024_05_30',
        r'H:\data\BAYLORCW046\python\2024_05_31',
            ],
             [r'F:\data\BAYLORCW032\python\2023_10_19',
            # r'F:\data\BAYLORCW034\python\2023_10_22',
            r'F:\data\BAYLORCW036\python\2023_10_19',
            r'F:\data\BAYLORCW035\python\2023_12_07',
            r'F:\data\BAYLORCW037\python\2023_12_08',

        r'F:\data\BAYLORCW032\python\2023_10_18',
        r'F:\data\BAYLORCW035\python\2023_10_25',
            r'F:\data\BAYLORCW035\python\2023_11_27',
            r'F:\data\BAYLORCW035\python\2023_11_29',
            r'F:\data\BAYLORCW037\python\2023_11_28',
            
        r'H:\data\BAYLORCW044\python\2024_06_06',
        r'H:\data\BAYLORCW044\python\2024_06_04',
        r'H:\data\BAYLORCW044\python\2024_06_03',
        r'H:\data\BAYLORCW044\python\2024_06_12',

        r'H:\data\BAYLORCW046\python\2024_06_07',
        r'H:\data\BAYLORCW046\python\2024_06_10',
        r'H:\data\BAYLORCW046\python\2024_06_11',
        r'H:\data\BAYLORCW046\python\2024_06_19',

        ],
        [r'F:\data\BAYLORCW032\python\2023_10_24',
            # r'F:\data\BAYLORCW034\python\2023_10_27',
            r'F:\data\BAYLORCW036\python\2023_10_30',
            r'F:\data\BAYLORCW035\python\2023_12_15',
            r'F:\data\BAYLORCW037\python\2023_12_15',
            
            r'F:\data\BAYLORCW036\python\2023_10_28',
        r'F:\data\BAYLORCW035\python\2023_12_12',
            r'F:\data\BAYLORCW035\python\2023_12_14',
            r'F:\data\BAYLORCW035\python\2023_12_16',
            r'F:\data\BAYLORCW037\python\2023_12_13',
            
            r'H:\data\BAYLORCW044\python\2024_06_19',
            r'H:\data\BAYLORCW044\python\2024_06_18',
            r'H:\data\BAYLORCW044\python\2024_06_17',
            
            r'H:\data\BAYLORCW046\python\2024_06_24',
            r'H:\data\BAYLORCW046\python\2024_06_27',
            r'H:\data\BAYLORCW046\python\2024_06_26',
            r'H:\data\BAYLORCW046\python\2024_06_25',

]]

all_deltas = []
all_mod = []

for paths in all_paths:

    deltas = []
    modularity = []

    for path in paths:

        l1 = session.Session(path)
        temp, _ = l1.modularity_proportion(p=0.01, period = range(l1.response - int(1.5*(1/l1.fs)), l1.response))
        
        if temp is None: # No selective neurons
            continue

        if temp > 0 and temp < 1: # Exclude values based on Chen et al method guidelines

            stim_trials = np.where(l1.stim_ON)[0]
            control_trials = np.where(~l1.stim_ON)[0]
            
            _, _, perf_all = l1.performance_in_trials(stim_trials)
            _, _, perf_all_c = l1.performance_in_trials(control_trials)
            
            if perf_all_c < 0.5 or perf_all / perf_all_c > 1: #Skip low performance sessions
                continue
            
            modularity += [temp]


            deltas += [perf_all / perf_all_c]
            # deltas += [perf_all_c - perf_all]

    all_deltas += [deltas]
    all_mod += [modularity]
    
    
fig = plt.figure(figsize=(7,6))

ls = ['Naive', 'Learning', 'Expert']

for i in range(3):
    plt.scatter(all_mod[i], all_deltas[i], label = ls[i], s=150, alpha = 0.7)

plt.xlabel('Modularity, unperturbed hemisphere')
plt.ylabel('Behvioral recovery, (frac. correct, photoinhib./control)')
# plt.ylabel('Behvioral recovery, (delta correct, control-photoinhib.)')
plt.legend()
plt.title('Pearsons correlation: {}, p-val: {}'.format(stats.pearsonr(cat(all_mod), cat(all_deltas))[0], 
                                                       stats.pearsonr(cat(all_mod), cat(all_deltas))[1]))

plt.savefig(r'F:\data\Fig 3\corr_behaviordiff_modularity_ALL_plus.pdf')
plt.show()

#%% Correlate modularity with robustness ALL SESS
    
all_paths = [[    r'F:\data\BAYLORCW032\python\2023_10_05',
            # r'F:\data\BAYLORCW034\python\2023_10_12',
            r'F:\data\BAYLORCW036\python\2023_10_09',
            r'F:\data\BAYLORCW035\python\2023_10_26',
            r'F:\data\BAYLORCW037\python\2023_11_21',
            
            r'F:\data\BAYLORCW036\python\2023_10_16',
            r'F:\data\BAYLORCW035\python\2023_10_12',
        r'F:\data\BAYLORCW035\python\2023_11_02',

        r'H:\data\BAYLORCW044\python\2024_05_22',
        r'H:\data\BAYLORCW044\python\2024_05_23',
        r'H:\data\BAYLORCW044\python\2024_05_24',
        
        r'H:\data\BAYLORCW046\python\2024_05_29',
        r'H:\data\BAYLORCW046\python\2024_05_30',
        r'H:\data\BAYLORCW046\python\2024_05_31',
            ],
             [r'F:\data\BAYLORCW032\python\2023_10_19',
            # r'F:\data\BAYLORCW034\python\2023_10_22',
            r'F:\data\BAYLORCW036\python\2023_10_19',
            r'F:\data\BAYLORCW035\python\2023_12_07',
            r'F:\data\BAYLORCW037\python\2023_12_08',

        r'F:\data\BAYLORCW032\python\2023_10_18',
        r'F:\data\BAYLORCW035\python\2023_10_25',
            r'F:\data\BAYLORCW035\python\2023_11_27',
            r'F:\data\BAYLORCW035\python\2023_11_29',
            r'F:\data\BAYLORCW037\python\2023_11_28',
            
        r'H:\data\BAYLORCW044\python\2024_06_06',
        r'H:\data\BAYLORCW044\python\2024_06_04',
        r'H:\data\BAYLORCW044\python\2024_06_03',
        r'H:\data\BAYLORCW044\python\2024_06_12',

        r'H:\data\BAYLORCW046\python\2024_06_07',
        r'H:\data\BAYLORCW046\python\2024_06_10',
        r'H:\data\BAYLORCW046\python\2024_06_11',
        r'H:\data\BAYLORCW046\python\2024_06_19',

        ],
        [r'F:\data\BAYLORCW032\python\2023_10_24',
            # r'F:\data\BAYLORCW034\python\2023_10_27',
            r'F:\data\BAYLORCW036\python\2023_10_30',
            r'F:\data\BAYLORCW035\python\2023_12_15',
            r'F:\data\BAYLORCW037\python\2023_12_15',
            
            r'F:\data\BAYLORCW036\python\2023_10_28',
        r'F:\data\BAYLORCW035\python\2023_12_12',
            r'F:\data\BAYLORCW035\python\2023_12_14',
            r'F:\data\BAYLORCW035\python\2023_12_16',
            r'F:\data\BAYLORCW037\python\2023_12_13',
            
            r'H:\data\BAYLORCW044\python\2024_06_19',
            r'H:\data\BAYLORCW044\python\2024_06_18',
            r'H:\data\BAYLORCW044\python\2024_06_17',
            
            r'H:\data\BAYLORCW046\python\2024_06_24',
            r'H:\data\BAYLORCW046\python\2024_06_27',
            r'H:\data\BAYLORCW046\python\2024_06_26',
            r'H:\data\BAYLORCW046\python\2024_06_25',

        ]]


all_deltas = []
all_mod = []
p=0.01

for paths in all_paths:

    deltas = []
    modularity = []

    for path in paths:

        l1 = session.Session(path)
        temp, _ = l1.modularity_proportion(p=p) # Robustness
        
        if temp is None: # No selective neurons
            continue

        if temp > 0 and temp < 1: # Exclude values based on Chen et al method guidelines
            
            mod, _ = l1.modularity_proportion(p=p, period=range(l1.delay, l1.delay + int(1/l1.fs))) # Modularity / Coupling using first second of delay
            
            if mod > 0 and mod < 1:
                
                deltas += [temp]
                modularity += [mod]

    all_deltas += [deltas]
    all_mod += [modularity]
    
    
fig = plt.figure(figsize=(7,6))

ls = ['Naive', 'Learning', 'Expert']

for i in range(3):
    plt.scatter(all_mod[i], all_deltas[i], label = ls[i], s=150, alpha = 0.7)

plt.xlabel('Modularity, unperturbed hemisphere')
plt.ylabel('Robustness')
# plt.ylabel('Behvioral recovery, (delta correct, control-photoinhib.)')
plt.legend()
plt.title('Pearsons correlation: {}, p-val: {}'.format(stats.pearsonr(cat(all_mod), cat(all_deltas))[0], 
                                                       stats.pearsonr(cat(all_mod), cat(all_deltas))[1]))

plt.savefig(r'F:\data\Fig 3\corr_modularity_robustness_ALL.pdf')
plt.show()



#%% CD recovery

# paths =  [r'F:\data\BAYLORCW036\python\2023_10_07',
#             r'F:\data\BAYLORCW036\python\2023_10_17',
#             r'F:\data\BAYLORCW036\python\2023_10_30',]


paths = [    r'F:\data\BAYLORCW032\python\2023_10_05',
            # r'F:\data\BAYLORCW034\python\2023_10_12',
            r'F:\data\BAYLORCW036\python\2023_10_09',
            r'F:\data\BAYLORCW035\python\2023_10_26',
            r'F:\data\BAYLORCW037\python\2023_11_21',]

paths = [r'F:\data\BAYLORCW032\python\2023_10_19',
            # r'F:\data\BAYLORCW034\python\2023_10_22',
            r'F:\data\BAYLORCW036\python\2023_10_19',
            r'F:\data\BAYLORCW035\python\2023_12_07',
            r'F:\data\BAYLORCW037\python\2023_12_08',]


paths = [r'F:\data\BAYLORCW032\python\2023_10_24',
            # r'F:\data\BAYLORCW034\python\2023_10_27',
            r'F:\data\BAYLORCW036\python\2023_10_30',
            r'F:\data\BAYLORCW035\python\2023_12_15',
            r'F:\data\BAYLORCW037\python\2023_12_15',]


control_r, control_l = np.zeros(61), np.zeros(61)
opto_r, opto_l = np.zeros(61), np.zeros(61)
error_r, error_l = np.zeros(61), np.zeros(61)

allsets = []
counter = 0
for path in paths:
    l1 = Mode(path, use_reg = True, triple=True)

    # Expert stage only
    control_traces, opto_traces, error_bars, orthonormal_basis, mean, meantrain, meanstd = l1.plot_CD_opto(return_traces=True, return_applied=True)
    sett = orthonormal_basis, mean, meantrain, meanstd
    allsets += [sett]

    # Learning and naive stages
    # orthonormal_basis, mean, meantrain, meanstd = allsets[counter]
    # counter += 1
    # control_traces, opto_traces, error_bars = l1.plot_CD_opto_applied(orthonormal_basis, mean, meantrain, meanstd, return_traces=True)
    
    control_r = np.vstack((control_r, control_traces[0]))
    control_l = np.vstack((control_l, control_traces[1]))
    
    opto_r = np.vstack((opto_r, opto_traces[0]))
    opto_l = np.vstack((opto_l, opto_traces[1]))
    
    error_r = np.vstack((error_r, error_bars[0]))
    error_l = np.vstack((error_l, error_bars[1]))

    
# Plotting    
# Control trace:
x = np.arange(-6.97,4,l1.fs)[:l1.time_cutoff]

plt.plot(x, np.mean(control_r[1:], axis=0), 'b', ls = '--', linewidth = 0.5)
plt.plot(x, np.mean(control_l[1:], axis=0), 'r', ls = '--', linewidth = 0.5)
plt.title("Choice decoder projections with opto")
plt.axvline(-4.3, color = 'grey', alpha=0.5, ls = '--')
plt.axvline(-3, color = 'grey', alpha=0.5, ls = '--')
plt.axvline(0, color = 'grey', alpha=0.5, ls = '--')
plt.ylabel('CD_delay projection (a.u.)')

# Opto trace:
plt.plot(x, np.mean(opto_r[1:], axis=0), 'b', linewidth = 2)
plt.plot(x, np.mean(opto_l[1:], axis=0), 'r', linewidth = 2)

plt.fill_between(x, np.mean(opto_r[1:], axis=0) - np.mean(error_r[1:], axis=0), 
                     np.mean(opto_r[1:], axis=0) + np.mean(error_r[1:], axis=0),
                     color=['#b4b2dc'])
plt.fill_between(x,  np.mean(opto_l[1:], axis=0) - np.mean(error_l[1:], axis=0), 
          np.mean(opto_l[1:], axis=0) + np.mean(error_l[1:], axis=0),
         color=['#ffaeb1'])

plt.hlines(y=1.5, xmin=-3, xmax=-2, linewidth=10, color='red')
# plt.savefig(r'F:\data\Fig 3\CD_recovery_learningv1.pdf')

plt.show()
#%% Sort by selectivity pre-perturbation:
    
# Recreate Fig 5F
l1 = Mode(path, use_reg = True, triple=True)

l1.plot_sorted_CD_opto()

#%%



