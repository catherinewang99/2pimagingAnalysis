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
from alm_2p import session
from matplotlib.pyplot import figure
# import decon
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
# # IPSI
# paths = [r'F:\data\BAYLORCW032\python\2023_10_08',
#           r'F:\data\BAYLORCW032\python\2023_10_16',
#           r'F:\data\BAYLORCW032\python\2023_10_25',]
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

# paths = [r'F:\data\BAYLORCW032\python\2023_10_19',
#             # r'F:\data\BAYLORCW034\python\2023_10_22',
#             r'F:\data\BAYLORCW036\python\2023_10_19',
#             r'F:\data\BAYLORCW035\python\2023_12_07',
#             r'F:\data\BAYLORCW037\python\2023_12_08',
            
#             r'H:\data\BAYLORCW044\python\2024_06_06',
#             r'H:\data\BAYLORCW044\python\2024_06_04',

#             # r'H:\data\BAYLORCW046\python\2024_06_07',
#             r'H:\data\BAYLORCW046\python\2024_06_24',
#             r'H:\data\BAYLORCW046\python\2024_06_10',
#             r'H:\data\BAYLORCW046\python\2024_06_11',
#             ]


# paths = [r'F:\data\BAYLORCW032\python\2023_10_24',
#             # r'F:\data\BAYLORCW034\python\2023_10_27',
#             r'F:\data\BAYLORCW036\python\2023_10_30',
#             r'F:\data\BAYLORCW035\python\2023_12_15',
#             r'F:\data\BAYLORCW037\python\2023_12_15',
            
#             r'H:\data\BAYLORCW044\python\2024_06_19',
#             r'H:\data\BAYLORCW044\python\2024_06_18',
            
#             r'H:\data\BAYLORCW046\python\2024_06_28',
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

# pref, nonpref, optop, optonp = np.zeros(61), np.zeros(61), np.zeros(61), np.zeros(61)
all_control_sel, all_opto_sel = np.zeros(61), np.zeros(61)
num_neurons = 0
by_FOV = True
for path in paths:
    
    l1 = session.Session(path, use_reg=True, triple=True, 
                         use_background_sub=True,
                         remove_consec_opto=False,
                         baseline_normalization="median_zscore")    
    
    adjusted_p = 0.05 / np.sqrt(len(l1.good_neurons))
    # adjusted_p = 0.01
    
    control_sel, opto_sel = l1.selectivity_optogenetics(p=adjusted_p, 
                                                        # exclude_unselective=True,
                                                        exclude_unselective=False,
                                                        lickdir=False, 
                                                        return_traces=True,
                                                        downsample='04' in path)
    
    if control_sel is None or len(control_sel) == 0 or np.sum(control_sel) == 0: # no selective neurons
        
        continue
    
    num_neurons_selective = len(control_sel)
    fov_selectivity = np.mean(np.mean(control_sel, axis=0)[range(28, 40)])
    
    print(num_neurons_selective, fov_selectivity)
    
    # if num_neurons_selective > 3 and fov_selectivity > 0.3:
    if True:
        if by_FOV:
            all_control_sel = np.vstack((all_control_sel, np.mean(control_sel, axis=0)))
            all_opto_sel = np.vstack((all_opto_sel, np.mean(opto_sel, axis=0)))
        else:
            all_control_sel = np.vstack((all_control_sel, control_sel))
            all_opto_sel = np.vstack((all_opto_sel, opto_sel))
        num_neurons += num_neurons_selective
    
all_control_sel, all_opto_sel = all_control_sel[1:], all_opto_sel[1:]

# sel = np.mean(pref, axis = 0) - np.mean(nonpref, axis = 0)
# err = np.std(pref, axis=0) / np.sqrt(len(pref)*2) 
# err += np.std(nonpref, axis=0) / np.sqrt(len(pref)*2)

# selo = np.mean(optop, axis = 0) - np.mean(optonp, axis = 0)
# erro = np.std(optop, axis=0) / np.sqrt(len(pref)*2) 
# erro += np.std(optonp, axis=0) / np.sqrt(len(pref)*2)  

sel = np.mean(all_control_sel, axis=0)
err = np.std(all_control_sel, axis=0) / np.sqrt(len(all_control_sel))
selo = np.mean(all_opto_sel, axis=0)
erro = np.std(all_opto_sel, axis=0) / np.sqrt(len(all_opto_sel))

f, axarr = plt.subplots(1,1, sharex='col', figsize=(5,5))  
x = np.arange(-6.97,4,1/6)[:sel.shape[0]]
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
axarr.set_ylim((-0.2, 1.1))

plt.savefig(r'F:\data\Fig 3\nai_sel_recovery_updatedsess.pdf')
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


# CONTRA PATHS BACKWARDS:
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

            # r'H:\data\BAYLORCW046\python\2024_06_07', #sub out for below
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

naive_sel_recovery,learning_sel_recovery,expert_sel_recovery = [],[],[]
all_recovery = []
for st, paths in enumerate(all_paths[1:]): # For each stage of training
    recovery = []
    for path in paths: # For each mouse
        
        l1 = session.Session(path, use_reg=True, triple=True, 
                             use_background_sub=True,
                             remove_consec_opto=False,
                             baseline_normalization="median_zscore")   

        
        
    
        adjusted_p = 0.05 / np.sqrt(len(l1.good_neurons))
        adjusted_p = 0.01
        
        control_sel, opto_sel = l1.selectivity_optogenetics(p=adjusted_p, 
                                                            # exclude_unselective=st > 0,
                                                            exclude_unselective=True,
                                                            lickdir=False, 
                                                            return_traces=True,
                                                            downsample='04' in path)
        
        l1.selectivity_optogenetics(p=adjusted_p, 
                                    exclude_unselective=st > 0,
                                    # exclude_unselective=False,
                                    lickdir=False, 
                                    return_traces=False,
                                    downsample='04' in path)

        if control_sel is None or len(control_sel) == 0 or np.sum(control_sel) == 0: # no selective neurons
            print(path)
            continue
        
        temp, _ = l1.modularity_proportion(p=adjusted_p, 
                                            exclude_unselective=True,
                                            # exclude_unselective=st > 0,
                                            period = range(l1.delay+int(1/l1.fs), l1.response),
                                           lickdir=False,
                                           bootstrap=True)
        
        num_neurons_selective = len(control_sel)
        fov_selectivity = np.mean(np.mean(control_sel, axis=0)[range(28, 40)])
        
        print(num_neurons_selective, fov_selectivity)
        
        # if num_neurons_selective > 3 and fov_selectivity > 0.3 or st == 0:
        if True:

            if by_FOV:
                all_control_sel = np.vstack((all_control_sel, np.mean(control_sel, axis=0)))
                all_opto_sel = np.vstack((all_opto_sel, np.mean(opto_sel, axis=0)))
            else:
                all_control_sel = np.vstack((all_control_sel, control_sel))
                all_opto_sel = np.vstack((all_opto_sel, opto_sel))
            num_neurons += num_neurons_selective
            
            
            
            # if temp > 0 and temp < 1: # Exclude values based on Chen et al method guidelines
            if True:
                recovery += [temp]
    
    all_recovery += [recovery]
        

plt.bar(range(3), [np.mean(a) for a in all_recovery])
plt.scatter(np.zeros(len(all_recovery[0])), all_recovery[0], color='navajowhite')
plt.scatter(np.ones(len(all_recovery[1])), all_recovery[1], color='khaki')
plt.scatter(np.ones(len(all_recovery[2]))+1, all_recovery[2], color='lightskyblue')

plt.xticks(range(3), ['Naive', 'Learning', 'Expert'])
plt.ylabel('Robustness')
# for i in range(len(all_recovery[0])):
#     plt.plot([0,1], [all_recovery[0][i], all_recovery[1][i]], color='grey', alpha=0.3)
#     plt.plot([1,2], [all_recovery[1][i], all_recovery[2][i]], color='grey', alpha=0.3)
# plt.ylim(0,1.1)
# plt.savefig(r'F:\data\Fig 3\updated_modularity_bargraph_updated.pdf')

plt.show()


#only show learning and expert

plt.bar(range(2), [np.mean(a) for a in all_recovery[1:]])
# plt.scatter(np.zeros(len(all_recovery[0])), all_recovery[0])
plt.scatter(np.ones(len(all_recovery[1]))-1, all_recovery[1])
plt.scatter(np.ones(len(all_recovery[2])), all_recovery[2])
for i in range(len(all_recovery[1])):
    plt.plot([0,1], [all_recovery[1][i], all_recovery[2][i]], color='grey', alpha=0.3)
plt.xticks(range(2), ['Learning', 'Expert'])
plt.ylabel('Modularity')
# plt.ylim(bottom=0)
# plt.savefig(r'F:\data\Fig 3\updated_modularity_bargraph_leaexp.pdf')

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
all_paths = [[r'F:\data\BAYLORCW032\python\2023_10_05',
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

            # r'H:\data\BAYLORCW046\python\2024_06_07', #sub out for below
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

# all_paths = [[    r'F:\data\BAYLORCW032\python\2023_10_05',
#             # r'F:\data\BAYLORCW034\python\2023_10_12',
#             r'F:\data\BAYLORCW036\python\2023_10_09',
#             r'F:\data\BAYLORCW035\python\2023_10_26',
#             r'F:\data\BAYLORCW037\python\2023_11_21',
            
#             r'F:\data\BAYLORCW036\python\2023_10_16',
#             r'F:\data\BAYLORCW035\python\2023_10_12',
#         r'F:\data\BAYLORCW035\python\2023_11_02',

#         r'H:\data\BAYLORCW044\python\2024_05_22',
#         r'H:\data\BAYLORCW044\python\2024_05_23',
#         r'H:\data\BAYLORCW044\python\2024_05_24',
        
#         r'H:\data\BAYLORCW046\python\2024_05_29',
#         r'H:\data\BAYLORCW046\python\2024_05_30',
#         r'H:\data\BAYLORCW046\python\2024_05_31',
#             ],
#               [r'F:\data\BAYLORCW032\python\2023_10_19',
#             # r'F:\data\BAYLORCW034\python\2023_10_22',
#             r'F:\data\BAYLORCW036\python\2023_10_19',
#             r'F:\data\BAYLORCW035\python\2023_12_07',
#             r'F:\data\BAYLORCW037\python\2023_12_08',

#         r'F:\data\BAYLORCW032\python\2023_10_18',
#         r'F:\data\BAYLORCW035\python\2023_10_25',
#             r'F:\data\BAYLORCW035\python\2023_11_27',
#             r'F:\data\BAYLORCW035\python\2023_11_29',
#             r'F:\data\BAYLORCW037\python\2023_11_28',
            
#         r'H:\data\BAYLORCW044\python\2024_06_06',
#         r'H:\data\BAYLORCW044\python\2024_06_04',
#         r'H:\data\BAYLORCW044\python\2024_06_03',
#         r'H:\data\BAYLORCW044\python\2024_06_12',

#         r'H:\data\BAYLORCW046\python\2024_06_07',
#         r'H:\data\BAYLORCW046\python\2024_06_10',
#         r'H:\data\BAYLORCW046\python\2024_06_11',
#         # r'H:\data\BAYLORCW046\python\2024_06_19',
#         r'H:\data\BAYLORCW046\python\2024_06_25',
#         r'H:\data\BAYLORCW046\python\2024_06_24',



#         ],
#         [r'F:\data\BAYLORCW032\python\2023_10_24',
#             # r'F:\data\BAYLORCW034\python\2023_10_27',
#             r'F:\data\BAYLORCW036\python\2023_10_30',
#             r'F:\data\BAYLORCW035\python\2023_12_15',
#             r'F:\data\BAYLORCW037\python\2023_12_15',
            
#             r'F:\data\BAYLORCW036\python\2023_10_28',
#         r'F:\data\BAYLORCW035\python\2023_12_12',
#             r'F:\data\BAYLORCW035\python\2023_12_14',
#             r'F:\data\BAYLORCW035\python\2023_12_16',
#             r'F:\data\BAYLORCW037\python\2023_12_13',
            
#             r'H:\data\BAYLORCW044\python\2024_06_19',
#             r'H:\data\BAYLORCW044\python\2024_06_18',
#             r'H:\data\BAYLORCW044\python\2024_06_17',
#             r'H:\data\BAYLORCW044\python\2024_06_20',
            
#             r'H:\data\BAYLORCW046\python\2024_06_27',
#             r'H:\data\BAYLORCW046\python\2024_06_26',
#             r'H:\data\BAYLORCW046\python\2024_06_28',

# ]]

all_deltas = []
all_mod = []

all_stats = []

for st, paths in enumerate(all_paths):

    deltas = []
    modularity = []
    stats_ = []
    
    for path in paths:

        # l1 = session.Session(path, use_reg=True, triple=True, 
        #                      use_background_sub=True,
        #                      remove_consec_opto=False,
        #                      baseline_normalization="median_zscore")   
        
        l1 = session.Session(path, use_background_sub=True,
                             remove_consec_opto=False,
                             baseline_normalization="median_zscore") 

    
        # adjusted_p = 0.05 / np.sqrt(len(l1.good_neurons))
        adjusted_p = 0.01
        
        control_sel, opto_sel = l1.selectivity_optogenetics(p=adjusted_p, 
                                                            exclude_unselective=st > 0,
                                                            # exclude_unselective=False,
                                                            lickdir=False, 
                                                            return_traces=True,
                                                            downsample='04' in path)
        
        # l1.selectivity_optogenetics(p=adjusted_p, 
        #                             exclude_unselective=st > 0,
        #                             # exclude_unselective=False,
        #                             lickdir=False, 
        #                             return_traces=False,
        #                             downsample='04' in path)

        if control_sel is None or len(control_sel) == 0 or np.sum(control_sel) == 0: # no selective neurons
            print(path)
            continue
        
        temp, _ = l1.modularity_proportion(p=adjusted_p, 
                                           period = range(l1.delay+int(1/l1.fs), l1.response),
                                            exclude_unselective=st > 0,
                                           lickdir=False,
                                           bootstrap=True)
        
        num_neurons_selective = len(control_sel)
        fov_selectivity = np.mean(np.mean(control_sel, axis=0)[range(28, 40)])
        
        print(num_neurons_selective, fov_selectivity)
        # if num_neurons_selective > 3 and fov_selectivity > 0.3 or st == 0:
        if True:
            
            # if temp > 0 and temp < 1: # Exclude values based on Chen et al method guidelines
            # if temp > 0:
            if True:

                stim_trials = np.where(l1.stim_ON)[0]
                control_trials = np.where(~l1.stim_ON)[0]
                stim_trials = [c for c in stim_trials if c in l1.i_good_trials]
                stim_trials = [c for c in stim_trials if ~l1.early_lick[c]]
                control_trials = [c for c in control_trials if c in l1.i_good_trials]
                control_trials = [c for c in control_trials if ~l1.early_lick[c]]
                
                _, _, perf_all = l1.performance_in_trials(stim_trials)
                _, _, perf_all_c = l1.performance_in_trials(control_trials)

                if perf_all_c < 0.5: #or perf_all / perf_all_c > 1: #Skip low performance sessions
                    continue
                
                modularity += [temp]
                deltas += [perf_all - perf_all_c]

                stats_ += [(num_neurons_selective, fov_selectivity)]

    all_deltas += [deltas]
    all_mod += [modularity]
    all_stats += [stats]
#%%
fig = plt.figure(figsize=(7,6))

ls = ['Naive', 'Learning', 'Expert']

# for i in range(3):
for i in [1,2]:
    plt.scatter(all_mod[i], all_deltas[i], label = ls[i], s=150)

plt.xlabel('Robustness, unperturbed hemisphere')
plt.ylabel('Behvioral recovery, (correct, control - photoinhib.)')
# plt.ylabel('Behvioral recovery, (delta correct, control-photoinhib.)')
# plt.xlim(left=0)
# plt.xlim(right=1.5)
plt.legend()
plt.title('Pearsons correlation: {}, p-val: {}'.format(stats.pearsonr(cat(all_mod[1:]), cat(all_deltas[1:]))[0], 
                                                        stats.pearsonr(cat(all_mod[1:]), cat(all_deltas[1:]))[1]))

# plt.savefig(r'F:\data\Fig 3\corr_behaviordiff_modularity_leaexp_lasttwoseconds.pdf')
plt.show()
#%% Plot using previous calculations

robustness = np.array(all_recovery[1:])
robustness = all_recovery[1:]
#%%

comb_deltas = [[0.25991828866542954,
  # -0.10974754492891847,
  0.25143672384526033,
  -0.040446413608178355,
  0.08840105141227828,
  0.3998064712712779,
  0.3792390259441073,
  0.011909693681845579,
  0.0676638086069663],
 [0.10368558462698452,
  0.17703423665335816,
  -0.0749101480648876,
  -0.10563080495356036,
  0.07627252252252248,
  0.10954672511276276,
  0.26868853586410846,
  0.23441575750652122,
  0.12969162035523596]]

all_proportions = [[0.9160794362588085,
  # 1.0418342719227676,
  0.7905027932960893,
  0.8628725891291643,
  0.9269066117012842,
  0.898062015503876,
  0.9961636828644501,
  0.7371814092953524,
  1.0238048780487805],
 [1.016112389872212,
  0.7683653765426915,
  0.9354550330770606,
  0.9914068745003998,
  1.0181086519114688,
  0.9529633771316746,
  0.8993243243243243,
  0.9794403198172472,
  0.9178245335450576]]
#%%
# all_deltas
plt.scatter(cat(robustness), cat(all_proportions))
plt.scatter(cat(robustness), cat(comb_deltas))
plt.scatter(cat(robustness), cat(all_deltas))
# stats.pearsonr(cat(robustness), cat(all_deltas))

plt.scatter(cat(robustness), cat(comb_deltas))
# plt.xlim(top=1)
stats.pearsonr(cat(robustness), cat(comb_deltas))

#drop low values of robustness
robustness_filt = []
deltas_filt = []
for i in range(2):
    a,b = [],[]
    for j in range(len(robustness[i])):
        if robustness[i][j] < 1.2:  #and comb_deltas[i][j] < 0.2:
            a += [robustness[i][j]]
            b += [all_proportions[i][j]]
    robustness_filt += [a]
    deltas_filt += [b]
    
plt.scatter(cat(robustness_filt), cat(deltas_filt))
stats.pearsonr(cat(robustness_filt), cat(deltas_filt))
#%%

fig = plt.figure(figsize=(7,6))

ls = ['Learning', 'Expert']
for i in range(2):
    plt.scatter(robustness_filt[i], deltas_filt[i], label = ls[i], s=100)
plt.xlabel('Robustness, unperturbed hemisphere')
plt.ylabel('Behvioral recovery, (correct, control - photoinhib.)')
plt.legend()
# plt.savefig(r'F:\data\Fig 3\corr_behaviordiff_robustness_matched_updated.pdf')

#%%
fig = plt.figure(figsize=(7,6))

ls = ['Learning', 'Expert']
for i in range(2):
    plt.scatter(robustness_filt[i], deltas_filt[i], label = ls[i])
plt.xlabel('Robustness, unperturbed hemisphere')
plt.ylabel('Behvioral recovery, (correct, control - photoinhib.)')
plt.legend()

# plt.scatter(cat(all_proportions), cat(robustness))
# stats.pearsonr(cat(all_proportions), cat(robustness))

#%% Plot only first few
fig = plt.figure(figsize=(7,6))

ls = ['Learning', 'Expert']

for i in range(2):
    plt.scatter(all_mod[i][:], all_deltas[i][:], label = ls[i])

plt.xlabel('Robustness, unperturbed hemisphere')
plt.ylabel('Behvioral recovery, (correct, control - photoinhib.)')
# plt.ylabel('Behvioral recovery, (delta correct, control-photoinhib.)')
plt.legend()
plt.title('Pearsons correlation: {}, p-val: {}'.format(stats.pearsonr(cat(all_mod), cat(all_deltas))[0], 
                                                       stats.pearsonr(cat(all_mod), cat(all_deltas))[1]))

# plt.savefig(r'F:\data\Fig 3\corr_behaviordiff_modularity_matched_updated.pdf')
plt.show()
print(stats.pearsonr((all_mod[0]), (all_deltas[0]))[0], stats.pearsonr((all_mod[0]), (all_deltas[0]))[1])
print(stats.pearsonr((all_mod[1]), (all_deltas[1]))[0], stats.pearsonr((all_mod[1]), (all_deltas[1]))[1])

#%% Correlate behavior recovery with modularity ALL SESS
    
all_paths = [[    r'F:\data\BAYLORCW032\python\2023_10_05',
            r'F:\data\BAYLORCW034\python\2023_10_12',
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
            r'F:\data\BAYLORCW034\python\2023_10_22',
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
        r'H:\data\BAYLORCW046\python\2024_06_25',
        r'H:\data\BAYLORCW046\python\2024_06_24',



        ],
        [r'F:\data\BAYLORCW032\python\2023_10_24',
            r'F:\data\BAYLORCW034\python\2023_10_27',
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
            r'H:\data\BAYLORCW044\python\2024_06_20',
            
            r'H:\data\BAYLORCW046\python\2024_06_27',
            r'H:\data\BAYLORCW046\python\2024_06_26',
            r'H:\data\BAYLORCW046\python\2024_06_28',

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
# all_paths = [[    r'F:\data\BAYLORCW032\python\2023_10_05',
#             # r'F:\data\BAYLORCW034\python\2023_10_12',
#             r'F:\data\BAYLORCW036\python\2023_10_09',
#             r'F:\data\BAYLORCW035\python\2023_10_26',
#             r'F:\data\BAYLORCW037\python\2023_11_21',
            
#             r'H:\data\BAYLORCW044\python\2024_05_22',
#             r'H:\data\BAYLORCW044\python\2024_05_23',
            
#             r'H:\data\BAYLORCW046\python\2024_05_29',
#             r'H:\data\BAYLORCW046\python\2024_05_30',
#             r'H:\data\BAYLORCW046\python\2024_05_31',
#             ],

#              [r'F:\data\BAYLORCW032\python\2023_10_19',
#             # r'F:\data\BAYLORCW034\python\2023_10_22',
#             r'F:\data\BAYLORCW036\python\2023_10_19',
#             r'F:\data\BAYLORCW035\python\2023_12_07',
#             r'F:\data\BAYLORCW037\python\2023_12_08',
            
#             r'H:\data\BAYLORCW044\python\2024_06_06',
#             r'H:\data\BAYLORCW044\python\2024_06_04',

#             r'H:\data\BAYLORCW046\python\2024_06_07',
#             r'H:\data\BAYLORCW046\python\2024_06_10',
#             r'H:\data\BAYLORCW046\python\2024_06_11',
#             ],


#              [r'F:\data\BAYLORCW032\python\2023_10_24',
#             # r'F:\data\BAYLORCW034\python\2023_10_27',
#             r'F:\data\BAYLORCW036\python\2023_10_30',
#             r'F:\data\BAYLORCW035\python\2023_12_15',
#             r'F:\data\BAYLORCW037\python\2023_12_15',
            
#             r'H:\data\BAYLORCW044\python\2024_06_19',
#             r'H:\data\BAYLORCW044\python\2024_06_18',
            
#             r'H:\data\BAYLORCW046\python\2024_06_24',
#             r'H:\data\BAYLORCW046\python\2024_06_27',
#             r'H:\data\BAYLORCW046\python\2024_06_26',
            
#             ]]


all_deltas = []
all_mod = []
p=0.01

for paths in all_paths:

    deltas = []
    modularity = []

    for path in paths:

        l1 = session.Session(path)
        temp, _ = l1.modularity_proportion(p=p, method=1) # Robustness
        
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
plt.legend()
plt.title('Pearsons correlation: {}, p-val: {}'.format(stats.pearsonr(cat(all_mod), cat(all_deltas))[0], 
                                                       stats.pearsonr(cat(all_mod), cat(all_deltas))[1]))

# plt.savefig(r'F:\data\Fig 3\corr_modularity_robustness_ALL.pdf')
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

#%% Modularity vs sample amplitude in learning sessions

mod = []
seln_idx = []
sample_ampl = []
for path in allpaths[1]:

    l1 = Mode(path, use_reg = True, triple=True)
    m, _ = l1.modularity_proportion(p=0.01, period = range(l1.delay, l1.delay + int(1.5 * 1/l1.fs)))
    mod += [m]
    idx = [np.where(l1.good_neurons == n)[0][0] for n in l1.selective_neurons]
    seln_idx += [idx]
    
    orthonormal_basis, mean, db, acc_learning_sample = l1.decision_boundary(mode_input='stimulus', persistence=False)
    lea_sample = np.mean(acc_learning_sample)
    lea_sample = lea_sample if lea_sample > 0.5 else 1-lea_sample
    sample_ampl += [lea_sample]

f = plt.figure(figsize = (5,5))
plt.scatter(mod, sample_ampl, marker='x')
plt.xlabel('Modularity')
plt.ylabel('Sample amplitude')
print(scipy.stats.pearsonr(mod, sample_ampl))
#%% Robustness vs sample amplitude in learning sessions

mod = []
seln_idx = []
sample_ampl = []
for path in allpaths[1]:

    l1 = Mode(path, use_reg = True, triple=True)
    m, _ = l1.modularity_proportion(p=0.01)
    mod += [m]
    idx = [np.where(l1.good_neurons == n)[0][0] for n in l1.selective_neurons]
    seln_idx += [idx]
    
    orthonormal_basis, mean, db, acc_learning_sample = l1.decision_boundary(mode_input='stimulus', persistence=False)
    lea_sample = np.mean(acc_learning_sample)
    lea_sample = lea_sample if lea_sample > 0.5 else 1-lea_sample
    sample_ampl += [lea_sample]

f = plt.figure(figsize = (5,5))
plt.scatter(mod, sample_ampl, marker='x')
plt.xlabel('Robustness')
plt.ylabel('Sample amplitude')
print(scipy.stats.pearsonr(mod, sample_ampl))

#%% Modularity vs sample amplitude in all sessions

f = plt.figure(figsize = (5,5))
stages = ['Naive', 'Learning', 'Expert']

for i in range(3):
    mod = []
    seln_idx = []
    sample_ampl = []
    for path in allpaths[i]:
    
        l1 = Mode(path, use_reg = True, triple=True)
        m, _ = l1.modularity_proportion(p=0.01, period = range(l1.delay, l1.delay + int(1.5 * 1/l1.fs)))
        mod += [m]
        idx = [np.where(l1.good_neurons == n)[0][0] for n in l1.selective_neurons]
        seln_idx += [idx]
        
        orthonormal_basis, mean, db, acc_learning_sample = l1.decision_boundary(mode_input='stimulus', persistence=False)
        lea_sample = np.mean(acc_learning_sample)
        lea_sample = lea_sample if lea_sample > 0.5 else 1-lea_sample
        sample_ampl += [lea_sample]
    
    plt.scatter(mod, sample_ampl, marker='x', label=stages[i])
    print(scipy.stats.pearsonr(mod, sample_ampl))

plt.xlabel('Modularity')
plt.ylabel('Sample amplitude')


