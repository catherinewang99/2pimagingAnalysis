# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 12:59:23 2023

@author: Catherine Wang
"""


import sys
sys.path.append("C:\scripts\Imaging analysis\src")
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
#%% paths
all_matched_paths = [
    
            [r'F:\data\BAYLORCW032\python\2023_10_05',
              r'F:\data\BAYLORCW032\python\2023_10_19',
              r'F:\data\BAYLORCW032\python\2023_10_24',
          ],
         
            # [ r'F:\data\BAYLORCW034\python\2023_10_12',
            #    r'F:\data\BAYLORCW034\python\2023_10_22',
            #    r'F:\data\BAYLORCW034\python\2023_10_27',
            #    r'F:\data\BAYLORCW034\python\cellreg\layer{}\1012_1022_1027pairs_proc.npy'],
         
            [r'F:\data\BAYLORCW036\python\2023_10_09',
            r'F:\data\BAYLORCW036\python\2023_10_19',
            r'F:\data\BAYLORCW036\python\2023_10_30',
           ],
         
         [r'F:\data\BAYLORCW037\python\2023_11_21',
            r'F:\data\BAYLORCW037\python\2023_12_08',
            r'F:\data\BAYLORCW037\python\2023_12_15',],
         
         [r'F:\data\BAYLORCW035\python\2023_10_26',
            r'F:\data\BAYLORCW035\python\2023_12_07',
            r'F:\data\BAYLORCW035\python\2023_12_15',],
         
         [r'H:\data\BAYLORCW044\python\2024_05_22',
          r'H:\data\BAYLORCW044\python\2024_06_06',
        r'H:\data\BAYLORCW044\python\2024_06_19'],

         
            [r'H:\data\BAYLORCW044\python\2024_05_23',
             r'H:\data\BAYLORCW044\python\2024_06_04',
        r'H:\data\BAYLORCW044\python\2024_06_18'],

            [r'H:\data\BAYLORCW046\python\2024_05_29',
             r'H:\data\BAYLORCW046\python\2024_06_24',
             r'H:\data\BAYLORCW046\python\2024_06_28'],


            [r'H:\data\BAYLORCW046\python\2024_05_30',
             r'H:\data\BAYLORCW046\python\2024_06_10',
             r'H:\data\BAYLORCW046\python\2024_06_27'],

            [r'H:\data\BAYLORCW046\python\2024_05_31',
             r'H:\data\BAYLORCW046\python\2024_06_11',
             r'H:\data\BAYLORCW046\python\2024_06_26'
             ]
         
        ]

#%% Decoding and behavior correlation

allpaths = [[r'F:\data\BAYLORCW032\python\2023_10_08',
          r'F:\data\BAYLORCW032\python\2023_10_16',
          r'F:\data\BAYLORCW032\python\2023_10_25',
          r'F:\data\BAYLORCW032\python\cellreg\layer{}\1008_1016_1025pairs_proc.npy'],
         
         # [ r'F:\data\BAYLORCW034\python\2023_10_12',
         #    r'F:\data\BAYLORCW034\python\2023_10_22',
         #    r'F:\data\BAYLORCW034\python\2023_10_27',
         #    r'F:\data\BAYLORCW034\python\cellreg\layer{}\1012_1022_1027pairs_proc.npy'],
         
         [r'F:\data\BAYLORCW036\python\2023_10_09',
            r'F:\data\BAYLORCW036\python\2023_10_19',
            r'F:\data\BAYLORCW036\python\2023_10_30',
            r'F:\data\BAYLORCW036\python\cellreg\layer{}\1009_1019_1030pairs_proc.npy'],
         
         [r'F:\data\BAYLORCW037\python\2023_11_21',
                     r'F:\data\BAYLORCW037\python\2023_12_08',
                     r'F:\data\BAYLORCW037\python\2023_12_15',],
         
          [r'F:\data\BAYLORCW035\python\2023_10_26',
                      r'F:\data\BAYLORCW035\python\2023_12_07',
                      r'F:\data\BAYLORCW035\python\2023_12_15',]
        ]


allaccs = []
allbeh = []
for paths in allpaths:
    
    l1 = Mode(paths[2], use_reg=True, triple=True) #Expert
    orthonormal_basis, mean, db, acc_expert = l1.decision_boundary(mode_input='choice')
    _, _, exp_beh = l1.performance_in_trials(l1.i_good_non_stim_trials)
    exp = np.mean(acc_expert)
    exp = exp if exp > 0.5 else 1-exp
    
    l1 = Mode(paths[1], use_reg=True, triple=True) #Learning
    acc_learning = l1.decision_boundary_appliedCD('choice', orthonormal_basis, mean, db)
    _, _, lea_beh = l1.performance_in_trials(l1.i_good_non_stim_trials)
    lea = np.mean(acc_learning)
    lea = lea if lea > 0.5 else 1-lea
    
    l1 = Mode(paths[0], use_reg=True, triple=True) #Naive
    acc_naive = l1.decision_boundary_appliedCD('choice', orthonormal_basis, mean, db)
    _, _, nai_beh = l1.performance_in_trials(l1.i_good_non_stim_trials)
    nai = np.mean(acc_naive)
    nai = nai if nai > 0.5 else 1-nai
    
    allaccs += [[nai, lea, exp]]
    allbeh += [[nai_beh, lea_beh, exp_beh]]

    plt.scatter([nai_beh, lea_beh, exp_beh], [nai, lea, exp])
    plt.plot([nai_beh, lea_beh, exp_beh], [nai, lea, exp])
    
plt.xlabel('Behavioral performance')
plt.ylabel('Decoder accuracy')
plt.savefig('F:\data\Fig 2\CD_delay_behavior_corrAGG.pdf')

plt.show()

#%% Decoding analysis for all mice applied across training stages


modes = ['choice', 'action', 'stimulus']
pers=False
for i in range(3):
    mode_input = modes[i]
    allaccs = []
    counter = 1
    if i == 0:
        continue
    for paths in all_matched_paths:
        
        l1 = Mode(paths[1], use_reg=True, triple=True) #Learning
        orthonormal_basis, mean, db, acc_learning = l1.decision_boundary(mode_input=mode_input, persistence=pers)
    
        
        l1 = Mode(paths[2], use_reg=True, triple=True) #Expert
        acc_expert = l1.decision_boundary_appliedCD(mode_input, orthonormal_basis, mean, db, persistence=pers)
    
        l1 = Mode(paths[0], use_reg=True, triple=True) # Naive
        acc_naive = l1.decision_boundary_appliedCD(mode_input, orthonormal_basis, mean, db, persistence=pers)
        
        nai = np.mean(acc_naive)
        nai = nai if nai > 0.5 else 1-nai
        exp = np.mean(acc_expert)
        exp = exp if exp > 0.5 else 1-exp
        lea = np.mean(acc_learning)
        lea = lea if lea > 0.5 else 1-lea
        
        allaccs += [[nai, lea, exp]]
        # plt.scatter([0,1,2], [nai, lea, exp], label=counter)
        counter += 1
    
    allaccs = np.array(allaccs)
    
    fig = plt.figure(figsize=(5,5))
    
    plt.bar([0,1,2], np.mean(allaccs, axis=0))
    # plt.errorbar([0,1,2], np.mean(allaccs, axis=0),
    #              stats.sem(allaccs, axis=0),
    #              color = 'r')
    for i in range(len(allaccs)): # For each FOV
        plt.plot([0, 1], [allaccs[i,0], allaccs[i,1]], color='lightgrey')
        plt.plot([1, 2], [allaccs[i,1], allaccs[i,2]], color='lightgrey')
    for i in range(len(allaccs)): # For each FOV
        plt.scatter([0,1,2], allaccs[i], facecolors='white', edgecolors='black')

    plt.xticks([0,1,2], ['Naive', 'Learning', 'Expert'])
    plt.ylim(bottom=0.4, top =1)
    plt.axhline(0.5, ls='--', color='black')
    plt.title(mode_input)
    plt.savefig(r'F:\data\Fig 2\CD_{}_allAGG_decoding_learningctl_NLE.pdf'.format(mode_input))
    plt.show()
    
    print(stats.ttest_ind(np.array(allaccs)[:, 0], np.array(allaccs)[:, 1]))
    print(stats.ttest_ind(np.array(allaccs)[:, 2], np.array(allaccs)[:, 1]))

#%% Decoding analysis applied across training stages for choice CW37
paths =[r'F:\data\BAYLORCW037\python\2023_11_21',
            r'F:\data\BAYLORCW037\python\2023_12_08',
            r'F:\data\BAYLORCW037\python\2023_12_15',]

l1 = Mode(paths[2], use_reg=True, triple=True) # Expert
orthonormal_basis, mean, db, acc_expert = l1.decision_boundary(mode_input='choice')
print(np.mean(acc_expert))

l1 = Mode(paths[1], use_reg=True, triple=True) # Learning
acc_learning = l1.decision_boundary_appliedCD('choice', orthonormal_basis, mean, db)

l1 = Mode(paths[0], use_reg=True, triple=True) # Naive
acc_naive = l1.decision_boundary_appliedCD('choice', orthonormal_basis, mean, db)

plt.bar([0,1,2], [np.mean(acc_naive), 1 - np.mean(acc_learning), np.mean(acc_expert)])
plt.errorbar([0,1,2], [np.mean(acc_naive), 1-np.mean(acc_learning), np.mean(acc_expert)],
             [np.std(acc_naive)/np.sqrt(len(acc_naive)), 
              np.std(acc_learning)/np.sqrt(len(acc_learning)),
              np.std(acc_expert)/np.sqrt(len(acc_expert))],
             color = 'r')

plt.xticks([0,1,2], ['Naive', 'Learning', 'Expert'])
plt.ylim(bottom=0.4, top =1)
# plt.savefig('F:\data\Fig 2\CD_delay_decoding_NLE.pdf')
plt.show()
#%% Decoding analysis applied across training stages for action CW37
paths =[r'F:\data\BAYLORCW037\python\2023_11_21',
            r'F:\data\BAYLORCW037\python\2023_12_08',
            r'F:\data\BAYLORCW037\python\2023_12_15',]

l1 = Mode(paths[2], use_reg=True, triple=True) #Expert
orthonormal_basis, mean, db, acc_expert = l1.decision_boundary(mode_input='action')
print(np.mean(acc_expert))

l1 = Mode(paths[1], use_reg=True, triple=True) #learning
acc_learning = l1.decision_boundary_appliedCD('action', orthonormal_basis, mean, db)

l1 = Mode(paths[0], use_reg=True, triple=True) #naive
acc_naive = l1.decision_boundary_appliedCD('action', orthonormal_basis, mean, db)

plt.bar([0,1,2], [1-np.mean(acc_naive), 1 - np.mean(acc_learning), np.mean(acc_expert)])
plt.errorbar([0,1,2], [1-np.mean(acc_naive), 1-np.mean(acc_learning), np.mean(acc_expert)],
             [np.std(acc_naive)/np.sqrt(len(acc_naive)), 
              np.std(acc_learning)/np.sqrt(len(acc_learning)),
              np.std(acc_expert)/np.sqrt(len(acc_expert))],
             color = 'r')

plt.xticks([0,1,2], ['Naive', 'Learning', 'Expert'])
plt.ylim(bottom=0.5, top =1)
plt.savefig('F:\data\Fig 2\CD_action_decoding_NLE.pdf')
plt.show()
#%% Decoding analysis applied across two training stages SFN
save = 'F:\data\SFN 2023\CD_delay_expert_trainedexpert.pdf'

# DECODER ANALYSIS
path = r'F:\data\BAYLORCW032\python\2023_10_25'
# path = r'F:\data\BAYLORCW036\python\2023_10_28'

l1 = Mode(path, use_reg = True)
orthonormal_basis, mean, db, acc_within = l1.decision_boundary()
print(np.mean(acc_within))


path = r'F:\data\BAYLORCW032\python\2023_10_08'
l1 = Mode(path, use_reg = True)
acc_without = l1.decision_boundary_appliedCD(orthonormal_basis, mean, db)

plt.bar([0,1], [np.mean(acc_within), np.mean(acc_without)])
plt.errorbar([0,1], [np.mean(acc_within), np.mean(acc_without)],
             [np.std(acc_within)/np.sqrt(len(acc_within)), np.std(acc_without)/np.sqrt(len(acc_without))],
             color = 'r')
plt.xticks([0,1], ['Expert:Expert', 'Expert:Naive'])
plt.ylim(bottom=0.4, top =1)
plt.savefig( 'F:\data\SFN 2023\CD_delay_DB_trainedexpert.pdf')
plt.show()

path = r'F:\data\BAYLORCW032\python\2023_10_08'
# path = r'F:\data\BAYLORCW036\python\2023_10_28'

l1 = Mode(path, use_reg = True)
orthonormal_basis, mean, db, acc_within = l1.decision_boundary()
print(np.mean(acc_within))


path = r'F:\data\BAYLORCW032\python\2023_10_25'
l1 = Mode(path, use_reg = True)
acc_without = l1.decision_boundary_appliedCD(orthonormal_basis, mean, db)

plt.bar([0,1], [np.mean(acc_within), np.mean(acc_without)])
plt.errorbar([0,1], [np.mean(acc_within), np.mean(acc_without)],
             [np.std(acc_within)/np.sqrt(len(acc_within)), np.std(acc_without)/np.sqrt(len(acc_without))],
             color = 'r')
plt.xticks([0,1], ['Naive:Naive', 'Naive:Expert'])
plt.ylim(bottom=0.4, top =1)
plt.savefig( 'F:\data\SFN 2023\CD_delay_DB_trainednaive.pdf')

plt.show()
# Plot choice decoding in control vs opto trials


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
for path in paths:
    
    l1 = Mode(path, use_reg=True, triple=True)
    

    orthonormal_basis, mean, db, acc_within = l1.decision_boundary()
    print(np.mean(acc_within))
    
    
    
    plt.bar([0], [np.mean(acc_within)])
    plt.scatter(np.zeros(len(acc_within)), np.mean(acc_within,axis=1), color = 'r')
    # plt.errorbar([0], [np.mean(acc_within)],
    #              [np.std(acc_within)/np.sqrt(len(acc_within))],
    #              color = 'r')
    plt.xticks([0], ['Left ALM -> behavior'])
    plt.ylim(bottom=0.4, top =1)
    plt.show()
#%% UNMATCHED ACROSS PERTURBATION/CONTROL TRIALS
# from activityMode import Mode
paths = [r'F:\data\BAYLORCW032\python\2023_10_08',
          r'F:\data\BAYLORCW032\python\2023_10_16',
          r'F:\data\BAYLORCW032\python\2023_10_25',]

# perf = []
# for path in paths:

#     l1 = Mode(path)
#     # l1.plot_CD()
#     # d = l1.plot_performance_distfromCD()
#     _,_,_,db = l1.decision_boundary(opto=True)
#     perf += [np.mean(db)]

# plt.bar(range(3), perf)
# plt.ylim([0.5,1])
# plt.show()

# paths = [r'F:\data\BAYLORCW032\python\2023_10_05',
#           r'F:\data\BAYLORCW032\python\2023_10_19',
#           r'F:\data\BAYLORCW032\python\2023_10_24',]

perf, perfctl = [], []
for path in paths:

    l1 = Mode(path)
    # l1.plot_CD()
    # d = l1.plot_performance_distfromCD()
    _,_,_,dbctl = l1.decision_boundary(opto=False)
    _,_,_,db = l1.decision_boundary(opto=True)
    perf += [np.mean(db)]
    perfctl += [np.mean(dbctl)]
    
plt.bar(np.arange(3)+0.2, perf, 0.4, label="Perturbation trials")

plt.bar(np.arange(3)-0.2, perfctl, 0.4, label="Control trials")

plt.xticks(range(3), ["Naive", "learning", "Expert"])
plt.ylim([0.4,1])
plt.legend()
plt.show()

#%% matched across sessions
# from activityMode import Mode
# paths = [r'F:\data\BAYLORCW032\python\2023_10_08',
#           r'F:\data\BAYLORCW032\python\2023_10_16',
#           r'F:\data\BAYLORCW032\python\2023_10_25',]

paths =[r'F:\data\BAYLORCW037\python\2023_11_21',
            r'F:\data\BAYLORCW037\python\2023_12_08',
            r'F:\data\BAYLORCW037\python\2023_12_15',]
# perf = []
# for path in paths:

#     l1 = Mode(path)
#     # l1.plot_CD()
#     # d = l1.plot_performance_distfromCD()
#     _,_,_,db = l1.decision_boundary(opto=True)
#     perf += [np.mean(db)]

# plt.bar(range(3), perf)
# plt.ylim([0.5,1])
# plt.show()

# paths = [r'F:\data\BAYLORCW032\python\2023_10_05',
#           r'F:\data\BAYLORCW032\python\2023_10_19',
#           r'F:\data\BAYLORCW032\python\2023_10_24',]

perf, perfctl = [], []
for path in paths:

    l1 = Mode(path)
    # l1.plot_CD()
    # d = l1.plot_performance_distfromCD()
    _,_,_,dbctl = l1.decision_boundary()
    perfctl += [np.mean(dbctl)]
    
# plt.bar(np.arange(3)+0.2, perf, 0.4, label="Perturbation trials")

plt.bar(np.arange(3), perfctl, label="Control trials")

plt.xticks(range(3), ["Naive", "learning", "Expert"])
plt.ylim([0.4,1])
plt.legend()
plt.show()
