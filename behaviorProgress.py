# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 15:29:46 2023

@author: Catherine Wang
"""

import sys
sys.path.append("C:\scripts\Imaging analysis")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import session
import behavior
cat = np.concatenate
plt.rcParams['pdf.fonttype'] = '42' 



#%% Plot behavior in opto trials over learning with left/right info

all_paths = [[r'F:\data\BAYLORCW032\python\2023_10_05',
          r'F:\data\BAYLORCW032\python\2023_10_19',
          r'F:\data\BAYLORCW032\python\2023_10_24',],


             [r'F:\data\BAYLORCW036\python\2023_10_09',
            r'F:\data\BAYLORCW036\python\2023_10_19',
            r'F:\data\BAYLORCW036\python\2023_10_30',],
             
             [ r'F:\data\BAYLORCW034\python\2023_10_12',
            r'F:\data\BAYLORCW034\python\2023_10_22',
            r'F:\data\BAYLORCW034\python\2023_10_27'],

        [r'F:\data\BAYLORCW035\python\2023_10_12',
            r'F:\data\BAYLORCW035\python\2023_10_26',
            r'F:\data\BAYLORCW035\python\2023_12_16',],

        [r'F:\data\BAYLORCW037\python\2023_11_21',
            r'F:\data\BAYLORCW037\python\2023_11_28',
            r'F:\data\BAYLORCW037\python\2023_12_15',],
        
         [r'H:\data\BAYLORCW044\python\2024_05_22',
          r'H:\data\BAYLORCW044\python\2024_06_06',
        r'H:\data\BAYLORCW044\python\2024_06_19'],
         
            [r'H:\data\BAYLORCW044\python\2024_05_23',
             r'H:\data\BAYLORCW044\python\2024_06_04',
        r'H:\data\BAYLORCW044\python\2024_06_18'],
            
        #     [r'H:\data\BAYLORCW044\python\2024_05_24',
        #      r'H:\data\BAYLORCW044\python\2024_06_05',
        # r'H:\data\BAYLORCW044\python\2024_06_20'],

            [r'H:\data\BAYLORCW046\python\2024_05_29',
             r'H:\data\BAYLORCW046\python\2024_06_24',
             r'H:\data\BAYLORCW046\python\2024_06_28'],


            [r'H:\data\BAYLORCW046\python\2024_05_30',
             r'H:\data\BAYLORCW046\python\2024_06_10',
             r'H:\data\BAYLORCW046\python\2024_06_27'],

            [r'H:\data\BAYLORCW046\python\2024_05_31',
             r'H:\data\BAYLORCW046\python\2024_06_11',
             r'H:\data\BAYLORCW046\python\2024_06_26']
            
            ]

performance_opto = []
performance_ctl = []
fig = plt.figure()
for paths in all_paths:
    counter = -1

    opt, ctl = [],[]
    for path in paths:
        counter += 1
        l1 = session.Session(path)
        stim_trials = np.where(l1.stim_ON)[0]
        control_trials = np.where(~l1.stim_ON)[0]
        
        perf_right, perf_left, perf_all = l1.performance_in_trials(stim_trials)
        opt += [perf_all]

        perf_rightctl, perf_leftctl, perf_all = l1.performance_in_trials(control_trials)
        ctl += [perf_all]
      
        plt.plot([counter - 0.2, counter + 0.2], [perf_all, perf_right], color='blue', alpha=0.3)
        plt.plot([counter - 0.2, counter + 0.2], [perf_all, perf_left], color='red', alpha=0.3)
        # plt.scatter(counter - 0.2, perf_rightctl, c='b', marker='o')
        # plt.scatter(counter - 0.2, perf_leftctl, c='r', marker='o')
        plt.scatter(counter - 0.2, perf_all, facecolors='white', edgecolors='black')
        plt.scatter(counter + 0.2, perf_right, c='b', marker='x')
        plt.scatter(counter + 0.2, perf_left, c='r', marker='x')
        
    performance_opto += [opt]
    performance_ctl += [ctl]

plt.scatter(counter + 0.2, perf_right, c='b', marker='x', label="Perturbation trials")
plt.scatter(counter - 0.2, perf_rightctl, c='b', marker='o', label="Control trials")

plt.bar(np.arange(3)+0.2, np.mean(performance_opto, axis=0), 0.4, fill=False)

plt.bar(np.arange(3)-0.2, np.mean(performance_ctl, axis=0), 0.4, fill=False)

plt.xticks(range(len(paths)), ["Naive", "Learning", "Expert"])
# plt.ylim([0.4,1])
plt.legend()
plt.savefig(r'F:\data\Fig 1\beh_opto_LR_updated_ctl.pdf')
plt.show()

#%% Plot behavior in opto trials over learning NO left/right info
all_paths = [[r'F:\data\BAYLORCW032\python\2023_10_05',
          r'F:\data\BAYLORCW032\python\2023_10_19',
          r'F:\data\BAYLORCW032\python\2023_10_24',],


             [r'F:\data\BAYLORCW036\python\2023_10_09',
            r'F:\data\BAYLORCW036\python\2023_10_19',
            r'F:\data\BAYLORCW036\python\2023_10_30',],
             
             [ r'F:\data\BAYLORCW034\python\2023_10_12',
            r'F:\data\BAYLORCW034\python\2023_10_22',
            r'F:\data\BAYLORCW034\python\2023_10_27'],

        [r'F:\data\BAYLORCW035\python\2023_10_12',
            r'F:\data\BAYLORCW035\python\2023_10_26',
            r'F:\data\BAYLORCW035\python\2023_12_16',],

        [r'F:\data\BAYLORCW037\python\2023_11_21',
            r'F:\data\BAYLORCW037\python\2023_11_28',
            r'F:\data\BAYLORCW037\python\2023_12_15',],
        
         [r'H:\data\BAYLORCW044\python\2024_05_22',
          r'H:\data\BAYLORCW044\python\2024_06_06',
        r'H:\data\BAYLORCW044\python\2024_06_19'],
         
            [r'H:\data\BAYLORCW044\python\2024_05_23',
             r'H:\data\BAYLORCW044\python\2024_06_04',
        r'H:\data\BAYLORCW044\python\2024_06_18'],
            
        #     [r'H:\data\BAYLORCW044\python\2024_05_24',
        #      r'H:\data\BAYLORCW044\python\2024_06_05',
        # r'H:\data\BAYLORCW044\python\2024_06_20'],

            [r'H:\data\BAYLORCW046\python\2024_05_29',
             r'H:\data\BAYLORCW046\python\2024_06_07',
             r'H:\data\BAYLORCW046\python\2024_06_28'],


            [r'H:\data\BAYLORCW046\python\2024_05_30',
             r'H:\data\BAYLORCW046\python\2024_06_10',
             r'H:\data\BAYLORCW046\python\2024_06_27'],

            [r'H:\data\BAYLORCW046\python\2024_05_31',
             r'H:\data\BAYLORCW046\python\2024_06_11',
             r'H:\data\BAYLORCW046\python\2024_06_26']
            
            ]

performance_opto = []
performance_ctl = []
fig = plt.figure()
for paths in all_paths:
    counter = -1

    opt, ctl = [],[]
    for path in paths:
        counter += 1
        l1 = session.Session(path)
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

plt.xticks(range(3), ["Naive", "Learning", "Expert"])
plt.axhline(0.5, ls='--')
plt.ylim([0.15,1])
plt.yticks(ticks=plt.yticks()[0][1:], labels=(100 * np.array(plt.yticks()[0][1:])).astype(int)) #Multiply all ticks by 100
plt.ylabel('Performance (%)')
# plt.legend()
plt.savefig(r'F:\data\Fig 1\updated_beh_opto.pdf')
plt.show()
    
#%% Plot the delta of behavior recovery from learning to expert
all_paths = [[r'F:\data\BAYLORCW032\python\2023_10_05',
          r'F:\data\BAYLORCW032\python\2023_10_19',
          r'F:\data\BAYLORCW032\python\2023_10_24',],


             [r'F:\data\BAYLORCW036\python\2023_10_09',
            r'F:\data\BAYLORCW036\python\2023_10_19',
            r'F:\data\BAYLORCW036\python\2023_10_30',],
             
            #  [ r'F:\data\BAYLORCW034\python\2023_10_12',
            # r'F:\data\BAYLORCW034\python\2023_10_22',
            # r'F:\data\BAYLORCW034\python\2023_10_27'],

        [r'F:\data\BAYLORCW035\python\2023_10_12',
            r'F:\data\BAYLORCW035\python\2023_10_26',
            r'F:\data\BAYLORCW035\python\2023_12_16',],

        [r'F:\data\BAYLORCW037\python\2023_11_21',
            r'F:\data\BAYLORCW037\python\2023_11_28',
            r'F:\data\BAYLORCW037\python\2023_12_15',],
        
        #  [r'H:\data\BAYLORCW044\python\2024_05_22',
        #   r'H:\data\BAYLORCW044\python\2024_06_06',
        # r'H:\data\BAYLORCW044\python\2024_06_19'],
         
            [r'H:\data\BAYLORCW044\python\2024_05_23',
             r'H:\data\BAYLORCW044\python\2024_06_04',
        r'H:\data\BAYLORCW044\python\2024_06_18'],
            
        #     [r'H:\data\BAYLORCW044\python\2024_05_24',
        #       r'H:\data\BAYLORCW044\python\2024_06_05',
        # r'H:\data\BAYLORCW044\python\2024_06_20'],

            # [r'H:\data\BAYLORCW046\python\2024_05_29',
            #  r'H:\data\BAYLORCW046\python\2024_06_24',
            #  r'H:\data\BAYLORCW046\python\2024_06_28'],


            [r'H:\data\BAYLORCW046\python\2024_05_30',
             r'H:\data\BAYLORCW046\python\2024_06_10',
             r'H:\data\BAYLORCW046\python\2024_06_27'],

            # [r'H:\data\BAYLORCW046\python\2024_05_31',
            #  r'H:\data\BAYLORCW046\python\2024_06_11',
            #  r'H:\data\BAYLORCW046\python\2024_06_26']
            
            ]
agg_paths = [[    r'F:\data\BAYLORCW032\python\2023_10_05',
            # r'F:\data\BAYLORCW034\python\2023_10_12',
            r'F:\data\BAYLORCW036\python\2023_10_09',
            r'F:\data\BAYLORCW035\python\2023_10_26',
            r'F:\data\BAYLORCW037\python\2023_11_21',
            
            # r'H:\data\BAYLORCW044\python\2024_05_22',
            r'H:\data\BAYLORCW044\python\2024_05_23',
            
            # r'H:\data\BAYLORCW046\python\2024_05_29',
            r'H:\data\BAYLORCW046\python\2024_05_30',
            # r'H:\data\BAYLORCW046\python\2024_05_31',
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
            # r'H:\data\BAYLORCW046\python\2024_06_10',
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
            # r'H:\data\BAYLORCW046\python\2024_06_27',
            r'H:\data\BAYLORCW046\python\2024_06_26',
            
            ]]

# agg_paths = [[    r'F:\data\BAYLORCW032\python\2023_10_05',
#             r'F:\data\BAYLORCW034\python\2023_10_12',
#             r'F:\data\BAYLORCW036\python\2023_10_09',
#             r'F:\data\BAYLORCW035\python\2023_10_26',
#             r'F:\data\BAYLORCW037\python\2023_11_21',
            
#             r'F:\data\BAYLORCW036\python\2023_10_16',
#             r'F:\data\BAYLORCW035\python\2023_10_12',
#         r'F:\data\BAYLORCW035\python\2023_11_02',

#         # r'H:\data\BAYLORCW044\python\2024_05_22',
#         r'H:\data\BAYLORCW044\python\2024_05_23',
#         # r'H:\data\BAYLORCW044\python\2024_05_24',
        
#         # r'H:\data\BAYLORCW046\python\2024_05_29',
#         r'H:\data\BAYLORCW046\python\2024_05_30',
#         # r'H:\data\BAYLORCW046\python\2024_05_31',
#             ],
#              [r'F:\data\BAYLORCW032\python\2023_10_19',
#             # r'F:\data\BAYLORCW034\python\2023_10_22',
#             r'F:\data\BAYLORCW036\python\2023_10_19',
#             r'F:\data\BAYLORCW035\python\2023_12_07',
#             r'F:\data\BAYLORCW037\python\2023_12_08',

#         r'F:\data\BAYLORCW032\python\2023_10_18',
#         r'F:\data\BAYLORCW035\python\2023_10_25',
#             r'F:\data\BAYLORCW035\python\2023_11_27',
#             r'F:\data\BAYLORCW035\python\2023_11_29',
#             r'F:\data\BAYLORCW037\python\2023_11_28',
            
#         # r'H:\data\BAYLORCW044\python\2024_06_06',
#         r'H:\data\BAYLORCW044\python\2024_06_04',
#         # r'H:\data\BAYLORCW044\python\2024_06_03',
#         # r'H:\data\BAYLORCW044\python\2024_06_12',

#         # r'H:\data\BAYLORCW046\python\2024_06_07',
#         r'H:\data\BAYLORCW046\python\2024_06_10',
#         # r'H:\data\BAYLORCW046\python\2024_06_11',
#         # r'H:\data\BAYLORCW046\python\2024_06_19',
#         # r'H:\data\BAYLORCW046\python\2024_06_25',
#         # r'H:\data\BAYLORCW046\python\2024_06_24',



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
            
#             # r'H:\data\BAYLORCW044\python\2024_06_19',
#             r'H:\data\BAYLORCW044\python\2024_06_18',
#             # r'H:\data\BAYLORCW044\python\2024_06_17',
#             # r'H:\data\BAYLORCW044\python\2024_06_20',
            
#             r'H:\data\BAYLORCW046\python\2024_06_27',
#             # r'H:\data\BAYLORCW046\python\2024_06_26',
#             # r'H:\data\BAYLORCW046\python\2024_06_28',

# ]]

all_deltas = []
all_proportions =[]
all_deltas_r, all_deltas_l = [], []
for paths in agg_paths[1:]: # per stage
    deltas = []
    dl, dr = [], []
    prop = []
    for path in paths: 

        l1 = session.Session(path, remove_consec_opto=True)
        stim_trials = np.where(l1.stim_ON)[0]
        control_trials = np.where(~l1.stim_ON)[0]
        
        stim_trials = [c for c in stim_trials if c in l1.i_good_trials]
        # stim_trials = [c for c in stim_trials if ~l1.early_lick[c]]
        control_trials = [c for c in control_trials if c in l1.i_good_trials]
        # control_trials = [c for c in control_trials if ~l1.early_lick[c]]
        
        perf_right, perf_left, perf_all = l1.performance_in_trials(stim_trials)
        perf_rightctl, perf_leftctl, perf_all_c = l1.performance_in_trials(control_trials)
        
        deltas += [perf_all_c - perf_all]
        prop += [perf_all/perf_all_c]
        dr += [perf_rightctl - perf_right]
        dl += [perf_leftctl - perf_left]
        
        
    all_deltas += [deltas]
    all_deltas_r += [dr]
    all_deltas_l += [dl]
    all_proportions += [prop]
    
    
#%% Plot combined deltas

# all_deltas = np.array(comb_deltas)
plt.bar([0,1], np.mean(all_deltas, axis=1))
plt.scatter(np.zeros(len(all_deltas[0])), all_deltas[0, :])
plt.scatter(np.ones(len(all_deltas[1])), all_deltas[1, :])
for i in range(len(all_deltas[0])):
    plt.plot([0,1], [all_deltas[0, i], all_deltas[1, i]], color='grey', alpha=0.5)
# plt.savefig(r'F:\data\Fig 1\updated_beh_opto_deltacomb.pdf')

#%% Deltas combo right and left
all_deltas = np.array(all_deltas)
all_deltas_r = np.array(all_deltas_r)
all_deltas_l = np.array(all_deltas_l)
comb_deltas = []
for j in range(2):
    all_deltas_new = []
    for i in range(all_deltas.shape[1]): # Each fov
        new_delta = -all_deltas_r[j, i]
        new_delta += all_deltas_l[j, i]
        all_deltas_new += [new_delta]
    comb_deltas += [all_deltas_new]
    
#%% Plot combined deltas

# all_deltas = np.array(comb_deltas)
comb_deltas = np.array(comb_deltas)
plt.bar([0,1], np.mean(comb_deltas, axis=1))
plt.scatter(np.zeros(len(comb_deltas[0])), comb_deltas[0, :])
plt.scatter(np.ones(len(comb_deltas[1])), comb_deltas[1, :])
for i in range(len(comb_deltas[0])):
    plt.plot([0,1], [comb_deltas[0, i], comb_deltas[1, i]], color='grey', alpha=0.5)
plt.savefig(r'F:\data\Fig 1\updated_beh_opto_deltacomb.pdf')

#%%
    
fig = plt.figure()

plt.bar(np.arange(3)+0.2, np.mean(performance_opto, axis=0), 0.4, fill=False)

plt.bar(np.arange(3)-0.2, np.mean(performance_ctl, axis=0), 0.4, fill=False)

plt.xticks(range(3), ["Naive", "Learning", "Expert"])
plt.axhline(0.5, ls='--')
plt.ylim([0.15,1])
plt.yticks(ticks=plt.yticks()[0][1:], labels=(100 * np.array(plt.yticks()[0][1:])).astype(int)) #Multiply all ticks by 100
plt.ylabel('Performance (%)')
# plt.legend()
plt.savefig(r'F:\data\Fig 1\updated_beh_opto.pdf')
plt.show()
    


#%% Plot learning progression

# b = behavior.Behavior('F:\data\Behavior data\BAYLORCW028\python_behavior', behavior_only=True)
# b.learning_progression(imaging=True)

# b = behavior.Behavior('F:\data\Behavior data\BAYLORCW032\python_behavior', behavior_only=True)
# # b.learning_progression(imaging=True)
# b.learning_progression(window=150, save =r'F:\data\Fig 1\CW32.pdf')

# b = behavior.Behavior('F:\data\Behavior data\BAYLORCW037\python_behavior', behavior_only=True)
# # b.learning_progression(imaging=True)
# b.learning_progression(window=200, save =r'F:\data\Fig 1\CW37.pdf')

b = behavior.Behavior('F:\data\Behavior data\BAYLORCW021\python_behavior', behavior_only=True)
b.learning_progression_no_EL(imaging=True)
b.learning_progression_no_EL(window = 150,save=True)

# b = behavior.Behavior('F:\data\Behavior data\BAYLORCW021\python_behavior', behavior_only=True)
# b.learning_progression(window = 100)
# b.learning_progression(window = 100, imaging=True)

# b = behavior.Behavior(r'H:\data\Behavior data\BAYLORCW038\python_behavior', behavior_only=True)
# # b.learning_progression(window = 150, include_delay=False, color_background=[2,3,4,5,6,7,8,9, 11, 12, 14,15,16,17,18, 19, 20, 21,22,23])
# b.plot_performance_over_sessions(all=True, color_background=[2,3,4,5,6,7,8,9, 11, 12, 14,15,16,17,18, 19, 20, 21,22,23])

# b = behavior.Behavior(r'H:\data\Behavior data\BAYLORCW039\python_behavior', behavior_only=True)
# b.learning_progression(window = 50, include_delay=False, color_background=[2,3,4,5,8,9,10,11,12,13,14,15,16,20,21,22,23,24])
# b.plot_performance_over_sessions(all=True, color_background=[2,3,4,5,8,9,10,11,12,13,14,15,16,20,21,22,23,24])


# b = behavior.Behavior(r'H:\data\Behavior data\BAYLORCW041\python_behavior', behavior_only=True)
# b.learning_progression(window = 75, include_delay=False, color_background=[3,4,5,6,7,8,9,13,14,15,16,17,18,19,20])
# b.plot_performance_over_sessions(all=True, color_background=[3,4,5,6,7,8,9,13,14,15,16,17,18,19,20])


# b = behavior.Behavior(r'H:\data\Behavior data\BAYLORCW043\python_behavior', behavior_only=True)
# b.learning_progression(window = 50, include_delay=False, color_background=[3,4,5,6,7,8,9,13,14,15,16,17,18])
# b.plot_performance_over_sessions(all=True, color_background=[3,4,5,6,7,8,9,13,14,15,16,17,18])

# b = behavior.Behavior(r'H:\data\Behavior data\BAYLORCW042\python_behavior', behavior_only=True)
# b.learning_progression(window = 50, include_delay=False, color_background=[2,3,4,5,6,7,8,10,11,12,14,15,16,17,18])
# b.plot_performance_over_sessions(all=True, color_background=[2,3,4,5,6,7,8,10,11,12,14,15,16,17,18])

# b = behavior.Behavior(r'H:\data\Behavior data\BAYLORCW044\python_behavior', behavior_only=True)
# b.learning_progression(window = 50)

# b = behavior.Behavior(r'H:\data\Behavior data\BAYLORCW046\python_behavior', behavior_only=True)
# b.learning_progression(window = 50, imaging=False)



#%% Compare learning curves ####

# window = 200
# b = behavior.Behavior('F:\data\Behavior data\BAYLORCW032\python_behavior', behavior_only=True)
# delays, acc_arr_32, numtrials_32 = b.learning_progression(return_results = True, window=window)


# b = behavior.Behavior('F:\data\Behavior data\BAYLORCW034\python_behavior', behavior_only=True)
# delays, acc_arr, numtrials = b.learning_progression(return_results = True, window=window)

# b = behavior.Behavior('F:\data\Behavior data\BAYLORCW036\python_behavior', behavior_only=True)
# delays, acc_arr_36, numtrials_36 = b.learning_progression(return_results = True, window=window)



# # New data 
# b = behavior.Behavior('F:\data\Behavior data\BAYLORCW035\python_behavior', behavior_only=True)
# delays, acc_arr_35, numtrials_35 = b.learning_progression(return_results = True, window=window)

# b = behavior.Behavior('F:\data\Behavior data\BAYLORCW037\python_behavior', behavior_only=True)
# delays, acc_arr_37, numtrials_37 = b.learning_progression(return_results = True, window=window)


# # Points
# plt.scatter(numtrials_32[4], (acc_arr_32[numtrials_32[0]:numtrials_32[4]] - acc_arr_32[numtrials_32[0]])[-1], marker='o', s=150, alpha = 0.5, color = 'r')
# plt.scatter(numtrials_32[10] - numtrials_32[6], (acc_arr_32[numtrials_32[6]:numtrials_32[10]] - acc_arr_32[numtrials_32[6]])[-1], marker='o', s=150, alpha = 0.5, color = 'r')
# plt.scatter(numtrials_32[6] - numtrials_32[4], (acc_arr_32[numtrials_32[4]:numtrials_32[6]] - acc_arr_32[numtrials_32[4]])[-1], marker='o', s=150, alpha = 0.5, color = 'g')
# plt.scatter(numtrials_32[12] - numtrials_32[10], (acc_arr_32[numtrials_32[10]:numtrials_32[12]] - acc_arr_32[numtrials_32[10]])[-1], marker='o', s=150, alpha = 0.5, color = 'g')


# plt.scatter(numtrials_36[2], (acc_arr_36[numtrials_36[0]:numtrials_36[2]] - acc_arr_36[numtrials_36[0]])[-1], marker='o', s=150, alpha = 0.5, color = 'r')
# plt.scatter(numtrials_36[5] - numtrials_36[2], (acc_arr_36[numtrials_36[2]:numtrials_36[5]] - acc_arr_36[numtrials_36[2]])[-1], marker='o', s=150, alpha = 0.5, color = 'g')

# plt.scatter(numtrials[2], (acc_arr[numtrials[0]:numtrials[2]] - acc_arr[numtrials[0]])[-1], marker='o', s=150, alpha = 0.5, color = 'r')
# plt.scatter(numtrials[6] - numtrials[4], (acc_arr[numtrials[4]:numtrials[6]] - acc_arr[numtrials[4]])[-1], marker='o', s=150, alpha = 0.5, color = 'r')
# plt.scatter(numtrials[4] - numtrials[2], (acc_arr[numtrials[2]:numtrials[4]] - acc_arr[numtrials[2]])[-1], marker='o', s=150, alpha = 0.5, color = 'g')
# plt.scatter(numtrials[8] - numtrials[6], (acc_arr[numtrials[6]:numtrials[8]] - acc_arr[numtrials[6]])[-1], marker='o', s=150, alpha = 0.5, color = 'g')

# # Lines
# plt.plot(acc_arr_32[numtrials_32[0]:numtrials_32[4]] - acc_arr_32[numtrials_32[0]], color='r')
# plt.plot(acc_arr_32[numtrials_32[6]:numtrials_32[10]] - acc_arr_32[numtrials_32[6]], color='r')
# plt.plot(acc_arr_32[numtrials_32[4]:numtrials_32[6]] - acc_arr_32[numtrials_32[4]], color='g')
# plt.plot(acc_arr_32[numtrials_32[10]:numtrials_32[12]] - acc_arr_32[numtrials_32[10]], color='g')

# plt.plot(acc_arr_36[numtrials_36[0]:numtrials_36[2]] - acc_arr_36[numtrials_36[0]], color='r')
# plt.plot(acc_arr_36[numtrials_36[2]:numtrials_36[5]] - acc_arr_36[numtrials_36[2]], color='g')

# plt.plot(acc_arr[numtrials[0]:numtrials[2]] - acc_arr[numtrials[0]], color='r')
# plt.plot(acc_arr[numtrials[4]:numtrials[6]] - acc_arr[numtrials[4]], color='r', label='Full delay')
# plt.plot(acc_arr[numtrials[2]:numtrials[4]] - acc_arr[numtrials[2]], color='g', label='Varied delay')
# plt.plot(acc_arr[numtrials[6]:numtrials[8]] - acc_arr[numtrials[6]], color='g')


# #lines
# plt.plot(acc_arr_35[numtrials_35[0]:numtrials_35[4]] - acc_arr_35[numtrials_35[0]], color='r')
# plt.plot(acc_arr_35[numtrials_35[-3]:] - acc_arr_35[numtrials_35[-3]], color='r')
# plt.plot(acc_arr_35[numtrials_35[4]:numtrials_35[7]] - acc_arr_35[numtrials_35[4]], color='g')
# plt.plot(acc_arr_35[numtrials_35[23]:numtrials_35[27]] - acc_arr_35[numtrials_35[23]], color='g')

# plt.plot(acc_arr_37[numtrials_37[0]:numtrials_37[5]] - acc_arr_37[numtrials_37[0]], color='r')
# plt.plot(acc_arr_37[numtrials_37[10]:numtrials_37[13]] - acc_arr_37[numtrials_37[10]], color='r')
# plt.plot(acc_arr_37[numtrials_37[5]:numtrials_37[10]] - acc_arr_37[numtrials_37[5]], color='g')
# plt.plot(acc_arr_37[numtrials_37[13]:numtrials_37[15]] - acc_arr_37[numtrials_37[13]], color='g')

# #points

# plt.scatter(numtrials_35[4], (acc_arr_35[numtrials_35[0]:numtrials_35[4]] - acc_arr_35[numtrials_35[0]])[-1], marker='o', s=150, alpha = 0.5, color = 'r')
# plt.scatter(len(acc_arr_35) - numtrials_35[-3], (acc_arr_35[numtrials_35[-3]:] - acc_arr_35[numtrials_35[-3]])[-1], marker='o', s=150, alpha = 0.5, color = 'r')
# plt.scatter(numtrials_35[7] - numtrials_35[4], (acc_arr_35[numtrials_35[4]:numtrials_35[7]] - acc_arr_35[numtrials_35[4]])[-1], marker='o', s=150, alpha = 0.5, color = 'g')
# plt.scatter(numtrials_35[27] - numtrials_35[23], (acc_arr_35[numtrials_35[23]:numtrials_35[27]] - acc_arr_35[numtrials_35[23]])[-1], marker='o', s=150, alpha = 0.5, color = 'g')

# plt.scatter(numtrials_37[5], (acc_arr_37[numtrials_37[0]:numtrials_37[5]] - acc_arr_37[numtrials_37[0]])[-1], marker='o', s=150, alpha = 0.5, color = 'r')
# plt.scatter(numtrials_37[13] - numtrials_37[10], (acc_arr_37[numtrials_37[10]:numtrials_37[13]] - acc_arr_37[numtrials_37[10]])[-1], marker='o', s=150, alpha = 0.5, color = 'r')
# plt.scatter(numtrials_37[10] - numtrials_37[5], (acc_arr_37[numtrials_37[5]:numtrials_37[10]] - acc_arr_37[numtrials_37[4]])[-1], marker='o', s=150, alpha = 0.5, color = 'g')
# plt.scatter(numtrials_37[15] - numtrials_37[13], (acc_arr_37[numtrials_37[13]:numtrials_37[15]] - acc_arr_37[numtrials_37[13]])[-1], marker='o', s=150, alpha = 0.5, color = 'g')


# plt.ylabel('Change in performance accuracy')
# plt.xlabel('Number of trials')

# plt.legend()
# plt.savefig(r'F:\data\Fig 1\comparedelayleearning.pdf',transparent=True)
# plt.show()


#%% Compare all learning curves together

f, axarr = plt.subplots(2, 1, sharex='col', figsize=(10,6))

all_acc = []
all_EL = []
for idx in [34, 32, 36, 37]:
    b = behavior.Behavior('F:\data\Behavior data\BAYLORCW0{}\python_behavior'.format(idx), behavior_only=True)
    earlylicksarr, correctarr, _ = b.get_acc_EL(window=100)
    axarr[0].plot(correctarr, 'g', alpha =0.75)        
    axarr[0].set_ylabel('% correct')
    axarr[0].axhline(y=0.7, alpha = 0.5, color='orange')
    axarr[0].axhline(y=0.5, alpha = 0.5, color='red', ls = '--')
    axarr[0].set_ylim(0, 1)
    
    # Early licking
    
    axarr[1].plot(earlylicksarr, 'b', alpha=0.75)        
    axarr[1].set_ylabel('% Early licks')
    axarr[1].set_xlabel('Trials')
plt.savefig(r'F:\data\Fig 1\alllearningcurves.pdf',transparent=True)
plt.show()

#%% Separate learning curves by naive, learning, expert AGG all mice

# f, axarr = plt.subplots(1, 3, sharex='col', figsize=(20,6))

all_acc = []
all_EL = []
mice_id = [32, 34, 36, 37, 35, 44, 46]
colors = ['red', 'orange', 'blue']
sess = [[(0,4),(0,2), (0,2),(0,5),(0,4), (0,4), (0,3)],
        [(4,11),(2,13),(2,15), (5,21), (4,40), (4, 18), (3,18)],
        [(11,14),(13,15), (15,17),(21,24), (40,43), (18,21), (18,23)]
        ]

length = [4, 35, 4]
name = ['naive', 'learning', 'expert']
for i in range(3):
    fig = plt.figure(figsize =(length[i], 12)) 

    for idx in range(len(mice_id)):
        if mice_id[idx] > 40:
            b = behavior.Behavior('H:\data\Behavior data\BAYLORCW0{}\python_behavior'.format(mice_id[idx]), 
                                  behavior_only=True)
        else:
            b = behavior.Behavior('F:\data\Behavior data\BAYLORCW0{}\python_behavior'.format(mice_id[idx]), 
                              behavior_only=True)
        _, correctarr, _ = b.get_acc_EL(window=150, sessions = sess[i][idx])
        
        plt.scatter(len(correctarr), correctarr[-1], marker='o', s=650, alpha = 0.85, color = colors[i])

        plt.plot(correctarr, 'g', alpha =0.75)        
        plt.ylabel('% correct')
        plt.axhline(y=0.7, alpha = 0.5, color='orange')
        plt.axhline(y=0.5, alpha = 0.5, color='red', ls = '--')
        plt.ylim(0.18, 0.99)
        
    plt.savefig(r'F:\data\Fig 1\updated_alllearningcurves_{}.pdf'.format(name[i]),transparent=True)

    plt.show()
        # axarr[i].plot(correctarr, 'g', alpha =0.75)        
        # axarr[i].set_ylabel('% correct')
        # axarr[i].axhline(y=0.7, alpha = 0.5, color='orange')
        # axarr[i].axhline(y=0.5, alpha = 0.5, color='red', ls = '--')
        # axarr[i].set_ylim(0.25, 0.92)
        
        # axarr[i].scatter(len(correctarr), correctarr[-1], marker='o', s=200, alpha = 0.5, color = 'g')
        
# plt.savefig(r'F:\data\Fig 1\alllearningcurves_sep.pdf',transparent=True)

plt.show()


#%% Plot session to match GLM HMM
# sessions = ['20230215', '20230322', '20230323',  '20230403', '20230406', '20230409', '20230411',
#             '20230413', '20230420', '20230421', '20230423', '20230424', '20230427',
#             '20230426', '20230503', '20230508', '20230509', '20230510', '20230511', '20230512',
#             '20230516', '20230517']

# b = behavior.Behavior('F:\data\Behavior data\BAYLORCW021\python_behavior', behavior_only=True, glmhmm=sessions)
# b.learning_progression()



#%% Plot over all imaging sessions

# b = behavior.Behavior('F:\data\BAYLORCW022\python')
# b.plot_LR_performance_over_sessions()
# b.plot_early_lick()

# b = behavior.Behavior('F:\data\BAYLORCW021\python')
# b.plot_LR_performance_over_sessions()
# b.plot_early_lick()

# b = behavior.Behavior('F:\data\BAYLORCW030\python')
# b.plot_LR_performance_over_sessions()
# b.plot_early_lick()

#%% Plot single session performance - diagnostic session

# b = behavior.Behavior('H:\\data\\BAYLORCW043\\python\\2024_06_13', single=True)
# b.plot_single_session(save=True)

b = behavior.Behavior('H:\\data\\BAYLORCW046\\python\\2024_06_10', single=True)
b.plot_single_session(save=True)

b = behavior.Behavior('H:\\data\\BAYLORCW044\\python\\2024_06_04', single=True)
b.plot_single_session(save=True)

# b = behavior.Behavior(r'F:\data\BAYLORCW036\python\2023_10_30', single=True)
# b.plot_single_session(save=True)


# b = behavior.Behavior('H:\\data\\BAYLORCW041\\python\\2024_05_15', single=True)
# b.plot_single_session(save=True)


# b = behavior.Behavior(r'F:\data\BAYLORCW022\python\2023_03_04', single=True)
# b.plot_single_session(save=True)

# b = behavior.Behavior(r'F:\data\BAYLORCW021\python\2023_05_17', single=True)
# b.plot_single_session_multidose(save=True)

# b = behavior.Behavior(r'F:\data\BAYLORCW027\python\2023_04_10', single=True)
# b.plot_single_session()

# b = behavior.Behavior(r'F:\data\BAYLORCW034\python\2023_10_11', single=True)
# b.plot_single_session()

# b = behavior.Behavior(r'F:\data\BAYLORCW021\python\2023_04_25', single=True)
# b.plot_single_session(save=True)

# b = behavior.Behavior(r'F:\data\BAYLORCW021\python\2023_05_23', single=True)
# b.plot_single_session_multidose(save=True)

# b = behavior.Behavior(r'F:\data\BAYLORCW021\python\2023_02_27', single=True)
# b.plot_single_session(save=True)

# b = behavior.Behavior(r'F:\data\BAYLORCW021\python\2023_03_03', single=True)
# b.plot_single_session(save=True)