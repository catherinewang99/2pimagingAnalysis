# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 17:18:42 2024

@author: catherinewang
"""

import sys
sys.path.append("C:\scripts\Imaging analysis")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from alm_2p import session
from matplotlib.pyplot import figure
from scipy.stats import chisquare
import pandas as pd
from activityMode import Mode
from scipy import stats
cat=np.concatenate
from scipy.stats import pearsonr
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

all_paths = [[    
            r'F:\data\BAYLORCW032\python\2023_10_05',
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

            # [r'H:\data\BAYLORCW046\python\2024_05_29',
            #  r'H:\data\BAYLORCW046\python\2024_06_07',
            #  r'H:\data\BAYLORCW046\python\2024_06_24'],

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


#%% Number/proportion of susceptible cells over learning
# Classify by SDR buckets

all_naive_susc, all_lea_susc, all_exp_susc = [],[],[]
p_naive, p_learning, p_expert = [],[],[]
p_s=0.01
p=0.01

for paths in all_matched_paths: # For each mouse/FOV
    s1 = session.Session(paths[0], use_reg=True, triple=True) # Naive
    s2 = session.Session(paths[1], use_reg=True, triple=True) # Learning
    s3 = session.Session(paths[2], use_reg=True, triple=True) # Expert
    
    # p_s = p_s / len(s1.good_neurons)
    stim_period = range(s1.delay+int(0.5/s1.fs), s1.delay+int(1.3/s1.fs))
    sample_epoch = range(s1.sample, s1.delay)
    delay_epoch = range(s1.delay+int(1.5 * 1/s1.fs), s1.response)
    response_epoch = range(s1.response, s1.response + int(2*1/s1.fs))
    
    naive_susc_all, p_n = s1.susceptibility(period = stim_period, p=p_s, return_n=True)
    lea_susc_all, p_l = s2.susceptibility(period = stim_period, p=p_s, return_n=True)
    exp_susc_all, p_e = s3.susceptibility(period = stim_period, p=p_s, return_n=True)

    # Delay selective
    # bucket = sample_epoch
    # sample_nai, sample_lea, sample_exp = s1.get_epoch_selective(sample_epoch, p=p), s2.get_epoch_selective(sample_epoch, p=p), s3.get_epoch_selective(sample_epoch, p=p)
    # delay_nai, delay_lea, delay_exp = s1.get_epoch_selective(delay_epoch, p=p), s2.get_epoch_selective(delay_epoch, p=p), s3.get_epoch_selective(delay_epoch, p=p)
    # naive_delay = [n for n in sample_nai if n not in delay_nai]
    # learning_delay = [n for n in sample_lea if n not in delay_lea]
    # expert_delay = [n for n in sample_exp if n not in delay_exp]
    
    
    # Sample selective
    # sample_nai, sample_lea, sample_exp = s1.get_epoch_selective(sample_epoch, p=p), s2.get_epoch_selective(sample_epoch, p=p), s3.get_epoch_selective(sample_epoch, p=p)
    
    # naive_delay = [n for n in sample_nai]
    # learning_delay = [n for n in sample_lea]
    # expert_delay = [n for n in sample_exp]
    
    all_naive_susc += [len(naive_susc_all) / len(s1.good_neurons)]
    all_lea_susc += [len(lea_susc_all)  / len(s2.good_neurons)]
    all_exp_susc += [len(exp_susc_all) / len(s3.good_neurons)]
    # if len(s1.selective_neurons) == 0:
    #     all_naive_susc += [0]
    # else:
    #     all_naive_susc += [len(naive_susc) / len(naive_delay)]
        
    # all_lea_susc += [len(lea_susc)  / len(learning_delay)]
    # all_exp_susc += [len(exp_susc) / len(expert_delay)]
    
    p_naive += [p_n]
    p_learning += [p_l]
    p_expert += [p_e]


p_naive = [np.array(p) for p in p_naive] 
p_learning = [np.array(p) for p in p_learning] 
p_expert = [np.array(p) for p in p_expert] 

# PLot

f=plt.figure()

plt.bar(range(3), [np.mean(all_naive_susc), np.mean(all_lea_susc), np.mean(all_exp_susc)])


for i in range(len(all_exp_susc)):
    plt.plot([0,1], [all_naive_susc[i], all_lea_susc[i]], color='grey')
    plt.plot([1,2], [all_lea_susc[i], all_exp_susc[i]], color='grey')
plt.scatter(np.zeros(len(all_naive_susc)), all_naive_susc,facecolors='white', edgecolors='red')
plt.scatter(np.ones(len(all_lea_susc)), all_lea_susc,facecolors='white', edgecolors='yellow')
plt.scatter(np.ones(len(all_exp_susc))*2, all_exp_susc,facecolors='white', edgecolors='blue')
plt.ylabel('Proportion of susc neurons')
plt.xticks([0,1,2], ['Naive', 'Learning', 'Expert'])
plt.title('Proportion of susceptible neurons over learning')
plt.savefig(r'H:\COSYNE 2025\proportion_susceptible.pdf')
# plt.ylim(bottom=0.5)
plt.show()

#%% Look at distribution of p-values of suscp


# Plot scatter with error bars
plt.errorbar(np.zeros(len(p_naive)) + np.random.normal(0, 0.1, size=len(p_naive)), 
             [np.mean(p[:, 0]) for p in p_naive], 
             yerr=[stats.sem(p[:, 0]) for p in p_naive], 
             fmt='o', capsize=5, alpha=0.8, ecolor='red')

plt.errorbar(np.ones(len(p_learning)) + np.random.normal(0, 0.1, size=len(p_learning)), 
             [np.mean(p[:, 0]) for p in p_learning], 
             yerr=[stats.sem(p[:, 0]) for p in p_learning], 
             fmt='o', capsize=5, alpha=0.8, ecolor='red')

plt.errorbar(np.ones(len(p_expert)) + np.random.normal(1, 0.1, size=len(p_expert)), 
             [np.mean(p[:, 0]) for p in p_expert], 
             yerr=[stats.sem(p[:, 0]) for p in p_expert], 
             fmt='o', capsize=5, alpha=0.8, ecolor='red')
# Labels and title
plt.xticks([0,1,2], ['Naive', 'Learning', 'Expert'])
plt.ylabel("p-values")
plt.title("Left trial susceptibility p-values over learning")
# plt.legend()
plt.grid(True)

plt.show()

# Plot scatter with error bars
plt.errorbar(np.zeros(len(p_naive)) + np.random.normal(0, 0.1, size=len(p_naive)), 
             [np.mean(p[:, 1]) for p in p_naive], 
             yerr=[stats.sem(p[:, 1]) for p in p_naive], 
             fmt='o', capsize=5, alpha=0.8, ecolor='red')

plt.errorbar(np.ones(len(p_learning)) + np.random.normal(0, 0.1, size=len(p_learning)), 
             [np.mean(p[:, 1]) for p in p_learning], 
             yerr=[stats.sem(p[:, 1]) for p in p_learning], 
             fmt='o', capsize=5, alpha=0.8, ecolor='red')

plt.errorbar(np.ones(len(p_expert)) + np.random.normal(1, 0.1, size=len(p_expert)), 
             [np.mean(p[:, 1]) for p in p_expert], 
             yerr=[stats.sem(p[:, 1]) for p in p_expert], 
             fmt='o', capsize=5, alpha=0.8, ecolor='red')
# Labels and title
plt.xticks([0,1,2], ['Naive', 'Learning', 'Expert'])
plt.ylabel("p-values")
plt.title("Right trial susceptibility p-values over learning")
# plt.legend()
plt.grid(True)

plt.show()

# Plot scatter with error bars
plt.errorbar(np.zeros(len(p_naive)),# + np.random.normal(0, 0.1, size=len(p_naive)), 
             [np.mean(p) for p in p_naive], 
             yerr=[np.mean(stats.sem(p)) for p in p_naive], 
             fmt='o', capsize=5, alpha=0.8, ecolor='red')

plt.errorbar(np.ones(len(p_learning)),# + np.random.normal(0, 0.1, size=len(p_learning)), 
             [np.mean(p) for p in p_learning], 
             yerr=[np.mean(stats.sem(p)) for p in p_learning], 
             fmt='o', capsize=5, alpha=0.8, ecolor='red')

plt.errorbar(np.ones(len(p_expert)) * 2,# + np.random.normal(1, 0.1, size=len(p_expert)), 
             [np.mean(p) for p in p_expert], 
             yerr=[np.mean(stats.sem(p)) for p in p_expert], 
             fmt='o', capsize=5, alpha=0.8, ecolor='red')

for i in range(len(p_naive)):
    plt.plot([0,1], [np.mean(p_naive[i]), np.mean(p_learning[i])], color='grey')
    plt.plot([1,2], [np.mean(p_learning[i]), np.mean(p_expert[i])], color='grey')
# Labels and title
plt.xticks([0,1,2], ['Naive', 'Learning', 'Expert'])
plt.ylabel("p-values")
plt.title("All trial susceptibility p-values over learning")
# plt.legend()
plt.grid(True)

plt.show()

#  Plot as a histogram instead (not useful actually)
cat_p_naive = cat(p_naive)
cat_p_learning = cat(p_learning)
cat_p_expert = cat(p_expert)

plt.hist(cat_p_naive[:, 0], color='yellow', alpha = 0.25, label='Naive')
plt.hist(cat_p_learning[:, 0], color='green', alpha = 0.25, label='Learning')
plt.hist(cat_p_expert[:, 0], color='purple', alpha = 0.25, label='Expert')
plt.legend()
plt.xlabel('p-value')
plt.ylabel('Count')
plt.show() 

plt.violinplot(cat_p_naive[:, 0], [0])
plt.violinplot(cat_p_learning[:, 0], [1])
plt.violinplot(cat_p_expert[:, 0], [2])
plt.xticks([0,1,2], ['Naive', 'Learning', 'Expert'])
plt.ylabel("p-values")



#%% Baseline activity vs susceptibility analysis

# all_naive_susc, all_lea_susc, all_exp_susc = [],[],[]
baseline_naive, baseline_learning, baseline_expert = [],[],[]
p_naive, p_learning, p_expert = [],[],[]

p_s=0.01

for paths in all_matched_paths: # For each mouse/FOV
    s1 = session.Session(paths[0], use_reg=True, triple=True) # Naive
    s2 = session.Session(paths[1], use_reg=True, triple=True) # Learning
    s3 = session.Session(paths[2], use_reg=True, triple=True) # Expert
    
    # p_s = p_s / len(s1.good_neurons)
    stim_period = range(s1.delay+int(0.5/s1.fs), s1.delay+int(1.2/s1.fs))
    
    bn, p_n = s1.susceptibility(period = stim_period, p=p_s, baseline=True, all_n = False, exc_supr=True)
    bl, p_l = s2.susceptibility(period = stim_period, p=p_s, baseline=True, all_n = False, exc_supr=True)
    be, p_e = s3.susceptibility(period = stim_period, p=p_s, baseline=True, all_n = False, exc_supr=True)

    p_naive += [p_n]
    p_learning += [p_l]
    p_expert += [p_e]

    baseline_naive += [bn]
    baseline_learning += [bl]
    baseline_expert += [be]
    
p_naive = [np.array(p) for p in p_naive] 
p_learning = [np.array(p) for p in p_learning] 
p_expert = [np.array(p) for p in p_expert] 

baseline_naive = [np.array(p) for p in baseline_naive] 
baseline_learning = [np.array(p) for p in baseline_learning] 
baseline_expert = [np.array(p) for p in baseline_expert] 

# Plot r_values p values?
    
# Plot average baseline level over learning
# Plot scatter with error bars
plt.errorbar(np.zeros(len(baseline_naive)),# + np.random.normal(0, 0.1, size=len(p_naive)), 
             [np.mean(p) for p in baseline_naive], 
             yerr=[np.mean(stats.sem(p)) for p in baseline_naive], 
             fmt='o', capsize=5, alpha=0.8, ecolor='red')

plt.errorbar(np.ones(len(baseline_learning)),# + np.random.normal(0, 0.1, size=len(p_learning)), 
             [np.mean(p) for p in baseline_learning], 
             yerr=[np.mean(stats.sem(p)) for p in baseline_learning], 
             fmt='o', capsize=5, alpha=0.8, ecolor='red')

plt.errorbar(np.ones(len(baseline_expert)) * 2,# + np.random.normal(1, 0.1, size=len(p_expert)), 
             [np.mean(p) for p in baseline_expert], 
             yerr=[np.mean(stats.sem(p)) for p in baseline_expert], 
             fmt='o', capsize=5, alpha=0.8, ecolor='red')

for i in range(len(p_naive)):
    plt.plot([0,1], [np.mean(baseline_naive[i]), np.mean(baseline_learning[i])], color='grey')
    plt.plot([1,2], [np.mean(baseline_learning[i]), np.mean(baseline_expert[i])], color='grey')
# Labels and title
plt.xticks([0,1,2], ['Naive', 'Learning', 'Expert'])
plt.ylabel("Z-scored change in dF/F0")
plt.title("All neuron z-scored delay activity over learning")
# plt.legend()
plt.grid(True)

plt.show()


# Plot as a bar graph the z-scored delay activity for exc vs supressed

f = plt.figure()
allpos, allneg = [],[]
for i in range(len(baseline_naive)):

    positive_group = baseline_naive[i][p_naive[i] >= 0]
    negative_group = baseline_naive[i][p_naive[i] < 0]
    
    # plt.scatter(np.zeros(len(positive_group)), positive_group)
    # plt.scatter(np.ones(len(negative_group)), negative_group)
    
    allpos += [positive_group]
    allneg += [negative_group]
    
    # allpos += [np.mean(positive_group)]
    # allneg += [np.mean(negative_group)]

plt.violinplot([cat(allpos), cat(allneg)], [0,1])
# plt.bar([0,1], [np.nanmean(allpos), np.nanmean(allneg)])
plt.xticks([0,1], ['Excited', 'Suppressed'])
plt.ylabel('Z-scored delay activity')
plt.title('Naive stage')
plt.show()

f = plt.figure()
allpos, allneg = [],[]
for i in range(len(baseline_learning)):

    positive_group = baseline_learning[i][p_learning[i] >= 0]
    negative_group = baseline_learning[i][p_learning[i] < 0]
    
    # plt.scatter(np.zeros(len(positive_group)), positive_group)
    # plt.scatter(np.ones(len(negative_group)), negative_group)
    allpos += [positive_group]
    allneg += [negative_group]
    # allpos += [np.mean(positive_group)]
    # allneg += [np.mean(negative_group)]
plt.violinplot([cat(allpos), cat(allneg)], [0,1])
# plt.bar([0,1], [np.nanmean(allpos), np.nanmean(allneg)])
plt.xticks([0,1], ['Excited', 'Suppressed'])
plt.ylabel('Z-scored delay activity')
plt.title('Learning stage')
plt.show()

f = plt.figure()
allpos, allneg = [],[]
for i in range(len(baseline_expert)):

    positive_group = baseline_expert[i][p_expert[i] >= 0]
    negative_group = baseline_expert[i][p_expert[i] < 0]
    
    # plt.scatter(np.zeros(len(positive_group)), positive_group)
    # plt.scatter(np.ones(len(negative_group)), negative_group)
    allpos += [positive_group]
    allneg += [negative_group]
    # allpos += [np.mean(positive_group)]
    # allneg += [np.mean(negative_group)]
plt.violinplot([cat(allpos), cat(allneg)], [0,1])
# plt.bar([0,1], [np.nanmean(allpos), np.nanmean(allneg)])
plt.xticks([0,1], ['Excited', 'Suppressed'])
plt.ylabel('Z-scored delay activity')
plt.title('Expert stage')
plt.show()

# Plot as a scatter every neuron

side = 1 # 0 for left 1 for right
# plot the correlation of baseline vs susc p-value in neurons - should we only look at supressed neurons?
for i in range(len(baseline_naive)):
    plt.scatter(p_naive[i][:, side], (baseline_naive[i][:, side]))
    r_value, p_value = pearsonr(p_naive[i][:, side], (baseline_naive[i][:, side]))
    print(r_value, p_value)

plt.ylabel('Z-scored delay activity')
plt.xlabel('t-value')
allr, allp = pearsonr(cat([p[:,side] for p in p_naive]), cat([b[:, side] for b in baseline_naive]))
plt.title('Naive (R: {}, p: {})'.format(allr, allp))
plt.show()


for i in range(len(baseline_learning)):
    plt.scatter(p_learning[i][:, side], (baseline_learning[i][:, side]))
    r_value, p_value = pearsonr(p_learning[i][:, side], (baseline_learning[i][:, side]))
    print(r_value, p_value)
plt.ylabel('Z-scored delay activity')
plt.xlabel('t-value')
allr, allp = pearsonr(cat([p[:,side] for p in p_learning]), cat([b[:, side] for b in baseline_learning]))
plt.title('Learning (R: {}, p: {})'.format(allr, allp))
plt.show()

for i in range(len(baseline_expert)):
    plt.scatter(p_expert[i][:, side], (baseline_expert[i][:, side]))
    r_value, p_value = pearsonr(p_expert[i][:, side], (baseline_expert[i][:, side]))
    print(r_value, p_value)
plt.ylabel('Z-scored delay activity')
plt.xlabel('t-value')
allr, allp = pearsonr(cat([p[:,side] for p in p_expert]), cat([b[:, side] for b in baseline_expert]))
plt.title('Expert (R: {}, p: {})'.format(allr, allp))
plt.show()

#%% Trial Side of susc  analysis

# all_naive_susc, all_lea_susc, all_exp_susc = [],[],[]
baseline_naive, baseline_learning, baseline_expert = [],[],[]
p_naive, p_learning, p_expert = [],[],[]

p_s=0.01

for paths in all_matched_paths: # For each mouse/FOV
    s1 = session.Session(paths[0], use_reg=True, triple=True) # Naive
    s2 = session.Session(paths[1], use_reg=True, triple=True) # Learning
    s3 = session.Session(paths[2], use_reg=True, triple=True) # Expert
    
    # p_s = p_s / len(s1.good_neurons)
    stim_period = range(s1.delay+int(0.5/s1.fs), s1.delay+int(1.2/s1.fs))
    
    bn, p_n = s1.susceptibility(period = stim_period, p=p_s, baseline=False, all_n = True, side=True)
    bl, p_l = s2.susceptibility(period = stim_period, p=p_s, baseline=False, all_n = True, side=True)
    be, p_e = s3.susceptibility(period = stim_period, p=p_s, baseline=False, all_n = True, side=True)

    p_naive += [p_n]
    p_learning += [p_l]
    p_expert += [p_e]

    baseline_naive += [bn]
    baseline_learning += [bl]
    baseline_expert += [be]
    
p_naive = [np.array(p) for p in p_naive] 
p_learning = [np.array(p) for p in p_learning] 
p_expert = [np.array(p) for p in p_expert] 

baseline_naive = [np.array(p) for p in baseline_naive] 
baseline_learning = [np.array(p) for p in baseline_learning] 
baseline_expert = [np.array(p) for p in baseline_expert] 

# Plot the breakdown 
proportion_both = [sum(b == 2)/len(b) for b in baseline_naive]
proportion_left = [sum(b == 0)/len(b) for b in baseline_naive]
proportion_right = [sum(b == 1)/len(b) for b in baseline_naive]

f, ax=plt.subplots(1,3, figsize=(10,5), sharey='row')

ax[0].bar([0,1,2], [np.mean(proportion_left), np.mean(proportion_right), np.mean(proportion_both)])
ax[0].scatter(np.zeros(len(proportion_left)), proportion_left)
ax[0].scatter(np.ones(len(proportion_right)), proportion_right)
ax[0].scatter(np.ones(len(proportion_both)) * 2, proportion_both)
ax[0].set_xticks([0,1,2], ['Left', 'Right', 'Both'])
ax[0].set_ylabel('Proportion of all neurons')

proportion_both = [sum(b == 2)/len(b) for b in baseline_learning]
proportion_left = [sum(b == 0)/len(b) for b in baseline_learning]
proportion_right = [sum(b == 1)/len(b) for b in baseline_learning]

ax[1].bar([0,1,2], [np.mean(proportion_left), np.mean(proportion_right), np.mean(proportion_both)])
ax[1].scatter(np.zeros(len(proportion_left)), proportion_left)
ax[1].scatter(np.ones(len(proportion_right)), proportion_right)
ax[1].scatter(np.ones(len(proportion_both)) * 2, proportion_both)
ax[1].set_xticks([0,1,2], ['Left', 'Right', 'Both'])

proportion_both = [sum(b == 2)/len(b) for b in baseline_expert]
proportion_left = [sum(b == 0)/len(b) for b in baseline_expert]
proportion_right = [sum(b == 1)/len(b) for b in baseline_expert]

ax[2].bar([0,1,2], [np.mean(proportion_left), np.mean(proportion_right), np.mean(proportion_both)])
ax[2].scatter(np.zeros(len(proportion_left)), proportion_left)
ax[2].scatter(np.ones(len(proportion_right)), proportion_right)
ax[2].scatter(np.ones(len(proportion_both)) * 2, proportion_both)
ax[2].set_xticks([0,1,2], ['Left', 'Right', 'Both'])
ax[0].set_xlabel('Trial type susceptible')

plt.suptitle('Which trial type is susceptible?')

#%% Sannkey of sig susceptibly cells - all neurons

p_s=0.01
p=0.01
retained_sample = []
recruited_sample = []
retained_delay = []
recruited_delay = []
dropped_delay = []
dropped_sample = []
alls1list, alld1, allr1, allns1 = [],[],[],[] # s1: susc ns: non susc

learning_SDR = []
expert_SDR = []

for paths in all_matched_paths: # For each mouse/FOV
    ret_s = []
    recr_s = []
    ret_d, recr_d = [],[]
    drop_d, drop_s = [], []
    
    s1list, d1, r1, ns1 = np.zeros(4),np.zeros(4),np.zeros(4),np.zeros(4)

    s1 = session.Session(paths[1], use_reg=True, triple=True, use_background_sub=False) # Learning
    stim_period = range(s1.delay+int(0.5/s1.fs), s1.delay+int(1.2/s1.fs))

    naive_sample_sel, _ = s1.susceptibility(period = stim_period, p=p_s, return_n=True)

    # Get functional group info
    sample_epoch = range(s1.sample, s1.delay)
    delay_epoch = range(s1.delay+int(1.5 * 1/s1.fs), s1.response)
    response_epoch = range(s1.response, s1.response + int(2*1/s1.fs))
    
    
    naive_sample_sel_mod = s1.get_epoch_selective(sample_epoch, p=p)
    naive_sample_sel_mod = [n for n in naive_sample_sel_mod if n in naive_sample_sel]
    
    naive_delay_sel = s1.get_epoch_selective(delay_epoch, p=p)
    naive_delay_sel = [n for n in naive_delay_sel if n not in naive_sample_sel_mod and n in naive_sample_sel]
    
    naive_response_sel = s1.get_epoch_selective(response_epoch, p=p)
    naive_response_sel = [n for n in naive_response_sel if n not in naive_sample_sel_mod and n not in naive_delay_sel and n in naive_sample_sel]

    naive_nonsel_mod = [n for n in s1.good_neurons if n not in naive_sample_sel_mod and n not in naive_delay_sel and n not in naive_response_sel and n in naive_sample_sel]
    
    learning_SDR += [[len(naive_sample_sel_mod), len(naive_delay_sel), len(naive_response_sel), len(naive_nonsel_mod)]]
    
    
    naive_nonsel = [n for n in s1.good_neurons if n not in naive_sample_sel]

    s2 = session.Session(paths[2], use_reg=True, triple=True) # Expert
    exp_susc, _ = s2.susceptibility(period = stim_period, p=p_s, return_n=True)
    
    # Get functional group info
    
    naive_sample_sel_mod = s2.get_epoch_selective(sample_epoch, p=p)
    naive_sample_sel_mod = [n for n in naive_sample_sel_mod if n in exp_susc]
    
    naive_delay_sel = s2.get_epoch_selective(delay_epoch, p=p)
    naive_delay_sel = [n for n in naive_delay_sel if n not in naive_sample_sel_mod and n in exp_susc]
    
    naive_response_sel = s2.get_epoch_selective(response_epoch, p=p)
    naive_response_sel = [n for n in naive_response_sel if n not in naive_sample_sel_mod and n not in naive_delay_sel and n in exp_susc]

    naive_nonsel_mod = [n for n in s2.good_neurons if n not in naive_sample_sel_mod and n not in naive_delay_sel and n not in naive_response_sel and n in exp_susc]
    
    expert_SDR += [[len(naive_sample_sel_mod), len(naive_delay_sel), len(naive_response_sel), len(naive_nonsel_mod)]]
    
    
    
    for n in naive_sample_sel:
        if s2.good_neurons[np.where(s1.good_neurons == n)[0][0]] in exp_susc:
            s1list[0] += 1
            ret_s += [(n, s2.good_neurons[np.where(s1.good_neurons == n)[0][0]])]

        else:
            s1list[3] += 1
            drop_s += [(n, s2.good_neurons[np.where(s1.good_neurons == n)[0][0]])]

    
    for n in naive_nonsel:
        if s2.good_neurons[np.where(s1.good_neurons == n)[0][0]] in exp_susc:
            ns1[0] += 1
            recr_s += [(n, s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]])]

        else:
            ns1[3] += 1
    print(s1list)

    s1list, d1, r1, ns1 = s1list / len(s1.good_neurons), d1 / len(s1.good_neurons), r1 / len(s1.good_neurons), ns1 / len(s1.good_neurons)

    alls1list += [s1list]
    alld1 += [d1]
    allr1 += [r1] 
    allns1 += [ns1]
    
    retained_sample += [ret_s]
    recruited_sample += [recr_s]
    dropped_sample += [drop_s]
    retained_delay += [ret_d]
    recruited_delay += [recr_d]
    dropped_delay += [drop_d]

alls1list = np.mean(alls1list, axis=0) 
alld1 = np.mean(alld1, axis=0)
allr1 = np.mean(allr1, axis=0)
allns1 = np.mean(allns1, axis=0)

# SDR proportion of susceptible population

learning_SDR = np.array(learning_SDR) # S, D, R, NS
expert_SDR = np.array(expert_SDR)

f, ax = plt.subplots(1,2, figsize=(9,5), sharey='row')

ax[0].bar(range(4), np.sum(learning_SDR, axis=0))
ax[0].set_xticks(range(4), ['Sample', 'Delay', 'Response', 'N.S.'])
ax[0].set_ylabel('Number of neurons')
ax[0].set_title('Learning stage susceptible neurons')

ax[1].bar(range(4), np.sum(expert_SDR, axis=0))
ax[1].set_xticks(range(4), ['Sample', 'Delay', 'Response', 'N.S.'])
ax[1].set_title('Expert stage susceptible neurons')

# Plot as stacked instead
sum_learning_SDR = np.sum(learning_SDR, axis=0)
sum_expert_SDR = np.sum(expert_SDR, axis=0)
f = plt.figure(figsize=(8,8))
labels = ['Sample', 'Delay', 'Response', 'N.S.']
bottom_exp, bottom_lea = 0,0
for i in range(4):
    plt.bar(range(2), [sum_learning_SDR[i], sum_expert_SDR[i]], label=labels[i], bottom = [bottom_lea, bottom_exp])
    bottom_lea += sum_learning_SDR[i]
    bottom_exp += sum_expert_SDR[i]
plt.legend()
plt.ylabel('Number of neurons')
plt.xticks([0,1], ['Learning', 'Expert'])

#%% Plot and save some susceptible neurons
        #     [r'H:\data\BAYLORCW044\python\2024_05_23',
        #      r'H:\data\BAYLORCW044\python\2024_06_04',
        # r'H:\data\BAYLORCW044\python\2024_06_18'],

            # [r'H:\data\BAYLORCW046\python\2024_05_29',
            #  r'H:\data\BAYLORCW046\python\2024_06_07',
            #  r'H:\data\BAYLORCW046\python\2024_06_24'],

            # [r'H:\data\BAYLORCW046\python\2024_05_29',
            #  r'H:\data\BAYLORCW046\python\2024_06_24',
            #  r'H:\data\BAYLORCW046\python\2024_06_28'],
_,    learningpath,  expertpath =                [r'H:\data\BAYLORCW046\python\2024_05_29',
             r'H:\data\BAYLORCW046\python\2024_06_24',
             r'H:\data\BAYLORCW046\python\2024_06_28']
counter = 0
# plt.savefig(r'H:\COSYNE 2025\proportion_susceptible.pdf')

for paths in all_matched_paths: # For each mouse/FOV
    nlist = dropped_sample[counter]
    # learningpath = r'H:\data\BAYLORCW044\python\2024_06_06'
    s1 = session.Session(paths[1], use_reg=True, triple=True, use_background_sub=False) # Learning
    # expertpath = r'H:\data\BAYLORCW044\python\2024_06_19'
    s2 = session.Session(paths[2], use_reg=True, triple=True, use_background_sub=False) # Learning
    for n in nlist:
        s1.plot_rasterPSTH_sidebyside(n[0])#, fixaxis=(-0.1,2.2))
        s2.plot_rasterPSTH_sidebyside(n[1])#, fixaxis=(-0.1,2.2))

    counter += 1
#%% Sankey of sig susc cells - delay neurons only (lea or exp)
p_s=0.05
p=0.05
retained_sample = []
recruited_sample = []
retained_delay = []
recruited_delay = []
dropped_delay = []
dropped_sample = []
alls1list, alld1, allr1, allns1 = [],[],[],[] # s1: susc ns: non susc

learning_SDR = []
expert_SDR = []

for paths in all_matched_paths: # For each mouse/FOV
    ret_s = []
    recr_s = []
    ret_d, recr_d = [],[]
    drop_d, drop_s = [], []
    
    s1list, d1, r1, ns1 = np.zeros(4),np.zeros(4),np.zeros(4),np.zeros(4)

    s1 = session.Session(paths[1], use_reg=True, triple=True, use_background_sub=False) # Learning
    stim_period = range(s1.delay+int(0.5/s1.fs), s1.delay+int(1.2/s1.fs))

    naive_sample_sel,_ = s1.susceptibility(period = stim_period, p=p_s, return_n=True)

    
    # Get functional group info
    # sample_epoch = range(s1.sample, s1.delay)
    delay_epoch = range(s1.delay+int(1.5 * 1/s1.fs), s1.response)
    # response_epoch = range(s1.response, s1.response + int(2*1/s1.fs))
    
    naive_delay_sel = s1.get_epoch_selective(delay_epoch, p=p) # Learning stage delay neurons
    


    s2 = session.Session(paths[2], use_reg=True, triple=True) # Expert
    exp_susc,_ = s2.susceptibility(period = stim_period, p=p_s, return_n=True)
    exp_delay_sel = s2.get_epoch_selective(delay_epoch, p=p)
    
    naive_sample_sel = [n for n in naive_sample_sel if n in naive_delay_sel or s2.good_neurons[np.where(s1.good_neurons == n)[0][0]] in exp_delay_sel]
    naive_nonsel = [n for n in s1.good_neurons if n not in naive_sample_sel and (n in naive_delay_sel or s2.good_neurons[np.where(s1.good_neurons == n)[0][0]] in exp_delay_sel)]
    
    exp_susc = [n for n in exp_susc if n in exp_delay_sel or s1.good_neurons[np.where(s2.good_neurons == n)[0][0]] in naive_delay_sel]
    
    num_delay_neurons = sum([len(exp_delay_sel), len(naive_delay_sel)])
    
    for n in naive_sample_sel:
        if s2.good_neurons[np.where(s1.good_neurons == n)[0][0]] in exp_susc:
            s1list[0] += 1
            ret_s += [(n, s2.good_neurons[np.where(s1.good_neurons == n)[0][0]])]
        # elif s2.is_selective(s2.good_neurons[np.where(s1.good_neurons == n)[0][0]], delay_epoch, p=p):
        #     s1list[1] += 1
        # elif s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], response_epoch, p=p):
        #     s1list[2] += 1
        else:
            s1list[3] += 1
            drop_s += [(n, s2.good_neurons[np.where(s1.good_neurons == n)[0][0]])]

    # for n in naive_delay_sel:
    #     if s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], sample_epoch, p=p):
    #         d1[0] += 1
    #     elif s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], delay_epoch, p=p):
    #         d1[1] += 1
    #         ret_d += [(n, s2.good_neurons[np.where(s1.good_neurons == n)[0][0]])]

    #     elif s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], response_epoch, p=p):
    #         d1[2] += 1
    #     else:
    #         d1[3] += 1
    #         drop_d += [(n, s2.good_neurons[np.where(s1.good_neurons == n)[0][0]])]
    
    
    # for n in naive_response_sel:
    #     if s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], sample_epoch, p=p):
    #         r1[0] += 1

    #     elif s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], delay_epoch, p=p):
    #         r1[1] += 1

    #     elif s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], response_epoch, p=p):
    #         r1[2] += 1
    #     else:
    #         r1[3] += 1
    
    
    for n in naive_nonsel:
        if s2.good_neurons[np.where(s1.good_neurons == n)[0][0]] in exp_susc:
            ns1[0] += 1
            recr_s += [(n, s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]])]
        # elif s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], delay_epoch, p=p):
        #     ns1[1] += 1
        #     recr_d += [(n, s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]])]

        # elif s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], response_epoch, p=p):
        #     ns1[2] += 1
        else:
            ns1[3] += 1
    print(s1list)

    # s1list, d1, r1, ns1 = s1list / len(s1.good_neurons), d1 / len(s1.good_neurons), r1 / len(s1.good_neurons), ns1 / len(s1.good_neurons)
    s1list, d1, r1, ns1 = s1list / num_delay_neurons, d1 / num_delay_neurons, r1 / num_delay_neurons, ns1 / num_delay_neurons

    alls1list += [s1list]
    alld1 += [d1]
    allr1 += [r1] 
    allns1 += [ns1]
    
    retained_sample += [ret_s]
    recruited_sample += [recr_s]
    dropped_sample += [drop_s]
    retained_delay += [ret_d]
    recruited_delay += [recr_d]
    dropped_delay += [drop_d]

alls1list = np.mean(alls1list, axis=0) 
alld1 = np.mean(alld1, axis=0)
allr1 = np.mean(allr1, axis=0)
allns1 = np.mean(allns1, axis=0)


#%% Compare variability of opto response within session vs across sessions FIXME 3/10

# compare just opto trial stim period activity
s1 = session.Session(paths[1], use_reg=True, triple=True, use_background_sub=False) # Learning
idx = 6
n = s1.good_neurons[idx]
stim_period = range(s1.delay+int(0.5/s1.fs), s1.delay+int(1.3/s1.fs))

good_non_stim_trials_set = set(s1.i_good_non_stim_trials)
good_stim_trials_set = set(s1.i_good_stim_trials)
    
control_trials_left = np.random.permutation([t for t in s1.L_trials if t in good_non_stim_trials_set])
pert_trials_left = np.random.permutation([t for t in s1.L_trials if t in good_stim_trials_set])

control_trials_right = np.random.permutation([t for t in s1.R_trials if t in good_non_stim_trials_set])
pert_trials_right = np.random.permutation([t for t in s1.R_trials if t in good_stim_trials_set])

L_trials_ctl_first, L_trials_ctl_second = control_trials_left[:int(len(control_trials_left) / 2)], control_trials_left[int(len(control_trials_left) / 2):]
L_trials_opto_first, L_trials_opto_second = pert_trials_left[:int(len(pert_trials_left) / 2)], pert_trials_left[int(len(pert_trials_left) / 2):]

R_trials_ctl_first, R_trials_ctl_second = control_trials_right[:int(len(control_trials_right) / 2)], control_trials_right[int(len(control_trials_right) / 2):]
R_trials_opto_first, R_trials_opto_second = pert_trials_right[:int(len(pert_trials_right) / 2)], pert_trials_right[int(len(pert_trials_right) / 2):]

first_half = cat((L_trials_opto_first, R_trials_opto_first))
second_half = cat((L_trials_opto_second, R_trials_opto_second))

within_sess_first = np.array([s1.dff[0, t][n, stim_period] for t in first_half])
within_sess_second = np.array([s1.dff[0, t][n, stim_period] for t in second_half])


s2 = session.Session(paths[2], use_reg=True, triple=True) # Expert
n = s2.good_neurons[idx]

good_stim_trials_set = set(s2.i_good_stim_trials)
across_session = np.array([s2.dff[0, t][n, stim_period] for t in good_stim_trials_set])

F = f_oneway(np.mean(within_sess_first,axis=0), 
             np.mean(within_sess_second,axis=0), 
             np.mean(across_session,axis=0))

if F.pvalue < 0.01:

    # Combine data into one array
    data = np.concatenate([np.mean(within_sess_first,axis=0), 
                             np.mean(within_sess_second,axis=0), 
                             np.mean(across_session,axis=0)])
    
    # Create group labels
    labels = (["Group 1"] * within_sess_first.shape[1]) + (["Group 2"] * within_sess_second.shape[1]) + (["Group 3"] * across_session.shape[1])
    
    tukey = pairwise_tukeyhsd(data, labels, alpha=0.05)
    
    print(tukey)
    # compare control activity - opto activity variance during stim period





#%% Sannkey of sig susceptibly cells WITHIN SESSION CONTROL

p_s=0.01
p=0.01
retained_sample = []
recruited_sample = []
retained_delay = []
recruited_delay = []
dropped_delay = []
dropped_sample = []
alls1list, alld1, allr1, allns1 = [],[],[],[] # s1: susc ns: non susc

learning_SDR = []
expert_SDR = []

for paths in all_matched_paths: # For each mouse/FOV
    ret_s = []
    recr_s = []
    ret_d, recr_d = [],[]
    drop_d, drop_s = [], []
    
    s1list, d1, r1, ns1 = np.zeros(4),np.zeros(4),np.zeros(4),np.zeros(4)

    s1 = session.Session(paths[1], use_reg=True, triple=True) # Learning
    stim_period = range(s1.delay+int(0.5/s1.fs), s1.delay+int(1.3/s1.fs))
    
    good_non_stim_trials_set = set(s1.i_good_non_stim_trials)
    good_stim_trials_set = set(s1.i_good_stim_trials)
        
    control_trials_left = np.random.permutation([t for t in s1.L_trials if t in good_non_stim_trials_set])
    pert_trials_left = np.random.permutation([t for t in s1.L_trials if t in good_stim_trials_set])
    
    control_trials_right = np.random.permutation([t for t in s1.R_trials if t in good_non_stim_trials_set])
    pert_trials_right = np.random.permutation([t for t in s1.R_trials if t in good_stim_trials_set])
    
    L_trials_ctl_first, L_trials_ctl_second = control_trials_left[:int(len(control_trials_left) / 2)], control_trials_left[int(len(control_trials_left) / 2):]
    L_trials_opto_first, L_trials_opto_second = pert_trials_left[:int(len(pert_trials_left) / 2)], pert_trials_left[int(len(pert_trials_left) / 2):]
    
    R_trials_ctl_first, R_trials_ctl_second = control_trials_right[:int(len(control_trials_right) / 2)], control_trials_right[int(len(control_trials_right) / 2):]
    R_trials_opto_first, R_trials_opto_second = pert_trials_right[:int(len(pert_trials_right) / 2)], pert_trials_right[int(len(pert_trials_right) / 2):]

    naive_sample_sel, _ = s1.susceptibility(period = stim_period, p=p_s, return_n=True,
                                            provide_trials = (L_trials_ctl_first, L_trials_opto_first,
                                                              R_trials_ctl_first, R_trials_opto_first))
    
    naive_nonsel = [n for n in s1.good_neurons if n not in naive_sample_sel] # non susc population

    exp_susc, _ = s1.susceptibility(period = stim_period, p=p_s, return_n=True,
                                    provide_trials = (L_trials_ctl_second, L_trials_opto_second,
                                                      R_trials_ctl_second, R_trials_opto_second))
    
    # Get functional group info
    
    
    for n in naive_sample_sel:
        if n in exp_susc:
            s1list[0] += 1
            # ret_s += [(n, s2.good_neurons[np.where(s1.good_neurons == n)[0][0]])]

        else:
            s1list[3] += 1
            # drop_s += [(n, s2.good_neurons[np.where(s1.good_neurons == n)[0][0]])]

    
    for n in naive_nonsel:
        if n in exp_susc:
            ns1[0] += 1
            # recr_s += [(n, s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]])]

        else:
            ns1[3] += 1
    print(s1list)

    s1list, d1, r1, ns1 = s1list / len(s1.good_neurons), d1 / len(s1.good_neurons), r1 / len(s1.good_neurons), ns1 / len(s1.good_neurons)

    alls1list += [s1list]
    alld1 += [d1]
    allr1 += [r1] 
    allns1 += [ns1]
    
    retained_sample += [ret_s]
    recruited_sample += [recr_s]
    dropped_sample += [drop_s]
    retained_delay += [ret_d]
    recruited_delay += [recr_d]
    dropped_delay += [drop_d]

alls1list = np.mean(alls1list, axis=0) 
alld1 = np.mean(alld1, axis=0)
allr1 = np.mean(allr1, axis=0)
allns1 = np.mean(allns1, axis=0)

#%% Plot top contributors from input vectors

s1 = Mode(paths[1], lickdir=False, use_reg = True, triple=True, proportion_train=1, proportion_opto_train=1)
input_vector, delta = s1.input_vector(by_trialtype=False, plot=True, return_delta = True)

stim_period = range(s1.delay+int(0.5/s1.fs), s1.delay+int(1.3/s1.fs))

good_non_stim_trials_set = set(s1.i_good_non_stim_trials)
good_stim_trials_set = set(s1.i_good_stim_trials)
    
control_trials_left = np.random.permutation([t for t in s1.L_trials if t in good_non_stim_trials_set])
pert_trials_left = np.random.permutation([t for t in s1.L_trials if t in good_stim_trials_set])

control_trials_right = np.random.permutation([t for t in s1.R_trials if t in good_non_stim_trials_set])
pert_trials_right = np.random.permutation([t for t in s1.R_trials if t in good_stim_trials_set])

L_trials_ctl_first, L_trials_ctl_second = control_trials_left[:int(len(control_trials_left) / 2)], control_trials_left[int(len(control_trials_left) / 2):]
L_trials_opto_first, L_trials_opto_second = pert_trials_left[:int(len(pert_trials_left) / 2)], pert_trials_left[int(len(pert_trials_left) / 2):]

R_trials_ctl_first, R_trials_ctl_second = control_trials_right[:int(len(control_trials_right) / 2)], control_trials_right[int(len(control_trials_right) / 2):]
R_trials_opto_first, R_trials_opto_second = pert_trials_right[:int(len(pert_trials_right) / 2)], pert_trials_right[int(len(pert_trials_right) / 2):]

naive_sample_sel, _ = s1.susceptibility(period = stim_period, p=p_s, return_n=True,
                                        provide_trials = (L_trials_ctl_first, L_trials_opto_first,
                                                          R_trials_ctl_first, R_trials_opto_first))

naive_nonsel = [n for n in s1.good_neurons if n not in naive_sample_sel] # non susc population

exp_susc, _ = s1.susceptibility(period = stim_period, p=p_s, return_n=True,
                                provide_trials = (L_trials_ctl_second, L_trials_opto_second,
                                                  R_trials_ctl_second, R_trials_opto_second))

plt.hist(input_vector)
plt.title('Input vector weight distr')
plt.show()

sorted_n = s1.good_neurons[np.argsort(input_vector)]
#%%
n=715

s1.plot_raster_and_PSTH(n,trials=(R_trials_opto_first, L_trials_opto_first))
s1.plot_raster_and_PSTH(n,trials=(R_trials_opto_second, L_trials_opto_second))


#%% Plot some susceptible neurons
s1 = session.Session(paths[1], use_reg=True, triple=True, use_background_sub=False) # Learning
s2 = session.Session(paths[2], use_reg=True, triple=True) # Expert

for pairs in retained_sample[-4]:
    s1.plot_rasterPSTH_sidebyside(pairs[0])
    s2.plot_rasterPSTH_sidebyside(pairs[1])

#%% Average weight on choice CD of susceptible population across learning
cd_weight_lea, cd_weight_exp = [], []
p_s=0.05

for paths in all_matched_paths: # For each mouse/FOV
    s1 = Mode(paths[1], use_reg=True, triple=True, use_background_sub=False) # Learning
    lea_cd, _ = s1.plot_CD(plot=False)
    naive_susc = s1.susceptibility(period = range(s1.delay, s1.delay+int(1/s1.fs)), p=p_s, return_n=True)
    susc_idx = [np.where(s1.good_neurons == n)[0][0] for n in naive_susc]
    cd_weight_lea += [lea_cd[susc_idx]]
    
    s2 = Mode(paths[2], use_reg=True, triple=True) # Expert
    exp_cd, _ = s1.plot_CD(plot=False)
    exp_susc = s2.susceptibility(period = range(s2.delay, s2.delay+int(1/s2.fs)), p=p_s, return_n=True)
    susc_idx = [np.where(s2.good_neurons == n)[0][0] for n in exp_susc]
    cd_weight_exp += [exp_cd[susc_idx]]

#%% PLot
catcd_weight_lea = np.abs(cat(cd_weight_lea))
catcd_weight_exp = np.abs(cat(cd_weight_exp))

plt.bar(range(2), [np.mean(catcd_weight_lea), np.mean(catcd_weight_exp)])
plt.scatter(np.zeros(len(catcd_weight_lea)), catcd_weight_lea)
plt.scatter(np.ones(len(catcd_weight_exp)), catcd_weight_exp)
plt.ylabel('CD weight')
plt.xticks([0,1], ['Learning', 'Expert'])

stats.ttest_ind(catcd_weight_lea, catcd_weight_exp)


#%% Stability of susceptibility 

r_sus = []

f = plt.figure(figsize=(5,5))
for paths in all_matched_paths:
    
    intialpath, finalpath = paths[1], paths[2]
    
    # sample CD
    s1 = session.Session(intialpath, use_reg=True, triple=True)
    s2 = session.Session(finalpath, use_reg = True, triple=True)
    
    s1_sus, _ = s1.susceptibility()
    s2_sus, _ = s2.susceptibility()
    
    plt.scatter(s1_sus, s2_sus)
    r_sus += [stats.pearsonr(s1_sus, s2_sus)[0]]

# Plot
plt.title('Pearsons correlation: {}, p-val: {}'.format(stats.pearsonr(s1_sus, s2_sus)[0], 
                                                       stats.pearsonr(s1_sus, s2_sus)[1]))
plt.xlabel('Initial susceptibility')
plt.ylabel('Final susceptibility')
plt.show()


f = plt.figure(figsize=(5,5))
plt.bar([1], np.mean(r_sus), fill=False)
plt.scatter(np.ones(len(r_sus)), r_sus)
plt.xticks([1], ['R-values'])


#%% Selectivity vs susceptibility

all_r_sus = []
agg_allsus = []
agg_allsel = []
titles = ['Naive', 'Learning', 'Expert']

# f, ax = plt.subplots(1,3,figsize=(15,5))
for i in range(3):
    r_sus = []
    allsus = []
    allsel = []
    f = plt.figure(figsize=(5,5))

    for path in all_paths[i]:
            
        # Use all neurons 
        s1 = Session(path)
        # s1 = Session(path, use_reg=True, triple=True)
        
        sus = s1.susceptibility()
        sel, _, _ = s1.get_epoch_selectivity(range(s1.delay, s1.response), s1.good_neurons)
        sel = np.abs(sel)
        
        # ax[i].scatter(sus, sel)
        plt.scatter(sus, sel)
        plt.xlim(right=125)
        plt.ylim(top=100)
        
        r_sus += [(stats.pearsonr(sus, sel)[0], stats.pearsonr(sus, sel)[1])]
        allsus += [sus]
        allsel += [sel]
    
    agg_allsus += [[allsus]]
    agg_allsel += [[allsel]]
    all_r_sus += [r_sus]
    # ax[i].set_xlim(right=125)
    # ax[i].set_ylim(top=100)
    
# ax[0].set_xlabel('Susceptibility')
# ax[0].set_ylabel('Selectivity')
# plt.show()

    f = plt.figure(figsize=(7,5))
    plt.hist(allsus)
    plt.xlabel('Susceptibility')

#%% Plot selectivity vs sus

# for i in range(3):
i=0
f = plt.figure(figsize=(5,5))
plt.scatter(cat(agg_allsus[i][0]), cat(agg_allsel[i][0]), alpha=0.5)
plt.xlim(-5, 125)
plt.ylim(-0.5, 25)
plt.ylabel('Selectivity')
plt.xlabel('Susceptibility')



#%% Investigate single neuron susceptibility changes


paths = all_matched_paths[6]


intialpath, finalpath = paths[1], paths[2]

# sample CD
s1 = Session(intialpath, use_reg=True, triple=True)
s2 = Session(finalpath, use_reg=True, triple=True)

s1_sus = s1.susceptibility()
s2_sus = s2.susceptibility()

f = plt.figure(figsize=(5,5))

plt.scatter(s1_sus, s2_sus)
plt.plot(range(40), range(40))
r_sus = [stats.pearsonr(s1_sus, s2_sus)[0]]
plt.xlabel('Learning')
plt.ylabel('Expert')
plt.title('Susceptibility')

f = plt.figure(figsize=(5,5))
less = len([i for i in range(len(s1_sus)) if s1_sus[i] > s2_sus[i]])
plt.bar([0,1],[less / len(s1_sus), 1 - (less / len(s1_sus))])
plt.xticks([0,1], ['Less susceptible', 'More'])

# Plot all neurons' susceptibility as a bar plot
f = plt.figure(figsize=(15,5))

scores = np.array(s2_sus) - np.array(s1_sus)

plt.bar(range(len(scores)), np.sort(scores))
plt.axhline(np.quantile(scores,0.25), color = 'r', ls = '--')
plt.axhline(np.quantile(scores,0.75), color = 'r', ls = '--')
plt.axhline(np.quantile(scores,0.95), color = 'y', ls = '--')
plt.axhline(np.quantile(scores,0.05), color = 'y', ls = '--')
plt.xlabel('Neuron #')
plt.ylabel('Delta of susceptibility')


# Plot bar graph but with sample neurons in red

f = plt.figure(figsize=(15,5))

scores = np.array(s2_sus) - np.array(s1_sus)


# plt.bar(range(len(scores)), np.sort(scores))
sus_order = np.argsort(scores)
for i in range(len(sus_order)):
    # If sample selective, plot in red
    if s1.is_selective(s1.good_neurons[sus_order[i]], range(s1.sample, s1.delay), p=0.01):
        plt.bar([i], scores[sus_order[i]], color='red')
        plt.scatter([i], [max(scores)], color='red')
    else:
        plt.bar([i], scores[sus_order[i]], fill=False)

    
plt.axhline(np.quantile(scores,0.25), color = 'r', ls = '--')
plt.axhline(np.quantile(scores,0.75), color = 'r', ls = '--')
plt.axhline(np.quantile(scores,0.95), color = 'y', ls = '--')
plt.axhline(np.quantile(scores,0.05), color = 'y', ls = '--')
plt.xlabel('Neuron #')
plt.ylabel('Delta of susceptibility')


# Look on individual neuron level
s1_sorted_n = np.take(s1.good_neurons, np.argsort(scores))
s2_sorted_n = np.take(s2.good_neurons, np.argsort(scores))

i=0
s1.plot_rasterPSTH_sidebyside(s1_sorted_n[i])
s2.plot_rasterPSTH_sidebyside(s2_sorted_n[i])

# look as population
s1_neurons = s1.good_neurons[np.where(scores > np.quantile(scores,0.95))[0]]
s1.selectivity_optogenetics(selective_neurons = s1_neurons)
s2_neurons = s2.good_neurons[np.where(scores > np.quantile(scores,0.95))[0]]
s2.selectivity_optogenetics(selective_neurons = s2_neurons)