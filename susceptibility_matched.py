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

            [r'H:\data\BAYLORCW046\python\2024_05_29',
             r'H:\data\BAYLORCW046\python\2024_06_07',
             r'H:\data\BAYLORCW046\python\2024_06_24'],


            [r'H:\data\BAYLORCW046\python\2024_05_30',
             r'H:\data\BAYLORCW046\python\2024_06_10',
             r'H:\data\BAYLORCW046\python\2024_06_27'],

            [r'H:\data\BAYLORCW046\python\2024_05_31',
             r'H:\data\BAYLORCW046\python\2024_06_11',
             r'H:\data\BAYLORCW046\python\2024_06_26'
             ]
         
        ]



#%% Sannkey of sig susceptibly cells - all neurons

p_s=0.05
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

    naive_sample_sel = s1.susceptibility(period = stim_period, p=p_s, return_n=True)

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
    exp_susc = s2.susceptibility(period = stim_period, p=p_s, return_n=True)
    
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


#%% Sankey of sig susc cells - delay neurons only (lea or exp)
p_s=0.05
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

    naive_sample_sel = s1.susceptibility(period = stim_period, p=p_s, return_n=True)

    
    # Get functional group info
    # sample_epoch = range(s1.sample, s1.delay)
    delay_epoch = range(s1.delay+int(1.5 * 1/s1.fs), s1.response)
    # response_epoch = range(s1.response, s1.response + int(2*1/s1.fs))
    
    naive_delay_sel = s1.get_epoch_selective(delay_epoch, p=p) # Learning stage delay neurons
    


    s2 = session.Session(paths[2], use_reg=True, triple=True) # Expert
    exp_susc = s2.susceptibility(period = stim_period, p=p_s, return_n=True)
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