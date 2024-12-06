# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 14:44:38 2024

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

#%% Individually plotted distributions
agg_mice_paths = [[[r'F:\data\BAYLORCW032\python\2023_10_08',
          r'F:\data\BAYLORCW032\python\2023_10_16',
          r'F:\data\BAYLORCW032\python\2023_10_25',
          r'F:\data\BAYLORCW032\python\cellreg\layer{}\1008_1016_1025pairs_proc.npy'],]]

         
agg_mice_paths=    [ [[ r'F:\data\BAYLORCW034\python\2023_10_12',
            r'F:\data\BAYLORCW034\python\2023_10_22',
            r'F:\data\BAYLORCW034\python\2023_10_27',
            r'F:\data\BAYLORCW034\python\cellreg\layer{}\1012_1022_1027pairs_proc.npy'],]]
         
# agg_mice_paths=   [[[r'F:\data\BAYLORCW036\python\2023_10_09',
#             r'F:\data\BAYLORCW036\python\2023_10_19',
#             r'F:\data\BAYLORCW036\python\2023_10_30',
#             r'F:\data\BAYLORCW036\python\cellreg\layer{}\1009_1019_1030pairs_proc.npy'],
#         ]]

# agg_mice_paths = [[[r'F:\data\BAYLORCW037\python\2023_11_21',
#             r'F:\data\BAYLORCW037\python\2023_12_08',
#             r'F:\data\BAYLORCW037\python\2023_12_15',]]]

#%% Plot all mice together
#%% Paths
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



# agg_mice_paths = [[[r'F:\data\BAYLORCW032\python\2023_10_08',
#           r'F:\data\BAYLORCW032\python\2023_10_16',
#           r'F:\data\BAYLORCW032\python\2023_10_25',
#           r'F:\data\BAYLORCW032\python\cellreg\layer{}\1008_1016_1025pairs_proc.npy'],],

         
#         [[ r'F:\data\BAYLORCW034\python\2023_10_12',
#               r'F:\data\BAYLORCW034\python\2023_10_22',
#               r'F:\data\BAYLORCW034\python\2023_10_27',
#               r'F:\data\BAYLORCW034\python\cellreg\layer{}\1012_1022_1027pairs_proc.npy'],],
         
#     [[r'F:\data\BAYLORCW036\python\2023_10_09',
#             r'F:\data\BAYLORCW036\python\2023_10_19',
#             r'F:\data\BAYLORCW036\python\2023_10_30',
#             r'F:\data\BAYLORCW036\python\cellreg\layer{}\1009_1019_1030pairs_proc.npy'],
#         ],
    
#     [[r'F:\data\BAYLORCW035\python\2023_10_26',
#             r'F:\data\BAYLORCW035\python\2023_12_07',
#             r'F:\data\BAYLORCW035\python\2023_12_15',],
#         ],
    
#     [[r'F:\data\BAYLORCW037\python\2023_11_21',
#             r'F:\data\BAYLORCW037\python\2023_12_08',
#             r'F:\data\BAYLORCW037\python\2023_12_15',],
#         ]    

#     ]

#%% Get number to make SANKEY diagram SDR

# naivepath, learningpath, expertpath =[r'F:\data\BAYLORCW037\python\2023_11_21',
#             r'F:\data\BAYLORCW037\python\2023_12_08',
#             r'F:\data\BAYLORCW037\python\2023_12_15',]

# p=0.0005
p=0.001
retained_sample = []
recruited_sample = []
retained_delay = []
recruited_delay = []
dropped_delay = []
dropped_sample = []
alls1list, alld1, allr1, allns1 = [],[],[],[]
for paths in all_matched_paths: # For each mouse/FOV
    ret_s = []
    recr_s = []
    ret_d, recr_d = [],[]
    drop_d, drop_s = [], []
    
    s1list, d1, r1, ns1 = np.zeros(4),np.zeros(4),np.zeros(4),np.zeros(4)

    s1 = session.Session(paths[1], use_reg=True, triple=True, use_background_sub=False) # Naive

    sample_epoch = range(s1.sample, s1.delay)
    delay_epoch = range(s1.delay+int(1.5 * 1/s1.fs), s1.response)
    response_epoch = range(s1.response, s1.response + int(2*1/s1.fs))
    
    naive_sample_sel = s1.get_epoch_selective(sample_epoch, p=p)
    
    naive_delay_sel = s1.get_epoch_selective(delay_epoch, p=p)
    naive_delay_sel = [n for n in naive_delay_sel if n not in naive_sample_sel]
    
    naive_response_sel = s1.get_epoch_selective(response_epoch, p=p)
    naive_response_sel = [n for n in naive_response_sel if n not in naive_sample_sel and n not in naive_delay_sel]

    naive_nonsel = [n for n in s1.good_neurons if n not in naive_sample_sel and n not in naive_delay_sel and n not in naive_response_sel]

    s2 = session.Session(paths[2], use_reg=True, triple=True) # Learning
    
    for n in naive_sample_sel:
        if s2.is_selective(s2.good_neurons[np.where(s1.good_neurons == n)[0][0]], sample_epoch, p=p):
            s1list[0] += 1
            ret_s += [(n, s2.good_neurons[np.where(s1.good_neurons == n)[0][0]])]
        elif s2.is_selective(s2.good_neurons[np.where(s1.good_neurons == n)[0][0]], delay_epoch, p=p):
            s1list[1] += 1
        elif s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], response_epoch, p=p):
            s1list[2] += 1
        else:
            s1list[3] += 1
            drop_s += [(n, s2.good_neurons[np.where(s1.good_neurons == n)[0][0]])]

    for n in naive_delay_sel:
        if s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], sample_epoch, p=p):
            d1[0] += 1
        elif s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], delay_epoch, p=p):
            d1[1] += 1
            ret_d += [(n, s2.good_neurons[np.where(s1.good_neurons == n)[0][0]])]

        elif s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], response_epoch, p=p):
            d1[2] += 1
        else:
            d1[3] += 1
            drop_d += [(n, s2.good_neurons[np.where(s1.good_neurons == n)[0][0]])]
    
    
    for n in naive_response_sel:
        if s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], sample_epoch, p=p):
            r1[0] += 1

        elif s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], delay_epoch, p=p):
            r1[1] += 1

        elif s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], response_epoch, p=p):
            r1[2] += 1
        else:
            r1[3] += 1
    
    
    for n in naive_nonsel:
        if s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], sample_epoch, p=p):
            ns1[0] += 1
            recr_s += [(n, s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]])]
        elif s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], delay_epoch, p=p):
            ns1[1] += 1
            recr_d += [(n, s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]])]

        elif s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], response_epoch, p=p):
            ns1[2] += 1
        else:
            ns1[3] += 1

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
#%% Look at specific neurons    
# retained_sample,recruited_sample
naivepaths = agg_matched_paths[0]

learnpaths = agg_matched_paths[1]

for i in range(len(retained_sample)):
    
    if len(retained_sample[i]) != 0:
        
        for nai,lea in retained_sample[i]:
            
            s1 = session.Session(all_matched_paths[i][0])
            s2 = session.Session(all_matched_paths[i][1])
            
            s1.plot_rasterPSTH_sidebyside(nai)
            s1.plot_rasterPSTH_sidebyside(lea)
#%%
allcats = [retained_sample, recruited_sample, retained_delay, recruited_delay, dropped_delay] 

by_FOV = True
stage1, stage2 = 1, 2
for cats in allcats:
    num_neurons, counter = 0, 0
    for paths in all_matched_paths:
        
        stos = np.array(cats[counter])
        counter += 1

        if len(stos) == 0: # no neurons
            # print()
            continue
        
        intialpath, finalpath =  paths[stage1], paths[stage2]
        s1 = session.Session(intialpath, use_reg=True, triple=True) # pre
        s2 = session.Session(finalpath, use_reg=True, triple=True) # post

        for i in range(len(stos)):
            
            if by_FOV and i == 0 :
                print("first addition")
                pre_sel = s1.plot_selectivity(stos[i,0], plot=False)
                post_sel = s2.plot_selectivity(stos[i,1], plot=False)
        
                pre_sel_opto = s1.plot_selectivity(stos[i,0], plot=False, opto=True)
                post_sel_opto = s2.plot_selectivity(stos[i,1], plot=False, opto=True)
                
            elif num_neurons + i == 0:
                print("first addition non FOV")
                pre_sel = s1.plot_selectivity(stos[i,0], plot=False)
                post_sel = s2.plot_selectivity(stos[i,1], plot=False)
        
                pre_sel_opto = s1.plot_selectivity(stos[i,0], plot=False, opto=True)
                post_sel_opto = s2.plot_selectivity(stos[i,1], plot=False, opto=True)
            else:
                pre_sel = np.vstack((pre_sel, s1.plot_selectivity(stos[i,0], plot=False)))
                post_sel = np.vstack((post_sel, s2.plot_selectivity(stos[i,1], plot=False)))
                
                pre_sel_opto = np.vstack((pre_sel_opto, s1.plot_selectivity(stos[i,0], plot=False, opto=True)))
                post_sel_opto = np.vstack((post_sel_opto, s2.plot_selectivity(stos[i,1], plot=False, opto=True)))
                
        if num_neurons == 0 and by_FOV:
            
            if stos.shape[0] == 1:
                agg_pre_sel = pre_sel
                agg_pre_sel_opto = pre_sel_opto
                
                agg_post_sel = post_sel
                agg_post_sel_opto = post_sel_opto
            else:
                agg_pre_sel = np.mean(pre_sel, axis=0)
                agg_pre_sel_opto = np.mean(pre_sel_opto, axis=0)
                
                agg_post_sel = np.mean(post_sel, axis=0)
                agg_post_sel_opto = np.mean(post_sel_opto, axis=0)
            
        elif by_FOV:
            if stos.shape[0] == 1:
                
                agg_pre_sel = np.vstack((agg_pre_sel, pre_sel))
                agg_pre_sel_opto = np.vstack((agg_pre_sel_opto, pre_sel_opto))
                
                agg_post_sel = np.vstack((agg_post_sel, post_sel))
                agg_post_sel_opto = np.vstack((agg_post_sel_opto, post_sel_opto))
                
            else:

                agg_pre_sel = np.vstack((agg_pre_sel, np.mean(pre_sel, axis=0)))
                agg_pre_sel_opto = np.vstack((agg_pre_sel_opto, np.mean(pre_sel_opto, axis=0)))
                
                agg_post_sel = np.vstack((agg_post_sel, np.mean(post_sel, axis=0)))
                agg_post_sel_opto = np.vstack((agg_post_sel_opto, np.mean(post_sel_opto, axis=0)))
                
        num_neurons += len(stos)
        
    f, ax = plt.subplots(1,2, sharey='row', figsize=(10,5))

    if by_FOV:
        sel = np.mean(agg_pre_sel, axis=0)
        err = np.std(agg_pre_sel, axis=0) / np.sqrt(len(agg_pre_sel)) 
        selo = np.mean(agg_pre_sel_opto, axis=0)
        erro = np.std(agg_pre_sel_opto, axis=0) / np.sqrt(len(agg_pre_sel_opto)) 
    else:
        sel = np.mean(pre_sel, axis=0)
        err = np.std(pre_sel, axis=0) / np.sqrt(len(pre_sel)) 
        selo = np.mean(pre_sel_opto, axis=0)
        erro = np.std(pre_sel_opto, axis=0) / np.sqrt(len(pre_sel_opto))
        
        
    x = np.arange(-6.97,5,1/6)[:61]
    ax[0].plot(x, sel, 'black')
            
    ax[0].fill_between(x, sel - err, 
              sel + err,
              color=['darkgray'])
    
    ax[0].plot(x, selo, 'r-')
            
    ax[0].fill_between(x, selo - erro, 
              selo + erro,
              color=['#ffaeb1']) 
    
    if by_FOV:
        sel = np.mean(agg_post_sel, axis=0)
        err = np.std(agg_post_sel, axis=0) / np.sqrt(len(agg_post_sel)) 
        selo = np.mean(agg_post_sel_opto, axis=0)
        erro = np.std(agg_post_sel_opto, axis=0) / np.sqrt(len(agg_post_sel_opto)) 
    else:
        sel = np.mean(post_sel, axis=0)
        err = np.std(post_sel, axis=0) / np.sqrt(len(post_sel)) 
        selo = np.mean(post_sel_opto, axis=0)
        erro = np.std(post_sel_opto, axis=0) / np.sqrt(len(post_sel_opto)) 
        
    ax[1].plot(x, sel, 'black')
            
    ax[1].fill_between(x, sel - err, 
              sel + err,
              color=['darkgray'])
    
    ax[1].plot(x, selo, 'r-')
            
    ax[1].fill_between(x, selo - erro, 
              selo + erro,
              color=['#ffaeb1']) 
    
    for i in range(2):
        ax[i].axvline(-4.3, color = 'grey', alpha=0.5, ls = '--')
        ax[i].axvline(-3, color = 'grey', alpha=0.5, ls = '--')
        ax[i].axvline(0, color = 'grey', alpha=0.5, ls = '--')
        ax[i].hlines(y=max(cat((selo, sel))), xmin=-3, xmax=-2, linewidth=10, color='red')
        
    ax[0].set_xlabel('Time from Go cue (s)')
    ax[0].set_ylabel('Selectivity')

    ax[0].set_title('Optogenetic effect on selectivity (n = {} neurons)'.format(num_neurons))                  
    # ax[1].set_title('Optogenetic effect on selectivity (n = {} neurons)'.format(num_neurons))         
    plt.show()         
    
    
#%% Count total number of neurons
neuron_count = []
for paths in agg_matched_paths[0]:
    l1 = session.Session(paths, use_reg=True, triple=True)
    neuron_count += [len(l1.good_neurons)]
    
    
#%% Calculate SDR signifiances
# allnums = np.vstack((s1list, d1, r1, ns1))
allnums = np.vstack((alls1list, alld1, allr1, allns1)) * 3028
exptotals = np.sum(allnums, axis=0)
# Calculate significance using binomial/proportion z-test

for i in range(4): # SDR
    print("####################")
    for j in range(4): #Where the bucket went
        # Input data
        count = allnums[i, j]   # number of successes
        nobs = np.sum(allnums[i])    # total number of observations
        value = exptotals[j] / np.sum(exptotals) # hypothesized proportion
        
        # Perform one-proportion z-test
        z_stat, p_value = proportions_ztest(count, nobs, value, alternative='larger')
        
        # Output the results
        print(f"Z-statistic: {z_stat}")
        print(f"P-value: {p_value}")


#%% Use Chi-square for overall comparison of proportions (no greater/less than expected info)
# Forward probabilities
for i in range(4):
    res = chisquare(f_obs=allnums[i], f_exp=exptotals / sum(exptotals) * sum(allnums[i]))
    print(res.pvalue)

allnums = allnums.T
exptotals = np.sum(allnums, axis=0)
# Backwards probabilities
for i in range(4):
    res = chisquare(f_obs=allnums[i], f_exp=exptotals / sum(exptotals) * sum(allnums[i]))
    print(res.pvalue)

    

    


#%% Get number to make SANKEY diagram contra ipsi


p=0.01

og_SDR = []
c1, i1, ns1 = np.zeros(3),np.zeros(3),np.zeros(3)
for paths in agg_mice_paths: # For each mouse

    s1 = session.Session(paths[0][1], use_reg=True, triple=True) # Naive
    # epoch = range(s1.response, s1.time_cutoff) # response selective
    epoch = range(s1.delay + 9, s1.response) # delay selective
    
    contra_neurons, ipsi_neurons, _, _ = s1.contra_ipsi_pop(epoch, p=p)
    naive_nonsel = [n for n in s1.good_neurons if n not in ipsi_neurons and n not in contra_neurons]

    og_SDR += [[len(contra_neurons), len(ipsi_neurons), len(naive_nonsel)]]

    # s2 = session.Session(paths[0][1], use_reg=True, triple=True) # Learning
    s2 = session.Session(paths[0][2], use_reg=True, triple=True) # Expert

    # learning = sum([s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], epoch) for n in naive_sel])
    # expert = sum([s3.is_selective(s3.good_neurons[np.where(s1.good_neurons ==n)[0][0]], epoch) for n in naive_sel])
    
    for n in contra_neurons:
        if s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], epoch, p=p):
            ipsi, _, _ = s2.screen_preference(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], epoch)
            if not ipsi:
                c1[0] += 1
            else:
                c1[1] += 1
        else:
            c1[2] += 1
    
    for n in ipsi_neurons:
        if s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], epoch, p=p):
            ipsi, _, _ = s2.screen_preference(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], epoch)
            if not ipsi:
                i1[0] += 1
            else:
                i1[1] += 1
        else:
            i1[2] += 1
    

    for n in naive_nonsel:
        if s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], epoch, p=p):
            ipsi, _, _ = s2.screen_preference(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], epoch)
            if not ipsi:
                ns1[0] += 1
            else:
                ns1[1] += 1
        else:
            ns1[2] += 1
    
og_SDR = np.sum(og_SDR, axis=0)
    
#RESULTS
    
# // Enter Flows between Nodes, like this:
# //         Source [AMOUNT] Target

# Sample[8] Sample1
# Sample[1] Delay1
# Sample[5] Response1
# Sample[24] NS1

# Delay[9] Sample1
# Delay[5] Delay1
# Delay[6] Response1
# Delay[41] NS1

# Response[10] Sample1
# Response[15] Delay1
# Response[45] Response1
# Response[110] NS1

# Non-selective[34] Sample1
# Non-selective[48] Delay1
# Non-selective[135] Response1
# Non-selective[1176] NS1

# // Learning expert
# Sample1[15] Sample2
# Sample1[3] Delay2
# Sample1[10] Response2
# Sample1[20] NS2

# Delay1[20] Sample2
# Delay1[16] Delay2
# Delay1[22] Response2
# Delay1[32] NS2

# Response1[15] Sample2
# Response1[30] Delay2
# Response1[92] Response2
# Response1[99] NS2

# NS1[30] Sample2
# NS1[83] Delay2
# NS1[204] Response2
# NS1[1006] NS2
    
    
    
    