# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 18:17:13 2023

@author: Catherine Wang
"""

import sys
sys.path.append("C:\scripts\Imaging analysis")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import session
from matplotlib.pyplot import figure
# import decon
from scipy.stats import chisquare
import pandas as pd
plt.rcParams['pdf.fonttype'] = '42' 


#%%#Expert --> naive, intermediate
p=0.01
all_naive_DS_exp, all_learning_DS_exp = 0, 0
all_exp_DS_exp = 0
propsnaive_exp, propslearning_exp = [], []
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

for allpath in paths:
    n, l, e = 0,0,0
    for l_num in range(1,6):
        path = allpath[2]
        s1 = session.Session(path, layer_num = l_num, use_reg=True, triple=True)
        s1_neurons = s1.get_epoch_selective(epoch=range(s1.delay, s1.response), p=p)
        all_exp_DS_exp += len(s1_neurons)
        
        ## Intermediate and naive
        path1 = allpath[1]
        s2 = session.Session(path1, layer_num = l_num, use_reg=True, triple=True)
        path2 = allpath[0]
        s3 = session.Session(path2, layer_num = l_num, use_reg=True, triple=True)
        
        match_n_path = allpath[3]
        matched_neurons=np.load(match_n_path.format(l_num-1))
        print(len(matched_neurons))
        for n in s1_neurons:
            idx = np.where(matched_neurons[:, 2] == n)[0][0]
            all_learning_DS_exp += s2.is_selective(matched_neurons[idx, 1], range(s1.delay, s1.response), p=p)
            all_naive_DS_exp += s3.is_selective(matched_neurons[idx, 0], range(s1.delay, s1.response), p=p)
        
        e += all_exp_DS_exp
        n += all_naive_DS_exp
        l += all_learning_DS_exp
    
    propsnaive_exp += [n/e]
    propslearning_exp += [l/e]
        
            
plt.bar([1,3,5], [all_naive_DS_exp / all_exp_DS_exp,
                  all_learning_DS_exp / all_exp_DS_exp, 1])

plt.errorbar([1,3,5],[all_naive_DS_exp / all_exp_DS_exp,
                  all_learning_DS_exp / all_exp_DS_exp, 1],
             yerr = [np.std(propsnaive_exp) / np.sqrt(3),
                     np.std(propslearning_exp) / np.sqrt(3), 0], color = 'r')

plt.xticks([1,3,5], ['Naive', 'Learning', 'Expert'])
        

        
#%% #Naive --> intermediate, expert
p=0.01
all_naive_DS, all_learning_DS = 0, 0
all_exp_DS = 0
propsexp, propslearning = [], []
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

for allpath in paths:
    n, l, e = 0,0,0
    for l_num in range(1,6):
        path = allpath[0]
        s1 = session.Session(path, layer_num = l_num, use_reg=True, triple=True)
        s1_neurons = s1.get_epoch_selective(epoch=range(s1.delay, s1.response), p=p)
        all_naive_DS += len(s1_neurons)
        
        ## Intermediate and naive
        path1 = allpath[1]
        s2 = session.Session(path1, layer_num = l_num, use_reg=True, triple=True)
        path2 = allpath[2]
        s3 = session.Session(path2, layer_num = l_num, use_reg=True, triple=True)
        
        match_n_path = allpath[3]
        matched_neurons=np.load(match_n_path.format(l_num-1))
        print(len(matched_neurons))
        for n in s1_neurons:
            idx = np.where(matched_neurons[:, 0] == n)[0][0]
            all_learning_DS += s2.is_selective(matched_neurons[idx, 1], range(s1.delay, s1.response), p=p)
            all_exp_DS += s3.is_selective(matched_neurons[idx, 2], range(s1.delay, s1.response), p=p)
        
        e += all_exp_DS
        n += all_naive_DS
        l += all_learning_DS
    
    propsexp += [e/n]
    propslearning += [l/n]
        
            
plt.bar([1,3,5], [1, all_learning_DS / all_naive_DS,
                  all_exp_DS / all_naive_DS])

plt.errorbar([1,3,5],[1, all_learning_DS / all_naive_DS,
                  all_exp_DS / all_naive_DS],
             yerr = [0, np.std(propslearning) / np.sqrt(3),
                     np.std(propsexp) / np.sqrt(3)], color = 'r')

plt.xticks([1,3,5], ['Naive', 'Learning', 'Expert'])

#%% Plot altogether

x = np.arange(3)


plt.xticks(x, ['Naive', 'Learning', 'Expert'])


plt.bar(x-0.2, [1, all_learning_DS / all_naive_DS,
                  all_exp_DS / all_naive_DS], 0.4, color = 'pink', label='Trained on expert session')

plt.errorbar(x-0.2,[1, all_learning_DS / all_naive_DS,
                  all_exp_DS / all_naive_DS],
             yerr = [0, np.std(propslearning) / np.sqrt(3),
                     np.std(propsexp) / np.sqrt(3)], color = 'r')



plt.bar(x+0.2, [all_naive_DS_exp / all_exp_DS_exp,
                  all_learning_DS_exp / all_exp_DS_exp, 1], 0.4, color = 'lightblue', label='Trained on expert session')

plt.errorbar(x+0.2,[all_naive_DS_exp / all_exp_DS_exp,
                  all_learning_DS_exp / all_exp_DS_exp, 1],
             yerr = [np.std(propsnaive_exp) / np.sqrt(3),
                     np.std(propslearning_exp) / np.sqrt(3), 0], color = 'r')
plt.legend()
plt.savefig(r'F:\data\SFN 2023\change_sel_doublebars.pdf')
plt.show()