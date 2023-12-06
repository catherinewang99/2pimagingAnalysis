# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 14:41:18 2023

@author: Catherine Wang

Replacting bi-modal LR pref histogram shuffling (JH paper)
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

paths = [[r'F:\data\BAYLORCW032\python\2023_10_08',
          r'F:\data\BAYLORCW032\python\2023_10_16',
          r'F:\data\BAYLORCW032\python\2023_10_25',
          r'F:\data\BAYLORCW032\python\cellreg\layer{}\1008_1016_1025pairs_proc.npy'],]

         
# paths=     [[ r'F:\data\BAYLORCW034\python\2023_10_12',
#             r'F:\data\BAYLORCW034\python\2023_10_22',
#             r'F:\data\BAYLORCW034\python\2023_10_27',
#             r'F:\data\BAYLORCW034\python\cellreg\layer{}\1012_1022_1027pairs_proc.npy'],]
         
# paths=   [[r'F:\data\BAYLORCW036\python\2023_10_09',
#             r'F:\data\BAYLORCW036\python\2023_10_19',
#             r'F:\data\BAYLORCW036\python\2023_10_30',
#             r'F:\data\BAYLORCW036\python\cellreg\layer{}\1009_1019_1030pairs_proc.npy'],
#         ]

p=0.05
rexpertsel, rlearningsel, rnaivesel = [],[],[]
lexpertsel, llearningsel, lnaivesel = [],[],[]
allsel = []
for allpath in paths:
    for l_num in range(1,6):
        path = allpath[2] # Expert session
        s1 = session.Session(path, layer_num = l_num, use_reg=True, triple=True)
        
        epoch = range(s1.response+6, s1.response+12) # Response selective
        # epoch = range(s1.response-9, s1.response) # Delay selective
        # epoch = range(s1.delay-3, s1.delay+3) # Stimulus selective

        s1_neurons = s1.get_epoch_selective(epoch, p=p)
        poststat, negtstat = s1.get_epoch_tstat(epoch, s1_neurons, split=True)
        rexpertsel += negtstat
        lexpertsel += poststat
        allsel += s1.get_epoch_tstat(epoch, s1_neurons)
        
        ## Intermediate and naive
        path1 = allpath[1]
        s2 = session.Session(path1, layer_num = l_num, use_reg=True, triple=True)
        path2 = allpath[0]
        s3 = session.Session(path2, layer_num = l_num, use_reg=True, triple=True)
        
        match_n_path = allpath[3]
        matched_neurons=np.load(match_n_path.format(l_num-1))
        print(len(matched_neurons))
        
        for i in range(len(s1_neurons)):
            
            n = s1_neurons[i]
            idx = np.where(matched_neurons[:, 2] == n)[0][0]
            
            if allsel[i] < 0:
                rlearningsel += s2.get_epoch_tstat(epoch, [matched_neurons[idx, 1]])
                rnaivesel += s3.get_epoch_tstat(epoch, [matched_neurons[idx, 0]])
            else:
                llearningsel += s2.get_epoch_tstat(epoch, [matched_neurons[idx, 1]])
                lnaivesel += s3.get_epoch_tstat(epoch, [matched_neurons[idx, 0]])

bins = 25                
plt.hist(rexpertsel, bins=bins, color='b', alpha = 0.7)
plt.hist(lexpertsel, bins=bins, color='r', alpha = 0.7)
plt.show()

plt.hist(rlearningsel, bins=bins, color='b', alpha = 0.7)
plt.hist(llearningsel, bins=bins, color='r', alpha = 0.7)
plt.show()

plt.hist(lnaivesel, bins=bins, color='r', alpha = 0.7)
plt.hist(rnaivesel, bins=bins, color='b', alpha = 0.7)
plt.show()
# plt.bar([1,3,5], [all_naive_DS_exp / all_exp_DS_exp,
#                   all_learning_DS_exp / all_exp_DS_exp, 1])

# plt.errorbar([1,3,5],[all_naive_DS_exp / all_exp_DS_exp,
#                   all_learning_DS_exp / all_exp_DS_exp, 1],
#              yerr = [np.std(propsnaive_exp) / np.sqrt(3),
#                      np.std(propslearning_exp) / np.sqrt(3), 0], color = 'r')

# plt.xticks([1,3,5], ['Naive', 'Learning', 'Expert'])
        
