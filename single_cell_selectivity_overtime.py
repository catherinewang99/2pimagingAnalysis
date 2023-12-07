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
import random

#%% Individually plotted distributions
# agg_mice_paths = [[r'F:\data\BAYLORCW032\python\2023_10_08',
#           r'F:\data\BAYLORCW032\python\2023_10_16',
#           r'F:\data\BAYLORCW032\python\2023_10_25',
#           r'F:\data\BAYLORCW032\python\cellreg\layer{}\1008_1016_1025pairs_proc.npy'],]

         
agg_mice_paths=    [ [[ r'F:\data\BAYLORCW034\python\2023_10_12',
            r'F:\data\BAYLORCW034\python\2023_10_22',
            r'F:\data\BAYLORCW034\python\2023_10_27',
            r'F:\data\BAYLORCW034\python\cellreg\layer{}\1012_1022_1027pairs_proc.npy'],]]
         
# agg_mice_paths=   [[r'F:\data\BAYLORCW036\python\2023_10_09',
#             r'F:\data\BAYLORCW036\python\2023_10_19',
#             r'F:\data\BAYLORCW036\python\2023_10_30',
#             r'F:\data\BAYLORCW036\python\cellreg\layer{}\1009_1019_1030pairs_proc.npy'],
#         ]

#%% Plot all mice together

agg_mice_paths = [[[r'F:\data\BAYLORCW032\python\2023_10_08',
          r'F:\data\BAYLORCW032\python\2023_10_16',
          r'F:\data\BAYLORCW032\python\2023_10_25',
          r'F:\data\BAYLORCW032\python\cellreg\layer{}\1008_1016_1025pairs_proc.npy'],],

         
       [[ r'F:\data\BAYLORCW034\python\2023_10_12',
            r'F:\data\BAYLORCW034\python\2023_10_22',
            r'F:\data\BAYLORCW034\python\2023_10_27',
            r'F:\data\BAYLORCW034\python\cellreg\layer{}\1012_1022_1027pairs_proc.npy'],],
         
    [[r'F:\data\BAYLORCW036\python\2023_10_09',
            r'F:\data\BAYLORCW036\python\2023_10_19',
            r'F:\data\BAYLORCW036\python\2023_10_30',
            r'F:\data\BAYLORCW036\python\cellreg\layer{}\1009_1019_1030pairs_proc.npy'],
        ]]

#%% Plot trained --> naive

p=0.05
rexpertsel, rlearningsel, rnaivesel = [],[],[]
lexpertsel, llearningsel, lnaivesel = [],[],[]

for paths in agg_mice_paths:
    allsel = []
    for allpath in paths:
        for l_num in range(1,6):
            path = allpath[2] # Expert session
            s1 = session.Session(path, layer_num = l_num, use_reg=True, triple=True)
            
            # epoch = range(s1.response+6, s1.response+12) # Response selective
            epoch = range(s1.response-9, s1.response) # Delay selective
            # epoch = range(s1.delay-3, s1.delay+3) # Stimulus selective
    
            rtrials = s1.lick_correct_direction('r')
            random.shuffle(rtrials)
            rtrials_train = rtrials[:50]
            rtrials_test = rtrials[50:]  
            
            ltrials = s1.lick_correct_direction('l')
            random.shuffle(ltrials)
            ltrials_train = ltrials[:50]
            ltrials_test = ltrials[50:]              
            
            s1_neurons = s1.get_epoch_selective(epoch, p=p, rtrials=rtrials_train, ltrials=ltrials_train)
            allstat, poststat, negtstat = s1.get_epoch_tstat(epoch, s1_neurons, rtrials=rtrials_test, ltrials=ltrials_test)
            rexpertsel += negtstat
            lexpertsel += poststat
            allsel += allstat
            
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
                    allstat, _, _ = s2.get_epoch_tstat(epoch, [matched_neurons[idx, 1]])
                    rlearningsel += allstat
                    allstat, _, _ = s3.get_epoch_tstat(epoch, [matched_neurons[idx, 0]])
                    rnaivesel += allstat
                else:
                    allstat, _, _ = s2.get_epoch_tstat(epoch, [matched_neurons[idx, 1]])
                    llearningsel += allstat
                    allstat, _, _ = s3.get_epoch_tstat(epoch, [matched_neurons[idx, 0]])
                    lnaivesel += allstat

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

        
#%% naive --> trained

p=0.05
rexpertsel, rlearningsel, rnaivesel = [],[],[]
lexpertsel, llearningsel, lnaivesel = [],[],[]

for paths in agg_mice_paths:
    allsel = []
    for allpath in paths:
        for l_num in range(1,6):
            path = allpath[0] # Naive session
            s1 = session.Session(path, layer_num = l_num, use_reg=True, triple=True)
            
            epoch = range(s1.response+6, s1.response+12) # Response selective
            # epoch = range(s1.response-9, s1.response) # Delay selective
            # epoch = range(s1.delay-3, s1.delay+3) # Stimulus selective
    
            rtrials = s1.lick_correct_direction('r')
            random.shuffle(rtrials)
            rtrials_train = rtrials[:50]
            rtrials_test = rtrials[50:]  
            
            ltrials = s1.lick_correct_direction('l')
            random.shuffle(ltrials)
            ltrials_train = ltrials[:50]
            ltrials_test = ltrials[50:]              
            
            s1_neurons = s1.get_epoch_selective(epoch, p=p, rtrials=rtrials_train, ltrials=ltrials_train)
            allstat, poststat, negtstat = s1.get_epoch_tstat(epoch, s1_neurons, rtrials=rtrials_test, ltrials=ltrials_test)
            rexpertsel += negtstat
            lexpertsel += poststat
            allsel += allstat
            
            ## Intermediate and expert
            path1 = allpath[1]
            s2 = session.Session(path1, layer_num = l_num, use_reg=True, triple=True)
            path2 = allpath[2]
            s3 = session.Session(path2, layer_num = l_num, use_reg=True, triple=True)
            
            match_n_path = allpath[3]
            matched_neurons=np.load(match_n_path.format(l_num-1))
            print(len(matched_neurons))
            
            for i in range(len(s1_neurons)):
                
                n = s1_neurons[i]
                idx = np.where(matched_neurons[:, 0] == n)[0][0]
                
                if allsel[i] < 0:
                    allstat, _, _ = s2.get_epoch_tstat(epoch, [matched_neurons[idx, 1]])
                    rlearningsel += allstat
                    allstat, _, _ = s3.get_epoch_tstat(epoch, [matched_neurons[idx, 2]])
                    rnaivesel += allstat
                else:
                    allstat, _, _ = s2.get_epoch_tstat(epoch, [matched_neurons[idx, 1]])
                    llearningsel += allstat
                    allstat, _, _ = s3.get_epoch_tstat(epoch, [matched_neurons[idx, 2]])
                    lnaivesel += allstat

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

        
