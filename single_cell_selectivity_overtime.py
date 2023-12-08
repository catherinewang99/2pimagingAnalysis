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
from scipy import stats

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

#%% Plot all mice together

agg_mice_paths = [[[r'F:\data\BAYLORCW032\python\2023_10_08',
          r'F:\data\BAYLORCW032\python\2023_10_16',
          r'F:\data\BAYLORCW032\python\2023_10_25',
          r'F:\data\BAYLORCW032\python\cellreg\layer{}\1008_1016_1025pairs_proc.npy'],],

         
        # [[ r'F:\data\BAYLORCW034\python\2023_10_12',
        #      r'F:\data\BAYLORCW034\python\2023_10_22',
        #      r'F:\data\BAYLORCW034\python\2023_10_27',
        #      r'F:\data\BAYLORCW034\python\cellreg\layer{}\1012_1022_1027pairs_proc.npy'],],
         
    [[r'F:\data\BAYLORCW036\python\2023_10_09',
            r'F:\data\BAYLORCW036\python\2023_10_19',
            r'F:\data\BAYLORCW036\python\2023_10_30',
            r'F:\data\BAYLORCW036\python\cellreg\layer{}\1009_1019_1030pairs_proc.npy'],
        ]]

#%% Plot expert --> naive

p=0.05
rexpertsel, rlearningsel, rnaivesel = [],[],[]
lexpertsel, llearningsel, lnaivesel = [],[],[]


for paths in agg_mice_paths:
    allsel = []
    for allpath in paths:
        path = allpath[2] # Expert session
        s1 = session.Session(path, use_reg=True, triple=True)
        
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
        
        idx = [np.where(s1.good_neurons == s)[0][0] for s in s1_neurons] # positions of selective neurons
        
        ## Intermediate and naive
        path1 = allpath[1]
        s2 = session.Session(path1, use_reg=True, triple=True)
        path2 = allpath[0]
        s3 = session.Session(path2, use_reg=True, triple=True)
        
        
        for i in range(len(idx)):
            
            if allsel[i] < 0:
                allstat, _, _ = s2.get_epoch_tstat(epoch, [s2.good_neurons[idx[i]]])
                rlearningsel += allstat
                allstat, _, _ = s3.get_epoch_tstat(epoch, [s3.good_neurons[idx[i]]])
                rnaivesel += allstat
            else:
                allstat, _, _ = s2.get_epoch_tstat(epoch, [s2.good_neurons[idx[i]]])
                llearningsel += allstat
                allstat, _, _ = s3.get_epoch_tstat(epoch, [s3.good_neurons[idx[i]]])
                lnaivesel += allstat

bins = 25                
plt.hist(rexpertsel, bins=bins, color='b', alpha = 0.7)
plt.hist(lexpertsel, bins=bins, color='r', alpha = 0.7)
plt.axvline(np.median(rexpertsel), color = 'b')
plt.axvline(np.median(lexpertsel), color = 'r')
plt.title('Expert Left/Right selectivity')
plt.show()

# T-test to see if there is a significant difference between left and right groups


tstat, p_val = stats.ttest_ind(llearningsel, rlearningsel)
print("Learning diff p-value: ", p_val)
plt.hist(rlearningsel, bins=bins, color='b', alpha = 0.7)
plt.hist(llearningsel, bins=bins, color='r', alpha = 0.7)
plt.axvline(np.median(rlearningsel), color = 'b')
plt.axvline(np.median(llearningsel), color = 'r')
plt.title('Learning Left/Right selectivity (p-value={})'.format(p_val))
plt.show()

tstat, p_val = stats.ttest_ind(lnaivesel, rnaivesel)
print("Naive diff p-value: ", p_val)
plt.hist(lnaivesel, bins=bins, color='r', alpha = 0.7)
plt.hist(rnaivesel, bins=bins, color='b', alpha = 0.7)
plt.axvline(np.median(rnaivesel), color = 'b')
plt.axvline(np.median(lnaivesel), color = 'r')
plt.title('Naive Left/Right selectivity (p-value={})'.format(p_val))
plt.show()



#%% naive --> trained

p=0.05
rexpertsel, rlearningsel, rnaivesel = [],[],[]
lexpertsel, llearningsel, lnaivesel = [],[],[]

for paths in agg_mice_paths:
    allsel = []
    for allpath in paths:
        path = allpath[0] # Naive session
        s1 = session.Session(path, use_reg=True, triple=True)
        
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
        
        idx = [np.where(s1.good_neurons == s)[0][0] for s in s1_neurons] # positions of selective neurons
        
        ## Intermediate and expert
        path1 = allpath[1]
        s2 = session.Session(path1, use_reg=True, triple=True)
        path2 = allpath[2]
        s3 = session.Session(path2, use_reg=True, triple=True)
        
        
        for i in range(len(idx)):
            
            if allsel[i] < 0:
                allstat, _, _ = s2.get_epoch_tstat(epoch, [s2.good_neurons[idx[i]]])
                rlearningsel += allstat
                allstat, _, _ = s3.get_epoch_tstat(epoch, [s3.good_neurons[idx[i]]])
                rnaivesel += allstat
            else:
                allstat, _, _ = s2.get_epoch_tstat(epoch, [s2.good_neurons[idx[i]]])
                llearningsel += allstat
                allstat, _, _ = s3.get_epoch_tstat(epoch, [s3.good_neurons[idx[i]]])
                lnaivesel += allstat






bins = 25                
plt.hist(rexpertsel, bins=bins, color='b', alpha = 0.7)
plt.hist(lexpertsel, bins=bins, color='r', alpha = 0.7)
plt.axvline(np.median(rexpertsel), color = 'b')
plt.axvline(np.median(lexpertsel), color = 'r')
plt.title('Naive Left/Right selectivity')
plt.show()

tstat, p_val = stats.ttest_ind(llearningsel, rlearningsel)
print("Learning diff p-value: ", p_val)
plt.hist(rlearningsel, bins=bins, color='b', alpha = 0.7)
plt.hist(llearningsel, bins=bins, color='r', alpha = 0.7)
plt.axvline(np.median(rlearningsel), color = 'b')
plt.axvline(np.median(llearningsel), color = 'r')
plt.title('Learning Left/Right selectivity (p-value={})'.format(p_val))
plt.show()

# T-test to see if there is a significant difference between left and right groups
tstat, p_val = stats.ttest_ind(lnaivesel, rnaivesel)
print("Expert diff p-value: ", p_val)
plt.hist(lnaivesel, bins=bins, color='r', alpha = 0.7)
plt.hist(rnaivesel, bins=bins, color='b', alpha = 0.7)
plt.axvline(np.median(rnaivesel), color = 'b')
plt.axvline(np.median(lnaivesel), color = 'r')
plt.title('Expert Left/Right selectivity (p-value={})'.format(p_val))
plt.show()



#%% Ranked neurons by selectivity over training expert --> naive

p=0.05
expertn, learningn, naiven = [],[],[]
expertsel, learningsel, naivesel = [],[],[]

for paths in agg_mice_paths:
    allsel = []
    for allpath in paths:
        path = allpath[2] # Expert session
        s1 = session.Session(path, use_reg=True, triple=True)
        
        # epoch = range(s1.response+6, s1.response+12) # Response selective
        epoch = range(s1.response-9, s1.response) # Delay selective
        # epoch = range(s1.delay-3, s1.delay+3) # Stimulus selective     
        
        order, _, (test_r, test_l) = s1.ranked_cells_by_selectivity(epoch=epoch)
        
        s1_neurons = []     
        for n in order:
            s1_neurons += [s1.good_neurons[n]]
        allstat, _, _ = s1.get_epoch_tstat(epoch, s1_neurons, rtrials=test_r, ltrials=test_l)
        expertsel += allstat
        
        ## Intermediate and naive
        path1 = allpath[1]
        s2 = session.Session(path1, use_reg=True, triple=True)
        path2 = allpath[0]
        s3 = session.Session(path2, use_reg=True, triple=True)
        
        s2_neurons = []     
        for n in order:
            s2_neurons += [s2.good_neurons[n]]
        allstat, _, _ = s2.get_epoch_tstat(epoch, s2_neurons)
        learningsel += allstat
        
        s3_neurons = []     
        for n in order:
            s3_neurons += [s3.good_neurons[n]]
        allstat, _, _ = s3.get_epoch_tstat(epoch, s3_neurons)
        naivesel += allstat

plt.plot(range(len(expertsel)), expertsel)
plt.axhline(y=0, color='grey')
plt.show()

plt.plot(range(len(learningsel)), learningsel)
plt.axhline(y=0, color='grey')
plt.show()

plt.plot(range(len(naivesel)), naivesel)
plt.axhline(y=0, color='grey')
plt.show()






