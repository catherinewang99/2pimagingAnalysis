# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 12:59:23 2023

@author: Catherine Wang
"""


import sys
sys.path.append("C:\scripts\Imaging analysis")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import session
from matplotlib.pyplot import figure
import decon
from scipy.stats import chisquare
import pandas as pd
from activityMode import Mode

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
    
    l1.selectivity_optogenetics(p=0.01)
    l1.plot_CD_opto()




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
