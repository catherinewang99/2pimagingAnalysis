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
from session import Session
from matplotlib.pyplot import figure
import decon
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
#%% Stability of susceptibility 

r_sus = []

f = plt.figure(figsize=(5,5))
for paths in all_matched_paths:
    
    intialpath, finalpath = paths[1], paths[2]
    
    # sample CD
    s1 = Session(intialpath, use_reg=True, triple=True)
    s2 = Session(finalpath, use_reg = True, triple=True)
    
    s1_sus = s1.susceptibility()
    s2_sus = s2.susceptibility()
    
    plt.scatter(s1_sus, s2_sus)
    r_sus += [stats.pearsonr(s1_sus, s2_sus)[0]]

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

#%% Plot selectivity vs sus

# for i in range(3):
i=0
f = plt.figure(figsize=(5,5))
plt.scatter(cat(agg_allsus[i][0]), cat(agg_allsel[i][0]), alpha=0.5)
plt.xlim(-5, 125)
plt.ylim(-0.5, 25)
plt.ylabel('Selectivity')
plt.xlabel('Susceptibility')














