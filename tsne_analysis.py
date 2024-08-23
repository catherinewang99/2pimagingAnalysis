# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 16:42:16 2024

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

from sklearn.manifold import TSNE
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

#%% Compute tsne using Yang et al method
paths = all_matched_paths[6]

intialpath, finalpath = paths[1], paths[2]
l1 = Mode(intialpath, use_reg=True, triple=True)
r,l = l1.get_trace_matrix_multiple(l1.good_neurons)
X = np.hstack((r,l))
X_embedded = TSNE(n_components=50, method='exact', metric='cosine', perplexity=50).fit_transform(X)

orthonormal_basis_initial, mean = l1.plot_CD(mode_input = 'stimulus')
orthonormal_basis_initial_choice, mean = l1.plot_CD(mode_input = 'choice')

# Plot the CD sample weights

f=plt.figure(figsize=(5,5))
for i in range(len(orthonormal_basis)):
    if orthonormal_basis[i] > 0:
        plt.scatter(X_embedded[i,0], X_embedded[i,1], s=[np.abs(orthonormal_basis)[i]*100], color = 'blue')
    else:
        plt.scatter(X_embedded[i,0], X_embedded[i,1], s=[np.abs(orthonormal_basis)[i]*100], color = 'red')
        
        
# Plot the CD choice weights

f=plt.figure(figsize=(5,5))
for i in range(len(orthonormal_basis_initial_choice)):
    if orthonormal_basis_initial_choice[i] > 0:
        plt.scatter(X_embedded[i,0], X_embedded[i,1], s=[np.abs(orthonormal_basis_initial_choice)[i]*100], color = 'blue')
    else:
        plt.scatter(X_embedded[i,0], X_embedded[i,1], s=[np.abs(orthonormal_basis_initial_choice)[i]*100], color = 'red')


# Plot the susceptibility score
f=plt.figure(figsize=(5,5))
for i in range(len(scores)):
    if scores[i] > 0:
        plt.scatter(X_embedded[i,0], X_embedded[i,1], s=[np.abs(scores)[i]], color = 'blue')
    else:
        plt.scatter(X_embedded[i,0], X_embedded[i,1], s=[np.abs(scores)[i]], color = 'red')



#%% Below: compute tsne using CD weights

paths = all_matched_paths[6]

intialpath, finalpath = paths[1], paths[2]

# sample CD
if '43' in paths[1] or '38' in paths[1]:
    l1 = Mode(intialpath, use_reg=True, triple=False)
    l2 = Mode(finalpath, use_reg = True, triple=False)
else:
    l1 = Mode(intialpath, use_reg=True, triple=True)
    l2 = Mode(finalpath, use_reg = True, triple=True)

orthonormal_basis_initial, mean = l1.plot_CD(mode_input = 'stimulus')
orthonormal_basis_initial_choice, mean = l1.plot_CD(mode_input = 'choice')

orthonormal_basis, mean = l2.plot_CD(mode_input = 'stimulus')
orthonormal_basis_choice, mean = l2.plot_CD(mode_input = 'choice')
    
X = np.vstack((orthonormal_basis, orthonormal_basis_choice)).T
X_embedded = TSNE(n_components=3, learning_rate='auto',init='random', perplexity=10).fit_transform(X)

# Plot the susceptibility scores

f=plt.figure(figsize=(5,5))
for i in range(len(scores)):
    if scores[i] > 0:
        plt.scatter(X_embedded[i,0], X_embedded[i,1], s=[np.abs(scores)[i]], color = 'blue')
    else:
        plt.scatter(X_embedded[i,0], X_embedded[i,1], s=[np.abs(scores)[i]], color = 'red')


X = np.vstack((orthonormal_basis_initial, orthonormal_basis_initial_choice)).T
X_embedded = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=10).fit_transform(X)

f=plt.figure(figsize=(5,5))
plt.scatter(X_embedded[:,0], X_embedded[:,1], s=np.abs(scores))

# Plot the CD weights

f=plt.figure(figsize=(5,5))
for i in range(len(orthonormal_basis)):
    if orthonormal_basis[i] > 0:
        plt.scatter(X_embedded[i,0], X_embedded[i,1], s=[np.abs(orthonormal_basis)[i]*100], color = 'blue')
    else:
        plt.scatter(X_embedded[i,0], X_embedded[i,1], s=[np.abs(orthonormal_basis)[i]*100], color = 'red')




