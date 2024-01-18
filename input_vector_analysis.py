# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 17:33:55 2024

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
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot
from numpy.linalg import norm

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def cos_sim(a,b):
    return np.dot(a, b)/(norm(a)*norm(b))

#%% Angle between input and CD
all_paths = [[    r'F:\data\BAYLORCW032\python\2023_10_05',
            # r'F:\data\BAYLORCW034\python\2023_10_12',
            r'F:\data\BAYLORCW036\python\2023_10_09',
            r'F:\data\BAYLORCW035\python\2023_10_26',
            r'F:\data\BAYLORCW037\python\2023_11_21',],

        [r'F:\data\BAYLORCW032\python\2023_10_19',
            # r'F:\data\BAYLORCW034\python\2023_10_22',
            r'F:\data\BAYLORCW036\python\2023_10_19',
            r'F:\data\BAYLORCW035\python\2023_12_07',
            r'F:\data\BAYLORCW037\python\2023_12_08',],


        [r'F:\data\BAYLORCW032\python\2023_10_24',
            # r'F:\data\BAYLORCW034\python\2023_10_27',
            r'F:\data\BAYLORCW036\python\2023_10_30',
            r'F:\data\BAYLORCW035\python\2023_12_15',
            r'F:\data\BAYLORCW037\python\2023_12_15',]]

all_recovery = []
for paths in all_paths: # For each stage of training
    recovery = []
    for path in paths: # For each mouse
        l1 = Mode(path, use_reg = True, triple=True)
        
        input_vector = l1.input_vector()
        indices = l1.get_stim_responsive_neurons()
        
        orthonormal_basis, mean = l1.plot_CD(mode_input='choice', plot=False)
        
        recovery += [cos_sim(input_vector,orthonormal_basis[indices])]

    all_recovery += [recovery]
    
plt.plot(range(3), [np.mean(a) for a in all_recovery], marker='x')
plt.scatter(np.zeros(len(all_recovery[0])), all_recovery[0])
plt.scatter(np.ones(len(all_recovery[1])), all_recovery[1])
plt.scatter(np.ones(len(all_recovery[2]))+1, all_recovery[2])

plt.xticks(range(3), ['Naive', 'Learning', 'Expert'])
plt.ylabel('Cosine similarity')
plt.axhline(0, color='grey', ls='--')
# plt.savefig(r'F:\data\Fig 3\modularity_bargraph.pdf')

plt.show()
stats.ttest_ind(all_recovery[1], all_recovery[2])
stats.ttest_ind(all_recovery[0], all_recovery[2])
stats.ttest_ind(all_recovery[0], all_recovery[1])

#%% Angle between input vectors across training
allpaths = [[r'F:\data\BAYLORCW032\python\2023_10_08',
          r'F:\data\BAYLORCW032\python\2023_10_16',
          r'F:\data\BAYLORCW032\python\2023_10_25',
          r'F:\data\BAYLORCW032\python\cellreg\layer{}\1008_1016_1025pairs_proc.npy'],
         
           # [ r'F:\data\BAYLORCW034\python\2023_10_12',
           #    r'F:\data\BAYLORCW034\python\2023_10_22',
           #    r'F:\data\BAYLORCW034\python\2023_10_27',
           #    r'F:\data\BAYLORCW034\python\cellreg\layer{}\1012_1022_1027pairs_proc.npy'],
         
         [r'F:\data\BAYLORCW036\python\2023_10_09',
            r'F:\data\BAYLORCW036\python\2023_10_19',
            r'F:\data\BAYLORCW036\python\2023_10_30',
            r'F:\data\BAYLORCW036\python\cellreg\layer{}\1009_1019_1030pairs_proc.npy'],
         
         [r'F:\data\BAYLORCW037\python\2023_11_21',
                     r'F:\data\BAYLORCW037\python\2023_12_08',
                     r'F:\data\BAYLORCW037\python\2023_12_15',],
         
         [r'F:\data\BAYLORCW035\python\2023_10_26',
                     r'F:\data\BAYLORCW035\python\2023_12_07',
                     r'F:\data\BAYLORCW035\python\2023_12_15',]
        ]

all_recovery = []
for paths in all_paths: # For each stage of training
    recovery = []
    l1 = Mode(paths[2], use_reg = True, triple=True) # expert
    
    expinput_vector = l1.input_vector()
    indices = l1.get_stim_responsive_neurons()
    
    # orthonormal_basis, mean = l1.plot_CD(mode_input='choice', plot=False)
    l1 = Mode(paths[1], use_reg = True, triple=True) # learning
    leainput_vector = l1.input_vector()

    l1 = Mode(paths[0], use_reg = True, triple=True) # naive
    naiinput_vector = l1.input_vector()

    all_recovery += [[cos_sim(naiinput_vector,expinput_vector),
                      cos_sim(leainput_vector,expinput_vector)]]

#%%

plt.plot(range(2), np.mean(all_recovery, axis=0), marker='x')
plt.scatter(np.zeros(len(all_recovery)), np.array(all_recovery).T[0])
plt.scatter(np.ones(len(all_recovery)), np.array(all_recovery).T[1])

plt.xticks(range(2), ['Naive:Expert', 'Learning:Expert'])
plt.ylabel('Cosine similarity')
plt.axhline(0, color='grey', ls='--')
# plt.savefig(r'F:\data\Fig 3\modularity_bargraph.pdf')

plt.show()


#%% Look at variance of input vectors:
    
opto_proj = l1.input_vector(return_opto=True)

np.mean(np.var(opto_proj, axis=0))


all_recovery = []
for paths in all_paths: # For each stage of training
    recovery = []
    for path in paths: # For each mouse
        l1 = Mode(path, use_reg = True, triple=True)
        
        opto_proj = l1.input_vector(return_opto=True)
        
        recovery += [np.mean(np.var(opto_proj, axis=0))]

    all_recovery += [recovery]
    
plt.bar(range(3), [np.mean(a) for a in all_recovery])
plt.scatter(np.zeros(len(all_recovery[0])), all_recovery[0])
plt.scatter(np.ones(len(all_recovery[1])), all_recovery[1])
plt.scatter(np.ones(len(all_recovery[2]))+1, all_recovery[2])

plt.xticks(range(3), ['Naive', 'Learning', 'Expert'])
plt.ylabel('Variance of input vec')
# plt.ylim(bottom=1.3)
# plt.savefig(r'F:\data\Fig 3\modularity_bargraph.pdf')

plt.show()
stats.ttest_ind(all_recovery[1], all_recovery[2])
stats.ttest_ind(all_recovery[0], all_recovery[2])
stats.ttest_ind(all_recovery[0], all_recovery[1])



#%% Project test trials on input vectors:

input_vector = l1.input_vector(plot=True)


    

#%% Recovery mode, Angle between recovery and CD


all_paths = [[    r'F:\data\BAYLORCW032\python\2023_10_05',
            # r'F:\data\BAYLORCW034\python\2023_10_12',
            r'F:\data\BAYLORCW036\python\2023_10_09',
            r'F:\data\BAYLORCW035\python\2023_10_26',
            r'F:\data\BAYLORCW037\python\2023_11_21',],

        [r'F:\data\BAYLORCW032\python\2023_10_19',
            # r'F:\data\BAYLORCW034\python\2023_10_22',
            r'F:\data\BAYLORCW036\python\2023_10_19',
            r'F:\data\BAYLORCW035\python\2023_12_07',
            r'F:\data\BAYLORCW037\python\2023_12_08',],


        [r'F:\data\BAYLORCW032\python\2023_10_24',
            # r'F:\data\BAYLORCW034\python\2023_10_27',
            r'F:\data\BAYLORCW036\python\2023_10_30',
            r'F:\data\BAYLORCW035\python\2023_12_15',
            r'F:\data\BAYLORCW037\python\2023_12_15',]]

all_recovery = []
for paths in all_paths: # For each stage of training
    recovery = []
    for path in paths: # For each mouse
        l1 = Mode(path, use_reg = True, triple=True)
        
        recovery_vector = l1.recovery_vector()
        orthonormal_basis, mean = l1.plot_CD(mode_input='choice', plot=False)
        
        
        recovery += [cos_sim(recovery_vector,orthonormal_basis)]

    all_recovery += [recovery]
    
plt.plot(range(3), [np.mean(a) for a in all_recovery], marker='x')
plt.scatter(np.zeros(len(all_recovery[0])), all_recovery[0])
plt.scatter(np.ones(len(all_recovery[1])), all_recovery[1])
plt.scatter(np.ones(len(all_recovery[2]))+1, all_recovery[2])

plt.xticks(range(3), ['Naive', 'Learning', 'Expert'])
plt.ylabel('Cosine similarity')
plt.axhline(0, color='grey', ls='--')

plt.show()


#%% Project test trials on recovery vectors:
l1 = Mode(path, use_reg = True, triple=True)

recovery_vector = l1.recovery_vector(plot=True)

#%% Look at variance of recovery vectors:
    
opto_proj = l1.input_vector(return_opto=True)

np.mean(np.var(opto_proj, axis=0))


all_recovery = []
for paths in all_paths: # For each stage of training
    recovery = []
    for path in paths: # For each mouse
        l1 = Mode(path, use_reg = True, triple=True)
        
        opto_proj = l1.recovery_vector(return_opto=True)

        recovery += [np.mean(np.var(opto_proj, axis=0))]

    all_recovery += [recovery]
    
plt.bar(range(3), [np.mean(a) for a in all_recovery])
plt.scatter(np.zeros(len(all_recovery[0])), all_recovery[0])
plt.scatter(np.ones(len(all_recovery[1])), all_recovery[1])
plt.scatter(np.ones(len(all_recovery[2]))+1, all_recovery[2])

plt.xticks(range(3), ['Naive', 'Learning', 'Expert'])
plt.ylabel('Variance of recovery vec')
# plt.ylim(bottom=1.3)
# plt.savefig(r'F:\data\Fig 3\modularity_bargraph.pdf')

plt.show()
# stats.ttest_ind(all_recovery[1], all_recovery[2])
# stats.ttest_ind(all_recovery[0], all_recovery[2])
# stats.ttest_ind(all_recovery[0], all_recovery[1])

#%% input+recovery angle with cd
all_paths = [[    r'F:\data\BAYLORCW032\python\2023_10_05',
            # r'F:\data\BAYLORCW034\python\2023_10_12',
            r'F:\data\BAYLORCW036\python\2023_10_09',
            r'F:\data\BAYLORCW035\python\2023_10_26',
            r'F:\data\BAYLORCW037\python\2023_11_21',],

        [r'F:\data\BAYLORCW032\python\2023_10_19',
            # r'F:\data\BAYLORCW034\python\2023_10_22',
            r'F:\data\BAYLORCW036\python\2023_10_19',
            r'F:\data\BAYLORCW035\python\2023_12_07',
            r'F:\data\BAYLORCW037\python\2023_12_08',],


        [r'F:\data\BAYLORCW032\python\2023_10_24',
            # r'F:\data\BAYLORCW034\python\2023_10_27',
            r'F:\data\BAYLORCW036\python\2023_10_30',
            r'F:\data\BAYLORCW035\python\2023_12_15',
            r'F:\data\BAYLORCW037\python\2023_12_15',]]

all_recovery = []
for paths in all_paths: # For each stage of training
    recovery = []
    for path in paths: # For each mouse
        l1 = Mode(path, use_reg = True, triple=True)
        
        recovery_vector = l1.recovery_vector()
        input_vector = l1.input_vector()

        orthonormal_basis, mean = l1.plot_CD(mode_input='choice', plot=False)
        
        
        recovery += [cos_sim(input_vector + recovery_vector,orthonormal_basis)]

    all_recovery += [recovery]

plt.plot(range(3), [np.mean(a) for a in all_recovery], marker='x')
plt.scatter(np.zeros(len(all_recovery[0])), all_recovery[0])
plt.scatter(np.ones(len(all_recovery[1])), all_recovery[1])
plt.scatter(np.ones(len(all_recovery[2]))+1, all_recovery[2])

plt.xticks(range(3), ['Naive', 'Learning', 'Expert'])
plt.ylabel('Cosine similarity')
plt.axhline(0, color='grey', ls='--')

plt.show()
