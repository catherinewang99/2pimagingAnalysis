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


# Angle between input and CD
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
        orthonormal_basis, mean = l1.plot_CD(mode_input='choice', plot=False)
        
        
        recovery += [angle_between(input_vector,orthonormal_basis)]

    all_recovery += [recovery]
    
plt.bar(range(3), [np.mean(a) for a in all_recovery])
plt.scatter(np.zeros(len(all_recovery[0])), all_recovery[0])
plt.scatter(np.ones(len(all_recovery[1])), all_recovery[1])
plt.scatter(np.ones(len(all_recovery[2]))+1, all_recovery[2])

plt.xticks(range(3), ['Naive', 'Learning', 'Expert'])
plt.ylabel('Angle (rad)')
# plt.savefig(r'F:\data\Fig 3\modularity_bargraph.pdf')

plt.show()
#%% Project test trials on input vectors:



    

#%% Recovery mode
# Angle between recovery and CD




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
        
        
        recovery += [angle_between(recovery_vector,orthonormal_basis)]

    all_recovery += [recovery]
    
plt.bar(range(3), [np.mean(a) for a in all_recovery])
plt.scatter(np.zeros(len(all_recovery[0])), all_recovery[0])
plt.scatter(np.ones(len(all_recovery[1])), all_recovery[1])
plt.scatter(np.ones(len(all_recovery[2]))+1, all_recovery[2])

plt.xticks(range(3), ['Naive', 'Learning', 'Expert'])
plt.ylabel('Angle (rad)')
# plt.savefig(r'F:\data\Fig 3\modularity_bargraph.pdf')

plt.show()

#%% Project test trials on recovery vectors:


