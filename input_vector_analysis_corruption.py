# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:03:41 2025

Calculate input vector but on the corruption dataset

@author: catherinewang
"""

import sys
sys.path.append("C:\scripts\Imaging analysis")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from alm_2p import session
from matplotlib.pyplot import figure
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

#%%PATHS

agg_mice_paths = [
                    [
                        r'H:\data\BAYLORCW038\python\2024_02_05',
                     'H:\\data\\BAYLORCW039\\python\\2024_04_17',
                     'H:\\data\\BAYLORCW039\\python\\2024_04_18',
                     'H:\\data\\BAYLORCW041\\python\\2024_05_14', 
                     'H:\\data\\BAYLORCW041\\python\\2024_05_13', 
                     'H:\\data\\BAYLORCW041\\python\\2024_05_15', 
                     'H:\\data\\BAYLORCW042\\python\\2024_06_05', 
                     ],
                    
                    [
                        # r'H:\data\BAYLORCW038\python\2024_02_15',
                     'H:\\data\\BAYLORCW039\\python\\2024_04_24',
                     'H:\\data\\BAYLORCW039\\python\\2024_04_25',
                     'H:\\data\\BAYLORCW041\\python\\2024_05_23',
                     'H:\\data\\BAYLORCW041\\python\\2024_05_24',
                     'H:\\data\\BAYLORCW041\\python\\2024_05_28',
                     'H:\\data\\BAYLORCW042\\python\\2024_06_14',
                     ],
                    
                    [
                        r'H:\data\BAYLORCW038\python\2024_03_15',
                     'H:\\data\\BAYLORCW039\\python\\2024_05_06',
                     'H:\\data\\BAYLORCW039\\python\\2024_05_08',
                     'H:\\data\\BAYLORCW041\\python\\2024_06_07',
                     'H:\\data\\BAYLORCW041\\python\\2024_06_12',
                     'H:\\data\\BAYLORCW041\\python\\2024_06_11',
                     'H:\\data\\BAYLORCW042\\python\\2024_06_24'
                     ]
    
                    ]

agg_mice_paths = [['H:\\data\\BAYLORCW038\\python\\2024_02_05', 
                    'H:\\data\\BAYLORCW038\\python\\2024_03_15'],
                   
                   ['H:\\data\\BAYLORCW039\\python\\2024_04_17', 
                    'H:\\data\\BAYLORCW039\\python\\2024_05_06'],
                   
                    [r'H:\\data\\BAYLORCW039\\python\\2024_04_18',
                     r'H:\\data\\BAYLORCW039\\python\\2024_05_08'],
                    
                    [r'H:\\data\\BAYLORCW041\\python\\2024_05_14',
                      r'H:\\data\\BAYLORCW041\\python\\2024_06_07'],
                    
                    [r'H:\\data\\BAYLORCW041\\python\\2024_05_13',
                      r'H:\\data\\BAYLORCW041\\python\\2024_06_12'],

                    [r'H:\\data\\BAYLORCW041\\python\\2024_05_15',
                     r'H:\\data\\BAYLORCW041\\python\\2024_06_11'],
                    
                    ['H:\\data\\BAYLORCW042\\python\\2024_06_05', 
                     'H:\\data\\BAYLORCW042\\python\\2024_06_24'],
                    
                      [r'H:\\data\\BAYLORCW043\\python\\2024_05_20',
                      r'H:\\data\\BAYLORCW043\\python\\2024_06_03'],
                    ]


                  #   [r'H:\\data\\BAYLORCW043\\python\\2024_05_21',
                  #    r'H:\\data\\BAYLORCW043\\python\\2024_06_14']
                  # ]]

#%% Example FOV
initialpath, finalpath = ['H:\\data\\BAYLORCW038\\python\\2024_02_05', 
                    'H:\\data\\BAYLORCW038\\python\\2024_03_15']

# ['H:\\data\\BAYLORCW039\\python\\2024_04_17', 
#  'H:\\data\\BAYLORCW039\\python\\2024_05_06']

l1 = Mode(initialpath, lickdir=False, use_reg = True, triple=True, proportion_opto_train=1, proportion_train=1)
input_vector_L, input_vector_R = l1.input_vector(by_trialtype=True, plot=True)
input_vec = l1.input_vector(by_trialtype=False, plot=True)
input_vec = l1.input_vector(by_trialtype=False, plot=True, plot_ctl_opto=False)
cd_choice, _ = l1.plot_CD(mode_input='choice', plot=False)



l2 = Mode(finalpath, lickdir=False, use_reg = True, triple=False, proportion_opto_train=1, proportion_train=1)
input_vector_Lexp, input_vector_Rexp = l2.input_vector(by_trialtype=True, plot=True)
input_vec = l2.input_vector(by_trialtype=False, plot=True)
input_vec = l2.input_vector(by_trialtype=False, plot=True, plot_ctl_opto=False)
cd_choice_exp, _ = l2.plot_CD(mode_input='choice', plot=False)





#%% Calculate t.t. independent input vector - all FOVs
CD_angle, rotation_learning = [], []

for paths in agg_mice_paths:

    if '43' in paths[0] or '38' in paths[0]:
        triple = False
    else:
        triple = True 
        
    l1 = Mode(paths[0], lickdir=False, use_reg = True, triple=triple, proportion_train=1, proportion_opto_train=1)
    input_vector = l1.input_vector(by_trialtype=False, plot=True)
    cd_choice, _ = l1.plot_CD(mode_input='choice', plot=False)


    l2 = Mode(paths[1], lickdir=False, use_reg = True, triple=triple, proportion_train=1, proportion_opto_train=1)
    input_vector_exp = l2.input_vector(by_trialtype=False, plot=True)
    cd_choice_exp, _ = l2.plot_CD(mode_input='choice', plot=False)

    # Angle between trial type input vector and CD
    CD_angle += [(cos_sim(input_vector, cd_choice), cos_sim(input_vector_exp, cd_choice_exp))]
    rotation_learning += [cos_sim(input_vector, input_vector_exp)]
    
CD_angle, rotation_learning = np.array(CD_angle), np.array(rotation_learning)

# Plot angle between input vectors
plt.bar([0],[np.mean(rotation_learning)])
plt.scatter(np.zeros(len(rotation_learning)), rotation_learning)
# for i in range(len(L_angles)):
#     plt.plot([0,1],[L_angles[i,0], L_angles[i,1]], color='grey')
plt.xticks([0],['learning-->expert'])
plt.ylabel('Dot product')
plt.title('Angle btw input vectors over learning')
plt.show()


# Plot angle between choice CD and input vector
plt.bar([0,1],np.mean(CD_angle, axis=0))
plt.scatter(np.zeros(len(CD_angle)), np.array(CD_angle)[:, 0])
plt.scatter(np.ones(len(CD_angle)), np.array(CD_angle)[:, 1])
for i in range(len(CD_angle)):
    plt.plot([0,1],[CD_angle[i,0], CD_angle[i,1]], color='grey')
plt.xticks([0,1],['Learning','Expert'])
plt.ylabel('Dot product')
plt.title('Input vector alignment to choice CD')
plt.show()



























