# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 12:06:24 2025

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

#%% Paths

all_matched_paths = [
    
            [r'F:\data\BAYLORCW032\python\2023_10_05',
              r'F:\data\BAYLORCW032\python\2023_10_19',
              r'F:\data\BAYLORCW032\python\2023_10_24',
          ],
         
            [ r'F:\data\BAYLORCW034\python\2023_10_12',
               r'F:\data\BAYLORCW034\python\2023_10_22',
               r'F:\data\BAYLORCW034\python\2023_10_27'],
         
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
             r'H:\data\BAYLORCW046\python\2024_06_24',
             r'H:\data\BAYLORCW046\python\2024_06_28'], # Modified dates (use 6/7, 6/24?)


            [r'H:\data\BAYLORCW046\python\2024_05_30',
             r'H:\data\BAYLORCW046\python\2024_06_10',
             r'H:\data\BAYLORCW046\python\2024_06_27'],

            [r'H:\data\BAYLORCW046\python\2024_05_31',
             r'H:\data\BAYLORCW046\python\2024_06_11',
             r'H:\data\BAYLORCW046\python\2024_06_26'
             ]
         
        ]

#%% Plot intrinsic vector on example FOV

naivepath, learningpath, expertpath = [r'F:\data\BAYLORCW032\python\2023_10_05',
  r'F:\data\BAYLORCW032\python\2023_10_19',
  r'F:\data\BAYLORCW032\python\2023_10_24',]
naivepath, learningpath, expertpath =[ r'F:\data\BAYLORCW034\python\2023_10_12',
    r'F:\data\BAYLORCW034\python\2023_10_22',
    r'F:\data\BAYLORCW034\python\2023_10_27']
# naivepath, learningpath, expertpath = [r'H:\data\BAYLORCW046\python\2024_05_31',
#                                           r'H:\data\BAYLORCW046\python\2024_06_11',
#                                           r'H:\data\BAYLORCW046\python\2024_06_26'
#                                           ]

naivepath, learningpath, expertpath = [r'H:\data\BAYLORCW044\python\2024_05_22',
                      r'H:\data\BAYLORCW044\python\2024_06_06',
                    r'H:\data\BAYLORCW044\python\2024_06_19']

l1 = Mode(naivepath, lickdir=False, use_reg = True, triple=True, proportion_opto_train=1, proportion_train=1)
intrinsic_vector, delta = l1.intrinsic_vector(return_delta=True)

# input_vector_L, input_vector_R = l1.input_vector(by_trialtype=True, plot=True)
# input_vec = l1.input_vector(by_trialtype=False, plot=True)
# input_vec = l1.input_vector(by_trialtype=False, plot=True, plot_ctl_opto=False)
# cd_choice, _ = l1.plot_CD(mode_input='choice', plot=False)



l1 = Mode(learningpath, lickdir=False, use_reg = True, triple=True, proportion_opto_train=1, proportion_train=1)
input_vec = l1.input_vector(by_trialtype=False, plot=True, plot_ctl_opto=False)
intrinsic_vector = l1.intrinsic_vector()
cd_choice, _ = l1.plot_CD(mode_input='choice', plot=False)



l2 = Mode(expertpath, lickdir=False, use_reg = True, triple=True, proportion_opto_train=1, proportion_train=1)
input_vec = l2.input_vector(by_trialtype=False, plot=True, plot_ctl_opto=False)
intrinsic_vector_exp = l2.intrinsic_vector()
cd_choice_exp, _ = l2.plot_CD(mode_input='choice', plot=False)


#%% Plot intrinsic CD over all FOVs

CD_angle, rotation_learning = [], []
all_deltas = []

for paths in all_matched_paths:
    
    l1 = Mode(paths[0], lickdir=False, use_reg = True, triple=True, proportion_train=1, proportion_opto_train=1)
    input_vector, delta_nai = l1.intrinsic_vector(return_delta=True)


    l1 = Mode(paths[1], lickdir=False, use_reg = True, triple=True, proportion_train=1, proportion_opto_train=1)
    input_vector, delta = l1.intrinsic_vector(return_delta=True)
    cd_choice, _ = l1.plot_CD(mode_input='choice', plot=False)


    l2 = Mode(paths[2], lickdir=False, use_reg = True, triple=True, proportion_train=1, proportion_opto_train=1)
    input_vector_exp, delta_exp = l2.intrinsic_vector(return_delta=True)
    cd_choice_exp, _ = l2.plot_CD(mode_input='choice', plot=False)

    # Angle between trial type input vector and CD
    CD_angle += [(cos_sim(input_vector, cd_choice), cos_sim(input_vector_exp, cd_choice_exp))]
    rotation_learning += [cos_sim(input_vector, input_vector_exp)]
    all_deltas += [(delta_nai, delta, delta_exp)]
    
CD_angle, rotation_learning, all_deltas = np.array(CD_angle), np.array(rotation_learning), np.array(all_deltas)

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

# Plot the deltas over learning
plt.bar([0,1,2],np.mean(all_deltas, axis=0))
plt.scatter(np.zeros(len(all_deltas)), np.array(all_deltas)[:, 0])
plt.scatter(np.ones(len(all_deltas)), np.array(all_deltas)[:, 1])
plt.scatter(np.ones(len(all_deltas))*2, np.array(all_deltas)[:, 2])
for i in range(len(all_deltas)):
    plt.plot([0,1],[all_deltas[i,0], all_deltas[i,1]], color='grey')
    plt.plot([1,2],[all_deltas[i,1], all_deltas[i,2]], color='grey')
plt.xticks([0,1,2],['Naive', 'Learning','Expert'])
plt.ylabel('Delta (ctl-stim)')
plt.title('Delta of input vector btw control and stim condition')
plt.show()


print(stats.ttest_rel(np.array(all_deltas)[:, 0], np.array(all_deltas)[:, 1]))
print(stats.ttest_rel(np.array(all_deltas)[:, 1], np.array(all_deltas)[:, 2]))

#%%








