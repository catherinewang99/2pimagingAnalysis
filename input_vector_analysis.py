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

#%% Project t.t. independent vector onto trial types




#%% Calculate input vector without trial type and project fwd
naivepath, learningpath, expertpath = [r'F:\data\BAYLORCW032\python\2023_10_05',
  r'F:\data\BAYLORCW032\python\2023_10_19',
  r'F:\data\BAYLORCW032\python\2023_10_24',]

naivepath, learningpath, expertpath = [r'H:\data\BAYLORCW046\python\2024_05_31',
                                         r'H:\data\BAYLORCW046\python\2024_06_11',
                                         r'H:\data\BAYLORCW046\python\2024_06_26'
                                         ]

l2 = Mode(expertpath, lickdir=False, use_reg = True, triple=True, proportion_train=1, proportion_opto_train=1)
input_vec, mean = l2.input_vector(by_trialtype=False, plot=True, return_applied=True)
# l2.applied_input_vector(input_vec, mean)
cd_choice_exp, _ = l2.plot_CD(mode_input='choice', plot=False)


l1 = Mode(learningpath, lickdir=False, use_reg = True, triple=True, proportion_opto_train=1)
# input_vec, mean = l1.input_vector(by_trialtype=False, plot=True, return_applied=True)
l1.applied_input_vector(input_vec, mean)
l1.applied_input_vector(input_vec, mean, plot_ctl_opto=False)
cd_choice, _ = l1.plot_CD(mode_input='choice', plot=False)




l1 = Mode(naivepath, lickdir=False, use_reg = True, triple=True, proportion_train=1, proportion_opto_train=1)
# input_vec, mean = l1.input_vector(by_trialtype=False, plot=True, return_applied=True)
l1.applied_input_vector(input_vec, mean)
l1.applied_input_vector(input_vec, mean, plot_ctl_opto=False)
cd_choice, _ = l1.plot_CD(mode_input='choice', plot=False)





#%% Calculate input vector by trial type  vs independent- one FOV
# Control -correct trials only, Opto - use trial type
naivepath, learningpath, expertpath = [r'F:\data\BAYLORCW032\python\2023_10_05',
  r'F:\data\BAYLORCW032\python\2023_10_19',
  r'F:\data\BAYLORCW032\python\2023_10_24',]
naivepath, learningpath, expertpath =[ r'F:\data\BAYLORCW034\python\2023_10_12',
   r'F:\data\BAYLORCW034\python\2023_10_22',
   r'F:\data\BAYLORCW034\python\2023_10_27']
naivepath, learningpath, expertpath = [r'H:\data\BAYLORCW046\python\2024_05_31',
                                         r'H:\data\BAYLORCW046\python\2024_06_11',
                                         r'H:\data\BAYLORCW046\python\2024_06_26'
                                         ]
l1 = Mode(naivepath, lickdir=False, use_reg = True, triple=True, proportion_opto_train=1, proportion_train=1)
input_vector_L, input_vector_R = l1.input_vector(by_trialtype=True, plot=True)
input_vec = l1.input_vector(by_trialtype=False, plot=True)
input_vec = l1.input_vector(by_trialtype=False, plot=True, plot_ctl_opto=False)
cd_choice, _ = l1.plot_CD(mode_input='choice', plot=False)



l1 = Mode(learningpath, lickdir=False, use_reg = True, triple=True, proportion_opto_train=1, proportion_train=1)
input_vector_L, input_vector_R = l1.input_vector(by_trialtype=True, plot=True)
input_vec = l1.input_vector(by_trialtype=False, plot=True)
input_vec = l1.input_vector(by_trialtype=False, plot=True, plot_ctl_opto=False)
cd_choice, _ = l1.plot_CD(mode_input='choice', plot=False)



l2 = Mode(expertpath, lickdir=False, use_reg = True, triple=True, proportion_opto_train=1, proportion_train=1)
input_vector_Lexp, input_vector_Rexp = l2.input_vector(by_trialtype=True, plot=True)
input_vec = l2.input_vector(by_trialtype=False, plot=True)
input_vec = l2.input_vector(by_trialtype=False, plot=True, plot_ctl_opto=False)
cd_choice_exp, _ = l2.plot_CD(mode_input='choice', plot=False)

# Angle between trial type input vector and CD
print(cos_sim(input_vector_L, cd_choice), cos_sim(input_vector_Lexp, cd_choice_exp))
print(cos_sim(input_vector_R, cd_choice), cos_sim(input_vector_Rexp, cd_choice_exp))
angle_lea = cos_sim(input_vector_L, cd_choice)
angle_lea = cos_sim(input_vector_R, cd_choice)

angle_exp = cos_sim(input_vector_Lexp, cd_choice_exp)
angle_exp = cos_sim(input_vector_Rexp, cd_choice_exp)

#%% Calculate t.t. independent input vector - all FOVs
CD_angle, rotation_learning = [], []
all_deltas = []

for paths in all_matched_paths:

    l1 = Mode(paths[1], lickdir=False, use_reg = True, triple=True, proportion_train=1, proportion_opto_train=1)
    input_vector, delta = l1.input_vector(by_trialtype=False, plot=True, return_delta = True)
    cd_choice, _ = l1.plot_CD(mode_input='choice', plot=False)


    l2 = Mode(paths[2], lickdir=False, use_reg = True, triple=True, proportion_train=1, proportion_opto_train=1)
    input_vector_exp, delta_exp = l2.input_vector(by_trialtype=False, plot=True, return_delta = True)
    cd_choice_exp, _ = l2.plot_CD(mode_input='choice', plot=False)

    # Angle between trial type input vector and CD
    CD_angle += [(cos_sim(input_vector, cd_choice), cos_sim(input_vector_exp, cd_choice_exp))]
    rotation_learning += [cos_sim(input_vector, input_vector_exp)]
    all_deltas += [(delta, delta_exp)]
    
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


#%% Calculate input vector by trial type - all FOVs
L_angles, R_angles = [], []
inputvector_angles_R, inputvector_angles_L = [], []

for paths in all_matched_paths:

    l1 = Mode(paths[1], lickdir=False, use_reg = True, triple=True, proportion_train=1, proportion_opto_train=1)
    input_vector_L, input_vector_R = l1.input_vector(by_trialtype=True, plot=True)
    cd_choice, _ = l1.plot_CD(mode_input='choice', plot=False)


    l2 = Mode(paths[2], lickdir=False, use_reg = True, triple=True, proportion_train=1, proportion_opto_train=1)
    input_vector_Lexp, input_vector_Rexp = l2.input_vector(by_trialtype=True, plot=True)
    cd_choice_exp, _ = l2.plot_CD(mode_input='choice', plot=False)

    # Angle between trial type input vector and CD
    L_angles += [(cos_sim(input_vector_L, cd_choice), cos_sim(input_vector_Lexp, cd_choice_exp))]
    R_angles += [(cos_sim(input_vector_R, cd_choice), cos_sim(input_vector_Rexp, cd_choice_exp))]
    inputvector_angles_R += [cos_sim(input_vector_R, input_vector_Rexp)]
    inputvector_angles_L += [cos_sim(input_vector_L, input_vector_Lexp)]
    
L_angles, R_angles = np.array(L_angles), np.array(R_angles)

# Plot angle between input vectors
plt.bar([0,1],[np.mean(inputvector_angles_L), np.mean(inputvector_angles_R)])
plt.scatter(np.zeros(len(inputvector_angles_L)), inputvector_angles_L)
plt.scatter(np.ones(len(inputvector_angles_R)), inputvector_angles_R)
# for i in range(len(L_angles)):
#     plt.plot([0,1],[L_angles[i,0], L_angles[i,1]], color='grey')
plt.xticks([0,1],['Left trials', 'Right trials'])
plt.ylabel('Dot product')
plt.title('Angle btw input vectors over learning')
plt.show()


# Plot angle between choice CD and input vector
plt.bar([0,1],np.mean(L_angles, axis=0))
plt.scatter(np.zeros(len(L_angles)), np.array(L_angles)[:, 0])
plt.scatter(np.ones(len(L_angles)), np.array(L_angles)[:, 1])
for i in range(len(L_angles)):
    plt.plot([0,1],[L_angles[i,0], L_angles[i,1]], color='grey')
plt.xticks([0,1],['Learning','Expert'])
plt.ylabel('Dot product')
plt.title('L trial input vector alignment to choice CD')
plt.show()

plt.bar([0,1],np.mean(R_angles, axis=0))
plt.scatter(np.zeros(len(R_angles)), np.array(R_angles)[:, 0])
plt.scatter(np.ones(len(R_angles)), np.array(R_angles)[:, 1])
for i in range(len(R_angles)):
    plt.plot([0,1],[R_angles[i,0], R_angles[i,1]], color='grey')
plt.xticks([0,1],['Learning', 'Expert'])
plt.ylabel('Dot product')
plt.title('R trial input vector alignment to choice CD')


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
for paths in allpaths: # For each mouse

    l1 = Mode(paths[2], use_reg = True, triple=True) # expert
    
    expinput_vector = l1.input_vector()
    indices = l1.get_stim_responsive_neurons()
    
    l1 = Mode(paths[2], use_reg = True, triple=True, responsive_neurons=indices) # expert
    orthonormal_basis, mean = l1.plot_CD(mode_input='choice', plot=False)
    
    l1 = Mode(paths[1], use_reg = True, triple=True, responsive_neurons=indices) # learning
    leainput_vector = l1.input_vector()

    l1 = Mode(paths[0], use_reg = True, triple=True, responsive_neurons=indices) # naive
    naiinput_vector = l1.input_vector()

    # all_recovery += [[cos_sim(naiinput_vector,expinput_vector),
    #                   cos_sim(naiinput_vector,leainput_vector),
    #                   cos_sim(leainput_vector,expinput_vector)]]
    
    all_recovery += [[cos_sim(orthonormal_basis,naiinput_vector),
                      cos_sim(orthonormal_basis,leainput_vector),
                      cos_sim(orthonormal_basis,expinput_vector)]]

#%%

plt.plot(range(3), np.mean(all_recovery, axis=0), marker='x')
plt.scatter(np.zeros(len(all_recovery)), np.array(all_recovery).T[0])
plt.scatter(np.ones(len(all_recovery)), np.array(all_recovery).T[1])
plt.scatter(np.ones(len(all_recovery)) + 1, np.array(all_recovery).T[2])

# plt.xticks(range(3), ['Naive:Expert', 'Naive:Learning', 'Learning:Expert'])
plt.xticks(range(3), ['Naive', 'Learning', 'Expert'])
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
