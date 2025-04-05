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
from scipy.stats import pearsonr
cat = np.concatenate
plt.rcParams['pdf.fonttype'] = 42 

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
         
            # [ r'F:\data\BAYLORCW034\python\2023_10_12',
            #    r'F:\data\BAYLORCW034\python\2023_10_22',
            #    r'F:\data\BAYLORCW034\python\2023_10_27'],
         
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

# Paths from beh/modularity correlation
all_paths = [[    r'F:\data\BAYLORCW032\python\2023_10_05',
            r'F:\data\BAYLORCW034\python\2023_10_12',
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
            r'F:\data\BAYLORCW034\python\2023_10_22',
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
        r'H:\data\BAYLORCW046\python\2024_06_25',
        r'H:\data\BAYLORCW046\python\2024_06_24',



        ],
        [r'F:\data\BAYLORCW032\python\2023_10_24',
            r'F:\data\BAYLORCW034\python\2023_10_27',
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
            r'H:\data\BAYLORCW044\python\2024_06_20',
            
            r'H:\data\BAYLORCW046\python\2024_06_27',
            r'H:\data\BAYLORCW046\python\2024_06_26',
            r'H:\data\BAYLORCW046\python\2024_06_28',

]]

#%% Project t.t. independent vector onto trial types




#%% Calculate input vector without trial type and project fwd
naivepath, learningpath, expertpath = [r'F:\data\BAYLORCW032\python\2023_10_05',
  r'F:\data\BAYLORCW032\python\2023_10_19',
  r'F:\data\BAYLORCW032\python\2023_10_24',]

# naivepath, learningpath, expertpath = [r'H:\data\BAYLORCW046\python\2024_05_31',
#                                           r'H:\data\BAYLORCW046\python\2024_06_11',
#                                           r'H:\data\BAYLORCW046\python\2024_06_26'
#                                           ]

# l2 = Mode(expertpath, lickdir=False, use_reg = True, triple=True, proportion_train=1, proportion_opto_train=1)
# input_vec, mean = l2.input_vector(by_trialtype=False, plot=True, return_applied=True)
# # l2.applied_input_vector(input_vec, mean)
# cd_choice_exp, _ = l2.plot_CD(mode_input='choice', plot=False)


l1 = Mode(learningpath, lickdir=False, use_reg = True, triple=True, proportion_opto_train=0.5)
input_vec, mean = l1.input_vector(by_trialtype=False, plot=True, return_applied=True, normalize=True)
                                   # save = r'H:\COSYNE 2025\CW32_learning_input_vector_projection_SEM_vxyz.pdf')
# l1 = Mode(expertpath, lickdir=False, use_reg = True, triple=True, proportion_opto_train=0.5)
# input_vec, mean = l1.input_vector(by_trialtype=False, plot=True, return_applied=True, normalize=True,
#                                   save = r'H:\COSYNE 2025\CW32_expert_input_vector_projection_SEM_vxy.pdf')
# l1.applied_input_vector(input_vec, mean)
# l1.applied_input_vector(input_vec, mean, plot_ctl_opto=False)
# cd_choice, _ = l1.plot_CD(mode_input='choice', plot=False)




# l1 = Mode(naivepath, lickdir=False, use_reg = True, triple=True, proportion_train=1, proportion_opto_train=1)
# # input_vec, mean = l1.input_vector(by_trialtype=False, plot=True, return_applied=True)
# l1.applied_input_vector(input_vec, mean)
# l1.applied_input_vector(input_vec, mean, plot_ctl_opto=False)
# cd_choice, _ = l1.plot_CD(mode_input='choice', plot=False)





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
decoding_acc = []
cd_delta = []
input_cd = []

for paths in all_matched_paths:
    
    l1 = Mode(paths[0], lickdir=False, use_reg = True, triple=True, proportion_train=1, proportion_opto_train=1)
    input_vector_nai, delta_nai = l1.input_vector(by_trialtype=False, plot=True, return_delta = True)
    _, cd_delta_nai = l1.input_vector(by_trialtype=False, plot=True, return_delta = True, plot_ctl_opto=False)


    l1 = Mode(paths[1], lickdir=False, use_reg = True, triple=True, proportion_train=1, proportion_opto_train=1)
    input_vector, delta = l1.input_vector(by_trialtype=False, plot=True, return_delta = True)
    cd_choice, _ = l1.plot_CD(mode_input='choice', plot=False)
    _, cd_delta_lea = l1.input_vector(by_trialtype=False, plot=True, return_delta = True, plot_ctl_opto=False)


    l2 = Mode(paths[2], lickdir=False, use_reg = True, triple=True, proportion_train=1, proportion_opto_train=1)
    input_vector_exp, delta_exp = l2.input_vector(by_trialtype=False, plot=True, return_delta = True)
    cd_choice_exp, _ = l2.plot_CD(mode_input='choice', plot=False)
    _, cd_delta_exp = l2.input_vector(by_trialtype=False, plot=True, return_delta = True, plot_ctl_opto=False)

    # Angle between trial type input vector and CD
    CD_angle += [(cos_sim(input_vector, cd_choice), cos_sim(input_vector_exp, cd_choice_exp))]
    rotation_learning += [(cos_sim(input_vector_nai, input_vector), cos_sim(input_vector, input_vector_exp))]
    all_deltas += [(delta_nai, delta, delta_exp)]
    cd_delta += [(cd_delta_nai, cd_delta_lea, cd_delta_exp)]
    input_cd += [(input_vector_nai, input_vector, input_vector_exp)]
    
CD_angle, rotation_learning, all_deltas, cd_delta = np.array(CD_angle), np.array(rotation_learning), np.array(all_deltas), np.array(np.abs(cd_delta))
input_cd = np.array(input_cd)


# Plot angle between input vectors
# plt.bar([0],[np.mean(rotation_learning)])
# plt.scatter(np.zeros(len(rotation_learning)), rotation_learning)
# for i in range(len(L_angles)):
#     plt.plot([0,1],[L_angles[i,0], L_angles[i,1]], color='grey')

plt.bar([0],np.mean(rotation_learning[:,1]), color='grey')

# plt.scatter(np.zeros(len(rotation_learning)), np.array(rotation_learning)[:, 0])
jitter_strength = 0.02
x_jittered = np.zeros(len(rotation_learning)) + np.random.normal(0, jitter_strength, size=len(np.zeros(len(rotation_learning))))
plt.scatter(x_jittered, np.array(rotation_learning)[:, 1], color = '#BE1E2D')
# for i in range(len(rotation_learning)):
#     plt.plot([0,1],[rotation_learning[i,0], rotation_learning[i,1]], color='grey')
# plt.xticks([0,1],['naive-->learning', 'learning-->expert'])
plt.ylabel('Dot product')
plt.title('Angle btw input vectors over learning')
plt.ylim(top=1, bottom=-1)
plt.savefig(r'H:\COSYNE 2025\input_vector_rotation_mod_jittered.pdf')
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
plt.savefig(r'H:\COSYNE 2025\input_vector_choicecd_angle.pdf')
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
stats.ttest_rel(np.array(all_deltas)[:, 1], np.array(all_deltas)[:, 2])

# Plot the correlation between cd delta and input delta
f=plt.figure()
plt.scatter(cd_delta[:,0], all_deltas[:,0], color='orange', label='Naive')
plt.scatter(cd_delta[:,1], all_deltas[:,1], color='purple', label='Learning')
plt.ylabel('Input vector delta')
plt.xlabel('Choice CD delta')
plt.legend()
plt.show()
r_value, p_value = pearsonr(cd_delta[:,0], all_deltas[:,0])
print(r_value, p_value)
r_value, p_value = pearsonr(cd_delta[:,1], all_deltas[:,1])
print(r_value, p_value)

# Plot the weights of the CDs as histograms across stages 
f, ax = plt.subplots(1,3, sharex='row', figsize=(10,6))

for i in range(len(input_cd)):
    
    ax[0].hist(np.abs(input_cd[i][0]))
    ax[1].hist(np.abs(input_cd[i][1]))
    ax[2].hist(np.abs(input_cd[i][2]))

# - against t-values of susc?

#%% Look at the delta over timecourse of trial average across FOVs

CD_angle, rotation_learning = [], []
all_deltas = []
decoding_acc = []
cd_delta = []
proportion_train, proportion_opto_train=0.5,0.5

for paths in all_matched_paths:
    
    l1 = Mode(paths[0], lickdir=False, use_reg = True, triple=True,
                proportion_train=proportion_train, proportion_opto_train=proportion_opto_train)
               # proportion_train=1, proportion_opto_train=1)
    input_vector_nai, delta_nai = l1.input_vector(by_trialtype=False, plot=True, return_delta = True, timecourse=True, normalize=False)
    # _, cd_delta_nai = l1.input_vector(by_trialtype=False, plot=True, return_delta = True, plot_ctl_opto=False)


    l1 = Mode(paths[1], lickdir=False, use_reg = True, triple=True, 
                proportion_train=proportion_train, proportion_opto_train=proportion_opto_train)
               # proportion_train=1, proportion_opto_train=1)

    input_vector, delta = l1.input_vector(by_trialtype=False, plot=True, return_delta = True, timecourse=True, normalize=False)
    # cd_choice, _ = l1.plot_CD(mode_input='choice', plot=False)
    # _, cd_delta_lea = l1.input_vector(by_trialtype=False, plot=True, return_delta = True, plot_ctl_opto=False)


    l2 = Mode(paths[2], lickdir=False, use_reg = True, triple=True, 
                proportion_train=proportion_train, proportion_opto_train=proportion_opto_train)
               # proportion_train=1, proportion_opto_train=1)

    input_vector_exp, delta_exp = l2.input_vector(by_trialtype=False, plot=True, return_delta = True, timecourse=True, normalize=False)
    # cd_choice_exp, _ = l2.plot_CD(mode_input='choice', plot=False)
    # _, cd_delta_exp = l2.input_vector(by_trialtype=False, plot=True, return_delta = True, plot_ctl_opto=False)

    # Angle between trial type input vector and CD
    # CD_angle += [(cos_sim(input_vector, cd_choice), cos_sim(input_vector_exp, cd_choice_exp))]
    # rotation_learning += [cos_sim(input_vector, input_vector_exp)]
    all_deltas += [[delta_nai, delta, delta_exp]]
    
# CD_angle, rotation_learning, all_deltas = np.array(CD_angle), np.array(rotation_learning), np.array(all_deltas)

# all_deltas = np.array(all_deltas)


f,ax=plt.subplots(1,3, figsize=(12,5), sharey='row')
for i in range(len(all_deltas)):

    if len(all_deltas[i][0]) > 100:
        
        if np.mean(all_deltas[i][2][int((3.8+0.5)*15):int((3.8+1.3)*15)]) < 0:
            all_deltas[i][2] = -all_deltas[i][2]
            
        if np.mean(all_deltas[i][1][int((3.8+0.5)*15):int((3.8+1.3)*15)]) < 0:
            all_deltas[i][1] = -all_deltas[i][1]
            
        if np.mean(all_deltas[i][0][int((3.8+0.5)*15):int((3.8+1.3)*15)]) < 0:
            all_deltas[i][0] = -all_deltas[i][0]
            
        x = np.arange(-6.97,4,1/15)[:151]

        ax[2].plot(x, all_deltas[i][2][:151], color='purple', alpha=0.15)
    else:
        x = np.arange(-6.97,4,1/6)[:61]
        if np.mean(all_deltas[i][2][int((3.8+0.5)*6):int((3.8+1.3)*6)]) < 0:
            all_deltas[i][2] = -all_deltas[i][2]
            
        if np.mean(all_deltas[i][1][int((3.8+0.5)*6):int((3.8+1.3)*6)]) < 0:
            all_deltas[i][1] = -all_deltas[i][1]
            
        if np.mean(all_deltas[i][0][int((3.8+0.5)*6):int((3.8+1.3)*6)]) < 0:
            all_deltas[i][0] = -all_deltas[i][0]
            
            
        ax[2].plot(x, all_deltas[i][2], color='purple', alpha=0.15)

    ax[0].plot(x, all_deltas[i][0], color='purple', alpha=0.15)
    ax[1].plot(x, all_deltas[i][1], color='purple', alpha=0.15)

for i in range(3):
    ax[i].axhline(0, ls='--', color='grey')
    ax[i].axvline(-4.3, ls='--', color='grey')
    ax[i].axvline(-3, ls='--', color='grey')
    ax[i].axvline(0, ls='--', color='grey')

ndelta, ldelta, edelta = [],[],[]
for i in range(len(all_deltas)):
    if len(all_deltas[i][0]) > 75:
        ndelta += [l1.dodownsample(all_deltas[i][0])[0]]
        ldelta += [l1.dodownsample(all_deltas[i][1])[0]]
        edelta += [l1.dodownsample(all_deltas[i][2][:151])[0]]

    else:
        ndelta += [all_deltas[i][0]]
        ldelta += [all_deltas[i][1]]
        edelta += [all_deltas[i][2]]

ax[0].plot(np.arange(-6.97,4,1/6)[:61], np.mean(ndelta, axis=0), color='purple')
ax[1].plot(np.arange(-6.97,4,1/6)[:61], np.mean(ldelta, axis=0), color='purple')
ax[2].plot(np.arange(-6.97,4,1/6)[:61], np.mean(edelta, axis=0), color='purple')

ax[0].set_ylabel('Delta between control vs stim')
ax[1].set_xlabel('Time from Go cue (s)')
# plt.savefig(r'H:\COSYNE 2025\inputvectordelta_timecourse.pdf')
plt.show()
# Bar plot the peak of every FOV
naive_peaks = [max(n[int(3.8*6):]) for n in ndelta]
learning_peaks = [max(n[int(3.8*6):]) for n in ldelta]
expert_peaks = [max(n[int(3.8*6):]) for n in edelta]

plt.bar([0,1,2], [np.mean(naive_peaks), np.mean(learning_peaks), np.mean(expert_peaks)])
plt.scatter(np.zeros(len(naive_peaks)), naive_peaks)
plt.scatter(np.ones(len(learning_peaks)), learning_peaks)
plt.scatter(np.ones(len(expert_peaks)) * 2, expert_peaks)
plt.ylabel('Peak deltas')
plt.xticks([0,1,2],['Naive', 'Learning', 'Expert'])
# plt.savefig(r'H:\COSYNE 2025\inputvectordelta_peak.pdf')
plt.show()


# Bar plot the delta during  the stim window, averaged
naive_peaks = [np.mean(n[int((3.8+0.5)*6):int((3.8+1.3)*6)]) for n in ndelta]
learning_peaks = [np.mean(n[int((3.8+0.5)*6):int((3.8+1.3)*6)]) for n in ldelta]
expert_peaks = [np.mean(n[int((3.8+0.5)*6):int((3.8+1.3)*6)]) for n in edelta]

plt.bar([0,1,2], [np.mean(naive_peaks), np.mean(learning_peaks), np.mean(expert_peaks)])
plt.scatter(np.zeros(len(naive_peaks)), naive_peaks)
plt.scatter(np.ones(len(learning_peaks)), learning_peaks)
plt.scatter(np.ones(len(expert_peaks)) * 2, expert_peaks)
plt.ylabel('Average deltas')
plt.xticks([0,1,2],['Naive', 'Learning', 'Expert'])
# plt.savefig(r'H:\COSYNE 2025\inputvectordelta_mean.pdf')
plt.show()

#%% instead of delta, look at all traces concatenated together?



#%% Correlate behavior recovery, robustness, etc.
CD_angle, deltas, beh, frac, modularity , cd_proj_delta = [],[],[],[],[],[]

for paths in all_matched_paths:
# for fov in range(len(all_paths[0])):
    # l1 = Mode(paths[0], lickdir=False, use_reg = True, triple=True, proportion_train=1, proportion_opto_train=1)
    # input_vector_nai, delta_nai = l1.input_vector(by_trialtype=False, plot=True, return_delta = True)
    # _, cd_delta_nai = l1.input_vector(by_trialtype=False, plot=True, return_delta = True, plot_ctl_opto=False)

    l1 = Mode(paths[1], lickdir=False, use_reg = True, triple=True, proportion_train=1, proportion_opto_train=1)
    # l1.exclude_sample_orthog = True
    input_vector, delta = l1.input_vector(by_trialtype=False, plot=True, return_delta = True)
    cd_choice, _ = l1.plot_CD(mode_input='stimulus', plot=False)
    _, cd_delta_lea = l1.input_vector(by_trialtype=False, plot=True, return_delta = True, plot_ctl_opto=False)


    l2 = Mode(paths[2], lickdir=False, use_reg = True, triple=True, proportion_train=1, proportion_opto_train=1)
    # l2.exclude_sample_orthog = True
    input_vector_exp, delta_exp = l2.input_vector(by_trialtype=False, plot=True, return_delta = True)
    cd_choice_exp, _ = l2.plot_CD(mode_input='stimulus', plot=False)
    _, cd_delta_exp = l2.input_vector(by_trialtype=False, plot=True, return_delta = True, plot_ctl_opto=False)

    adjusted_p = 0.01

    temp, _ = l1.modularity_proportion(p=adjusted_p, 
                                       period = range(l1.delay+int(2/l1.fs), l1.response),
                                       # exclude_unselective=st > 0,
                                       lickdir=False,
                                       bootstrap=True)
    
    temp_exp, _ = l2.modularity_proportion(p=adjusted_p, 
                                       period = range(l1.delay+int(2/l1.fs), l1.response),
                                       # exclude_unselective=st > 0,
                                       lickdir=False,
                                       bootstrap=True)  
    ## BEHAVIOR PERFORMANCE 
    stim_trials = np.where(l1.stim_ON)[0]
    control_trials = np.where(~l1.stim_ON)[0]
    stim_trials = [c for c in stim_trials if c in l1.i_good_trials]
    stim_trials = [c for c in stim_trials if ~l1.early_lick[c]]
    control_trials = [c for c in control_trials if c in l1.i_good_trials]
    control_trials = [c for c in control_trials if ~l1.early_lick[c]]
    
    _, _, perf_all = l1.performance_in_trials(stim_trials)
    _, _, perf_all_c = l1.performance_in_trials(control_trials)

    if perf_all_c < 0.5: #or perf_all / perf_all_c > 1: #Skip low performance sessions
        print(l1.path)
        continue
    
    stim_trials = np.where(l2.stim_ON)[0]
    control_trials = np.where(~l2.stim_ON)[0]
    stim_trials = [c for c in stim_trials if c in l2.i_good_trials]
    stim_trials = [c for c in stim_trials if ~l2.early_lick[c]]
    control_trials = [c for c in control_trials if c in l2.i_good_trials]
    control_trials = [c for c in control_trials if ~l2.early_lick[c]]
    
    _, _, perf_all_exp = l2.performance_in_trials(stim_trials)
    _, _, perf_all_c_exp = l2.performance_in_trials(control_trials)

    if perf_all_c_exp < 0.5: #or perf_all / perf_all_c > 1: #Skip low performance sessions
        print(l2.path)
        continue
    
    modularity += [(temp, temp_exp)]
    deltas += [(perf_all - perf_all_c, perf_all_exp - perf_all_c_exp)]
    frac += [(perf_all / perf_all_c, perf_all_exp / perf_all_c_exp)]
    beh += [(perf_all_c, perf_all_c_exp)]
    cd_proj_delta += [(cd_delta_lea, cd_delta_exp)]
    # Angle between trial type input vector and CD
    CD_angle += [(cos_sim(input_vector, cd_choice), cos_sim(input_vector_exp, cd_choice_exp))]

CD_angle, deltas, beh, frac = np.array(CD_angle), np.array(deltas), np.array(beh), np.array(frac)
cd_proj_delta, modularity = np.array(cd_proj_delta), np.array(modularity)




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

idx = [0,1,2,4,5,7,8]


plt.bar([0,1],np.mean(cd_proj_delta[idx, :], axis=0))
plt.scatter(np.zeros(len(cd_proj_delta[idx, :])), np.array(cd_proj_delta)[idx, 0])
plt.scatter(np.ones(len(cd_proj_delta[idx, :])), np.array(cd_proj_delta)[idx, 1])
for i in idx:
    plt.plot([0,1],[cd_proj_delta[i,0], cd_proj_delta[i,1]], color='grey')
plt.xticks([0,1],['Learning','Expert'])
plt.ylabel('Selectivity projection delta')
plt.title('Input vector alignment to choice CD')
plt.show()

## LOOK FOR CORRELATIONS

# Plot correlation between angle and behavioral recovery

plt.scatter(np.array(CD_angle)[:, 0], deltas[:, 0], label='Learning')
plt.scatter(np.array(CD_angle)[:, 1], deltas[:, 1], label='Expert')
plt.xlabel('Angle between cd_choice and input')
plt.ylabel('Behavior recovery')
plt.title('Angle vs behavior')
plt.legend()
plt.show()


r_value, p_value = pearsonr(CD_angle[:,0], deltas[:,0])
print(r_value, p_value)
r_value, p_value = pearsonr(CD_angle[:,1], deltas[:,1])
print(r_value, p_value)
r_value, p_value = pearsonr(cat(CD_angle), cat(deltas))
print(r_value, p_value)


# Plot correlation between diff in angle and behavioral recovery over learning

plt.scatter(np.array(CD_angle)[:, 1] - np.array(CD_angle)[:, 0], deltas[:, 1] - deltas[:, 0])
# plt.scatter(np.array(CD_angle)[:, 1], deltas[:, 1], label='Expert')
plt.xlabel('Angle between cd_choice and input')
plt.ylabel('Behavior recovery')
plt.title('Angle vs behavior')
# plt.legend()
plt.show()


r_value, p_value = pearsonr(CD_angle[:,0], deltas[:,0])
print(r_value, p_value)
r_value, p_value = pearsonr(CD_angle[:,1], deltas[:,1])
print(r_value, p_value)
r_value, p_value = pearsonr(cat(CD_angle), cat(deltas))
print(r_value, p_value)


# Plot correlation between angle and behavioral performance
plt.scatter(np.array(CD_angle)[:, 0], beh[:, 0], label='Learning')
plt.scatter(np.array(CD_angle)[:, 1], beh[:, 1], label='EXpert')
plt.xlabel('Angle between cd_choice and input')
plt.ylabel('Behavior')
plt.title('Angle vs behavior')
plt.legend()
plt.show()
r_value, p_value = pearsonr(CD_angle[:,0], beh[:,0])
print(r_value, p_value)
r_value, p_value = pearsonr(CD_angle[:,1], beh[:,1])
print(r_value, p_value)
r_value, p_value = pearsonr(cat(CD_angle), cat(beh))
print(r_value, p_value)


# Plot correlation between angle and robustness
idx = [0,1,2,4,5,7,8]
idx = np.arange(9)
plt.scatter(np.array(CD_angle)[idx, 0], modularity[idx, 0], label='Learning')
plt.scatter(np.array(CD_angle)[idx, 1], modularity[idx, 1], label='EXpert')
plt.xlabel('Angle between cd_choice and input')
plt.ylabel('Modularity')
plt.title('Alignment vs modularity')
plt.legend()
plt.show()
r_value, p_value = pearsonr(CD_angle[idx,0], modularity[idx,0])
print(r_value, p_value)
r_value, p_value = pearsonr(CD_angle[idx,1], modularity[idx,1])
print(r_value, p_value)
r_value, p_value = pearsonr(cat(CD_angle[idx]), cat(modularity[idx]))
print(r_value, p_value)


### Plot all with projection on CD instead of dot product

plt.scatter(cd_proj_delta[:, 0], deltas[:, 0], label='Learning')
plt.scatter(cd_proj_delta[:, 1], deltas[:, 1], label='Expert')
plt.xlabel('Selectivity projection delta')
plt.ylabel('Behavior recovery')
plt.title('Angle vs behavior')
plt.legend()
plt.show()

r_value, p_value = pearsonr(cd_proj_delta[:,0], deltas[:,0])
print(r_value, p_value)
r_value, p_value = pearsonr(cd_proj_delta[:,1], deltas[:,1])
print(r_value, p_value)
r_value, p_value = pearsonr(cat(cd_proj_delta), cat(deltas))
print(r_value, p_value)


plt.scatter(np.array(cd_proj_delta)[:, 0], beh[:, 0], label='Learning')
plt.scatter(np.array(cd_proj_delta)[:, 1], beh[:, 1], label='EXpert')
plt.xlabel('Selectivity projection delta')
plt.ylabel('Behavior')
plt.title('Angle vs behavior')
plt.legend()
plt.show()
r_value, p_value = pearsonr(cd_proj_delta[:,0], beh[:,0])
print(r_value, p_value)
r_value, p_value = pearsonr(cd_proj_delta[:,1], beh[:,1])
print(r_value, p_value)
r_value, p_value = pearsonr(cat(cd_proj_delta), cat(beh))
print(r_value, p_value)

# Plot correlation between angle and robustness
idx = [0,1,2,4,5,7,8]
idx = np.arange(9)
plt.scatter(np.array(cd_proj_delta)[idx, 0], modularity[idx, 0], label='Learning')
plt.scatter(np.array(cd_proj_delta)[idx, 1], modularity[idx, 1], label='EXpert')
plt.xlabel('Selectivity projection delta')
plt.ylabel('Modularity')
plt.title('Alignment vs modularity')
plt.legend()
plt.show()
r_value, p_value = pearsonr(cd_proj_delta[idx,0], modularity[idx,0])
print(r_value, p_value)
r_value, p_value = pearsonr(cd_proj_delta[idx,1], modularity[idx,1])
print(r_value, p_value)
r_value, p_value = pearsonr(cat(cd_proj_delta[idx]), cat(modularity[idx]))
print(r_value, p_value)

#%% Correlate behavior recovery, robustness, etc. UNMATCHED
CD_angle, deltas, beh, frac, modularity , cd_proj_delta = [],[],[],[],[],[]
cd_opto_proj_delta = []
for pid in [1,2]:
    CD_angle_, deltas_, beh_, frac_, modularity_, cd_proj_delta_ = [],[],[],[],[],[]
    cd_opto_proj_delta_ = []
    for path in all_paths[pid]:
        # l1 = Mode(paths[0], lickdir=False, use_reg = True, triple=True, proportion_train=1, proportion_opto_train=1)
        # input_vector_nai, delta_nai = l1.input_vector(by_trialtype=False, plot=True, return_delta = True)
        # _, cd_delta_nai = l1.input_vector(by_trialtype=False, plot=True, return_delta = True, plot_ctl_opto=False)
    
        l1 = Mode(path, lickdir=False, proportion_train=1, proportion_opto_train=1)
        input_vector, delta = l1.input_vector(by_trialtype=False, plot=True, return_delta = True)
        cd_choice, _ = l1.plot_CD(mode_input='choice', plot=False)
        _, cd_delta_lea = l1.input_vector(by_trialtype=False, plot=True, return_delta = True, plot_ctl_opto=False)
    
        _, mean, meantrain, meanstd = l1.plot_CD_opto(mode_input='choice', plot=False, return_applied=True)
    
        opto_delta = l1.plot_CD_opto_applied(input_vector, mean, meantrain, meanstd, return_delta=True)
        
        
        # l2 = Mode(paths[2], lickdir=False, use_reg = True, triple=True, proportion_train=1, proportion_opto_train=1)
        # input_vector_exp, delta_exp = l2.input_vector(by_trialtype=False, plot=True, return_delta = True)
        # cd_choice_exp, _ = l2.plot_CD(mode_input='choice', plot=False)
        # _, cd_delta_exp = l2.input_vector(by_trialtype=False, plot=True, return_delta = True, plot_ctl_opto=False)
    
        # adjusted_p = 0.01
        adjusted_p = 0.05 / np.sqrt(len(l1.good_neurons))
    
    
        temp, _ = l1.modularity_proportion(p=adjusted_p, 
                                           period = range(l1.delay+int(2/l1.fs), l1.response),
                                           # exclude_unselective=st > 0,
                                           lickdir=False,
                                           bootstrap=True)
        
        
        ## BEHAVIOR PERFORMANCE 
        stim_trials = np.where(l1.stim_ON)[0]
        control_trials = np.where(~l1.stim_ON)[0]
        stim_trials = [c for c in stim_trials if c in l1.i_good_trials]
        stim_trials = [c for c in stim_trials if ~l1.early_lick[c]]
        control_trials = [c for c in control_trials if c in l1.i_good_trials]
        control_trials = [c for c in control_trials if ~l1.early_lick[c]]
        
        _, _, perf_all = l1.performance_in_trials(stim_trials)
        _, _, perf_all_c = l1.performance_in_trials(control_trials)
    
        if perf_all_c < 0.5: #or perf_all / perf_all_c > 1: #Skip low performance sessions
            print(l1.path)
            continue
        
        
        
        modularity_ += [temp]
        deltas_ += [perf_all - perf_all_c]
        frac_ += [perf_all / perf_all_c]
        beh_ += [perf_all_c]
        cd_proj_delta_ += [cd_delta_lea]
        # Angle between trial type input vector and CD
        CD_angle_ += [cos_sim(input_vector, cd_choice)]
        cd_opto_proj_delta_ += [opto_delta]
        
        
    modularity += [modularity_]
    deltas += [deltas_]
    frac += [frac_]
    beh += [beh_]
    cd_proj_delta += [cd_proj_delta_]
    # Angle between trial type input vector and CD
    CD_angle += [CD_angle_]
    cd_opto_proj_delta += [cd_opto_proj_delta_]
    
CD_angle, deltas, beh, frac = np.array(CD_angle), np.array(deltas), np.array(beh), np.array(frac)
cd_proj_delta, modularity,cd_opto_proj_delta = np.array(cd_proj_delta), np.array(modularity),np.array(cd_opto_proj_delta)


#%% Plot the correlations

# Plot angle between choice CD and input vector
plt.bar([0],np.mean(CD_angle))
plt.scatter(np.zeros(len(CD_angle)), np.array(CD_angle))
# plt.scatter(np.ones(len(CD_angle)), np.array(CD_angle)[:, 1])
# for i in range(len(CD_angle)):
#     plt.plot([0,1],[CD_angle[i,0], CD_angle[i,1]], color='grey')
# plt.xticks([0,1],['Learning','Expert'])
plt.ylabel('Dot product')
plt.title('Input vector alignment to choice CD')
plt.show()

idx = [0,1,2,4,5,7,8]

plt.bar([0],np.mean(cd_proj_delta))
plt.scatter(np.zeros(len(cd_proj_delta)), np.array(cd_proj_delta))
# plt.scatter(np.ones(len(cd_proj_delta[idx, :])), np.array(cd_proj_delta)[idx, 1])
# for i in idx:
#     plt.plot([0,1],[cd_proj_delta[i,0], cd_proj_delta[i,1]], color='grey')
# plt.xticks([0,1],['Learning','Expert'])
plt.ylabel('Selectivity projection delta')
plt.title('Input vector alignment to choice CD')
plt.show()


plt.bar([0],np.mean(cd_opto_proj_delta))
plt.scatter(np.zeros(len(cd_opto_proj_delta)), np.array(cd_opto_proj_delta))
# plt.scatter(np.ones(len(cd_proj_delta[idx, :])), np.array(cd_proj_delta)[idx, 1])
# for i in idx:
#     plt.plot([0,1],[cd_proj_delta[i,0], cd_proj_delta[i,1]], color='grey')
# plt.xticks([0,1],['Learning','Expert'])
plt.ylabel('Opto CD projection delta')
plt.title('Input vector alignment to choice CD')
plt.show()

## LOOK FOR CORRELATIONS ### (pls)

# Plot correlation between angle and behavioral recovery

plt.scatter(np.array(CD_angle), deltas, label='Learning')
plt.xlabel('Angle between cd_choice and input')
plt.ylabel('Behavior recovery')
plt.title('Angle vs behavior')
plt.legend()
plt.show()


r_value, p_value = pearsonr(CD_angle, deltas)
print(r_value, p_value)
# r_value, p_value = pearsonr(CD_angle[:,1], deltas[:,1])
# print(r_value, p_value)
# r_value, p_value = pearsonr(cat(CD_angle), cat(deltas))
# print(r_value, p_value)

# Plot correlation between angle and behavioral performance
plt.scatter(np.array(CD_angle), beh, label='Learning')
# plt.scatter(np.array(CD_angle)[:, 1], beh[:, 1], label='EXpert')
plt.xlabel('Angle between cd_choice and input')
plt.ylabel('Behavior')
plt.title('Angle vs behavior')
plt.legend()
plt.show()
r_value, p_value = pearsonr(CD_angle, beh)
print(r_value, p_value)
# r_value, p_value = pearsonr(CD_angle[:,1], beh[:,1])
# print(r_value, p_value)
# r_value, p_value = pearsonr(cat(CD_angle), cat(beh))
# print(r_value, p_value)


# Plot correlation between angle and robustness
idx = [0,1,2,4,5,7,8]
idx = np.arange(9)
plt.scatter(np.array(CD_angle), modularity, label='Learning')
# plt.scatter(np.array(CD_angle)[idx, 1], modularity[idx, 1], label='EXpert')
plt.xlabel('Angle between cd_choice and input')
plt.ylabel('Modularity')
plt.title('Alignment vs modularity')
plt.legend()
plt.show()
r_value, p_value = pearsonr(CD_angle, modularity)
print(r_value, p_value)
# r_value, p_value = pearsonr(CD_angle[idx,1], modularity[idx,1])
# print(r_value, p_value)
# r_value, p_value = pearsonr(cat(CD_angle[idx]), cat(modularity[idx]))
# print(r_value, p_value)


### Plot all with projection on CD instead of dot product

plt.scatter(cd_proj_delta, deltas, label='Learning')
# plt.scatter(cd_proj_delta[:, 1], deltas[:, 1], label='Expert')
plt.xlabel('Selectivity projection delta')
plt.ylabel('Behavior recovery')
plt.title('Angle vs behavior')
plt.legend()
plt.show()

# r_value, p_value = pearsonr(cd_proj_delta[:,0], deltas[:,0])
# print(r_value, p_value)
# r_value, p_value = pearsonr(cd_proj_delta[:,1], deltas[:,1])
# print(r_value, p_value)
# r_value, p_value = pearsonr(cat(cd_proj_delta), cat(deltas))
# print(r_value, p_value)


plt.scatter(np.array(cd_proj_delta), beh, label='Learning')
# plt.scatter(np.array(cd_proj_delta)[:, 1], beh[:, 1], label='EXpert')
plt.xlabel('Selectivity projection delta')
plt.ylabel('Behavior')
plt.title('Angle vs behavior')
plt.legend()
plt.show()
# r_value, p_value = pearsonr(cd_proj_delta[:,0], beh[:,0])
# print(r_value, p_value)
# r_value, p_value = pearsonr(cd_proj_delta[:,1], beh[:,1])
# print(r_value, p_value)
# r_value, p_value = pearsonr(cat(cd_proj_delta), cat(beh))
# print(r_value, p_value)

# Plot correlation between angle and robustness
idx = [0,1,2,4,5,7,8]
idx = np.arange(9)
plt.scatter(np.array(cd_proj_delta), modularity, label='Learning')
# plt.scatter(np.array(cd_proj_delta)[idx, 1], modularity[idx, 1], label='EXpert')
plt.xlabel('Selectivity projection delta')
plt.ylabel('Modularity')
plt.title('Alignment vs modularity')
plt.legend()
plt.show()
# r_value, p_value = pearsonr(cd_proj_delta[idx,0], modularity[idx,0])
# print(r_value, p_value)
# r_value, p_value = pearsonr(cd_proj_delta[idx,1], modularity[idx,1])
# print(r_value, p_value)
# r_value, p_value = pearsonr(cat(cd_proj_delta[idx]), cat(modularity[idx]))
# print(r_value, p_value)

### Plot all with projection on CD OPTO instead of dot product

plt.scatter(cd_opto_proj_delta, deltas, label='Learning')
# plt.scatter(cd_proj_delta[:, 1], deltas[:, 1], label='Expert')
plt.xlabel('Opto CD projection delta')
plt.ylabel('Behavior recovery')
plt.title('Angle vs behavior')
plt.legend()
plt.show()

r_value, p_value = pearsonr(cd_opto_proj_delta, deltas)
print(r_value, p_value)
# r_value, p_value = pearsonr(cd_proj_delta[:,1], deltas[:,1])
# print(r_value, p_value)
# r_value, p_value = pearsonr(cat(cd_proj_delta), cat(deltas))
# print(r_value, p_value)


plt.scatter(np.array(cd_opto_proj_delta), beh, label='Learning')
# plt.scatter(np.array(cd_proj_delta)[:, 1], beh[:, 1], label='EXpert')
plt.xlabel('Opto CD projection delta')
plt.ylabel('Behavior')
plt.title('Angle vs behavior')
plt.legend()
plt.show()
# r_value, p_value = pearsonr(cd_proj_delta[:,0], beh[:,0])
# print(r_value, p_value)
# r_value, p_value = pearsonr(cd_proj_delta[:,1], beh[:,1])
# print(r_value, p_value)
# r_value, p_value = pearsonr(cat(cd_proj_delta), cat(beh))
# print(r_value, p_value)

# Plot correlation between angle and robustness
idx = [0,1,2,4,5,7,8]
idx = np.arange(9)
plt.scatter(np.array(cd_opto_proj_delta), modularity, label='Learning')
# plt.scatter(np.array(cd_proj_delta)[idx, 1], modularity[idx, 1], label='EXpert')
plt.xlabel('Opto CD projection delta')
plt.ylabel('Modularity')
plt.title('Alignment vs modularity')
plt.legend()
plt.show()
r_value, p_value = pearsonr(cd_opto_proj_delta, modularity)
print(r_value, p_value)


#%% Plot input vector on stim vs control trials L/R

CD_angle, rotation_learning = [], []
all_deltas = []
opto_deltas = []

for paths in all_matched_paths:
    
    # l1 = Mode(paths[0], lickdir=False, use_reg = True, triple=True, proportion_train=1, proportion_opto_train=1)
    # _, mean, meantrain, meanstd = l1.plot_CD_opto(mode_input='choice', plot=True, return_applied=True)
    # intrinsic_nai, delta_nai = l1.intrinsic_vector(return_delta=True)
    # l1.plot_CD_opto_applied(intrinsic_nai, mean, meantrain, meanstd)

    l1 = Mode(paths[1], lickdir=False, use_reg = True, triple=True, proportion_train=1, proportion_opto_train=1)
    _, mean, meantrain, meanstd = l1.plot_CD_opto(mode_input='choice', plot=False, return_applied=True)
    # intrinsic_CD, delta = l1.intrinsic_vector(return_delta=True)
    input_vector, delta = l1.input_vector(by_trialtype=False, plot=True, return_delta = True)
    opto_delta = l1.plot_CD_opto_applied(input_vector, mean, meantrain, meanstd, return_delta=True)


    l2 = Mode(paths[2], lickdir=False, use_reg = True, triple=True, proportion_train=1, proportion_opto_train=1)
    _, mean, meantrain, meanstd = l2.plot_CD_opto(mode_input='choice', plot=False, return_applied=True)
    # intrinsic_exp, delta_exp = l2.intrinsic_vector(return_delta=True)
    input_vector_exp, delta_exp = l2.input_vector(by_trialtype=False, plot=True, return_delta = True)
    opto_delta_exp = l2.plot_CD_opto_applied(input_vector_exp, mean, meantrain, meanstd, return_delta=True)

    # Angle between trial type input vector and CD
    # CD_angle += [(cos_sim(input_vector, cd_choice), cos_sim(input_vector_exp, cd_choice_exp))]
    rotation_learning += [cos_sim(input_vector, input_vector_exp)]
    all_deltas += [(delta, delta_exp)]
    opto_deltas += [(opto_delta, opto_delta_exp)]
    
CD_angle, rotation_learning, all_deltas, opto_deltas = np.array(CD_angle), np.array(rotation_learning), np.array(all_deltas), np.array(opto_deltas)

# Plot the deltas of opto traces over learning
f=plt.figure()
plt.bar([0,1], np.mean(opto_deltas, axis=0))
plt.scatter(np.zeros(len(opto_deltas)), np.array(opto_deltas)[:, 0])
plt.scatter(np.ones(len(opto_deltas)), np.array(opto_deltas)[:, 1])
for i in range(len(opto_deltas)):
    plt.plot([0,1],[opto_deltas[i,0], opto_deltas[i,1]], color='grey')
plt.xticks([0,1],['Learning','Expert'])
plt.ylabel('Delta (L-R, z-scored)')
plt.title('Persistence of choice information in input vector')
plt.show()

f=plt.figure()
plt.bar([0,1], np.mean(all_deltas, axis=0))
plt.scatter(np.zeros(len(all_deltas)), np.array(all_deltas)[:, 0])
plt.scatter(np.ones(len(all_deltas)), np.array(all_deltas)[:, 1])
for i in range(len(all_deltas)):
    plt.plot([0,1],[all_deltas[i,0], all_deltas[i,1]], color='grey')
plt.xticks([0,1],['Learning','Expert'])
plt.ylabel('Delta (ctl-stim, z-scored)')
plt.title('Amplitude of input vector')
plt.show()

# Correlate with other measures, like CD angle alignment
plt.scatter(rotation_learning[:], opto_deltas[:, 0], label='Learning')
# plt.scatter(rotation_learning[:], opto_deltas[:, 1], label='Expert')
plt.xlabel('Rotation of input vector')
plt.ylabel('Persistence of choice information')
plt.title('Rotation vs Persistence')
plt.legend()
plt.show()
# plt.bar([0,1],np.mean(cd_proj_delta[idx, :], axis=0))
# plt.scatter(np.zeros(len(cd_proj_delta[idx, :])), np.array(cd_proj_delta)[idx, 0])
# plt.scatter(np.ones(len(cd_proj_delta[idx, :])), np.array(cd_proj_delta)[idx, 1])
# for i in idx:
#     plt.plot([0,1],[cd_proj_delta[i,0], cd_proj_delta[i,1]], color='grey')
# plt.xticks([0,1],['Learning','Expert'])
# plt.ylabel('Selectivity projection delta')
# plt.title('Input vector alignment to choice CD')
# plt.show()

# Correlate the persistence with behavior performance
plt.scatter(opto_deltas[:, 0], deltas[:, 0], label='Learning')
plt.scatter(opto_deltas[:, 1], deltas[:, 1], label='Expert')
plt.xlabel('Persistence of choice information')
plt.ylabel('Behavior recovery')
plt.title('Angle vs behavior')
plt.legend()
plt.show()

r_value, p_value = pearsonr(opto_deltas[:,0], deltas[:,0])
print(r_value, p_value)
r_value, p_value = pearsonr(opto_deltas[:,1], deltas[:,1])
print(r_value, p_value)
r_value, p_value = pearsonr(cat(opto_deltas), cat(deltas))
print(r_value, p_value)


plt.scatter(np.array(opto_deltas)[:, 0], beh[:, 0], label='Learning')
plt.scatter(np.array(opto_deltas)[:, 1], beh[:, 1], label='EXpert')
plt.xlabel('Persistence of choice information')
plt.ylabel('Behavior')
plt.title('Angle vs behavior')
plt.legend()
plt.show()
# r_value, p_value = pearsonr(cd_proj_delta[:,0], beh[:,0])
# print(r_value, p_value)
# r_value, p_value = pearsonr(cd_proj_delta[:,1], beh[:,1])
# print(r_value, p_value)
# r_value, p_value = pearsonr(cat(cd_proj_delta), cat(beh))
# print(r_value, p_value)

# Plot correlation between angle and robustness
# idx = [0,1,2,4,5,7,8]
idx = np.arange(9)
plt.scatter(np.array(opto_deltas)[idx, 0], modularity[idx, 0], label='Learning')
plt.scatter(np.array(opto_deltas)[idx, 1], modularity[idx, 1], label='EXpert')
plt.xlabel('Persistence of choice information')
plt.ylabel('Modularity')
plt.title('Alignment vs modularity')
plt.legend()
plt.show()
# r_value, p_value = pearsonr(opto_deltas[idx,0], modularity[idx,0])
# print(r_value, p_value)
# r_value, p_value = pearsonr(opto_deltas[idx,1], modularity[idx,1])
# print(r_value, p_value)
r_value, p_value = pearsonr(cat(opto_deltas[idx]), cat(modularity[idx]))
print(r_value, p_value)

#%% Plot modularity WEIGHTED BY INPUT VECTOR WEIGHTS 

all_control_sel, all_opto_sel = np.zeros(61), np.zeros(61)
num_neurons = 0
by_FOV = False
for paths in all_matched_paths:
    
    l1 = Mode(paths[2], use_reg=True, triple=True, 
                         # use_background_sub=True,
                         # remove_consec_opto=False,
                         baseline_normalization="median_zscore",
                         proportion_train=1, proportion_opto_train=1)    
    
    input_vector, delta = l1.input_vector(by_trialtype=False, plot=True, return_delta = True)

    adjusted_p = 0.05 / np.sqrt(len(l1.good_neurons))
    # adjusted_p = 0.01
    
    control_sel, opto_sel = l1.selectivity_optogenetics(p=adjusted_p, 
                                                        # exclude_unselective=True,
                                                        exclude_unselective=False,
                                                        lickdir=False, 
                                                        return_traces=True,
                                                        downsample='04' in paths[1])
    
    # weight by input vector
    
    delay_n_idx = [np.where(n == l1.good_neurons)[0][0] for n in l1.selective_neurons]
    input_vector_weights = np.abs(input_vector[delay_n_idx])
    input_vector_weights_norm = input_vector_weights / np.sum(input_vector_weights)
    
    control_sel = (control_sel.T * input_vector_weights_norm).T
    opto_sel = (opto_sel.T * input_vector_weights_norm).T
    
    if control_sel is None or len(control_sel) == 0 or np.sum(control_sel) == 0: # no selective neurons
        
        continue
    
    num_neurons_selective = len(control_sel)
    fov_selectivity = np.mean(np.mean(control_sel, axis=0)[range(28, 40)])
    
    print(num_neurons_selective, fov_selectivity)
    
    # if num_neurons_selective > 3 and fov_selectivity > 0.3:
    if True:
        if by_FOV:
            all_control_sel = np.vstack((all_control_sel, np.mean(control_sel, axis=0)))
            all_opto_sel = np.vstack((all_opto_sel, np.mean(opto_sel, axis=0)))
        else:
            all_control_sel = np.vstack((all_control_sel, control_sel))
            all_opto_sel = np.vstack((all_opto_sel, opto_sel))
        num_neurons += num_neurons_selective
    
all_control_sel, all_opto_sel = all_control_sel[1:], all_opto_sel[1:]

# sel = np.mean(pref, axis = 0) - np.mean(nonpref, axis = 0)
# err = np.std(pref, axis=0) / np.sqrt(len(pref)*2) 
# err += np.std(nonpref, axis=0) / np.sqrt(len(pref)*2)

# selo = np.mean(optop, axis = 0) - np.mean(optonp, axis = 0)
# erro = np.std(optop, axis=0) / np.sqrt(len(pref)*2) 
# erro += np.std(optonp, axis=0) / np.sqrt(len(pref)*2)  

sel = np.mean(all_control_sel, axis=0)
err = np.std(all_control_sel, axis=0) / np.sqrt(len(all_control_sel))
selo = np.mean(all_opto_sel, axis=0)
erro = np.std(all_opto_sel, axis=0) / np.sqrt(len(all_opto_sel))

f, axarr = plt.subplots(1,1, sharex='col', figsize=(5,5))  
x = np.arange(-6.97,4,1/6)[:sel.shape[0]]
axarr.plot(x, sel, 'black')
        
axarr.fill_between(x, sel - err, 
          sel + err,
          color=['darkgray'])

axarr.plot(x, selo, 'r-')
        
axarr.fill_between(x, selo - erro, 
          selo + erro,
          color=['#ffaeb1'])       

axarr.axvline(-4.3, color = 'grey', alpha=0.5, ls = '--')
axarr.axvline(-3, color = 'grey', alpha=0.5, ls = '--')
axarr.axvline(0, color = 'grey', alpha=0.5, ls = '--')
axarr.hlines(y=max(cat((selo, sel))), xmin=-3, xmax=-2, linewidth=10, color='red')

axarr.set_title('Optogenetic effect on selectivity (n = {} neurons)'.format(num_neurons))                  
axarr.set_xlabel('Time from Go cue (s)')
axarr.set_ylabel('Selectivity')
# axarr.set_ylim((-0.2, 1.1))





#%% Calculate input vector by trial type - all FOVs
L_angles, R_angles = [], []
inputvector_angles_R, inputvector_angles_L = [], []
cd_delta = []
all_deltas_l, all_deltas_r = [], []


for paths in all_matched_paths:

    l1 = Mode(paths[0], lickdir=False, use_reg = True, triple=True, proportion_train=1, proportion_opto_train=1)
    input_vector_L, input_vector_R, delta_nai_l, delta_nai_r = l1.input_vector(by_trialtype=True, plot=True, return_delta=True)
    cd_choice, _ = l1.plot_CD(mode_input='choice', plot=False)
    
    l1 = Mode(paths[1], lickdir=False, use_reg = True, triple=True, proportion_train=1, proportion_opto_train=1)
    input_vector_L, input_vector_R, delta_l, delta_r = l1.input_vector(by_trialtype=True, plot=True, return_delta=True)
    cd_choice, _ = l1.plot_CD(mode_input='choice', plot=False)


    l2 = Mode(paths[2], lickdir=False, use_reg = True, triple=True, proportion_train=1, proportion_opto_train=1)
    input_vector_Lexp, input_vector_Rexp, delta_exp_l, delta_exp_r = l2.input_vector(by_trialtype=True, plot=True, return_delta=True)
    cd_choice_exp, _ = l2.plot_CD(mode_input='choice', plot=False)

    # Angle between trial type input vector and CD
    L_angles += [(cos_sim(input_vector_L, cd_choice), cos_sim(input_vector_Lexp, cd_choice_exp))]
    R_angles += [(cos_sim(input_vector_R, cd_choice), cos_sim(input_vector_Rexp, cd_choice_exp))]
    inputvector_angles_R += [cos_sim(input_vector_R, input_vector_Rexp)]
    inputvector_angles_L += [cos_sim(input_vector_L, input_vector_Lexp)]
    all_deltas_l += [(delta_nai_l, delta_l, delta_exp_l)]
    all_deltas_r += [(delta_nai_r, delta_r, delta_exp_r)]

    
L_angles, R_angles = np.array(L_angles), np.array(R_angles)
cd_delta, all_deltas_l, all_deltas_r = np.array(cd_delta), np.array(all_deltas_l), np.array(all_deltas_r)

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
plt.show()

# Plot deltas over both
plt.bar([0,1,2],np.sum([np.mean(all_deltas_l, axis=0), np.mean(all_deltas_r, axis=0)], axis=0))
plt.scatter(np.zeros(len(all_deltas_l)), np.sum([np.array(all_deltas_l)[:, 0], np.array(all_deltas_r)[:, 0]], axis=0), color='red')
plt.scatter(np.ones(len(all_deltas_l)), np.sum([np.array(all_deltas_l)[:, 1], np.array(all_deltas_r)[:, 1]], axis=0), color='orange')
plt.scatter(np.ones(len(all_deltas_l))*2, np.sum([np.array(all_deltas_l)[:, 2], np.array(all_deltas_r)[:, 2]], axis=0), color='green')
for i in range(len(all_deltas_l)):
    plt.plot([0,1],[np.sum([np.array(all_deltas_l)[i, 0], np.array(all_deltas_r)[i, 0]], axis=0), 
                    np.sum([np.array(all_deltas_l)[i, 1], np.array(all_deltas_r)[i, 1]], axis=0)], color='grey')
    plt.plot([1,2],[np.sum([np.array(all_deltas_l)[i, 1], np.array(all_deltas_r)[i, 1]], axis=0), 
                    np.sum([np.array(all_deltas_l)[i, 2], np.array(all_deltas_r)[i, 2]], axis=0)], color='grey')
plt.xticks([0,1,2],['Naive', 'Learning','Expert'])
plt.ylabel('Delta (ctl-stim)')
plt.title('Delta of input vector btw control and stim condition (summed over L/R)')
plt.show()
stats.ttest_rel(np.sum([np.array(all_deltas_l)[:, 0], np.array(all_deltas_r)[:, 0]], axis=0),
                np.sum([np.array(all_deltas_l)[:, 1], np.array(all_deltas_r)[:, 1]], axis=0))
stats.ttest_rel(np.sum([np.array(all_deltas_l)[:, 1], np.array(all_deltas_r)[:, 1]], axis=0), 
                np.sum([np.array(all_deltas_l)[:, 2], np.array(all_deltas_r)[:, 2]], axis=0))


# Plot the deltas over learning - LEFT
plt.bar([0,1,2],np.mean(all_deltas_l, axis=0))
plt.scatter(np.zeros(len(all_deltas_l)), np.array(all_deltas_l)[:, 0])
plt.scatter(np.ones(len(all_deltas_l)), np.array(all_deltas_l)[:, 1])
plt.scatter(np.ones(len(all_deltas_l))*2, np.array(all_deltas_l)[:, 2])
for i in range(len(all_deltas_l)):
    plt.plot([0,1],[all_deltas_l[i,0], all_deltas_l[i,1]], color='grey')
    plt.plot([1,2],[all_deltas_l[i,1], all_deltas_l[i,2]], color='grey')
plt.xticks([0,1,2],['Naive', 'Learning','Expert'])
plt.ylabel('Delta (ctl-stim)')
plt.title('Delta of left input vector btw control and stim condition')
plt.show()
stats.ttest_rel(np.array(all_deltas_l)[:, 1], np.array(all_deltas_l)[:, 2])
stats.ttest_rel(np.array(all_deltas_l)[:, 0], np.array(all_deltas_l)[:, 1])


# Plot the deltas over learning - RIGHT
plt.bar([0,1,2],np.mean(all_deltas_r, axis=0))
plt.scatter(np.zeros(len(all_deltas_r)), np.array(all_deltas_r)[:, 0])
plt.scatter(np.ones(len(all_deltas_r)), np.array(all_deltas_r)[:, 1])
plt.scatter(np.ones(len(all_deltas_r))*2, np.array(all_deltas_r)[:, 2])
for i in range(len(all_deltas_r)):
    plt.plot([0,1],[all_deltas_r[i,0], all_deltas_r[i,1]], color='grey')
    plt.plot([1,2],[all_deltas_r[i,1], all_deltas_r[i,2]], color='grey')
plt.xticks([0,1,2],['Naive', 'Learning','Expert'])
plt.ylabel('Delta (ctl-stim)')
plt.title('Delta of right input vector btw control and stim condition')
plt.show()
stats.ttest_rel(np.array(all_deltas_r)[:, 1], np.array(all_deltas_r)[:, 2])
stats.ttest_rel(np.array(all_deltas_r)[:, 0], np.array(all_deltas_r)[:, 1])


# diff between left and right input vector deltas over NLE



#  angle between left and right input vector over NLE







#%% Angle between input and CD OLD
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
