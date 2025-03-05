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
cat = np.concatenate

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
    cd_choice_nai, _ = l1.plot_CD(mode_input='choice', plot=False)
    input_vector_nai, delta_nai = l1.intrinsic_vector(return_delta=True)


    l1 = Mode(paths[1], lickdir=False, use_reg = True, triple=True, proportion_train=1, proportion_opto_train=1)
    input_vector, delta = l1.intrinsic_vector(return_delta=True)
    cd_choice, _ = l1.plot_CD(mode_input='choice', plot=False)


    l2 = Mode(paths[2], lickdir=False, use_reg = True, triple=True, proportion_train=1, proportion_opto_train=1)
    input_vector_exp, delta_exp = l2.intrinsic_vector(return_delta=True)
    cd_choice_exp, _ = l2.plot_CD(mode_input='choice', plot=False)

    # Angle between trial type input vector and CD
    CD_angle += [(cos_sim(input_vector_nai, cd_choice_nai), cos_sim(input_vector, cd_choice), cos_sim(input_vector_exp, cd_choice_exp))]
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
plt.bar([0,1,2],np.mean(CD_angle, axis=0))
plt.scatter(np.zeros(len(CD_angle)), np.array(CD_angle)[:, 0])
plt.scatter(np.ones(len(CD_angle)), np.array(CD_angle)[:, 1])
plt.scatter(np.ones(len(CD_angle)) * 2, np.array(CD_angle)[:, 2])
for i in range(len(CD_angle)):
    plt.plot([0,1],[CD_angle[i,0], CD_angle[i,1]], color='grey')
    plt.plot([1,2],[CD_angle[i,1], CD_angle[i,2]], color='grey')
plt.xticks([0,1,2],['Naive','Learning','Expert'])
plt.ylabel('Dot product')
plt.title('Intrinsic vector alignment to choice CD')
plt.ylim(bottom=0.9)
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

#%% Project intrinsic CD onto stim trials

CD_angle, rotation_learning = [], []
all_deltas, all_opto_deltas = [],[]

for paths in all_matched_paths:
    
    # l1 = Mode(paths[0], lickdir=False, use_reg = True, triple=True, proportion_train=1, proportion_opto_train=1)
    # _, mean, meantrain, meanstd = l1.plot_CD_opto(mode_input='choice', plot=True, return_applied=True)
    # intrinsic_nai, delta_nai = l1.intrinsic_vector(return_delta=True)
    # l1.plot_CD_opto_applied(intrinsic_nai, mean, meantrain, meanstd)

    l1 = Mode(paths[1], lickdir=False, use_reg = True, triple=True, proportion_train=1, proportion_opto_train=1)
    _, mean, meantrain, meanstd = l1.plot_CD_opto(mode_input='choice', plot=True, return_applied=True)
    intrinsic_CD, delta = l1.intrinsic_vector(return_delta=True)
    delta_opto = l1.plot_CD_opto_applied(intrinsic_CD, mean, meantrain, meanstd, return_delta=True)


    l2 = Mode(paths[2], lickdir=False, use_reg = True, triple=True, proportion_train=1, proportion_opto_train=1)
    _, mean, meantrain, meanstd = l2.plot_CD_opto(mode_input='choice', plot=True, return_applied=True)
    intrinsic_exp, delta_exp = l2.intrinsic_vector(return_delta=True)
    delta_opto_exp = l2.plot_CD_opto_applied(intrinsic_exp, mean, meantrain, meanstd, return_delta=True)

    # Angle between trial type input vector and CD
    # CD_angle += [(cos_sim(input_vector, cd_choice), cos_sim(input_vector_exp, cd_choice_exp))]
    rotation_learning += [cos_sim(intrinsic_CD, intrinsic_exp)]
    all_deltas += [(delta, delta_exp)]
    all_opto_deltas += [(delta_opto, delta_opto_exp)]
CD_angle, rotation_learning, all_deltas = np.array(CD_angle), np.array(rotation_learning), np.array(all_deltas)
all_opto_deltas = np.array(all_opto_deltas)

# Plot the opto trial deltas across learning

f=plt.figure()
plt.bar([0,1],np.mean(all_opto_deltas,axis=0))
plt.scatter(np.zeros(all_opto_deltas.shape[1]), all_opto_deltas[:,0])
plt.scatter(np.ones(all_opto_deltas.shape[1]), all_opto_deltas[:,1])
for i in range(len(all_opto_deltas)):
    plt.plot([0,1],[all_opto_deltas[i,0],all_opto_deltas[i,1]], color='grey')
plt.xticks([0,1], ['Learning', 'Expert'])

#%% weigh by intriinsisic vector

all_control_sel, all_opto_sel = np.zeros(61), np.zeros(61)
num_neurons = 0
by_FOV = False
for paths in all_matched_paths:
    
    l1 = Mode(paths[2], use_reg=True, triple=True, 
                         # use_background_sub=True,
                         # remove_consec_opto=False,
                         baseline_normalization="median_zscore",
                         proportion_train=1, proportion_opto_train=1)    
    
    intrinsic_CD, delta = l1.intrinsic_vector(return_delta=True)

    adjusted_p = 0.05 / np.sqrt(len(l1.good_neurons))
    # adjusted_p = 0.01
    
    control_sel, opto_sel = l1.selectivity_optogenetics(p=adjusted_p, 
                                                        # exclude_unselective=True,
                                                        exclude_unselective=False,
                                                        lickdir=False, 
                                                        return_traces=True,
                                                        downsample='04' in paths[1])
    
    # weight by input vector
    
    # delay_n_idx = [np.where(n == l1.good_neurons)[0][0] for n in l1.selective_neurons]
    # input_vector_weights = np.abs(intrinsic_CD[delay_n_idx])
    # input_vector_weights_norm = input_vector_weights / np.sum(input_vector_weights)
    
    # control_sel = (control_sel.T * input_vector_weights_norm).T
    # opto_sel = (opto_sel.T * input_vector_weights_norm).T
    
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

#%% Correlate with robustbess - beh, neural NONMATCHED

CD_angle, deltas, beh, frac, modularity , cd_proj_delta = [],[],[],[],[],[]
cd_opto_proj_delta = []
for path in all_paths[2]:
    # l1 = Mode(paths[0], lickdir=False, use_reg = True, triple=True, proportion_train=1, proportion_opto_train=1)
    # input_vector_nai, delta_nai = l1.input_vector(by_trialtype=False, plot=True, return_delta = True)
    # _, cd_delta_nai = l1.input_vector(by_trialtype=False, plot=True, return_delta = True, plot_ctl_opto=False)

    l1 = Mode(path, lickdir=False, proportion_train=1, proportion_opto_train=1)
    # input_vector, delta = l1.input_vector(by_trialtype=False, plot=True, return_delta = True)
    intrinsic_CD, cd_delta_lea = l1.intrinsic_vector(return_delta=True) # delta between L/R

    cd_choice, _ = l1.plot_CD(mode_input='choice', plot=False)

    _, mean, meantrain, meanstd = l1.plot_CD_opto(mode_input='choice', plot=False, return_applied=True)

    opto_delta = l1.plot_CD_opto_applied(intrinsic_CD, mean, meantrain, meanstd, return_delta=True)
    
    
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
    
    
    
    modularity += [temp]
    deltas += [perf_all - perf_all_c]
    frac += [perf_all / perf_all_c]
    beh += [perf_all_c]
    cd_proj_delta += [cd_delta_lea]
    # Angle between trial type input vector and CD
    CD_angle += [cos_sim(input_vector, cd_choice)]
    cd_opto_proj_delta += [opto_delta]
    
CD_angle, deltas, beh, frac = np.array(CD_angle), np.array(deltas), np.array(beh), np.array(frac)
cd_proj_delta, modularity,cd_opto_proj_delta = np.array(cd_proj_delta), np.array(modularity),np.array(cd_opto_proj_delta)







