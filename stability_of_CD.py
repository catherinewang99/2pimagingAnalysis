# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 12:09:11 2023

@author: Catherine Wang

Calculate CD for trained and apply to naive and learning sessions for
choice, stim, and action
"""


import sys
sys.path.append("C:\scripts\Imaging analysis")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from alm_2p import session
from activityMode import Mode
from matplotlib.pyplot import figure
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import stats
from numpy.linalg import norm
cat = np.concatenate
plt.rcParams['pdf.fonttype'] = '42' 


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

naivepath =r'F:\data\BAYLORCW032\python\2023_10_05'
learningpath =  r'F:\data\BAYLORCW032\python\2023_10_19'
expertpath =r'F:\data\BAYLORCW032\python\2023_10_24'

naivepath, learningpath, expertpath = [ r'F:\data\BAYLORCW034\python\2023_10_12',
    r'F:\data\BAYLORCW034\python\2023_10_22',
    r'F:\data\BAYLORCW034\python\2023_10_27',]

naivepath, learningpath, expertpath =[r'F:\data\BAYLORCW036\python\2023_10_09',
            r'F:\data\BAYLORCW036\python\2023_10_19',
            r'F:\data\BAYLORCW036\python\2023_10_30',]

naivepath, learningpath, expertpath = [r'H:\data\BAYLORCW044\python\2024_05_23',
                   r'H:\data\BAYLORCW044\python\2024_06_04',
                  r'H:\data\BAYLORCW044\python\2024_06_18',]

naivepath, learningpath, expertpath = [r'H:\data\BAYLORCW046\python\2024_05_31',
                   r'H:\data\BAYLORCW046\python\2024_06_11',
                  r'H:\data\BAYLORCW046\python\2024_06_26',]

# naivepath, learningpath, expertpath =[r'F:\data\BAYLORCW035\python\2023_10_12',
#             r'F:\data\BAYLORCW035\python\2023_10_26',
#             r'F:\data\BAYLORCW035\python\2023_12_12',]

# naivepath, learningpath, expertpath =[r'F:\data\BAYLORCW035\python\2023_10_26',
#             r'F:\data\BAYLORCW035\python\2023_12_07',
#             r'F:\data\BAYLORCW035\python\2023_12_15',]

    
# naivepath, learningpath, expertpath =[r'F:\data\BAYLORCW037\python\2023_11_21',
#             r'F:\data\BAYLORCW037\python\2023_12_08',
#             r'F:\data\BAYLORCW037\python\2023_12_15',]
#%% Choice dimension unmatched

path = expertpath
l1 = Mode(path)
orthonormal_basis, mean = l1.plot_CD(mode_input='action')

path = learningpath
l1 = Mode(path)
orthonormal_basis, mean = l1.plot_CD(mode_input='action')

path = naivepath
l1 = Mode(path)
orthonormal_basis, mean = l1.plot_CD(mode_input='action')

#%% CD defined on naive sess

path = naivepath
l1 = Mode(path, use_reg = True, triple=True)
orthonormal_basis, mean = l1.plot_CD(mode_input='action', save = r'F:\data\Fig 2\CDact_naive_nctl_CW37.pdf')

path = learningpath
l1 = Mode(path, use_reg = True, triple=True)
l1.plot_appliedCD(orthonormal_basis, mean,save = r'F:\data\Fig 2\CDact_learn_nctl_CW37.pdf')

path = expertpath
l1 = Mode(path, use_reg = True, triple=True)
l1.plot_appliedCD(orthonormal_basis, mean,save = r'F:\data\Fig 2\CDact_exp_nctl_CW37.pdf') 

#%% Choice dimension

path = expertpath
l1 = Mode(path, use_reg = True, triple=True)
orthonormal_basis, mean = l1.plot_CD(fix_axis = (-15, 17), save = r'F:\data\Fig 2\CDchoice_expert_CW37.pdf')

path =learningpath
l1 = Mode(path, use_reg = True, triple=True)
l1.plot_appliedCD(orthonormal_basis, mean, fix_axis = (-15, 17), save = r'F:\data\Fig 2\CDchoice_learning_CW37.pdf')

path = naivepath
l1 = Mode(path, use_reg = True, triple=True)
l1.plot_appliedCD(orthonormal_basis, mean, fix_axis = (-15, 17), save = r'F:\data\Fig 2\CDchoice_naive_CW37.pdf')

#%% Stim dimension

path = learningpath
l1 = Mode(path, use_reg = True, triple=True)
orthonormal_basis, mean = l1.plot_CD(mode_input='stimulus', ctl=True)#, save = r'F:\data\Fig 2\CDstim_expert_CW37.pdf')

path = expertpath
l1 = Mode(path, use_reg = True, triple=True)
l1.plot_appliedCD(orthonormal_basis, mean)#, save = r'F:\data\Fig 2\CDstim_learning_CW37.pdf')

path = naivepath
l1 = Mode(path, use_reg = True, triple=True)
l1.plot_appliedCD(orthonormal_basis, mean)#,  save = r'F:\data\Fig 2\CDstim_naive_CW37.pdf')


#%% Action dimension

path = expertpath
l1 = Mode(path, use_reg = True, triple=True)
orthonormal_basis, mean = l1.plot_CD(mode_input='action',fix_axis = (-12, 27), save = r'F:\data\Fig 2\CDaction_expert_CW36.pdf')

path = learningpath
l1 = Mode(path, use_reg = True, triple=True)
l1.plot_appliedCD(orthonormal_basis, mean,fix_axis = (-12, 27), save = r'F:\data\Fig 2\CDaction_learning_CW36.pdf')

path = naivepath
l1 = Mode(path, use_reg = True, triple=True)
l1.plot_appliedCD(orthonormal_basis, mean,fix_axis = (-12, 27), save = r'F:\data\Fig 2\CDaction_naive_CW36.pdf')

#%% Use Full method

naivepath, learningpath, expertpath = [r'F:\data\BAYLORCW032\python\2023_10_05',
          r'F:\data\BAYLORCW032\python\2023_10_19',
          r'F:\data\BAYLORCW032\python\2023_10_24',]
# naivepath, learningpath, expertpath =[r'F:\data\BAYLORCW036\python\2023_10_09',
#             r'F:\data\BAYLORCW036\python\2023_10_19',
#             r'F:\data\BAYLORCW036\python\2023_10_30',]
path = expertpath
l1 = Mode(path, use_reg = True, triple=True)
orthonormal_basis, mean = l1.plot_behaviorally_relevant_modes()

path = learningpath
l1 = Mode(path, use_reg = True, triple=True)
l1.plot_behaviorally_relevant_modes_appliedCD(orthonormal_basis, mean)

path = naivepath
l1 = Mode(path, use_reg = True, triple=True)
l1.plot_behaviorally_relevant_modes_appliedCD(orthonormal_basis, mean)

#%% Remove top contributing neurons
path = r'F:\data\BAYLORCW035\python\2023_12_15'

l1 = Mode(path, use_reg = True, triple=True)
inds = l1.plot_CD(mode_input='stimulus', remove_top=True)

#%% CD autocorrelogram
path = expertpath
l1 = Mode(path, use_reg = True, triple=True)
# orthonormal_basis, mean = l1.plot_CD(plot=False)
projR, projL = l1.plot_CD(plot=False, auto_corr_return=True)

allproj = np.vstack((projR, projL))
df = pd.DataFrame(allproj,
                  columns=range(61))

corrs = df.corr()
plt.imshow(corrs)
plt.axhline(l1.sample, color = 'white', ls='--', linewidth = 0.5)
plt.axvline(l1.sample, color = 'white', ls='--', linewidth = 0.5)

plt.axhline(l1.delay, color = 'white', ls='--', linewidth = 0.5)
plt.axvline(l1.delay, color = 'white', ls='--', linewidth = 0.5)

plt.axhline(l1.response, color = 'white', ls='--', linewidth = 0.5)
plt.axvline(l1.response, color = 'white', ls='--', linewidth = 0.5)

plt.xticks([l1.sample, l1.delay, l1.response], [-4.3, -3, 0])    
plt.yticks([l1.sample, l1.delay, l1.response], [-4.3, -3, 0])    
plt.colorbar()
#%% Plot R2 values for CD stability over learning
paths = [naivepath, learningpath, expertpath]
allr = []
for i in range(3):
    
    r_choice = []
    path = paths[i]
    l1 = Mode(path, use_reg = True, triple=True)
    orthonormal_basis_initial, mean = l1.plot_CD()
    for _ in range(25):
    
        l1 = Mode(path, use_reg = True, triple=True)
        orthonormal_basis, mean = l1.plot_CD()
        
        r_choice += [stats.pearsonr(orthonormal_basis_initial, orthonormal_basis)[0]]
        
    allr += [np.abs(r_choice)]

f = plt.figure(figsize = (5,5))
plt.bar([0,1,2], np.mean(allr, axis=1))
for i in range(3):
    plt.scatter(np.ones(len(allr[i])) * i, allr[i])
plt.xticks([0,1,2], ['Naive', 'Learning', 'Expert'])
plt.ylabel('R2 value')
plt.ylim(bottom=0.5)


#%% Plot R2 values of same CD over many pairs of runs to show stability of calculation

r_choice = []
path = learningpath
l1 = Mode(path, use_reg = True, triple=True)
orthonormal_basis_initial, mean = l1.plot_CD()
for _ in range(25):

    l1 = Mode(path, use_reg = True, triple=True)
    orthonormal_basis, mean = l1.plot_CD()
    
    r_choice += [stats.pearsonr(orthonormal_basis_initial, orthonormal_basis)[0]]

# Compare to only the selective neurons included:
all_sel_neurons = []
l1 = session.Session(naivepath, use_reg = True, triple=True)
sel = l1.get_epoch_selective(range(l1.time_cutoff), p=0.05)
all_sel_neurons += [np.where(l1.good_neurons == s)[0] for s in sel]
l1 = session.Session(learningpath, use_reg = True, triple=True)
sel = l1.get_epoch_selective(range(l1.time_cutoff), p=0.05)
all_sel_neurons += [np.where(l1.good_neurons == s)[0] for s in sel]
l1 = session.Session(expertpath, use_reg = True, triple=True)
sel = l1.get_epoch_selective(range(l1.time_cutoff), p=0.05)
all_sel_neurons += [np.where(l1.good_neurons == s)[0] for s in sel]

all_sel_neurons = list(set(cat(all_sel_neurons)))


r_choice_stab = []
path = learningpath
l1 = Mode(path, use_reg = True, triple=True, responsive_neurons=all_sel_neurons)
orthonormal_basis_initial, mean = l1.plot_CD()
for _ in range(25):

    l1 = Mode(path, use_reg = True, triple=True, responsive_neurons=all_sel_neurons)
    orthonormal_basis, mean = l1.plot_CD()
    
    r_choice_stab += [stats.pearsonr(orthonormal_basis_initial, orthonormal_basis)[0]]

r_choice, r_choice_stab = np.abs(r_choice), np.abs(r_choice_stab)
f = plt.figure(figsize=(5,7))
plt.bar([0,1], [np.mean(r_choice), np.mean(r_choice_stab)])
plt.scatter(np.zeros(len(r_choice)), r_choice)
plt.scatter(np.ones(len(r_choice_stab)), r_choice_stab)
plt.xticks([0,1], ['Control', 'Selective neurons only'])
plt.ylabel('R2 value')

#%% Stability of CD_delay in learning vs expert, measured by unit r2 values


agg_mice_paths = [
    
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


r_stim, r_delay = [], []

for paths in agg_mice_paths:
    
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
    
    plt.scatter(orthonormal_basis_initial, orthonormal_basis)
    plt.title('Pearsons correlation: {}, p-val: {}'.format(stats.pearsonr(orthonormal_basis_initial, orthonormal_basis)[0], 
                                                           stats.pearsonr(orthonormal_basis_initial, orthonormal_basis)[1]))
    plt.xlabel('Initial sample CD values')
    plt.ylabel('Final sample CD values')
    plt.show()
    r_stim += [stats.pearsonr(orthonormal_basis_initial, orthonormal_basis)[0]]
    
    # delay CD
    
    
    plt.scatter(orthonormal_basis_initial_choice, orthonormal_basis_choice)
    plt.title('Pearsons correlation: {}, p-val: {}'.format(stats.pearsonr(orthonormal_basis_initial_choice, orthonormal_basis_choice)[0], 
                                                           stats.pearsonr(orthonormal_basis_initial_choice, orthonormal_basis_choice)[1]))
    plt.xlabel('Initial delay CD values')
    plt.ylabel('Final delay CD values')
    plt.show()
    r_delay += [stats.pearsonr(orthonormal_basis_initial_choice, orthonormal_basis_choice)[0]]
#%% Plot the R squared values of each FOV
# r_stimr1, r_delayr1 = r_stim, r_delay

f = plt.figure(figsize = (5,5))
# plt.scatter(np.abs(r_stimr1), np.abs(r_delayr1), label="Round 1")
plt.scatter(np.abs(r_stim), np.abs(r_delay))
plt.xlabel('R2 values for sample mode')
plt.ylabel('R2 values for delay mode')
plt.axhline(0, ls='--')
plt.axvline(0, ls='--')
plt.axhline(0.5, ls='--', alpha = 0.5)
plt.axvline(0.5, ls='--', alpha = 0.5)
# plt.legend()


#%% Stability of choice mode vs amplitude of sample mode
# Here: amplitude is decoding accuracy and stability is dot product across

sample_ampl = []
choice_stability = []

for paths in agg_mice_paths:
    
    l1 = Mode(paths[1], use_reg=True, triple=True, use_selective=True, baseline_normalization="median_zscore") #Learning
    orthonormal_basis, mean, db, acc_learning = l1.decision_boundary(mode_input='stimulus', persistence=False)
    lea = np.mean(acc_learning)
    lea = lea if lea > 0.5 else 1-lea
    sample_ampl += [lea]
    
    l2 = Mode(paths[2], use_reg=True, triple=True, use_selective=True,  baseline_normalization="median_zscore") #Expert
    orthonormal_basis_initial_choice, mean = l1.plot_CD(mode_input = 'choice')
    orthonormal_basis_choice, mean = l2.plot_CD(mode_input = 'choice')

    choice_stability += [cos_sim(orthonormal_basis_initial_choice, orthonormal_basis_choice)]

f = plt.figure(figsize = (5,5))
plt.scatter(sample_ampl, np.abs(choice_stability))
plt.xlabel('Sample amplitude (learning)')
plt.ylabel('Stability of CD_choice (learning-->expert)')
# plt.axhline(0, ls='--')
# plt.axvline(0, ls='--')
# plt.axhline(0.5, ls='--', alpha = 0.5)
# plt.axvline(0.5, ls='--', alpha = 0.5)

scipy.stats.pearsonr(sample_ampl, np.abs(choice_stability))


#%% Rotation of choice mode vs robustness
# Robustness defined in expert stage?