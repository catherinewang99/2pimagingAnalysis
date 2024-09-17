# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 12:09:11 2023

@author: Catherine Wang

Calculate CD for trained and apply to naive and learning sessions for
choice, stim, and action
"""


import sys
sys.path.append("C:\scripts\Imaging analysis")
sys.path.append("Users/catherinewang/Desktop/Imaging analysis/2pimagingAnalysis/src")
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
# from scipy.stats import norm
from sklearn import preprocessing

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
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
#%% PATHS

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


naivepath, learningpath, expertpath = [r'H:\data\BAYLORCW046\python\2024_05_31',
                    r'/Users/catherinewang/Desktop/Imaging analysis/CW46/2024_06_11',
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

allpaths = [[    r'F:\data\BAYLORCW032\python\2023_10_05',
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

path = expertpath
l1 = Mode(path, use_reg = True, triple=True)
orthonormal_basis, mean = l1.plot_CD(mode_input='stimulus', ctl=True, save = r'F:\data\Fig 2\CDstim_expert_CW46.pdf')

path = learningpath
l1 = Mode(path, use_reg = True, triple=True)
l1.plot_appliedCD(orthonormal_basis, mean, save = r'F:\data\Fig 2\CDstim_learning_CW46.pdf')

path = naivepath
l1 = Mode(path, use_reg = True, triple=True)
l1.plot_appliedCD(orthonormal_basis, mean,  save = r'F:\data\Fig 2\CDstim_naive_CW46.pdf')


#%% Action dimension

path = expertpath
l1 = Mode(path, use_reg = True, triple=True)
orthonormal_basis, mean = l1.plot_CD(mode_input='action')#, save = r'F:\data\Fig 2\CDaction_expert_CW46.pdf')

path = learningpath
l1 = Mode(path, use_reg = True, triple=True)
l1.plot_appliedCD(orthonormal_basis, mean)#, save = r'F:\data\Fig 2\CDaction_learning_CW46.pdf')

path = naivepath
l1 = Mode(path, use_reg = True, triple=True)
l1.plot_appliedCD(orthonormal_basis, mean)#, save = r'F:\data\Fig 2\CDaction_naive_CW46.pdf')

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
                  columns=range(l1.time_cutoff))

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
#%% Plot R2 values for CD stability over learning TAKES A LONG TIME TO RUN
paths = [naivepath, learningpath, expertpath]
allrs = []
allmeans = []
medians = []
weights = []
n=25

for i in range(3):
    means = []
    med=[]
    rs = []
    for path in allpaths[i]:

        # path = paths[i]
        l1 = Mode(path, use_reg = True, triple=True)
        orthonormal_basis_initial, mean = l1.plot_CD()
        maxval = max(orthonormal_basis_initial)
        maxn = np.where(orthonormal_basis_initial == maxval)[0][0]
        r_choice = []
        for _ in range(n-1):
        
            l1 = Mode(path, use_reg = True, triple=True)
            orthonormal_basis, mean = l1.plot_CD()
            sign = np.sign(orthonormal_basis[maxn])
            orthonormal_basis_initial = np.vstack((orthonormal_basis_initial, orthonormal_basis * sign))
            
            r_choice += [scipy.stats.pearsonr(orthonormal_basis_initial[0], orthonormal_basis)[0]]
            
        rs += [np.mean(np.abs(r_choice))]
        means += [np.mean(np.var(orthonormal_basis_initial, axis=0))]
        med += [np.median(np.var(orthonormal_basis_initial, axis=0))]
        
        
    allrs += [rs]
    allmeans += [means]
    medians += [med]
    weights += [orthonormal_basis_initial]
    
    
np.save(r'H:\data\weights.npy', weights)
np.save(r'H:\data\allrs.npy', allrs)
np.save(r'H:\data\allmeans.npy', allmeans)
np.save(r'H:\data\medians.npy', medians)


f = plt.figure(figsize = (5,5))
plt.bar([0,1,2], np.mean(medians, axis=1))
for i in range(3):
    plt.scatter(np.ones(len(medians[i])) * i, medians[i])
plt.xticks([0,1,2], ['Naive', 'Learning', 'Expert'])
plt.ylabel('Median variance')
# plt.ylim(bottom=0.4)
plt.title('Median within session variance across FOVs')


#%% Correlate across FOV R2 values with amplitude of modes (decoding acc)

learning_rs = allrs[1]

all_choice_ampl = []
all_sample_ampl = []

for i in range(3):

    choice_ampl = []
    sample_ampl = []
    
    for path in allpaths[i]:
        
        l1 = Mode(path, use_reg = True, triple=True)
        orthonormal_basis, mean, db, acc_learning = l1.decision_boundary(mode_input='choice', persistence=False)
        lea = np.mean(acc_learning)
        lea = lea if lea > 0.5 else 1-lea
        choice_ampl += [lea]
        
        orthonormal_basis, mean, db, acc_learning_sample = l1.decision_boundary(mode_input='stimulus', persistence=False)
        lea_sample = np.mean(acc_learning_sample)
        lea_sample = lea_sample if lea_sample > 0.5 else 1-lea_sample
        sample_ampl += [lea_sample]
        
    all_choice_ampl += [choice_ampl]
    all_sample_ampl += [sample_ampl]
    

stage = ['Naive', 'Learning', 'Expert']
f = plt.figure(figsize = (5,5))
for i in range(3):
    plt.scatter(all_choice_ampl[i], allrs[i], marker='x', label = stage[i])
plt.xlabel('Choice amplitude (decoding accuracy)')
plt.ylabel('Stability of mode (r2 value)')
plt.title('CD_choice amplitude vs CD_choice stability')
plt.legend()
plt.ylim(bottom = 0.45)
print(scipy.stats.pearsonr(cat(all_choice_ampl), cat(allrs)))


f = plt.figure(figsize = (5,5))
for i in range(3):
    plt.scatter(all_sample_ampl[i], allrs[i], marker='x', label = stage[i])
plt.xlabel('Stimulus amplitude (decoding accuracy)')
plt.ylabel('Stability of mode (r2 value)')
plt.title('CD_sample amplitude vs CD_choice stability')
plt.legend()
plt.ylim(bottom = 0.45)
print(scipy.stats.pearsonr(cat(all_sample_ampl), cat(allrs)))


for i in range(3):
    f = plt.figure(figsize = (5,5))
    plt.scatter(all_choice_ampl[i], allrs[i], marker='x', label = stage[i])
    plt.xlabel('Choice amplitude (decoding accuracy)')
    plt.ylabel('Stability of mode (r2 value)')
    plt.title('CD_choice amplitude vs CD_choice stability')
    plt.legend()
    print(scipy.stats.pearsonr(all_choice_ampl[i], allrs[i]))


#%% Correlate across FOV R2 values with amplitude of modes (decoding acc) LEARNING ONLY


choice_ampl = []
sample_ampl = []

for path in allpaths[1]:
    
    l1 = Mode(path, use_reg = True, triple=True)
    orthonormal_basis, mean, db, acc_learning = l1.decision_boundary(mode_input='choice', persistence=False)
    lea = np.mean(acc_learning)
    lea = lea if lea > 0.5 else 1-lea
    choice_ampl += [lea]
    
    orthonormal_basis, mean, db, acc_learning_sample = l1.decision_boundary(mode_input='stimulus', persistence=True)
    lea_sample = np.mean(acc_learning_sample)
    lea_sample = lea_sample if lea_sample > 0.5 else 1-lea_sample
    sample_ampl += [lea_sample]
    

f = plt.figure(figsize = (5,5))
plt.scatter(choice_ampl,  allrs[1], marker='x', label = stage[i])
plt.xlabel('Sample persistence (decoding accuracy)')
plt.ylabel('Stability of mode (r2 value)')
plt.title('CD_choice amplitude vs CD_choice stability')
plt.legend()
print(scipy.stats.pearsonr(choice_ampl, allrs[1]))
    

f = plt.figure(figsize = (5,5))
plt.scatter(sample_ampl,  allrs[1], marker='x', label = stage[i])
plt.xlabel('Sample persistence (decoding accuracy)')
plt.ylabel('Stability of mode (r2 value)')
plt.title('CD_sample persistence vs CD_choice stability')
plt.legend()
print(scipy.stats.pearsonr(sample_ampl, allrs[1]))
    
    

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
    
#%% Stability of learning vs expert by showing runs on 10% train set sizes non separated
naivepath, learningpath, expertpath = [r'H:\data\BAYLORCW046\python\2024_05_31',
                    r'H:\data\BAYLORCW046\python\2024_06_11',
                  r'H:\data\BAYLORCW046\python\2024_06_26',]
##LEARNING
path = learningpath
l1 = Mode(path, use_reg = True, triple=True, 
          baseline_normalization="median_zscore",
          proportion_train = 0.1)
projR, projL = l1.plot_CD(mode_input = 'choice', plot=False, auto_corr_return=True)
l1.plot_CD()
l1 = Mode(path, use_reg = True, triple=True, 
          baseline_normalization="median_zscore",
          proportion_train = 0.1)
projR1, projL1 = l1.plot_CD(mode_input = 'choice', plot=False, auto_corr_return=True)
l1.plot_CD()

#Plot the autocorrelogram
f = plt.figure(figsize=(5,5))
allproj = np.vstack((projR, projL))
allproj1 = np.vstack((projR1, projL1))
corrs = np.corrcoef(allproj, allproj1)
corrs = np.corrcoef(allproj, allproj1, rowvar=False)
corrs = corrs[:l1.time_cutoff, :l1.time_cutoff]
plt.imshow(corrs, vmin=0, vmax=0.8)
plt.axhline(l1.sample, color = 'white', ls='--', linewidth = 0.5)
plt.axvline(l1.sample, color = 'white', ls='--', linewidth = 0.5)

plt.axhline(l1.delay, color = 'white', ls='--', linewidth = 0.5)
plt.axvline(l1.delay, color = 'white', ls='--', linewidth = 0.5)

plt.axhline(l1.response, color = 'white', ls='--', linewidth = 0.5)
plt.axvline(l1.response, color = 'white', ls='--', linewidth = 0.5)

plt.xticks([l1.sample, l1.delay, l1.response], [-4.3, -3, 0])    
plt.yticks([l1.sample, l1.delay, l1.response], [-4.3, -3, 0])    
plt.colorbar()
plt.show()

##EXPERT
path = expertpath
l2 = Mode(path, use_reg = True, triple=True, 
          baseline_normalization="median_zscore",
          proportion_train = 0.1)

projR, projL = l2.plot_CD(mode_input = 'choice', plot=False, auto_corr_return=True)
l2.plot_CD()

l2 = Mode(path, use_reg = True, triple=True, 
          baseline_normalization="median_zscore",
          proportion_train = 0.1)
projR1, projL1 = l2.plot_CD(mode_input = 'choice', plot=False, auto_corr_return=True)
l2.plot_CD()


#Plot the autocorrelogram
f = plt.figure(figsize=(5,5))
allproj = np.vstack((projR, projL))
allproj1 = np.vstack((projR1, projL1))
corrs = np.corrcoef(allproj, allproj1, rowvar=False)
corrs = corrs[:l2.time_cutoff, :l2.time_cutoff]
plt.imshow(corrs, vmin=0, vmax=0.8)
plt.axhline(l1.sample, color = 'white', ls='--', linewidth = 0.5)
plt.axvline(l1.sample, color = 'white', ls='--', linewidth = 0.5)

plt.axhline(l1.delay, color = 'white', ls='--', linewidth = 0.5)
plt.axvline(l1.delay, color = 'white', ls='--', linewidth = 0.5)

plt.axhline(l1.response, color = 'white', ls='--', linewidth = 0.5)
plt.axvline(l1.response, color = 'white', ls='--', linewidth = 0.5)

plt.xticks([l1.sample, l1.delay, l1.response], [-4.3, -3, 0])    
plt.yticks([l1.sample, l1.delay, l1.response], [-4.3, -3, 0])    
plt.colorbar()
#%% Stability of learning vs expert by showing runs on 50% train set sizes but independent
naivepath, learningpath, expertpath = [r'H:\data\BAYLORCW046\python\2024_05_31',
                    r'H:\data\BAYLORCW046\python\2024_06_11',
                  r'H:\data\BAYLORCW046\python\2024_06_26',]

split = 1/5
splitnum = int(1/split)
ctl=True
##LEARNING

path = learningpath
l1 = Mode(path, use_reg = True, triple=True)
numr = sum([l1.R_correct[i] for i in l1.i_good_non_stim_trials if not l1.early_lick[i]])
numl = sum([l1.L_correct[i] for i in l1.i_good_non_stim_trials if not l1.early_lick[i]])
numr = numr-numr%splitnum if numr%splitnum else numr
numl = numl-numl%splitnum if numl%splitnum else numl
r_trials = np.random.permutation(numr) # shuffle the indices
l_trials = np.random.permutation(numl)
numr_err = sum([l1.R_wrong[i] for i in l1.i_good_non_stim_trials if not l1.early_lick[i]])
numl_err = sum([l1.L_wrong[i] for i in l1.i_good_non_stim_trials if not l1.early_lick[i]])
numr_err = numr_err-numr_err%splitnum if numr_err%splitnum else numr_err
numl_err = numl_err-numl_err%splitnum if numl_err%splitnum else numl_err
r_trials_err = np.random.permutation(numr_err) # shuffle the indices
l_trials_err = np.random.permutation(numl_err)

# First half
r_train_idx, l_train_idx = r_trials[:int(split * numr)], l_trials[:int(split * numl)] #Take a portion of the trials for train
r_test_idx, l_test_idx = r_trials[int(split * numr):], l_trials[int(split * numl):]

r_train_err_idx, l_train_err_idx = r_trials_err[:int(split* numr_err)], l_trials_err[:int(split * numl_err)]
r_test_err_idx, l_test_err_idx = r_trials_err[int(split* numr_err):], l_trials_err[int(split * numl_err):]

train_test_trials = (r_train_idx, l_train_idx, r_test_idx, l_test_idx)
train_test_trials_err = (r_train_err_idx, l_train_err_idx, r_test_err_idx, l_test_err_idx)

l1 = Mode(path, use_reg = True, triple=True, 
          baseline_normalization="median_zscore",
          train_test_trials = [train_test_trials, train_test_trials_err])
projR, projL = l1.plot_CD(mode_input = 'choice', plot=False, auto_corr_return=True, ctl=ctl)
l1.plot_CD(ctl=ctl)


#second half
r_train_idx, l_train_idx = r_trials[int((1-split) * numr):], l_trials[int((1-split) * numl):] #Take a portion of the trials for train
r_test_idx, l_test_idx = r_trials[:int((1-split) * numr)], l_trials[:int((1-split) * numl)]

r_train_err_idx, l_train_err_idx = r_trials_err[int((1-split) * numr_err):], l_trials_err[int((1-split) * numl_err):]
r_test_err_idx, l_test_err_idx = r_trials_err[:int((1-split) * numr_err)], l_trials_err[:int((1-split) * numl_err)]

train_test_trials = (r_train_idx, l_train_idx, r_test_idx, l_test_idx)
train_test_trials_err = (r_train_err_idx, l_train_err_idx, r_test_err_idx, l_test_err_idx)

l1 = Mode(path, use_reg = True, triple=True, 
          baseline_normalization="median_zscore",
          train_test_trials = [train_test_trials, train_test_trials_err])
projR1, projL1 = l1.plot_CD(mode_input = 'choice', plot=False, auto_corr_return=True, ctl=ctl)
l1.plot_CD(ctl=ctl)

#Plot the autocorrelogram
f = plt.figure(figsize=(5,5))
allproj = np.vstack((projR, projL))
allproj1 = np.vstack((projR1, projL1))
corrs = np.corrcoef(allproj, allproj1, rowvar=False)
corrs = corrs[:l1.time_cutoff, :l1.time_cutoff]
plt.imshow(corrs, vmin=0, vmax=0.8)
plt.axhline(l1.sample, color = 'white', ls='--', linewidth = 0.5)
plt.axvline(l1.sample, color = 'white', ls='--', linewidth = 0.5)

plt.axhline(l1.delay, color = 'white', ls='--', linewidth = 0.5)
plt.axvline(l1.delay, color = 'white', ls='--', linewidth = 0.5)

plt.axhline(l1.response, color = 'white', ls='--', linewidth = 0.5)
plt.axvline(l1.response, color = 'white', ls='--', linewidth = 0.5)

plt.xticks([l1.sample, l1.delay, l1.response], [-4.3, -3, 0])    
plt.yticks([l1.sample, l1.delay, l1.response], [-4.3, -3, 0])    
plt.colorbar()
plt.show()

##EXPERT

path = expertpath
l1 = Mode(path, use_reg = True, triple=True)
numr = sum([l1.R_correct[i] for i in l1.i_good_non_stim_trials if not l1.early_lick[i]])
numl = sum([l1.L_correct[i] for i in l1.i_good_non_stim_trials if not l1.early_lick[i]])
numr = numr-numr%splitnum if numr%splitnum else numr
numl = numl-numl%splitnum if numl%splitnum else numl
r_trials = np.random.permutation(numr) # shuffle the indices
l_trials = np.random.permutation(numl)
numr_err = sum([l1.R_wrong[i] for i in l1.i_good_non_stim_trials if not l1.early_lick[i]])
numl_err = sum([l1.L_wrong[i] for i in l1.i_good_non_stim_trials if not l1.early_lick[i]])
numr_err = numr_err-numr_err%splitnum if numr_err%splitnum else numr_err
numl_err = numl_err-numl_err%splitnum if numl_err%splitnum else numl_err
r_trials_err = np.random.permutation(numr_err) # shuffle the indices
l_trials_err = np.random.permutation(numl_err)

# First half
r_train_idx, l_train_idx = r_trials[:int(split * numr)], l_trials[:int(split * numl)] #Take a portion of the trials for train
r_test_idx, l_test_idx = r_trials[int(split * numr):], l_trials[int(split * numl):]

r_train_err_idx, l_train_err_idx = r_trials_err[:int(split* numr_err)], l_trials_err[:int(split * numl_err)]
r_test_err_idx, l_test_err_idx = r_trials_err[int(split* numr_err):], l_trials_err[int(split * numl_err):]

train_test_trials = (r_train_idx, l_train_idx, r_test_idx, l_test_idx)
train_test_trials_err = (r_train_err_idx, l_train_err_idx, r_test_err_idx, l_test_err_idx)

l1 = Mode(path, use_reg = True, triple=True, 
          baseline_normalization="median_zscore",
          train_test_trials = [train_test_trials, train_test_trials_err])
projR, projL = l1.plot_CD(mode_input = 'choice', plot=False, auto_corr_return=True, ctl=ctl)
l1.plot_CD(ctl=ctl)


#second half
r_train_idx, l_train_idx = r_trials[int((1-split) * numr):], l_trials[int((1-split) * numl):] #Take a portion of the trials for train
r_test_idx, l_test_idx = r_trials[:int((1-split) * numr)], l_trials[:int((1-split) * numl)]

r_train_err_idx, l_train_err_idx = r_trials_err[int((1-split) * numr_err):], l_trials_err[int((1-split) * numl_err):]
r_test_err_idx, l_test_err_idx = r_trials_err[:int((1-split) * numr_err)], l_trials_err[:int((1-split) * numl_err)]

train_test_trials = (r_train_idx, l_train_idx, r_test_idx, l_test_idx)
train_test_trials_err = (r_train_err_idx, l_train_err_idx, r_test_err_idx, l_test_err_idx)

l1 = Mode(path, use_reg = True, triple=True, 
          baseline_normalization="median_zscore",
          train_test_trials = [train_test_trials, train_test_trials_err])
projR1, projL1 = l1.plot_CD(mode_input = 'choice', plot=False, auto_corr_return=True, ctl=ctl)
l1.plot_CD(ctl=ctl)


#Plot the autocorrelogram
f = plt.figure(figsize=(5,5))
allproj = np.vstack((projR, projL))
allproj1 = np.vstack((projR1, projL1))
corrs = np.corrcoef(allproj, allproj1, rowvar=False)
corrs = corrs[:l2.time_cutoff, :l2.time_cutoff]
plt.imshow(corrs, vmin=0, vmax=0.8)
plt.axhline(l1.sample, color = 'white', ls='--', linewidth = 0.5)
plt.axvline(l1.sample, color = 'white', ls='--', linewidth = 0.5)

plt.axhline(l1.delay, color = 'white', ls='--', linewidth = 0.5)
plt.axvline(l1.delay, color = 'white', ls='--', linewidth = 0.5)

plt.axhline(l1.response, color = 'white', ls='--', linewidth = 0.5)
plt.axvline(l1.response, color = 'white', ls='--', linewidth = 0.5)

plt.xticks([l1.sample, l1.delay, l1.response], [-4.3, -3, 0])    
plt.yticks([l1.sample, l1.delay, l1.response], [-4.3, -3, 0])    
plt.colorbar()
#%% Run over the 10 different possible splits for train set size
path = learningpath
l1 = Mode(path, use_reg = True, triple=True)
numr = sum([l1.R_correct[i] for i in l1.i_good_non_stim_trials if not l1.early_lick[i]])
numl = sum([l1.L_correct[i] for i in l1.i_good_non_stim_trials if not l1.early_lick[i]])
r_trials = np.random.permutation(numr) # shuffle the indices
l_trials = np.random.permutation(numl)
numr_err = sum([l1.R_wrong[i] for i in l1.i_good_non_stim_trials if not l1.early_lick[i]])
numl_err = sum([l1.L_wrong[i] for i in l1.i_good_non_stim_trials if not l1.early_lick[i]])
r_trials_err = np.random.permutation(numr_err) # shuffle the indices
l_trials_err = np.random.permutation(numl_err)

splits = 10
all_weights_learn = []
for i in range(splits-1):
    r_train_idx, l_train_idx = r_trials[int(i/splits*numr):int((i+1)/splits * numr)], l_trials[int(i/splits*numl):int((i+1)/splits * numl)] #Take a portion of the trials for train
    rtest = [n for n in range(numr) if n not in range(int(i/splits*numr), int((i+1)/splits * numr))]
    ltest = [n for n in range(numl) if n not in range(int(i/splits*numl), int((i+1)/splits * numl))]
    r_test_idx, l_test_idx = r_trials[rtest], l_trials[ltest]
    
    r_train_err_idx, l_train_err_idx = r_trials_err[int(i/splits*numr_err):int((i+1)/splits * numr_err)], l_trials_err[int(i/splits*numl_err):int((i+1)/splits * numl_err)]
    rtest = [n for n in range(numr_err) if n not in range(int(i/splits*numr_err), int((i+1)/splits * numr_err))]
    ltest = [n for n in range(numl_err) if n not in range(int(i/splits*numl_err), int((i+1)/splits * numl_err))]
    r_test_err_idx, l_test_err_idx = r_trials_err[rtest], l_trials_err[ltest]

    train_test_trials = (r_train_idx, l_train_idx, r_test_idx, l_test_idx)
    train_test_trials_err = (r_train_err_idx, l_train_err_idx, r_test_err_idx, l_test_err_idx)
        
    l1 = Mode(path, use_reg = True, triple=True, 
              baseline_normalization="median_zscore",
              train_test_trials = [train_test_trials, train_test_trials_err])
    orthonormal_basis_initial_choice, mean = l1.plot_CD(mode_input = 'choice')
    all_weights_learn += [orthonormal_basis_initial_choice]




#%% learning sess
path = expertpath
l1 = Mode(path, use_reg = True, triple=True)
numr = sum([l1.R_correct[i] for i in l1.i_good_non_stim_trials if not l1.early_lick[i]])
numl = sum([l1.L_correct[i] for i in l1.i_good_non_stim_trials if not l1.early_lick[i]])
r_trials = np.random.permutation(numr) # shuffle the indices
l_trials = np.random.permutation(numl)
numr_err = sum([l1.R_wrong[i] for i in l1.i_good_non_stim_trials if not l1.early_lick[i]])
numl_err = sum([l1.L_wrong[i] for i in l1.i_good_non_stim_trials if not l1.early_lick[i]])
r_trials_err = np.random.permutation(numr_err) # shuffle the indices
l_trials_err = np.random.permutation(numl_err)

splits = 10
all_weights = []
for i in range(splits-1):
    r_train_idx, l_train_idx = r_trials[int(i/splits*numr):int((i+1)/splits * numr)], l_trials[int(i/splits*numl):int((i+1)/splits * numl)] #Take a portion of the trials for train
    rtest = [n for n in range(numr) if n not in range(int(i/splits*numr), int((i+1)/splits * numr))]
    ltest = [n for n in range(numl) if n not in range(int(i/splits*numl), int((i+1)/splits * numl))]
    r_test_idx, l_test_idx = r_trials[rtest], l_trials[ltest]
    
    r_train_err_idx, l_train_err_idx = r_trials_err[int(i/splits*numr_err):int((i+1)/splits * numr_err)], l_trials_err[int(i/splits*numl_err):int((i+1)/splits * numl_err)]
    rtest = [n for n in range(numr_err) if n not in range(int(i/splits*numr_err), int((i+1)/splits * numr_err))]
    ltest = [n for n in range(numl_err) if n not in range(int(i/splits*numl_err), int((i+1)/splits * numl_err))]
    r_test_err_idx, l_test_err_idx = r_trials_err[rtest], l_trials_err[ltest]

    train_test_trials = (r_train_idx, l_train_idx, r_test_idx, l_test_idx)
    train_test_trials_err = (r_train_err_idx, l_train_err_idx, r_test_err_idx, l_test_err_idx)
        
    l1 = Mode(path, use_reg = True, triple=True, 
              baseline_normalization="median_zscore",
              train_test_trials = [train_test_trials, train_test_trials_err])
    orthonormal_basis_initial_choice, mean = l1.plot_CD(mode_input = 'choice')
    all_weights += [orthonormal_basis_initial_choice]
    
#%% Pairwise dot products
alldotslearn, alldotsexp = [], []
for i in range(splits-2):
    for j in range(i+1, splits-1):
        
        alldotslearn += [np.dot(all_weights_learn[i], all_weights_learn[j])]
        alldotsexp += [np.dot(all_weights[i], all_weights[j])]
                
f = plt.figure(figsize=(5,5))
plt.bar([0,1], [np.mean(np.abs(alldotslearn)), np.mean(np.abs(alldotsexp))])
plt.scatter(np.zeros(len(alldotslearn)), np.abs(alldotslearn))
plt.scatter(np.ones(len(alldotsexp)), np.abs(alldotsexp))
plt.ylabel('Dot product')
plt.title('Pairwise dot product of runs over 10% independent train sets')
plt.xticks([0,1], ['Learning', 'Expert'])
# f = plt.figure(figsize=(5,5))
# plt.bar([0,1], [np.mean(np.var(all_weights_learn, axis=0)), np.mean(np.var(all_weights, axis=0))])
    
#%% Stability represented by a few example neurons over few example trials
naivepath, learningpath, expertpath = [r'H:\data\BAYLORCW046\python\2024_05_31',
                    r'H:\data\BAYLORCW046\python\2024_06_11',
                  r'H:\data\BAYLORCW046\python\2024_06_26',]

path = learningpath
l1 = Mode(path, use_reg = True, triple=True, baseline_normalization="median_zscore")
path = expertpath
l2 = Mode(path, use_reg = True, triple=True, baseline_normalization="median_zscore")

p=0.001
learning_delay_sel = l1.get_epoch_selective(range(l1.delay+30, l1.response), p=0.001)
expert_delay_sel = l2.get_epoch_selective(range(l2.delay+30, l2.response), p=0.001)
learning_delay_sel_idx = [np.where(l1.good_neurons == i)[0][0] for i in learning_delay_sel]
expert_delay_sel_idx = [np.where(l2.good_neurons == i)[0][0] for i in expert_delay_sel]

shared_delay_idx = [n for n in cat((learning_delay_sel_idx, expert_delay_sel_idx)) if n in learning_delay_sel_idx and n in expert_delay_sel_idx]

#%% What do delay neurons look like:
for idx in shared_delay_idx:
    l1.plot_rasterPSTH_sidebyside(l1.good_neurons[idx])
    l2.plot_rasterPSTH_sidebyside(l2.good_neurons[idx])

#%%
idx_highvar = np.where(np.array(allr) > 0.003)[0] # high variance neurons
idx_highvar = np.where(np.array(avg_weights) > 0.1)[0] # high weight neurons
#%% BY INSPECTION (all left prferring)
learn_neurons = [4,385,380,8,87,118]
exp_neurons = [18,503,434,38,331,130]



#%%
trials = [160, 161, 165, 166, 169]
# trials = np.random.choice(np.where(l1.L_correct)[0], 5, replace=False)
trials = np.where(l1.L_correct)[0]
trials_learn = np.random.permutation(trials)[:57]#[:int(len(trials)/2)] # randomly shuffle and take half
agg_traces_all = []
neuron_av_learn = []
# for trial in trials:
for idx in learn_neurons:
    agg_traces = []
    for trial in trials_learn:
        agg_traces_all += [np.array(l1.dff[0,trial][idx, :l1.time_cutoff])]
        agg_traces += [np.array(l1.dff[0,trial][idx, :l1.time_cutoff])]
    neuron_av_learn += [np.mean(agg_traces, axis=0)]
    # mean = np.mean(l1.dff[0,trial][l1.good_neurons[idx], :l1.time_cutoff])
    # std = np.std(l1.dff[0,trial][l1.good_neurons[idx], :l1.time_cutoff])
    # agg_traces += [np.array(l1.dff[0,trial][l1.good_neurons[idx], :l1.time_cutoff]) - mean / std]
    # agg_traces_all += [agg_traces]
    


# trials = [195, 196, 200, 201, 203]
# trialsexp = np.random.choice(np.where(l2.L_correct)[0], 5, replace=False)
trials = np.where(l2.L_correct)[0]
trials = np.random.permutation(trials)[:57]#[:int(len(trials)/2)] # randomly shuffle and take half

# trialsexp = np.where(l2.R_correct)[0][15:]
agg_traces_all_exp = []
neuron_av_exp = []
# for trial in trials:
#     agg_traces = []
for idx in exp_neurons:
    agg_traces = []
    for trial in trials:
        agg_traces_all_exp += [np.array(l2.dff[0,trial][idx,:l2.time_cutoff])]
        agg_traces += [np.array(l1.dff[0,trial][idx, :l1.time_cutoff])]
    neuron_av_exp += [np.mean(agg_traces, axis=0)]
    # mean = np.mean(l1.dff[0,trial][l1.good_neurons[idx], :l1.time_cutoff])
    # std = np.std(l1.dff[0,trial][l1.good_neurons[idx], :l1.time_cutoff])
    # agg_traces += [np.array(l1.dff[0,trial][l1.good_neurons[idx], :l1.time_cutoff]) - mean / std]
    # agg_traces_all_exp += [agg_traces]
        
# for i in range(len(trials)):
f, ax = plt.subplots(1, 2, figsize = (10,5))
# ax[0].matshow(agg_traces_all[i], cmap='gray', interpolation='nearest', aspect='auto')
ax[0].matshow(agg_traces_all, vmin=0, vmax=3, aspect='auto') #interpolation='nearest', a
ax[0].axis('off')
# ax[1].matshow(agg_traces_all_exp[i], cmap='gray', interpolation='nearest', aspect='auto')
ax[1].matshow(agg_traces_all_exp, vmin=0, vmax=3, aspect='auto')
ax[1].axis('off')
for j in range(2):
    ax[j].axvline(l1.sample, color='white', ls='--')
    ax[j].axvline(l1.delay, color='white', ls='--')
    ax[j].axvline(l1.response, color='white', ls='--')
    for t in range(0,len(agg_traces_all),len(trials_learn)):
        ax[0].axhline(t, color='white')
    for t in range(0,len(agg_traces_all_exp), len(trials)):
        ax[1].axhline(t, color='white')
           
# Then, plot the averaged traces for these five neurons below:
f, ax = plt.subplots(1, 2, figsize = (10,3))
for i in range(len(learn_neurons)):
    ax[0].plot(np.mean(agg_traces_all[i*57:(i+1)*57], axis=0), label='Neuron {}'.format(i+1))
    # ax[0].plot(np.mean(agg_traces_all[i*57:(i+1)*57], axis=0))
    
for i in range(len(exp_neurons)):
    ax[1].plot(np.mean(agg_traces_all_exp[i*57:(i+1)*57], axis=0), label='Neuron {}'.format(i+1))
for j in range(2):
    ax[j].axvline(l1.sample, color='grey', ls='--')
    ax[j].axvline(l1.delay, color='grey', ls='--')
    ax[j].axvline(l1.response, color='grey', ls='--')
    
plt.legend()

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