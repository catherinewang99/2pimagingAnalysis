# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 10:05:17 2024

Investigate why the sample mode is more persistent in learning vs expert sessions

@author: catherinewang
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
import pandas as pd
from sklearn import preprocessing
import joblib
from sklearn.preprocessing import normalize
from scipy.stats import ttest_rel
from scipy.stats import ttest_ind

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


# naivepath, learningpath, expertpath = [r'H:\data\BAYLORCW046\python\2024_05_31',
#                     r'/Users/catherinewang/Desktop/Imaging analysis/CW46/2024_06_11',
#                   r'H:\data\BAYLORCW046\python\2024_06_26',]


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

naivepath, learningpath, expertpath, clusterpath = [r'H:\data\BAYLORCW044\python\2024_05_22',
                    r'H:\data\BAYLORCW044\python\2024_06_06',
                  r'H:\data\BAYLORCW044\python\2024_06_19',
                  r'H:\data\matched_topic_params\CW44_FOV1_table']

# naivepath, learningpath, expertpath, clusterpath = [
#                     r'H:\data\BAYLORCW046\python\2024_05_31',
#                     r'H:\data\BAYLORCW046\python\2024_06_11',
#                     r'H:\data\BAYLORCW046\python\2024_06_26',
                    # r'H:\data\matched_topic_params\CW46_FOV3_table']
                    
#%% Look at example FOVs

path = expertpath
s1 = Mode(path, use_reg = True, triple=True, baseline_normalization="median_zscore")
orthonormal_basis, mean_exp = s1.plot_CD(ctl=True)
s1.exclude_sample_orthog = True

orthonormal_basis, mean_exp = s1.plot_CD(ctl=True)



path = learningpath
s2 = Mode(path, use_reg = True, triple=True, baseline_normalization="median_zscore")
orthonormal_basis_learning, mean = s2.plot_CD(ctl=True)
s2.exclude_sample_orthog = True
orthonormal_basis_learning, mean = s2.plot_CD(ctl=True)

#%% Across FOVs, get angle between CDs
plot=False
all_choice_stability = []

for paths in allpaths[1:]:
    choice_stability = []
    for path in paths:
        
        s2 = Mode(path, use_reg = True, triple=True, 
                  baseline_normalization="median_zscore",
                  proportion_train=1)
        orthonormal_basis, mean = s2.plot_CD(ctl=True, plot=plot)
        s2.exclude_sample_orthog = True
        orthonormal_basis_no_sam, mean = s2.plot_CD(ctl=True, plot=plot)

        choice_stability += [cos_sim(orthonormal_basis, orthonormal_basis_no_sam)]

    all_choice_stability += [choice_stability]
    
    
all_choice_stability = np.abs(all_choice_stability)
t_stat, p_value = ttest_rel(all_choice_stability[0], all_choice_stability[1]) # Paired t-test
print(t_stat, p_value)

f = plt.figure(figsize=(6,6))
plt.bar(np.arange(len(all_choice_stability)), np.mean(all_choice_stability, axis=1))
for i in range(len(all_choice_stability)):
    plt.scatter(np.ones(len(all_choice_stability[i]))*i, all_choice_stability[i])
    
for j in range(len(all_choice_stability[i])):
    plt.plot([0,1], [all_choice_stability[0][j],
                     all_choice_stability[1][j]],
             color='grey')

plt.ylim(bottom=0.5)
plt.xticks([0,1],['Learning', 'Expert'])
plt.ylabel('Cosine similarity')
plt.title('Simialrity of CD_delay with/without sample mode')

#%% Contribution of sample selective vs delay selective cells to selectivity at end of delay ONE FOV
# Contribution can be thought of as weight to CD_delay wo sample exclusion

p=0.01

# First, separate neurons into sample vs sample/delay vs delay selective
s2 = Mode(learningpath, use_reg = True, triple=True,
          proportion_train=1,
          baseline_normalization="median_zscore")

sample_selective = s2.get_epoch_selective(range(s2.sample, s2.delay), p=p)
delay_selective = s2.get_epoch_selective(range(s2.delay + int(1.5*(1/s2.fs)), s2.response), p=p)

sample_delay_selective = np.intersect1d(sample_selective, delay_selective)
sample_selective = [n for n in sample_selective if n not in sample_delay_selective]
delay_selective = [n for n in delay_selective if n not in sample_delay_selective]

# Then look at weights on CDdelay of each of these pools
s2.exclude_sample_orthog = True
orthonormal_basis_learning, mean = s2.plot_CD(ctl=True)

# Get idx
sample_delay_selective_idx = [np.where(s2.good_neurons == n)[0][0] for n in sample_delay_selective]
sample_selective_idx = [np.where(s2.good_neurons == n)[0][0] for n in sample_selective]
delay_selective_idx = [np.where(s2.good_neurons == n)[0][0] for n in delay_selective]

plt.bar([0,1,2], [np.mean(np.abs(orthonormal_basis_learning[sample_selective_idx])),
                  np.mean(np.abs(orthonormal_basis_learning[sample_delay_selective_idx])),
                  np.mean(np.abs(orthonormal_basis_learning[delay_selective_idx]))])

plt.scatter(np.zeros(len(sample_selective_idx)), np.abs(orthonormal_basis_learning[sample_selective_idx]))
plt.scatter(np.ones(len(sample_delay_selective_idx)), np.abs(orthonormal_basis_learning[sample_delay_selective_idx]))
plt.scatter(np.ones(len(delay_selective_idx)) * 2, np.abs(orthonormal_basis_learning[delay_selective_idx]))

plt.xticks([0,1,2], ['Sample', 'Sample and delay', 'Delay'])
plt.ylabel('Weight on CD_delay')
plt.ylim(top=0.37)

#%% Look at the psth of sample_delay_selective neurons

for n in sample_delay_selective:
    s2.plot_rasterPSTH_sidebyside(n)
#%% Contribution of sample selective vs delay selective cells to selectivity at end of delay ALL FOVs


p=0.01

all_weights = []

for paths in allpaths[1:]:
    
    weights = []
    
    for path in paths:
    
        # First, separate neurons into sample vs sample/delay vs delay selective
        s2 = Mode(path, use_reg = True, triple=True,
                  proportion_train=1,
                  baseline_normalization="median_zscore")
        
        sample_selective = s2.get_epoch_selective(range(s2.sample, s2.delay), p=p)
        delay_selective = s2.get_epoch_selective(range(s2.delay + int(1.5*(1/s2.fs)), s2.response), p=p)
        
        sample_delay_selective = np.intersect1d(sample_selective, delay_selective)
        sample_selective = [n for n in sample_selective if n not in sample_delay_selective]
        delay_selective = [n for n in delay_selective if n not in sample_delay_selective]
        
        # Then look at weights on CDdelay of each of these pools
        s2.exclude_sample_orthog = True
        orthonormal_basis_learning, mean = s2.plot_CD(ctl=True)
        
        # Get idx
        sample_delay_selective_idx = [np.where(s2.good_neurons == n)[0][0] for n in sample_delay_selective]
        sample_selective_idx = [np.where(s2.good_neurons == n)[0][0] for n in sample_selective]
        delay_selective_idx = [np.where(s2.good_neurons == n)[0][0] for n in delay_selective]
        
        weights += [[np.mean(np.abs(orthonormal_basis_learning[sample_selective_idx])),
                          np.mean(np.abs(orthonormal_basis_learning[sample_delay_selective_idx])),
                          np.mean(np.abs(orthonormal_basis_learning[delay_selective_idx]))]]

    all_weights += [weights]


f = plt.figure(figsize=(10,6))

plt.bar(np.arange(3) - 0.2, np.mean(all_weights[0], axis=0), 0.4, 
        color='gold', label='Learning')
plt.bar(np.arange(3) + 0.2, np.mean(all_weights[1], axis=0), 0.4, 
        color='lightblue', label='Expert')

for i in range(len(all_weights[0])): # per FOV
    plt.scatter(np.arange(3) - 0.2, all_weights[0][i], color='yellow')
    plt.scatter(np.arange(3) + 0.2, all_weights[1][i], color='blue')
    
    for j in range(3):
        plt.plot([j-0.2, j+0.2],  [all_weights[0][i][j],  all_weights[1][i][j]],
                 color='grey')
        
plt.legend()
plt.xticks([0,1,2], ['Sample', 'Sample and delay', 'Delay'])
plt.ylabel('Weight on CD_delay')

all_weights = np.array(all_weights)

t_stat, p_value = ttest_rel(all_weights[0][:,0], all_weights[1][:,0]) # Paired t-test
print(t_stat, p_value)

t_stat, p_value = ttest_rel(all_weights[0][:,1], all_weights[1][:,1]) # Paired t-test
print(t_stat, p_value)

t_stat, p_value = ttest_rel(all_weights[0][:,2], all_weights[1][:,2]) # Paired t-test
print(t_stat, p_value)


#%% Compare the histogram of CDdelay distributions from learning to expert
all_weights = []

for paths in allpaths[1:]:
    
    weights = []
    
    for path in paths:
    
        # First, separate neurons into sample vs sample/delay vs delay selective
        s2 = Mode(path, use_reg = True, triple=True,
                  proportion_train=1,
                  baseline_normalization="median_zscore")
        
        # Then look at weights on CDdelay of each of these pools
        s2.exclude_sample_orthog = True
        orthonormal_basis_learning, mean = s2.plot_CD(ctl=True)
        
        weights += [orthonormal_basis_learning]

    all_weights += [weights]

f = plt.figure(figsize=(7,4))
plt.hist(np.abs(cat(all_weights[0])), color='red', alpha=0.2, label='Learning', bins=100)
plt.hist(np.abs(cat(all_weights[1])), color='blue', alpha=0.2, label='Expert', bins=100)
plt.xlim(right=0.2)
plt.legend()

f = plt.figure(figsize=(4,4))
plt.bar([0,1], [np.mean(np.abs(cat(all_weights[0]))),
                np.mean(np.abs(cat(all_weights[1])))])

plt.scatter(np.zeros(len(cat(all_weights[0]))),
            np.abs(cat(all_weights[0])))
plt.scatter(np.ones(len(cat(all_weights[1]))),
            np.abs(cat(all_weights[1])))

t_stat, p_value = ttest_ind(np.abs(cat(all_weights[0])),
                            np.abs(cat(all_weights[1]))) # Paired t-test
print(t_stat, p_value)



