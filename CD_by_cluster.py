# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 14:31:45 2024

@author: catherinewang

Use clusters to build various CD's in session instead of orthogonalizing to look 
for robust dimension
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

naivepath, learningpath, expertpath = [r'H:\data\BAYLORCW044\python\2024_05_22',
                    r'H:\data\BAYLORCW044\python\2024_06_06',
                  r'H:\data\BAYLORCW044\python\2024_06_19',]


#%% Import LDA cluster info

clusters = pd.read_pickle("/Users/catherinewang/Desktop/matched_topic_params/CW44_FOV1_table")
trialparams = clusters.trial_params[0]
num_clusters = len(trialparams.columns)
idx = pd.IndexSlice

learning = trialparams.loc[idx['learning', :]]
learning_normalized = pd.DataFrame(normalize(learning, norm='l1'), columns=learning.columns)#, index=learning.index)

expert = trialparams.loc[idx['expert', :]]
expert_normalized = pd.DataFrame(normalize(expert, norm='l1'), columns=expert.columns)#, index=expert.index)

#%% Run analysis
# lda = joblib.load(r'H:\data\BAYLORCW044\python\2024_06_19\full_model')
# expert_counts = pd.read_csv(r'H:\data\BAYLORCW044\python\2024_06_19\expert_counts')
# learning_counts = pd.read_csv(r'H:\data\BAYLORCW044\python\2024_06_06\learning_counts')
# ldaclusters_exp = lda.transform(expert_counts.filter(regex="^neuron"))
# ldaclusters = lda.transform(learning_counts.filter(regex="^neuron"))

path = expertpath
s1 = Mode(path, use_reg = True, triple=True)
orthonormal_basis, mean_exp = s1.plot_CD(ctl=True)

path = learningpath
s2 = Mode(path, use_reg = True, triple=True)
orthonormal_basis_learning, mean = s2.plot_CD(ctl=True)

#%% Get different CDs from the clusters for learning session applied to expert
# A trial belongs to a CD if the probability > 1/num clusters
cluster = 4 # focus on one cluster for now
learning_CDs = []
for cluster in range(num_clusters):
    # cluster_trials_all_idx = np.where(ldaclusters[:,cluster] > 1/ldaclusters.shape[1])[0]
    # cluster_trials_all = s2.i_good_trials[cluster_trials_all_idx]
    
    cluster_trials_all_idx = learning_normalized.index[learning_normalized['topic_{}'.format(cluster)] > 0.25].to_numpy()

    
    all_r_idx = [i for i in s2.i_good_non_stim_trials if s2.R_correct[i] and not bool(s2.early_lick[i])]
    all_l_idx = [i for i in s2.i_good_non_stim_trials if s2.L_correct[i] and not bool(s2.early_lick[i])]
    
    r_train_idx = [r for r in range(len(all_r_idx)) if all_r_idx[r] in cluster_trials_all_idx]
    l_train_idx = [r for r in range(len(all_l_idx)) if all_l_idx[r] in cluster_trials_all_idx]
    
    # Get R and L correct
    
    r_test_idx = r_train_idx
    l_test_idx = l_train_idx
    
    all_r_idx = [i for i in s2.i_good_non_stim_trials if s2.R_wrong[i] and not bool(s2.early_lick[i])]
    all_l_idx = [i for i in s2.i_good_non_stim_trials if s2.L_wrong[i] and not bool(s2.early_lick[i])]
    
    r_train_err_idx = [r for r in range(len(all_r_idx)) if all_r_idx[r] in cluster_trials_all_idx]
    l_train_err_idx = [r for r in range(len(all_l_idx)) if all_l_idx[r] in cluster_trials_all_idx]
    
    r_test_err_idx = r_train_err_idx
    l_test_err_idx = l_train_err_idx
    
    train_test_trials = (r_train_idx, l_train_idx, r_test_idx, l_test_idx)
    train_test_trials_err = (r_train_err_idx, l_train_err_idx, r_test_err_idx, l_test_err_idx)
    
    s2 = Mode(learningpath, use_reg = True, triple=True, 
              baseline_normalization="median_zscore",
              train_test_trials = [train_test_trials, train_test_trials_err])
    
    orthonormal_basis_learning, mean = s2.plot_CD(ctl=True)
    learning_CDs += [orthonormal_basis_learning]
    # s1.plot_appliedCD(orthonormal_basis_learning, mean)
    s2.plot_CD_opto(ctl=True)

    _, mean, meantrain, std = s1.plot_CD_opto(ctl=True, return_applied=True)
    s1.plot_CD_opto_applied(orthonormal_basis_learning, mean, meantrain, std)

#%% Look at scatter of weights compared to CD expert:
path = expertpath
s1 = Mode(path, use_reg = True, triple=True, proportion_train=1)
orthonormal_basis, mean_exp = s1.plot_CD(ctl=True)

for i in range(5):
    plt.scatter(orthonormal_basis, learning_CDs[i])
    plt.xlabel('Expert CD weights')
    plt.ylabel('Learning CD weights')
    plt.title('Cluster {} weights'.format(i+1))
    plt.show()

#%% look at cluster projections for opto trials
_, mean_exp = s2.plot_CD(ctl=True)
orthonormal_basis_learning, mean, meantrain, std = s1.plot_CD_opto(ctl=True, return_applied=True)
s1.plot_CD_opto_applied(orthonormal_basis_learning, mean, meantrain, std)
