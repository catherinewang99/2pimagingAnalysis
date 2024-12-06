# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 14:31:45 2024

@author: catherinewang

Use clusters to build various CD's in session instead of orthogonalizing to look 
for robust dimensions

Looks at analysis TRIAL wise
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
import os
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import ttest_rel
from scipy.stats import ttest_ind
import scipy 

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
# naivepath, learningpath, expertpath, clusterpath = [r'F:\data\BAYLORCW034\python\2023_10_22',
#                    r'F:\data\BAYLORCW034\python\2023_10_22',
#                    r'F:\data\BAYLORCW034\python\2023_10_27',
#                   r'H:\data\matched_topic_params\CW34_table']


agg_mice_paths = [[r'F:\data\BAYLORCW032\python\2023_10_08',
          r'F:\data\BAYLORCW032\python\2023_10_16',
          r'F:\data\BAYLORCW032\python\2023_10_25',
          r'H:\data\matched_topic_params\CW32_table'],
         
        [ r'F:\data\BAYLORCW034\python\2023_10_12',
              r'F:\data\BAYLORCW034\python\2023_10_22',
              r'F:\data\BAYLORCW034\python\2023_10_27',
              r'H:\data\matched_topic_params\CW34_table'],

        [r'F:\data\BAYLORCW036\python\2023_10_09',
            r'F:\data\BAYLORCW036\python\2023_10_19',
            r'F:\data\BAYLORCW036\python\2023_10_30',
            r'H:\data\matched_topic_params\CW36_table'],
    
        [r'F:\data\BAYLORCW035\python\2023_10_26',
            r'F:\data\BAYLORCW035\python\2023_12_07',
            r'F:\data\BAYLORCW035\python\2023_12_15',
            r'H:\data\matched_topic_params\CW35_table'],
     
        [r'F:\data\BAYLORCW037\python\2023_11_21',
             r'F:\data\BAYLORCW037\python\2023_12_08',
             r'F:\data\BAYLORCW037\python\2023_12_15',
             r'H:\data\matched_topic_params\CW37_table'],
        
        [r'H:\data\BAYLORCW044\python\2024_05_22',
              r'H:\data\BAYLORCW044\python\2024_06_06',
            r'H:\data\BAYLORCW044\python\2024_06_19',
            r'H:\data\matched_topic_params\CW44_FOV1_table'],
        
        [r'H:\data\BAYLORCW044\python\2024_05_23',
            r'H:\data\BAYLORCW044\python\2024_06_04',
            r'H:\data\BAYLORCW044\python\2024_06_18',
            r'H:\data\matched_topic_params\CW44_FOV2_table'],
        
        # [r'H:\data\BAYLORCW046\python\2024_05_29',
        #     r'H:\data\BAYLORCW046\python\2024_06_07',
        #     r'H:\data\BAYLORCW046\python\2024_06_24',
        #     r'H:\data\matched_topic_params\CW46_FOV1_table'],
        
        [r'H:\data\BAYLORCW046\python\2024_05_30',
            r'H:\data\BAYLORCW046\python\2024_06_10',
            r'H:\data\BAYLORCW046\python\2024_06_27',
            r'H:\data\matched_topic_params\CW46_FOV2_table'],
        
        [r'H:\data\BAYLORCW046\python\2024_05_31',
            r'H:\data\BAYLORCW046\python\2024_06_11',
            r'H:\data\BAYLORCW046\python\2024_06_26',
            r'H:\data\matched_topic_params\CW46_FOV3_table']
                            
        ]
# naivepath, learningpath, expertpath, clusterpath = [
#                     r'H:\data\BAYLORCW046\python\2024_05_31',
#                     r'H:\data\BAYLORCW046\python\2024_06_11',
#                     r'H:\data\BAYLORCW046\python\2024_06_26',
                    # r'H:\data\matched_topic_params\CW46_FOV3_table']

#%% Import LDA cluster info

clusters = pd.read_pickle(clusterpath)
trialparams = np.mean(clusters.trial_params.to_numpy())
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
s1 = Mode(path, use_reg = True, triple=True, baseline_normalization="median_zscore")
orthonormal_basis, mean_exp = s1.plot_CD(ctl=True)

path = learningpath
s2 = Mode(path, use_reg = True, triple=True, baseline_normalization="median_zscore")
orthonormal_basis_learning, mean = s2.plot_CD(ctl=True)

#%% Number of trials per cluster

# Max method:
max_cluster_by_trial = np.argmax(learning_normalized, axis=1)
max_cluster_by_trial_exp = np.argmax(expert_normalized, axis=1)
for c in set(max_cluster_by_trial):
    plt.scatter([c-0.2], [sum(max_cluster_by_trial == c)], color='black')
    plt.scatter([c+0.2], [sum(max_cluster_by_trial_exp == c)], color='black')
    plt.plot([c-0.2, c+0.2], [sum(max_cluster_by_trial == c),
                              sum(max_cluster_by_trial_exp == c)],
             color='grey', ls='--')
plt.scatter([c-0.2], [sum(max_cluster_by_trial == c)], color='black', label='Max')

# plt.show()
for cluster in range(num_clusters):
    cluster_trials_all_idx = learning_normalized.index[learning_normalized['topic_{}'.format(cluster)] > 1/num_clusters].to_numpy()
    plt.scatter([cluster-0.2], [len(cluster_trials_all_idx)], color='red')
    cluster_trials_all_idx_exp = expert_normalized.index[expert_normalized['topic_{}'.format(cluster)] > 1/num_clusters].to_numpy()
    plt.scatter([cluster+0.2], [len(cluster_trials_all_idx_exp)], color='red')
    plt.plot([cluster-0.2, cluster+0.2], [len(cluster_trials_all_idx),
                              len(cluster_trials_all_idx_exp)],
             color='grey', ls='--')
plt.scatter([cluster-0.2], [len(cluster_trials_all_idx)], color='red', label='Threshold')

plt.ylabel('Number of trials belong to cluster')
plt.xlabel('Cluster #')
plt.legend()
plt.show()

# Plot heatmap over trials
f,ax=plt.subplots(1,2,figsize=(15,2))

max_cluster_by_trial = np.argmax(learning_normalized, axis=1)
trial_stack = np.zeros(learning_normalized.shape[0])
for c in set(max_cluster_by_trial):
    trial_stack = np.vstack((trial_stack, max_cluster_by_trial == c))
ax[0].imshow(trial_stack[1:],aspect='auto',interpolation='none')
ax[0].set_title('Learning trials')
ax[0].set_ylabel('Cluster #')
ax[0].set_xlabel('Trial #')

max_cluster_by_trial = np.argmax(expert_normalized, axis=1)
trial_stack = np.zeros(expert_normalized.shape[0])
for c in set(max_cluster_by_trial):
    trial_stack = np.vstack((trial_stack, max_cluster_by_trial == c))
ax[1].imshow(trial_stack[1:],aspect='auto',interpolation='none')
ax[1].set_title('Expert trials')
plt.tight_layout()
plt.show()

# Left/right trials per cluster
# Max method:
f,ax=plt.subplots(1,2,figsize=(10,5), sharey='row')
max_cluster_by_trial = np.argmax(learning_normalized, axis=1)
max_cluster_by_trial_exp = np.argmax(expert_normalized, axis=1)
for c in set(max_cluster_by_trial):
    ax[0].scatter([c-0.2], [len([i for i in range(len(max_cluster_by_trial)) if s2.L_correct[s2.i_good_trials[i]] and max_cluster_by_trial[i] == c])], color='red')
    ax[0].scatter([c+0.2], [len([i for i in range(len(max_cluster_by_trial)) if s2.R_correct[s2.i_good_trials[i]] and max_cluster_by_trial[i] == c])], color='blue')
    ax[0].plot([c-0.2, c+0.2], [len([i for i in range(len(max_cluster_by_trial)) if s2.L_correct[s2.i_good_trials[i]] and max_cluster_by_trial[i] == c]),
                                len([i for i in range(len(max_cluster_by_trial)) if s2.R_correct[s2.i_good_trials[i]] and max_cluster_by_trial[i] == c])],
               color='grey', ls='--')
    ax[1].scatter([c-0.2], [len([i for i in range(len(max_cluster_by_trial_exp)) if s1.L_correct[s1.i_good_trials[i]] and max_cluster_by_trial_exp[i] == c])], color='red')
    ax[1].scatter([c+0.2], [len([i for i in range(len(max_cluster_by_trial_exp)) if s1.R_correct[s1.i_good_trials[i]] and max_cluster_by_trial_exp[i] == c])], color='blue')
    ax[1].plot([c-0.2, c+0.2], [len([i for i in range(len(max_cluster_by_trial_exp)) if s1.L_correct[s1.i_good_trials[i]] and max_cluster_by_trial_exp[i] == c]),
                                len([i for i in range(len(max_cluster_by_trial_exp)) if s1.R_correct[s1.i_good_trials[i]] and max_cluster_by_trial_exp[i] == c])],
               color='grey', ls='--')
    
ax[0].set_title('Learning clusters')
ax[1].set_title('Expert clusters')
ax[0].set_ylabel('Number of trials belong to cluster')
ax[0].set_xlabel('Cluster #')
plt.legend()
plt.show()



#%% Decoding accuracy of clusters within

all_learn_accs = []
for cluster in range(num_clusters):
    # cluster_trials_all_idx = np.where(ldaclusters[:,cluster] > 1/ldaclusters.shape[1])[0]
    # cluster_trials_all = s2.i_good_trials[cluster_trials_all_idx]
    
    cluster_trials_all_idx = learning_normalized.index[learning_normalized['topic_{}'.format(cluster)] > 1/num_clusters].to_numpy()
    # cluster_trials_all_idx = np.where(np.argmax(learning_normalized, axis=1) == cluster)[0]
   
    s2 = Mode(learningpath, use_reg = True, triple=True, 
              baseline_normalization="median_zscore",
              i_good = cluster_trials_all_idx,
              proportion_train = 0.5,
              lda_cluster=True)
    try:
        _, _, db, acc_learning = s2.decision_boundary(mode_input='choice', persistence=False)
    except np.linalg.LinAlgError:
        all_learn_accs += [0]
        continue
    
    all_learn_accs += [np.mean(acc_learning)]
  
all_exp_accs = []
for cluster in range(num_clusters):
    # cluster_trials_all_idx = np.where(ldaclusters[:,cluster] > 1/ldaclusters.shape[1])[0]
    # cluster_trials_all = s2.i_good_trials[cluster_trials_all_idx]
    
    cluster_trials_all_idx = expert_normalized.index[expert_normalized['topic_{}'.format(cluster)] > 1/num_clusters].to_numpy()
    # cluster_trials_all_idx = np.where(np.argmax(expert_normalized, axis=1) == cluster)[0]
   
    s2 = Mode(expertpath, use_reg = True, triple=True, 
              baseline_normalization="median_zscore",
              i_good = cluster_trials_all_idx,
              proportion_train = 0.5,
              lda_cluster=True)
    try:
        _, _, db, acc_learning = s2.decision_boundary(mode_input='choice', persistence=False)
    except np.linalg.LinAlgError:
        all_exp_accs += [0]
        continue
    all_exp_accs += [np.mean(acc_learning)]  
    
plt.bar(np.arange(num_clusters)-0.2, all_learn_accs, 0.4, label='Learning')
plt.bar(np.arange(num_clusters)+0.2, all_exp_accs, 0.4, label='Expert')
plt.legend()
plt.ylim(bottom=0.5)
plt.xlabel('Cluster number')
plt.ylabel('Accuracy %')

#%% Decoding acc across clusters
#FIXME: Need to do the correct within cluster CD assignment
main_cluster = 0
agg_accs = []
for main_cluster in range(num_clusters):
    all_learn_accs = []

    for cluster in range(num_clusters):
        # cluster_trials_all_idx = np.where(ldaclusters[:,cluster] > 1/ldaclusters.shape[1])[0]
        # cluster_trials_all = s2.i_good_trials[cluster_trials_all_idx]
        
        cluster_trials_all_idx = learning_normalized.index[learning_normalized['topic_{}'.format(cluster)] > 0.25].to_numpy()
        # cluster_trials_all_idx = np.where(np.argmax(learning_normalized, axis=1) == cluster)[0]
       
        s2 = Mode(learningpath, use_reg = True, triple=True, 
                  baseline_normalization="median_zscore",
                  # i_good = cluster_trials_all_idx,
                  # proportion_train = 0.4,
                  lda_cluster=True,
                  train_test_trials = cluster_trials_all_idx)
        if main_cluster == cluster:
            orthonormal_basis, mean, db, acc_learning = s2.decision_boundary(mode_input='choice', persistence=False)
        else:
            acc_learning = s2.decision_boundary_appliedCD('choice', orthonormal_basis, mean, db, persistence=False)
    
        all_learn_accs += [np.mean(acc_learning)]
    agg_accs += [all_learn_accs]


agg_accs = np.array(agg_accs)
plt.bar(np.arange(num_clusters)-0.15, agg_accs[:, 0], 0.1)
plt.bar(np.arange(num_clusters)-0.05, agg_accs[:, 1], 0.1,)
plt.bar(np.arange(num_clusters)+0.05, agg_accs[:, 2], 0.1)
plt.bar(np.arange(num_clusters)+0.15, agg_accs[:, 3], 0.1)
plt.legend()
plt.ylim(bottom=0.5)
plt.xlabel('Cluster number')
plt.ylabel('Accuracy %')

#%% End point analysis across clusters in learning session vs expert
main_cluster = 1
compare_cluster = 0

cluster_trials_all_idx = learning_normalized.index[learning_normalized['topic_{}'.format(compare_cluster)] > 0.25].to_numpy()
cluster_trials_all_idx = np.where(np.argmax(learning_normalized, axis=1) == compare_cluster)[0]


s2 = Mode(learningpath, use_reg = True, triple=True, 
          baseline_normalization="median_zscore",
            train_test_trials = cluster_trials_all_idx,
            lda_cluster=True)

orthonormal_basis_learning, mean = s2.plot_CD(ctl=True, plot=False)

## Look at main cluster
cluster_trials_all_idx = learning_normalized.index[learning_normalized['topic_{}'.format(main_cluster)] > 0.25].to_numpy()
cluster_trials_all_idx = np.where(np.argmax(learning_normalized, axis=1) == main_cluster)[0]

s2 = Mode(learningpath, use_reg = True, triple=True, 
          baseline_normalization="median_zscore",
            train_test_trials = cluster_trials_all_idx,
            lda_cluster=True)

proj_allDimR, proj_allDimL = s2.plot_CD(ctl=True, plot=False, auto_corr_return=True)
proj_allDimR_applied, proj_allDimL_applied = s2.plot_appliedCD(orthonormal_basis_learning, mean, auto_corr_return=True, plot=False)

# Plot end points

plt.scatter(proj_allDimR[:, s2.response-1], proj_allDimR_applied[:, s2.response-1], color='b')
plt.scatter(proj_allDimL[:, s2.response-1], proj_allDimL_applied[:, s2.response-1], color='r')
plt.axhline(0, ls = '--', color='black')
plt.axvline(0, ls = '--', color='black')
plt.ylabel('Outside cluster CD')
plt.xlabel('Within cluster CD')
plt.title('Learning: cluster main {} vs. cluster {}'.format(main_cluster, compare_cluster))
plt.show()

#Expert
cluster_trials_all_idx = expert_normalized.index[expert_normalized['topic_{}'.format(compare_cluster)] > 0.25].to_numpy()
cluster_trials_all_idx = np.where(np.argmax(expert_normalized, axis=1) == compare_cluster)[0]

s2 = Mode(expertpath, use_reg = True, triple=True, 
          baseline_normalization="median_zscore",
            train_test_trials = cluster_trials_all_idx,
            lda_cluster=True)

orthonormal_basis_learning, mean = s2.plot_CD(ctl=True, plot=False)

## Look at main cluster
cluster_trials_all_idx = expert_normalized.index[expert_normalized['topic_{}'.format(main_cluster)] > 0.25].to_numpy()
cluster_trials_all_idx = np.where(np.argmax(expert_normalized, axis=1) == main_cluster)[0]

s2 = Mode(expertpath, use_reg = True, triple=True, 
          baseline_normalization="median_zscore",
            train_test_trials = cluster_trials_all_idx,
            lda_cluster=True)

proj_allDimR, proj_allDimL = s2.plot_CD(ctl=True, plot=False, auto_corr_return=True)
proj_allDimR_applied, proj_allDimL_applied = s2.plot_appliedCD(orthonormal_basis_learning, mean, auto_corr_return=True, plot=False)

# Plot end points

plt.scatter(proj_allDimR[:, s2.response-1], proj_allDimR_applied[:, s2.response-1], color='b')
plt.scatter(proj_allDimL[:, s2.response-1], proj_allDimL_applied[:, s2.response-1], color='r')
plt.axhline(0, ls = '--', color='black')
plt.axvline(0, ls = '--', color='black')
plt.ylabel('Outside cluster CD')
plt.xlabel('Within cluster CD')
plt.title('Expert: cluster main {} vs. cluster {}'.format(main_cluster, compare_cluster))
plt.show()

#%% Get different CDs from the clusters for learning session applied to expert and opto projections
# A trial belongs to a CD if the probability > 1/num clusters OR max method

learning_CDs = []
for cluster in range(num_clusters):
    # cluster_trials_all_idx = np.where(ldaclusters[:,cluster] > 1/ldaclusters.shape[1])[0]
    # cluster_trials_all = s2.i_good_trials[cluster_trials_all_idx]
    
    cluster_trials_all_idx = learning_normalized.index[learning_normalized['topic_{}'.format(cluster)] > 1/num_clusters].to_numpy()
    cluster_trials_all_idx = np.where(np.argmax(learning_normalized, axis=1) == cluster)[0]

    s2 = Mode(learningpath, use_reg = True, triple=True, 
              baseline_normalization="median_zscore",
              train_test_trials = cluster_trials_all_idx,
              lda_cluster=True)
    
    orthonormal_basis_learning, mean = s2.plot_CD(ctl=True, plot=False)
    learning_CDs += [orthonormal_basis_learning]
    # s1.plot_appliedCD(orthonormal_basis_learning, mean)
    s2.plot_CD_opto(ctl=True)

    _, mean, meantrain, std = s1.plot_CD_opto(ctl=True, return_applied=True, plot=False)
    s1.plot_CD_opto_applied(orthonormal_basis_learning, mean, meantrain, std)

#%% Get different CDs from the clusters for expert session
# A trial belongs to a CD if the probability > 1/num clusters OR max method

learning_CDs = []
for cluster in range(num_clusters):
    # cluster_trials_all_idx = np.where(ldaclusters[:,cluster] > 1/ldaclusters.shape[1])[0]
    # cluster_trials_all = s2.i_good_trials[cluster_trials_all_idx]
    
    cluster_trials_all_idx = expert_normalized.index[expert_normalized['topic_{}'.format(cluster)] > 1/num_clusters].to_numpy()
    cluster_trials_all_idx = np.where(np.argmax(expert_normalized, axis=1) == cluster)[0]

    s2 = Mode(expertpath, use_reg = True, triple=True, 
              baseline_normalization="median_zscore",
              train_test_trials = cluster_trials_all_idx,
              lda_cluster=True)
    
    orthonormal_basis_learning, mean = s2.plot_CD(ctl=True, plot=False)

    s2.plot_CD_opto(ctl=True)

    
    
#%% Get different CDs directly from learning/expert clusters

for cluster in range(num_clusters):
    
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
    
    # orthonormal_basis_learning, mean = s2.plot_CD(ctl=True, plot=False)
    s2.plot_CD_opto(ctl=True)
    
    cluster_trials_all_idx = expert_normalized.index[expert_normalized['topic_{}'.format(cluster)] > 0.25].to_numpy()

    all_r_idx = [i for i in s1.i_good_non_stim_trials if s1.R_correct[i] and not bool(s1.early_lick[i])]
    all_l_idx = [i for i in s1.i_good_non_stim_trials if s1.L_correct[i] and not bool(s1.early_lick[i])]
    
    r_train_idx = [r for r in range(len(all_r_idx)) if all_r_idx[r] in cluster_trials_all_idx]
    l_train_idx = [r for r in range(len(all_l_idx)) if all_l_idx[r] in cluster_trials_all_idx]
    
    # Get R and L correct
    
    r_test_idx = r_train_idx
    l_test_idx = l_train_idx
    
    all_r_idx = [i for i in s1.i_good_non_stim_trials if s1.R_wrong[i] and not bool(s1.early_lick[i])]
    all_l_idx = [i for i in s1.i_good_non_stim_trials if s1.L_wrong[i] and not bool(s1.early_lick[i])]
    
    r_train_err_idx = [r for r in range(len(all_r_idx)) if all_r_idx[r] in cluster_trials_all_idx]
    l_train_err_idx = [r for r in range(len(all_l_idx)) if all_l_idx[r] in cluster_trials_all_idx]
    
    r_test_err_idx = r_train_err_idx
    l_test_err_idx = l_train_err_idx
    
    train_test_trials = (r_train_idx, l_train_idx, r_test_idx, l_test_idx)
    train_test_trials_err = (r_train_err_idx, l_train_err_idx, r_test_err_idx, l_test_err_idx)
    
    s1 = Mode(expertpath, use_reg = True, triple=True, 
              baseline_normalization="median_zscore",
              train_test_trials = [train_test_trials, train_test_trials_err])
    s1.plot_CD_opto(ctl=True)
    
    
    

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


#%% Iterate over all LDA clusters to get some general info

path = r'H:\data\matched_topic_params'
files = [f for f in os.listdir(path) if 'CW' in f]
num_clusters = []
num_trials_learning = []
num_trials_expert = []

for file in files:
    clusters = pd.read_pickle(path + '\\' + file)
    trialparams = np.mean(clusters.trial_params.to_numpy())
    num_clusters += [len(trialparams.columns)]
    idx = pd.IndexSlice
    
    learning = trialparams.loc[idx['learning', :]]
    learning_normalized = pd.DataFrame(normalize(learning, norm='l1'), columns=learning.columns)#, index=learning.index)
    
    expert = trialparams.loc[idx['expert', :]]
    expert_normalized = pd.DataFrame(normalize(expert, norm='l1'), columns=expert.columns)#, index=expert.index)
    max_cluster_by_trial = np.argmax(learning_normalized, axis=1)
    max_cluster_by_trial_exp = np.argmax(expert_normalized, axis=1)
    _, counts = np.unique(max_cluster_by_trial, return_counts=True)
    num_trials_learning += [counts]    
    _, counts = np.unique(max_cluster_by_trial_exp, return_counts=True)
    num_trials_expert += [counts]
num_trials_learning = cat(num_trials_learning)
num_trials_expert = cat(num_trials_expert)
    
# Make plots

# Number of clusters over FOVS
f = plt.figure(figsize=(5,5))
plt.bar([0], [np.mean(num_clusters)])
jitter_x = np.random.normal(0, 0.1, len(num_clusters))  # Jitter for x (mean=0, std=0.1)
plt.scatter(np.zeros(len(files)) + jitter_x, num_clusters)
plt.ylabel('Number of clusters')

# Number of trials per cluster all FOVs
f = plt.figure(figsize=(5,5))
plt.bar([0,1], [np.mean(num_trials_learning), np.mean(num_trials_expert)])
plt.scatter(np.zeros(len(num_trials_learning)), num_trials_learning)
plt.scatter(np.ones(len(num_trials_expert)), num_trials_expert)
for i in range(len(num_trials_expert)):
    plt.plot([0,1], [num_trials_learning[i],
                     num_trials_expert[i]],
             color='grey', alpha=0.5)
plt.xticks([0,1],['Learning', 'Expert'])
plt.ylabel('Number of trials')

# Delta of number of trials per cluster all FOVs
f = plt.figure(figsize=(5,5))
plt.hist(num_trials_expert-num_trials_learning, bins=20)
plt.xlabel('Clusters')
plt.ylabel('Delta in num of trials')

# Delta vs size of cluster scatter
f = plt.figure(figsize=(5,5))
plt.scatter(num_trials_expert, num_trials_expert-num_trials_learning)
plt.xlabel('Number of trials in cluster: expert')
plt.ylabel('Delta in num of trials (exp - learning)')


#%% Similarity of CDs over all sessions learning vs expert

# Takes a while to run
cd_learning_sim = []
cd_expert_sim = []
cd_learning_var = []
cd_expert_var = []
cd_learning_all = []
cd_expert_all = []

for paths in agg_mice_paths:
    clusters = pd.read_pickle(paths[3])
    trialparams = np.mean(clusters.trial_params.to_numpy())
    num_clusters = len(trialparams.columns)

    idx = pd.IndexSlice
    
    learning = trialparams.loc[idx['learning', :]]
    learning_normalized = pd.DataFrame(normalize(learning, norm='l1'), columns=learning.columns)#, index=learning.index)
    cds = []
    for cluster in range(num_clusters):
        cluster_trials_all_idx = np.where(np.argmax(learning_normalized, axis=1) == cluster)[0]

        s2 = Mode(paths[1], use_reg = True, triple=True, 
                  baseline_normalization="median_zscore",
                  train_test_trials = cluster_trials_all_idx,
                  lda_cluster=True)
        
        orthonormal_basis, mean = s2.plot_CD(ctl=True, plot=False)
        cds += [orthonormal_basis]
    cos_sim = cosine_similarity(cds)
    overall_similarity = np.mean(cos_sim[np.triu_indices_from(cos_sim, k=1)])  # Mean of upper triangle
    cd_learning_sim += [overall_similarity]
    cd_learning_var += [np.var(cos_sim[np.triu_indices_from(cos_sim, k=1)])]
    cd_learning_all += [cds]                 
       
    expert = trialparams.loc[idx['expert', :]]
    expert_normalized = pd.DataFrame(normalize(expert, norm='l1'), columns=expert.columns)#, index=expert.index)
    cds = []
    for cluster in range(num_clusters):
        cluster_trials_all_idx = np.where(np.argmax(expert_normalized, axis=1) == cluster)[0]

        s2 = Mode(paths[2], use_reg = True, triple=True, 
                  baseline_normalization="median_zscore",
                  train_test_trials = cluster_trials_all_idx,
                  lda_cluster=True)
        orthonormal_basis, mean = s2.plot_CD(ctl=True, plot=False)
        cds += [orthonormal_basis]
    cos_sim = cosine_similarity(cds)
    overall_similarity = np.mean(cos_sim[np.triu_indices_from(cos_sim, k=1)])  # Mean of upper triangle
    cd_expert_sim += [overall_similarity]
    cd_expert_var += [np.var(cos_sim[np.triu_indices_from(cos_sim, k=1)])]
    cd_expert_all += [cds]
# Plot

f = plt.figure(figsize=(5,5))
plt.bar([0,1],[np.mean(np.abs(cd_learning_sim)), np.mean(np.abs(cd_expert_sim))])
plt.scatter(np.zeros(len(cd_learning_sim)), np.abs(cd_learning_sim))
plt.scatter(np.ones(len(cd_expert_sim)), np.abs(cd_expert_sim))
for i in range(len(cd_learning_sim)):
    plt.plot([0,1], [np.abs(cd_learning_sim)[i], np.abs(cd_expert_sim)[i]], color='grey')
plt.xticks([0,1], ['Learning', 'Expert'])
plt.ylabel('Cosine similarity')
plt.title('Mean cosine similarity between cluster CDs')

t_stat, p_value = ttest_rel(np.abs(cd_learning_sim), np.abs(cd_expert_sim)) # Paired t-test
print(t_stat, p_value)

# Variance of CD similarities

f = plt.figure(figsize=(5,5))
plt.bar([0,1],[np.mean(np.abs(cd_learning_var)), np.mean(np.abs(cd_expert_var))])
plt.scatter(np.zeros(len(cd_learning_var)), np.abs(cd_learning_var))
plt.scatter(np.ones(len(cd_expert_var)), np.abs(cd_expert_var))
for i in range(len(cd_learning_var)):
    plt.plot([0,1], [np.abs(cd_learning_var)[i], np.abs(cd_expert_var)[i]], color='grey')
plt.xticks([0,1], ['Learning', 'Expert'])
plt.ylabel('Variance')
plt.title('Variance of cosine similarity between cluster CDs')

t_stat, p_value = ttest_rel(np.abs(cd_learning_var), np.abs(cd_expert_var)) # Paired t-test
print(t_stat, p_value)

#%% Rotation of CDs within cluster
all_cd_rotations = []
for fov in range(len(cd_expert_all)):
    cd_rotations = []
    for cl in range(len(cd_expert_all[fov])):
        if len(cd_expert_all[fov][cl]) == len(cd_learning_all[fov][cl]): # Catch bug case of CW46
            cos_sim = cosine_similarity(cd_learning_all[fov][cl].reshape(1, -1), 
                                        cd_expert_all[fov][cl].reshape(1, -1))[0][0]
            cd_rotations += [cos_sim]
        else:
            print('{} field of view excluded'.format(agg_mice_paths[fov][3]))
    all_cd_rotations += [cd_rotations]

# Plot all angles
cat_all_cd_rotations = np.abs(cat(all_cd_rotations))         
plt.hist(cat_all_cd_rotations)
plt.ylabel('Number of cluster CDs')
plt.xlabel('Cosine similarity (learning-->expert)')
plt.show()

# Plot all angles grouped by FOV
for fov in range(len(all_cd_rotations)):
    plt.scatter(np.ones(len(all_cd_rotations[fov]))*fov, np.abs(all_cd_rotations[fov]))
plt.ylabel('Cosine similarity')
plt.xlabel('Field of view')
plt.show()


#%% Robustness of CDs compared to trial numbers in expert session
# Takes a while to run
maxmethod = False
robustness_learning = []
robustness_expert = []
num_trials_learning = []
num_trials_expert = []

for paths in agg_mice_paths:
    clusters = pd.read_pickle(paths[3])
    trialparams = np.mean(clusters.trial_params.to_numpy())
    num_clusters = len(trialparams.columns)

    idx = pd.IndexSlice
    
    learning = trialparams.loc[idx['learning', :]]
    learning_normalized = pd.DataFrame(normalize(learning, norm='l1'), columns=learning.columns)#, index=learning.index)
    
    cds = []
    trials = []
    for cluster in range(num_clusters):
        if maxmethod:
            cluster_trials_all_idx = np.where(np.argmax(learning_normalized, axis=1) == cluster)[0]
        else:
            cluster_trials_all_idx = learning_normalized.index[learning_normalized['topic_{}'.format(cluster)] > 1/num_clusters].to_numpy()

        s2 = Mode(paths[1], use_reg = True, triple=True, 
                  baseline_normalization="median_zscore",
                  train_test_trials = cluster_trials_all_idx,
                  lda_cluster=True)
        try:
            rob = s2.modularity_proportion_by_CD()
        except np.linalg.LinAlgError:
            rob = 0
            
        cds += [rob]
        trials += [len(cluster_trials_all_idx)]
        
    robustness_learning += [cds]
    num_trials_learning += [trials]
    
    expert = trialparams.loc[idx['expert', :]]
    expert_normalized = pd.DataFrame(normalize(expert, norm='l1'), columns=expert.columns)#, index=expert.index)
    cds = []
    trials = []
    for cluster in range(num_clusters):
        if maxmethod:
            cluster_trials_all_idx = np.where(np.argmax(expert_normalized, axis=1) == cluster)[0]
        else:
            cluster_trials_all_idx = expert_normalized.index[expert_normalized['topic_{}'.format(cluster)] > 1/num_clusters].to_numpy()

        s2 = Mode(paths[2], use_reg = True, triple=True, 
                  baseline_normalization="median_zscore",
                  train_test_trials = cluster_trials_all_idx,
                  lda_cluster=True)
        try:
            rob = s2.modularity_proportion_by_CD()
        except np.linalg.LinAlgError:
            rob = 0
            
        cds += [rob]
        trials += [len(cluster_trials_all_idx)]
        
    robustness_expert += [cds]
    num_trials_expert += [trials]
    
#%% Decoding accuracy of CDs clusters
# Takes a while to run

# robustness_learning = []
# robustness_expert = []
acc_learning = []
acc_expert = []

for paths in agg_mice_paths:
    clusters = pd.read_pickle(paths[3])
    trialparams = np.mean(clusters.trial_params.to_numpy())
    num_clusters = len(trialparams.columns)

    idx = pd.IndexSlice
    
    learning = trialparams.loc[idx['learning', :]]
    learning_normalized = pd.DataFrame(normalize(learning, norm='l1'), columns=learning.columns)#, index=learning.index)
    
    cds = []
    allaccs = []
    for cluster in range(num_clusters):
        if maxmethod:
            cluster_trials_all_idx = np.where(np.argmax(learning_normalized, axis=1) == cluster)[0]
        else:
            cluster_trials_all_idx = learning_normalized.index[learning_normalized['topic_{}'.format(cluster)] > 1/num_clusters].to_numpy()

        s2 = Mode(paths[1], use_reg = True, triple=True, 
                  baseline_normalization="median_zscore",
                  i_good = cluster_trials_all_idx,                  
                  lda_cluster=True)
        # try:
        #     rob = s2.modularity_proportion_by_CD()
        # except np.linalg.LinAlgError:
        #     rob = 0
            
        try:
            _, _, db, acc = s2.decision_boundary(mode_input='choice', persistence=False)
        except np.linalg.LinAlgError:
            acc = [0]
            
        # cds += [rob]
        allaccs += [np.mean(acc)]
        
    robustness_learning += [cds]
    acc_learning += [allaccs]
    
    expert = trialparams.loc[idx['expert', :]]
    expert_normalized = pd.DataFrame(normalize(expert, norm='l1'), columns=expert.columns)#, index=expert.index)
    cds = []
    allaccs = []
    for cluster in range(num_clusters):
        if maxmethod:
            cluster_trials_all_idx = np.where(np.argmax(expert_normalized, axis=1) == cluster)[0]
        else:
            cluster_trials_all_idx = expert_normalized.index[expert_normalized['topic_{}'.format(cluster)] > 1/num_clusters].to_numpy()
            
        s2 = Mode(paths[2], use_reg = True, triple=True, 
                  baseline_normalization="median_zscore",
                  i_good = cluster_trials_all_idx,                  
                  lda_cluster=True)
        # try:
        #     rob = s2.modularity_proportion_by_CD()
        # except np.linalg.LinAlgError:
        #     rob = 0
            
        try:
            _, _, db, acc = s2.decision_boundary(mode_input='choice', persistence=False)
        except np.linalg.LinAlgError:
            acc = [0]
            
        # cds += [rob]
        allaccs += [np.mean(acc)]
        
    robustness_expert += [cds]
    acc_expert += [allaccs]
    
#%% Distribution of accuracies
all_acc_learning = cat(acc_learning)
all_acc_expert = cat(acc_expert)
all_acc_learning_filt, all_acc_expert_filt = [], []
for i in range(len(all_acc_learning)):
    if all_acc_expert[i] != 0 and all_acc_learning[i] != 0:
        all_acc_learning_filt += [all_acc_learning[i]]
        all_acc_expert_filt += [all_acc_expert[i]]



# Plot all decoding accuracry values across learning filtered
plt.bar([0,1], [np.mean(all_acc_learning_filt), np.mean(all_acc_expert_filt)])
plt.scatter(np.zeros(len(all_acc_learning_filt)), all_acc_learning_filt)
plt.scatter(np.ones(len(all_acc_expert_filt)), all_acc_expert_filt)
for i in range(len(all_acc_expert_filt)):
    plt.plot([0,1],[all_acc_learning_filt[i], all_acc_expert_filt[i]],
             color='grey')
plt.axhline(0.5, ls='--', color='black')
plt.ylabel('Decoding accuracy')
plt.xticks([0,1], ['Learning', 'Expert'])
plt.title('Decoding accuracy of cluster CDs across learning')
plt.ylim(bottom=0.4)
plt.show()

t_stat, p_value = ttest_ind(all_acc_learning_filt, all_acc_expert_filt) # Paired t-test
print(t_stat, p_value)

plt.hist(np.array(all_acc_expert_filt) - np.array(all_acc_learning_filt))
plt.axvline(0, ls = '--', color='black')
plt.xlabel('Delta of decoding accuracies')
plt.ylabel('Number of clusters')


#%% Robustness of CDs vs size of clusters
f, ax = plt.subplots(2,2, figsize=(10,10), sharey='row', sharex='col')

ax[0,0].scatter(cat(num_trials_learning), cat(robustness_learning))
ax[0,0].set_ylabel('Learning CD robustness')
ax[1,0].scatter(cat(num_trials_learning), cat(robustness_expert))
ax[1,0].set_ylabel('Expert CD robustness')

ax[0,1].scatter(cat(num_trials_expert), cat(robustness_learning))
ax[1,0].set_xlabel('Learning number of trials')
ax[1,1].scatter(cat(num_trials_expert), cat(robustness_expert))
ax[1,1].set_xlabel('Expert number of trials')


print(scipy.stats.pearsonr(cat(num_trials_learning), cat(robustness_learning)))
print(scipy.stats.pearsonr(cat(num_trials_learning), cat(robustness_expert)))


#%% Delta of robustness distribution
all_robustness_expert = cat(robustness_expert)
all_robustness_learning = cat(robustness_learning)
all_robustness_learning_filt, all_robustness_expert_filt = [],[]
for i in range(len(all_robustness_learning)):
    if all_robustness_expert[i] != 0 and all_robustness_learning[i] != 0:
        all_robustness_learning_filt += [all_robustness_learning[i]]
        all_robustness_expert_filt += [all_robustness_expert[i]]
        
delta_rob = np.array(all_robustness_expert_filt) - np.array(all_robustness_learning_filt)

plt.hist(delta_rob)
plt.axvline(x=0, ls='--', color='purple')
plt.xlabel('Delta of robustness')
plt.ylabel('Number of clusters')


#%% Plot all robustness values across learning

all_robustness_expert = cat(robustness_expert)
all_robustness_learning = cat(robustness_learning)

all_robustness_learning_filt = [i for i in all_robustness_learning if i != 0]
all_robustness_expert_filt = [i for i in all_robustness_expert if i != 0]


plt.bar([0,1], [np.mean(all_robustness_learning_filt), np.mean(all_robustness_expert_filt)])
plt.scatter(np.zeros(len(all_robustness_learning_filt)), all_robustness_learning_filt)
plt.scatter(np.ones(len(all_robustness_expert_filt)), all_robustness_expert_filt)
plt.ylabel('Robustness (delta from control)')
plt.xticks([0,1], ['Learning', 'Expert'])
plt.title('Robustness of clusters across learning')
plt.show()

t_stat, p_value = ttest_ind(all_robustness_learning_filt, all_robustness_expert_filt) # Paired t-test
print(t_stat, p_value)

all_robustness_learning_filt, all_robustness_expert_filt = [],[]
for i in range(len(all_robustness_learning)):
    if all_robustness_expert[i] != 0 and all_robustness_learning[i] != 0:
        all_robustness_learning_filt += [all_robustness_learning[i]]
        all_robustness_expert_filt += [all_robustness_expert[i]]

plt.bar([0,1], [np.mean(all_robustness_learning_filt), np.mean(all_robustness_expert_filt)])
plt.scatter(np.zeros(len(all_robustness_learning_filt)), all_robustness_learning_filt)
plt.scatter(np.ones(len(all_robustness_expert_filt)), all_robustness_expert_filt)
for i in range(len(all_robustness_expert_filt)):
    # if all_robustness_learning[i] or all_robustness
    plt.plot([0,1], [all_robustness_learning_filt[i], all_robustness_expert_filt[i]],
             color='grey')
    
plt.ylabel('Robustness (delta from control)')
plt.xticks([0,1], ['Learning', 'Expert'])
plt.title('Robustness of clusters across learning')
plt.show()

t_stat, p_value = ttest_rel(all_robustness_learning_filt, all_robustness_expert_filt) # Paired t-test
print(t_stat, p_value)

#%% Plot delta of cluster trial size vs robustness

all_robustness_expert = cat(robustness_expert)
all_robustness_learning = cat(robustness_learning)
all_num_trials_learning = cat(num_trials_learning)
all_num_trials_expert = cat(num_trials_expert)


all_robustness_learning_filt, all_robustness_expert_filt = [],[]
num_trials_learning_filt, num_trials_expert_filt = [],[]
for i in range(len(all_robustness_learning)):
    if all_robustness_expert[i] != 0 and all_robustness_learning[i] != 0:
        all_robustness_learning_filt += [all_robustness_learning[i]]
        all_robustness_expert_filt += [all_robustness_expert[i]]
        num_trials_learning_filt += [all_num_trials_learning[i]]
        num_trials_expert_filt += [all_num_trials_expert[i]]

delta = np.array(num_trials_expert_filt) - np.array(num_trials_learning_filt)
delta_rob = np.array(all_robustness_expert_filt) - np.array(all_robustness_learning_filt)
# 
plt.scatter(all_robustness_expert_filt, delta)
plt.axhline(0, ls = '--', color='grey')
plt.ylabel('Delta in number of trials (expert-learning)')
plt.xlabel('Robustness in expert stage (delta from control)')
plt.show()

# learning
plt.scatter(all_robustness_learning_filt, delta)
plt.axhline(0, ls = '--', color='grey')
plt.ylabel('Delta in number of trials (expert-learning)')
plt.xlabel('Robustness in learning stage (delta from control)')
# print(scipy.stats.pearsonr(all_robustness_learning_filt, delta))
plt.show()

plt.scatter(delta_rob, num_trials_expert_filt)
plt.axvline(0, ls = '--', color='grey')
plt.ylabel('Number of trials (expert)')
plt.xlabel('Delta robustness (delta from control)')
plt.show()
# print(scipy.stats.pearsonr(delta_rob, delta))

plt.scatter(delta_rob, num_trials_learning_filt)
plt.axvline(0, ls = '--', color='grey')
plt.ylabel('Number of trials (learning)')
plt.xlabel('Delta robustness (delta from control)')
plt.show()

#%% Filter out clusters with very few trials

all_robustness_learning_filt, all_robustness_expert_filt = [],[]
num_trials_learning_filt, num_trials_expert_filt = [],[]
for i in range(len(all_robustness_learning)):
    if all_num_trials_learning[i] > 50:
        all_robustness_learning_filt += [all_robustness_learning[i]]
        all_robustness_expert_filt += [all_robustness_expert[i]]
        num_trials_learning_filt += [all_num_trials_learning[i]]
        num_trials_expert_filt += [all_num_trials_expert[i]]

delta = np.array(num_trials_expert_filt) - np.array(num_trials_learning_filt)
delta_rob = np.array(all_robustness_expert_filt) - np.array(all_robustness_learning_filt)
# 
plt.scatter(all_robustness_expert_filt, delta)
plt.axhline(0, ls = '--', color='grey')
plt.ylabel('Delta in number of trials (expert-learning)')
plt.xlabel('Robustness in expert stage (delta from control)')
plt.show()

# learning
plt.scatter(all_robustness_learning_filt, delta)
plt.axhline(0, ls = '--', color='grey')
plt.ylabel('Delta in number of trials (expert-learning)')
plt.xlabel('Robustness in learning stage (delta from control)')
print(scipy.stats.pearsonr(all_robustness_learning_filt, delta))
plt.show()

plt.scatter(delta_rob, num_trials_expert_filt)
plt.axvline(0, ls = '--', color='grey')
plt.ylabel('Number of trials (expert)')
plt.xlabel('Delta robustness (delta from control)')
plt.show()
# print(scipy.stats.pearsonr(delta_rob, delta))

plt.scatter(delta_rob, num_trials_learning_filt)
plt.axvline(0, ls = '--', color='grey')
plt.ylabel('Number of trials (learning)')
plt.xlabel('Delta robustness (delta from control)')
plt.show()

#%% Plot the accuracy vs robustness

all_acc_learning = cat(acc_learning)
all_acc_expert = cat(acc_expert)

# all_robustness_learning_filt, all_robustness_expert_filt = [],[]
# all_acc_learning_filt, all_acc_expert_filt = [], []

# for i in range(len(all_robustness_learning)):
    
#     if all_acc_expert[i] != 0 and all_robustness_learning[i] != 0:
#         all_acc_expert_filt += [all_acc_expert[i]]
#         all_robustness_learning_filt += [all_robustness_learning[i]]

# # Plot learning robustness vs expert decoding acc
# plt.scatter(all_acc_expert_filt, all_robustness_learning_filt)
# plt.xlabel('Expert CD decoding accuracy')
# plt.ylabel('Learning CD robustness')
# print(scipy.stats.pearsonr(all_acc_expert_filt, all_robustness_learning_filt))

all_robustness_learning_filt, all_robustness_expert_filt = [],[]
all_acc_learning_filt, all_acc_expert_filt = [], []

for i in range(len(all_robustness_learning)):
    
    if all_acc_learning[i] != 0 and all_acc_expert[i] != 0 and all_robustness_expert[i] != 0 and all_robustness_learning[i] != 0:
        if all_num_trials_learning[i] > 50 and all_num_trials_expert[i] > 50:
            all_acc_learning_filt += [all_acc_learning[i]]
            all_acc_expert_filt += [all_acc_expert[i]]
            all_robustness_learning_filt += [all_robustness_learning[i]]
            all_robustness_expert_filt += [all_robustness_expert[i]]

# Plot all four
f, ax = plt.subplots(2,2, figsize=(10,10), sharey='row', sharex='col')

ax[0,0].scatter(all_acc_learning_filt, all_robustness_learning_filt)
ax[0,0].set_ylabel('Learning CD robustness')
ax[1,0].scatter(all_acc_learning_filt, all_robustness_expert_filt)
ax[1,0].set_ylabel('Expert CD robustness')

ax[0,1].scatter(all_acc_expert_filt, all_robustness_learning_filt)
ax[1,0].set_xlabel('Learning decoding acc')
ax[1,1].scatter(all_acc_expert_filt, all_robustness_expert_filt)
ax[1,1].set_xlabel('Expert decoding acc')


print(scipy.stats.pearsonr(all_acc_learning_filt, all_robustness_learning_filt))
print(scipy.stats.pearsonr(all_acc_learning_filt, all_robustness_expert_filt))        

print(scipy.stats.pearsonr(all_acc_expert_filt, all_robustness_learning_filt))
print(scipy.stats.pearsonr(all_acc_expert_filt, all_robustness_expert_filt))            




#%% Plot robustness vs rotation
# Do less robust CDs get rotated more?
# Plot robustness vs rotation
# drop empty from all_cd_rotations
all_cd_rotations = [i for i in all_cd_rotations if len(i) != 0]
robustness_expert = [i for i in robustness_expert if len(i) != 0]
robustness_learning = [i for i in robustness_learning if len(i) != 0]

cat_robustness_expert = cat(robustness_expert)
cat_robustness_learning = cat(robustness_learning)
cat_all_cd_rotations = cat(all_cd_rotations)

plt.scatter(cat_robustness_expert, np.abs(cat_all_cd_rotations))
plt.ylabel('CD rotation')
plt.xlabel('Robustness expert')
plt.show()

plt.scatter(cat_robustness_learning, np.abs(cat_all_cd_rotations))
plt.ylabel('CD rotation')
plt.xlabel('Robustness learning')
plt.show()
print(scipy.stats.pearsonr(cat_robustness_learning, np.abs(cat_all_cd_rotations)))

# Do more rotated CDs get more trials over learning?
cat_num_trials_expert = cat(num_trials_expert)
cat_num_trials_learning = cat(num_trials_learning)


plt.scatter(cat_num_trials_expert, np.abs(cat_all_cd_rotations))
plt.ylabel('CD rotation')
plt.xlabel('Number of trials in expert session')
plt.show()
print(scipy.stats.pearsonr(cat_num_trials_expert-cat_num_trials_learning, np.abs(cat_all_cd_rotations)))

plt.scatter(cat_num_trials_expert-cat_num_trials_learning, np.abs(cat_all_cd_rotations))
plt.ylabel('CD rotation')
plt.xlabel('Delta num of trials')
plt.show()





















