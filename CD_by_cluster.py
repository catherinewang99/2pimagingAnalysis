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

naivepath, learningpath, expertpath, clusterpath = [r'H:\data\BAYLORCW044\python\2024_05_22',
                    r'H:\data\BAYLORCW044\python\2024_06_06',
                  r'H:\data\BAYLORCW044\python\2024_06_19',
                  r'H:\data\matched_topic_params\CW44_FOV1_table']

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

#%% Decoding accuracy of clusters within

all_learn_accs = []
for cluster in range(num_clusters):
    # cluster_trials_all_idx = np.where(ldaclusters[:,cluster] > 1/ldaclusters.shape[1])[0]
    # cluster_trials_all = s2.i_good_trials[cluster_trials_all_idx]
    
    cluster_trials_all_idx = learning_normalized.index[learning_normalized['topic_{}'.format(cluster)] > 0.25].to_numpy()
    # cluster_trials_all_idx = np.where(np.argmax(learning_normalized, axis=1) == cluster)[0]
   
    s2 = Mode(learningpath, use_reg = True, triple=True, 
              baseline_normalization="median_zscore",
              i_good = cluster_trials_all_idx,
              proportion_train = 0.4,
              lda_cluster=True)
    
    _, _, db, acc_learning = s2.decision_boundary(mode_input='choice', persistence=False)
    all_learn_accs += [np.mean(acc_learning)]
  
all_exp_accs = []
for cluster in range(num_clusters):
    # cluster_trials_all_idx = np.where(ldaclusters[:,cluster] > 1/ldaclusters.shape[1])[0]
    # cluster_trials_all = s2.i_good_trials[cluster_trials_all_idx]
    
    cluster_trials_all_idx = expert_normalized.index[expert_normalized['topic_{}'.format(cluster)] > 0.25].to_numpy()
    # cluster_trials_all_idx = np.where(np.argmax(learning_normalized, axis=1) == cluster)[0]
   
    s2 = Mode(expertpath, use_reg = True, triple=True, 
              baseline_normalization="median_zscore",
              i_good = cluster_trials_all_idx,
              proportion_train = 0.6,
              lda_cluster=True)
    
    _, _, db, acc_learning = s2.decision_boundary(mode_input='choice', persistence=False)
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

#%% 
agg_accs = np.array(agg_accs)
plt.bar(np.arange(num_clusters)-0.15, agg_accs[:, 0], 0.1)
plt.bar(np.arange(num_clusters)-0.05, agg_accs[:, 1], 0.1,)
plt.bar(np.arange(num_clusters)+0.05, agg_accs[:, 2], 0.1)
plt.bar(np.arange(num_clusters)+0.15, agg_accs[:, 3], 0.1)
plt.legend()
plt.ylim(bottom=0.5)
plt.xlabel('Cluster number')
plt.ylabel('Accuracy %')

#%% End point analysis across clusters
main_cluster = 2
compare_cluster = 1

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
plt.title('Cluster main {} vs. cluster {}'.format(main_cluster, compare_cluster))

#%% Get different CDs from the clusters for learning session applied to expert and opto projections
# A trial belongs to a CD if the probability > 1/num clusters
# cluster = 4 # focus on one cluster for now
learning_CDs = []
for cluster in range(num_clusters):
    # cluster_trials_all_idx = np.where(ldaclusters[:,cluster] > 1/ldaclusters.shape[1])[0]
    # cluster_trials_all = s2.i_good_trials[cluster_trials_all_idx]
    
    cluster_trials_all_idx = learning_normalized.index[learning_normalized['topic_{}'.format(cluster)] > 0.25].to_numpy()

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
