# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 12:24:27 2024

@author: catherinewang

Use clusters to build various CD's in session instead of orthogonalizing to look 
for robust dimensions

Looks at analysis TRIAL AND NEURONS wise


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
trialparams = np.mean(clusters.trial_params.to_numpy()) # trial info
num_clusters = len(trialparams.columns)
idx = pd.IndexSlice

learning = trialparams.loc[idx['learning', :]]
learning_normalized = pd.DataFrame(normalize(learning, norm='l1'), columns=learning.columns)#, index=learning.index)

expert = trialparams.loc[idx['expert', :]]
expert_normalized = pd.DataFrame(normalize(expert, norm='l1'), columns=expert.columns)#, index=expert.index)

neuronparams = np.mean(clusters.components.to_numpy()).T
neurons_norm = pd.DataFrame(normalize(neuronparams, norm='l1'), columns=neuronparams.columns, index=neuronparams.index)
neurons_norm_arr = neurons_norm.to_numpy()
num_clusters = len(neurons_norm.columns)

#%% Read into the object

path = expertpath
s1 = Mode(path, use_reg = True, triple=True, baseline_normalization="median_zscore",
          filter_good_neurons=neurons_norm.index)
orthonormal_basis, mean_exp = s1.plot_CD(ctl=True)


#%% Characterize clusters
# Plot heatmap of all neurons
f=plt.figure(figsize=(5,10))
plt.imshow(neurons_norm_arr, aspect='auto',interpolation='none')
plt.xticks([0,1,2,3])
plt.xlabel('Cluster')
plt.ylabel('Neurons')
plt.show()

# Number of clusters
f=plt.figure(figsize=(5,5))
for cl in range(num_clusters):
    max_neurons_norm_arr = np.argmax(neurons_norm_arr, axis=1)
    cluster_neurons_all_idx = np.where(neurons_norm['topic_{}'.format(cl)] > 1/num_clusters)[0]
    
    plt.scatter([cl-0.2, cl+0.2], [sum(max_neurons_norm_arr == cl),
                                   len(cluster_neurons_all_idx)])
    plt.plot([cl-0.2, cl+0.2], [sum(max_neurons_norm_arr == cl),
                                   len(cluster_neurons_all_idx)],
             color='grey', ls='--')
plt.xticks([0,1,2,3])
plt.xlabel('Cluster')
plt.ylabel('Number of neurons')
plt.show()

# Binarized heatmap
max_neurons_norm_arr = np.argmax(neurons_norm_arr, axis=1)
trial_stack = np.zeros(neurons_norm_arr.shape[0])
for c in set(max_neurons_norm_arr):
    trial_stack = np.vstack((trial_stack, max_neurons_norm_arr == c))

plt.imshow(trial_stack[1:],aspect='auto',interpolation='none')
plt.title('Learning trials')
plt.ylabel('Cluster #')
plt.xlabel('Neuron #')

#%% Selectivity of every neuron

s1 = Mode(learningpath, use_reg = True, triple=True, baseline_normalization="median_zscore",
          filter_good_neurons=neurons_norm.index)
           # cluster_neurons=max_neurons_norm_arr)
orthonormal_basis, mean_exp = s1.plot_CD(ctl=True)
all_sel, _, _ = s1.get_epoch_selectivity(range(s1.delay, s1.response), s1.good_neurons)

# Plot average selectivity vs cluster size learning
all_sel =  np.abs(all_sel)
for cl in range(num_clusters):
    max_neurons_norm_arr = np.where(np.argmax(neurons_norm_arr, axis=1) == cl)[0]
    plt.scatter([len(max_neurons_norm_arr)], [np.mean(all_sel[max_neurons_norm_arr])], label=cl)
    plt.errorbar([len(max_neurons_norm_arr)], [np.mean(all_sel[max_neurons_norm_arr])], 
                 yerr = [np.var(all_sel[max_neurons_norm_arr])])
plt.legend()
plt.ylabel('Avg selectivity (t-stat)')
plt.xlabel('Cluster size')
plt.show()

# Repeat for expert

s1 = Mode(expertpath, use_reg = True, triple=True, baseline_normalization="median_zscore",
          filter_good_neurons=neurons_norm.index)
           # cluster_neurons=max_neurons_norm_arr)
orthonormal_basis, mean_exp = s1.plot_CD(ctl=True)
all_sel, _, _ = s1.get_epoch_selectivity(range(s1.delay, s1.response), s1.good_neurons)

# Plot average selectivity vs cluster size learning
all_sel =  np.abs(all_sel)
for cl in range(num_clusters):
    max_neurons_norm_arr = np.where(np.argmax(neurons_norm_arr, axis=1) == cl)[0]
    plt.scatter([len(max_neurons_norm_arr)], [np.mean(all_sel[max_neurons_norm_arr])], label=cl)
    plt.errorbar([len(max_neurons_norm_arr)], [np.mean(all_sel[max_neurons_norm_arr])], 
                 yerr = [np.var(all_sel[max_neurons_norm_arr])])
plt.legend()
plt.ylabel('Avg selectivity (t-stat)')
plt.xlabel('Cluster size')
plt.show()


#%% Plot the CDs 
for cl in range(num_clusters):
    max_neurons_norm_arr = np.where(np.argmax(neurons_norm_arr, axis=1) == cl)[0]
    cluster_trials_all_idx = learning_normalized.index[learning_normalized['topic_{}'.format(cl)] > 1/num_clusters].to_numpy()

    s1 = Mode(learningpath, use_reg = True, triple=True, baseline_normalization="median_zscore",
              filter_good_neurons=neurons_norm.index,
              cluster_neurons=max_neurons_norm_arr,
              lda_cluster=True,
              train_test_trials = cluster_trials_all_idx)
    
    orthonormal_basis, mean_exp = s1.plot_CD(ctl=True)
    s1.plot_CD_opto()

for cl in range(num_clusters):
    max_neurons_norm_arr = np.where(np.argmax(neurons_norm_arr, axis=1) == cl)[0]
    cluster_trials_all_idx = expert_normalized.index[expert_normalized['topic_{}'.format(cl)] > 1/num_clusters].to_numpy()

    s1 = Mode(expertpath, use_reg = True, triple=True, baseline_normalization="median_zscore",
              filter_good_neurons=neurons_norm.index,
              cluster_neurons=max_neurons_norm_arr,
              lda_cluster=True,
              train_test_trials = cluster_trials_all_idx)
    
    orthonormal_basis, mean_exp = s1.plot_CD(ctl=True)
    s1.plot_CD_opto()


#%% Get robustness vs t-stat avg
learning_tstat, expert_tstat = [], []
learning_rob, expert_rob = [], []
for cl in range(num_clusters):
    max_neurons_norm_arr = np.where(np.argmax(neurons_norm_arr, axis=1) == cl)[0]
    cluster_trials_all_idx = learning_normalized.index[learning_normalized['topic_{}'.format(cl)] > 1/num_clusters].to_numpy()

    s1 = Mode(learningpath, use_reg = True, triple=True, baseline_normalization="median_zscore",
              filter_good_neurons=neurons_norm.index,
              cluster_neurons=max_neurons_norm_arr,
              lda_cluster=True,
              train_test_trials = cluster_trials_all_idx)
    all_sel, _, _ = s1.get_epoch_selectivity(range(s1.delay, s1.response), s1.good_neurons)
    learning_tstat += [np.mean(np.abs(all_sel))]
    learning_rob += [s1.modularity_proportion_by_CD(ctl=True)]
    
    cluster_trials_all_idx = expert_normalized.index[expert_normalized['topic_{}'.format(cl)] > 1/num_clusters].to_numpy()
    
    s1 = Mode(expertpath, use_reg = True, triple=True, baseline_normalization="median_zscore",
              filter_good_neurons=neurons_norm.index,
              cluster_neurons=max_neurons_norm_arr,
              lda_cluster=True,
              train_test_trials = cluster_trials_all_idx)
    all_sel, _, _ = s1.get_epoch_selectivity(range(s1.delay, s1.response), s1.good_neurons)
    expert_tstat += [np.mean(np.abs(all_sel))]
    expert_rob += [s1.modularity_proportion_by_CD(ctl=True)]
    
# Plot
plt.scatter(learning_tstat, learning_rob, label='Learning')
plt.scatter(expert_tstat, expert_rob, label='Expert')
plt.ylabel('Robustness')
plt.xlabel('T-statistic')
plt.legend()

#%% Plot the delta
delta_tstat = np.array(expert_tstat) - np.array(learning_tstat)
delta_rob = np.array(expert_rob) - np.array(learning_rob)
plt.scatter(delta_tstat, delta_rob)
plt.ylabel('delta Robustness')
plt.xlabel('delta T-statistic')


#%% Decoding accuracy

# Use max method
allaccs_expert = []
allaccs_learning = []
num_neurons = []

for cl in range(num_clusters):
    
    max_neurons_norm_arr = np.where(np.argmax(neurons_norm_arr, axis=1) == cl)[0]
    
    s1 = Mode(expertpath, use_reg = True, triple=True, baseline_normalization="median_zscore",
              filter_good_neurons=neurons_norm.index,
              cluster_neurons=max_neurons_norm_arr)
    
    _, _, db, acc_learning = s1.decision_boundary(mode_input='choice', persistence=False)

    allaccs_expert += [np.mean(acc_learning)]

    s1 = Mode(learningpath, use_reg = True, triple=True, baseline_normalization="median_zscore",
              filter_good_neurons=neurons_norm.index,
              cluster_neurons=max_neurons_norm_arr)
    
    _, _, db, acc_learning = s1.decision_boundary(mode_input='choice', persistence=False)

    allaccs_learning += [np.mean(acc_learning)]
    
    num_neurons += [len(max_neurons_norm_arr)]

# Plot decoding accuracy vs number of neurons

fig, ax1 = plt.subplots()
ax1.bar(np.arange(num_clusters)-0.2, allaccs_learning, width= 0.4, label='Learning')
ax1.bar(np.arange(num_clusters)+0.2, allaccs_expert, width=0.4, label='Expert')

ax1.set_ylabel('Decoding accuracy')

ax2 = ax1.twinx()
ax2.scatter(np.arange(num_clusters), num_neurons, color='red')
ax2.plot(np.arange(num_clusters), num_neurons, color='red', ls='--')
ax2.set_ylabel('Number of neurons')

ax1.legend()
ax1.set_xticks([0,1,2,3])
ax1.set_xlabel('Cluster')
ax1.set_ylim(bottom=0.5)

#%% Look at cluster 1 neurons decoding on cluster 1 trials vs other cluster trials learning FOV
wi_decodingaccs = []
wo_decodingaccs = []
for cl in range(num_clusters):
    
    max_neurons_norm_arr = np.where(np.argmax(neurons_norm_arr, axis=1) == cl)[0]
    cluster_trials_all_idx = learning_normalized.index[learning_normalized['topic_{}'.format(cl)] > 0.25].to_numpy()
    # cluster_trials_all_idx = np.where(np.argmax(learning_normalized, axis=1) == cluster)[0]
    all_acc_learning = []
    for _ in range(5):

        s1 = Mode(learningpath, use_reg = True, triple=True, baseline_normalization="median_zscore",
                  filter_good_neurons=neurons_norm.index,
                    cluster_neurons=max_neurons_norm_arr,
                      i_good = cluster_trials_all_idx,
                      
                      lda_cluster=True)
                      # train_test_trials = cluster_trials_all_idx)
    
        _, _, db, acc_learning = s1.decision_boundary(mode_input='choice', persistence=False)
        all_acc_learning += [np.mean(acc_learning)]
    
    wi_decodingaccs += [np.mean(all_acc_learning)]
    
    wo_trials = [i for i in learning_normalized.index if i not in cluster_trials_all_idx]
    
    all_acc_learning = []
    for _ in range(5):

        s1 = Mode(learningpath, use_reg = True, triple=True, baseline_normalization="median_zscore",
                  filter_good_neurons=neurons_norm.index,
                    cluster_neurons=max_neurons_norm_arr,
                      i_good = wo_trials,
                      lda_cluster=True)
                      # train_test_trials = cluster_trials_all_idx)

        _, _, db, acc_learning = s1.decision_boundary(mode_input='choice', persistence=False)
        all_acc_learning += [np.mean(acc_learning)]
    
    
    wo_decodingaccs += [np.mean(all_acc_learning)]

#Plot

plt.bar(np.arange(num_clusters)-0.2, wi_decodingaccs, 0.4, label='within-cluster')
plt.bar(np.arange(num_clusters)+0.2, wo_decodingaccs, 0.4, label='without cluster')

plt.ylabel('Decoding accuracy')
plt.xticks([0,1,2,3])
plt.xlabel('Cluster')
plt.legend()
plt.ylim(bottom=0.5)
    
    

#%% Look at cluster 1 neurons decoding on cluster 1 trials vs other cluster trials expert FOV
wi_decodingaccs = []
wo_decodingaccs = []
for cl in range(num_clusters):
    
    max_neurons_norm_arr = np.where(np.argmax(neurons_norm_arr, axis=1) == cl)[0]
    cluster_trials_all_idx = expert_normalized.index[expert_normalized['topic_{}'.format(cl)] > 0.25].to_numpy()
    # cluster_trials_all_idx = np.where(np.argmax(learning_normalized, axis=1) == cluster)[0]
    all_acc_learning = []
    for _ in range(5):

        s1 = Mode(expertpath, use_reg = True, triple=True, baseline_normalization="median_zscore",
                  filter_good_neurons=neurons_norm.index,
                    cluster_neurons=max_neurons_norm_arr,
                      i_good = cluster_trials_all_idx,
                      
                      lda_cluster=True)
                      # train_test_trials = cluster_trials_all_idx)
    
        try:
            _, _, db, acc_learning = s1.decision_boundary(mode_input='choice', persistence=False)
        except np.linalg.LinAlgError:
            all_acc_learning += [0]
            continue
        all_acc_learning += [np.mean(acc_learning)]
    
    wi_decodingaccs += [np.mean(all_acc_learning)]
    
    wo_trials = [i for i in expert_normalized.index if i not in cluster_trials_all_idx]
    
    all_acc_learning = []
    for _ in range(5):

        s1 = Mode(expertpath, use_reg = True, triple=True, baseline_normalization="median_zscore",
                  filter_good_neurons=neurons_norm.index,
                    cluster_neurons=max_neurons_norm_arr,
                      i_good = wo_trials,
                      lda_cluster=True)
                      # train_test_trials = cluster_trials_all_idx)
        try:
            _, _, db, acc_learning = s1.decision_boundary(mode_input='choice', persistence=False)
        except np.linalg.LinAlgError:
            all_acc_learning += [0]
            continue
            
        all_acc_learning += [np.mean(acc_learning)]
    
    
    wo_decodingaccs += [np.mean(all_acc_learning)]

#Plot

plt.bar(np.arange(num_clusters)-0.2, wi_decodingaccs, 0.4, label='within-cluster')
plt.bar(np.arange(num_clusters)+0.2, wo_decodingaccs, 0.4, label='without cluster')

plt.ylabel('Decoding accuracy')
plt.xticks([0,1,2,3])
plt.xlabel('Cluster')
plt.legend()
plt.ylim(bottom=0.5)

#%% Look at endpoints of CDs on different neuron clusters / trial pairs
main_cluster = 2
compare_cluster = 0
max_neurons_norm_arr = np.where(np.argmax(neurons_norm_arr, axis=1) == main_cluster)[0]
# cluster_trials_all_idx = learning_normalized.index[learning_normalized['topic_{}'.format(main_cluster)] > 1/num_clusters].to_numpy()
cluster_trials_all_idx = np.where(np.argmax(learning_normalized, axis=1) == main_cluster)[0]

s1 = Mode(learningpath, use_reg = True, triple=True, baseline_normalization="median_zscore",
          filter_good_neurons=neurons_norm.index,
            cluster_neurons=max_neurons_norm_arr,
              train_test_trials = cluster_trials_all_idx,
              lda_cluster=True)

proj_allDimR, proj_allDimL = s1.plot_CD(ctl=True, plot=False, auto_corr_return=True)


max_neurons_norm_arr = np.where(np.argmax(neurons_norm_arr, axis=1) == compare_cluster)[0]
s2 = Mode(learningpath, use_reg = True, triple=True, baseline_normalization="median_zscore",
          filter_good_neurons=neurons_norm.index,
            cluster_neurons=max_neurons_norm_arr,
              train_test_trials = cluster_trials_all_idx,
              lda_cluster=True)

proj_allDimR_across, proj_allDimL_across = s2.plot_CD(ctl=True, plot=False, auto_corr_return=True)

plt.scatter(proj_allDimR[:, s2.response-1], proj_allDimR_across[:, s2.response-1], color='b')
plt.scatter(proj_allDimL[:, s2.response-1], proj_allDimL_across[:, s2.response-1], color='r')
plt.axhline(0, ls = '--', color='black')
plt.axvline(0, ls = '--', color='black')
plt.ylabel('Outside cluster CD')
plt.xlabel('Within cluster CD')
plt.title('Learning: cluster main {} vs. cluster {}'.format(main_cluster, compare_cluster))
plt.show()




#%% Look at cluster 1 neurons vs other cluster neurons decoding on cluster 1 trials learning FOV
wi_decodingaccs = []
wo_decodingaccs = []

for cl in range(num_clusters):
    
    max_neurons_norm_arr = np.where(np.argmax(neurons_norm_arr, axis=1) == cl)[0]
    cluster_trials_all_idx = learning_normalized.index[learning_normalized['topic_{}'.format(cl)] > 1/num_clusters].to_numpy()
    # cluster_trials_all_idx = np.where(np.argmax(learning_normalized, axis=1) == cluster)[0]
    all_acc_learning = []
    for _ in range(5):

        s1 = Mode(learningpath, use_reg = True, triple=True, baseline_normalization="median_zscore",
                  filter_good_neurons=neurons_norm.index,
                    cluster_neurons=max_neurons_norm_arr,
                      i_good = cluster_trials_all_idx,
                      
                      lda_cluster=True)
                      # train_test_trials = cluster_trials_all_idx)
    
        _, _, db, acc_learning = s1.decision_boundary(mode_input='choice', persistence=False)
        all_acc_learning += [np.mean(acc_learning)]
    
    wi_decodingaccs += [np.mean(all_acc_learning)]
    
    # wo_trials = [i for i in learning_normalized.index if i not in cluster_trials_all_idx]
    wo_neurons = np.array([i for i in range(neurons_norm_arr.shape[0]) if i not in max_neurons_norm_arr])
    
    all_acc_learning = []
    for _ in range(5):

        s1 = Mode(learningpath, use_reg = True, triple=True, baseline_normalization="median_zscore",
                  filter_good_neurons=neurons_norm.index,
                    cluster_neurons=wo_neurons,
                      i_good = cluster_trials_all_idx,
                      lda_cluster=True)
                      # train_test_trials = cluster_trials_all_idx)

        _, _, db, acc_learning = s1.decision_boundary(mode_input='choice', persistence=False)
        all_acc_learning += [np.mean(acc_learning)]
    
    
    wo_decodingaccs += [np.mean(all_acc_learning)]

#Plot

plt.bar(np.arange(num_clusters)-0.2, wi_decodingaccs, 0.4, label='within-cluster')
plt.bar(np.arange(num_clusters)+0.2, wo_decodingaccs, 0.4, label='without cluster')

plt.ylabel('Decoding accuracy')
plt.xticks([0,1,2,3])
plt.xlabel('Cluster')
plt.legend()
plt.ylim(bottom=0.5)

#%% Look at cluster 1 neurons vs other cluster neurons decoding on cluster 1 trials expert FOV
wi_decodingaccs = []
wo_decodingaccs = []

for cl in range(num_clusters):
    
    max_neurons_norm_arr = np.where(np.argmax(neurons_norm_arr, axis=1) == cl)[0]
    cluster_trials_all_idx = expert_normalized.index[expert_normalized['topic_{}'.format(cl)] > 1/num_clusters].to_numpy()
    # cluster_trials_all_idx = np.where(np.argmax(learning_normalized, axis=1) == cluster)[0]
    all_acc_learning = []
    for _ in range(5):

        s1 = Mode(expertpath, use_reg = True, triple=True, baseline_normalization="median_zscore",
                  filter_good_neurons=neurons_norm.index,
                    cluster_neurons=max_neurons_norm_arr,
                      i_good = cluster_trials_all_idx,
                      
                      lda_cluster=True)
                      # train_test_trials = cluster_trials_all_idx)
    
        _, _, db, acc_learning = s1.decision_boundary(mode_input='choice', persistence=False, ctl=True)
        all_acc_learning += [np.mean(acc_learning)]
    
    wi_decodingaccs += [np.mean(all_acc_learning)]
    
    # wo_trials = [i for i in learning_normalized.index if i not in cluster_trials_all_idx]
    wo_neurons = np.array([i for i in range(neurons_norm_arr.shape[0]) if i not in max_neurons_norm_arr])
    
    all_acc_learning = []
    for _ in range(5):

        s1 = Mode(expertpath, use_reg = True, triple=True, baseline_normalization="median_zscore",
                  filter_good_neurons=neurons_norm.index,
                    cluster_neurons=wo_neurons,
                      i_good = cluster_trials_all_idx,
                      lda_cluster=True)
                      # train_test_trials = cluster_trials_all_idx)

        _, _, db, acc_learning = s1.decision_boundary(mode_input='choice', persistence=False, ctl=True)
        all_acc_learning += [np.mean(acc_learning)]
    
    
    wo_decodingaccs += [np.mean(all_acc_learning)]

#Plot

plt.bar(np.arange(num_clusters)-0.2, wi_decodingaccs, 0.4, label='within-cluster')
plt.bar(np.arange(num_clusters)+0.2, wo_decodingaccs, 0.4, label='without cluster')

plt.ylabel('Decoding accuracy')
plt.xticks([0,1,2,3])
plt.xlabel('Cluster')
plt.legend()
plt.ylim(bottom=0.5)

#%% Iterate through all fovs to get robustness and tstat

all_learning_rob = []
all_learning_tstat = []

all_expert_rob = []
all_expert_tstat = []

all_wo_learning_tstat = []
all_wo_expert_tstat = []

for paths in agg_mice_paths:
    clusters = pd.read_pickle(paths[3])
    trialparams = np.mean(clusters.trial_params.to_numpy())
    num_clusters = len(trialparams.columns)

    learning = trialparams.loc[idx['learning', :]]
    learning_normalized = pd.DataFrame(normalize(learning, norm='l1'), columns=learning.columns)#, index=learning.index)
    
    expert = trialparams.loc[idx['expert', :]]
    expert_normalized = pd.DataFrame(normalize(expert, norm='l1'), columns=expert.columns)#, index=expert.index)
    
    neuronparams = np.mean(clusters.components.to_numpy()).T
    neurons_norm = pd.DataFrame(normalize(neuronparams, norm='l1'), columns=neuronparams.columns, index=neuronparams.index)
    neurons_norm_arr = neurons_norm.to_numpy()
    
    learning_rob = []
    learning_tstat = []

    expert_rob = []
    expert_tstat = []
    
    wo_learning_tstat = []
    wo_expert_tstat = []
    
    for cl in range(num_clusters):
        max_neurons_norm_arr = np.where(np.argmax(neurons_norm_arr, axis=1) == cl)[0]
        cluster_trials_all_idx = learning_normalized.index[learning_normalized['topic_{}'.format(cl)] > 1/num_clusters].to_numpy()
    
        s1 = Mode(paths[1], use_reg = True, triple=True, baseline_normalization="median_zscore",
                  filter_good_neurons=neurons_norm.index,
                  cluster_neurons=max_neurons_norm_arr,
                  lda_cluster=True,
                  train_test_trials = cluster_trials_all_idx)
        learning_rob += [s1.modularity_proportion_by_CD(ctl=True)]
        
        outside_cluster_trials = [i for i in s1.i_good_trials if i not in cluster_trials_all_idx]
        s1.i_good_trials = cluster_trials_all_idx
        all_sel, _, _ = s1.get_epoch_selectivity(range(s1.delay, s1.response), s1.good_neurons)
        # learning_tstat += [np.mean(np.abs(all_sel))]
        learning_tstat += [all_sel]
        
        s1.i_good_trials = outside_cluster_trials
        all_sel, _, _ = s1.get_epoch_selectivity(range(s1.delay, s1.response), s1.good_neurons)
        wo_learning_tstat += [all_sel]
        
        cluster_trials_all_idx = expert_normalized.index[expert_normalized['topic_{}'.format(cl)] > 1/num_clusters].to_numpy()
        
        s1 = Mode(paths[2], use_reg = True, triple=True, baseline_normalization="median_zscore",
                  filter_good_neurons=neurons_norm.index,
                  cluster_neurons=max_neurons_norm_arr,
                  lda_cluster=True,
                  train_test_trials = cluster_trials_all_idx)
        expert_rob += [s1.modularity_proportion_by_CD(ctl=True)]
        
        outside_cluster_trials = [i for i in s1.i_good_trials if i not in cluster_trials_all_idx]
        s1.i_good_trials = cluster_trials_all_idx
        all_sel, _, _ = s1.get_epoch_selectivity(range(s1.delay, s1.response), s1.good_neurons)
        # expert_tstat += [np.mean(np.abs(all_sel))]
        expert_tstat += [all_sel]
        
        s1.i_good_trials = outside_cluster_trials
        all_sel, _, _ = s1.get_epoch_selectivity(range(s1.delay, s1.response), s1.good_neurons)
        wo_expert_tstat += [all_sel]
        
    all_learning_rob += [learning_rob]
    all_learning_tstat += [learning_tstat]

    all_expert_rob += [expert_rob]
    all_expert_tstat += [expert_tstat]
    
    all_wo_learning_tstat += [wo_learning_tstat]
    all_wo_expert_tstat += [wo_expert_tstat]
    
#%% Plot tstat vs rob across all FOVs

for fov in range(len(all_learning_tstat)):
    plt.scatter(all_learning_tstat[fov], all_learning_rob[fov])
plt.ylabel('Robustness')
plt.xlabel('T-statistic')
plt.xlim(left=-0.2, right=1.0)
plt.ylim(bottom=0, top=4.0)
plt.legend()
plt.show()

for fov in range(len(all_expert_rob)):
    plt.scatter(all_expert_tstat[fov], all_expert_rob[fov])
plt.ylabel('Robustness')
plt.xlabel('T-statistic')
plt.xlim(left=-0.2, right=1.0)
plt.ylim(bottom=0, top=4.0)
plt.legend()
plt.show()

#%% Plot deltas
for fov in range(len(all_expert_rob)):
    delta_tsat = np.array(all_expert_tstat[fov]) - np.array(all_learning_tstat[fov])
    delta_rob = np.array(all_expert_rob[fov]) - np.array(all_learning_rob[fov])
    plt.scatter(delta_tsat, delta_rob)
    print(scipy.stats.pearsonr(delta_tsat, delta_rob))

plt.xlim(left=-0.7)
plt.ylabel('delta Robustness')
plt.xlabel('delta T-statistic')


#%% Iterate through all the FOVs to get decoding accs
all_learning_wi_decodingaccs = []
all_learning_wo_decodingaccs = []

all_expert_wi_decodingaccs = []
all_expert_wo_decodingaccs = []


for paths in agg_mice_paths:
    clusters = pd.read_pickle(paths[3])
    trialparams = np.mean(clusters.trial_params.to_numpy())
    num_clusters = len(trialparams.columns)

    learning = trialparams.loc[idx['learning', :]]
    learning_normalized = pd.DataFrame(normalize(learning, norm='l1'), columns=learning.columns)#, index=learning.index)
    
    expert = trialparams.loc[idx['expert', :]]
    expert_normalized = pd.DataFrame(normalize(expert, norm='l1'), columns=expert.columns)#, index=expert.index)
    
    neuronparams = np.mean(clusters.components.to_numpy()).T
    neurons_norm = pd.DataFrame(normalize(neuronparams, norm='l1'), columns=neuronparams.columns, index=neuronparams.index)
    neurons_norm_arr = neurons_norm.to_numpy()
        
    learning_wi_decodingaccs = []
    learning_wo_decodingaccs = []
    
    expert_wi_decodingaccs = []
    expert_wo_decodingaccs = []

    for cl in range(num_clusters):
        
        ## LEARNING ## 
        
        max_neurons_norm_arr = np.where(np.argmax(neurons_norm_arr, axis=1) == cl)[0]
        cluster_trials_all_idx = learning_normalized.index[learning_normalized['topic_{}'.format(cl)] > 0.25].to_numpy()
        # cluster_trials_all_idx = np.where(np.argmax(learning_normalized, axis=1) == cluster)[0]
        all_acc_learning = []
        for _ in range(5):

            s1 = Mode(paths[1], use_reg = True, triple=True, baseline_normalization="median_zscore",
                      filter_good_neurons=neurons_norm.index,
                        cluster_neurons=max_neurons_norm_arr,
                          i_good = cluster_trials_all_idx,
                          lda_cluster=True)
                          # train_test_trials = cluster_trials_all_idx)
        
            try:
                _, _, db, acc_learning = s1.decision_boundary(mode_input='choice', persistence=False, ctl=True)
            except np.linalg.LinAlgError:
                all_acc_learning += [0]
                continue
            all_acc_learning += [np.mean(acc_learning)]
        
        learning_wi_decodingaccs += [np.mean(all_acc_learning)]
        
        # wo_trials = [i for i in learning_normalized.index if i not in cluster_trials_all_idx]
        wo_neurons = np.array([i for i in range(neurons_norm_arr.shape[0]) if i not in max_neurons_norm_arr])

        all_acc_learning = []
        for _ in range(5):

            s1 = Mode(paths[1], use_reg = True, triple=True, baseline_normalization="median_zscore",
                      filter_good_neurons=neurons_norm.index,
                        cluster_neurons=wo_neurons,
                          i_good = cluster_trials_all_idx,
                          lda_cluster=True)
                          # train_test_trials = cluster_trials_all_idx)

            try:
                _, _, db, acc_learning = s1.decision_boundary(mode_input='choice', persistence=False, ctl=True)
            except np.linalg.LinAlgError:
                all_acc_learning += [0]
                continue
            all_acc_learning += [np.mean(acc_learning)]
        
        learning_wo_decodingaccs += [np.mean(all_acc_learning)]
        
        ## EXPERT ## 
        
        cluster_trials_all_idx = expert_normalized.index[expert_normalized['topic_{}'.format(cl)] > 0.25].to_numpy()
        # cluster_trials_all_idx = np.where(np.argmax(learning_normalized, axis=1) == cluster)[0]
        all_acc_learning = []
        for _ in range(5):

            s1 = Mode(paths[2], use_reg = True, triple=True, baseline_normalization="median_zscore",
                      filter_good_neurons=neurons_norm.index,
                        cluster_neurons=max_neurons_norm_arr,
                          i_good = cluster_trials_all_idx,
                          
                          lda_cluster=True)
                          # train_test_trials = cluster_trials_all_idx)
        
            try:
                _, _, db, acc_learning = s1.decision_boundary(mode_input='choice', persistence=False, ctl=True)
            except np.linalg.LinAlgError:
                all_acc_learning += [0]
                continue
            all_acc_learning += [np.mean(acc_learning)]
        
        expert_wi_decodingaccs += [np.mean(all_acc_learning)]
        
        # wo_trials = [i for i in learning_normalized.index if i not in cluster_trials_all_idx]
        wo_neurons = np.array([i for i in range(neurons_norm_arr.shape[0]) if i not in max_neurons_norm_arr])

        all_acc_learning = []
        for _ in range(5):

            s1 = Mode(paths[2], use_reg = True, triple=True, baseline_normalization="median_zscore",
                      filter_good_neurons=neurons_norm.index,
                        cluster_neurons=wo_neurons,
                          i_good = cluster_trials_all_idx,
                          lda_cluster=True)
                          # train_test_trials = cluster_trials_all_idx)

            try:
                _, _, db, acc_learning = s1.decision_boundary(mode_input='choice', persistence=False, ctl=True)
            except np.linalg.LinAlgError:
                all_acc_learning += [0]
                continue
            all_acc_learning += [np.mean(acc_learning)]
        
        expert_wo_decodingaccs += [np.mean(all_acc_learning)]
        
        
        
    all_learning_wi_decodingaccs += [learning_wi_decodingaccs]
    all_learning_wo_decodingaccs += [learning_wo_decodingaccs]
    
    all_expert_wi_decodingaccs += [expert_wi_decodingaccs]
    all_expert_wo_decodingaccs += [expert_wo_decodingaccs]


#%% Plot results (old, for trials clusters)
all_wi_decodingaccs_cat = cat(all_wi_decodingaccs)
all_wi_decodingaccs_cat_filt = [i for i in all_wi_decodingaccs_cat if i != 0]

all_wo_decodingaccs_cat = cat(all_wo_decodingaccs)
all_wo_decodingaccs_cat_filt = [i for i in all_wo_decodingaccs_cat if i != 0]


all_wi_decodingaccs_cat_filt, all_wo_decodingaccs_cat_filt = [], []

# plot the deltas
for i in range(len(all_wo_decodingaccs_cat)):
    if all_wo_decodingaccs_cat[i] !=0 and all_wi_decodingaccs_cat[i] != 0:
        all_wi_decodingaccs_cat_filt += [all_wi_decodingaccs_cat[i]]
        all_wo_decodingaccs_cat_filt += [all_wo_decodingaccs_cat[i]]

alldeltas = np.array(all_wi_decodingaccs_cat_filt) - np.array(all_wo_decodingaccs_cat_filt)

plt.hist(alldeltas, bins=20)
plt.xlabel('Delta of decoding accuracy')
plt.ylabel('Number of clusters')
plt.show()

plt.bar([0], np.mean(all_wi_decodingaccs_cat_filt), label='within-cluster')
plt.scatter(np.zeros(len(all_wi_decodingaccs_cat_filt)), all_wi_decodingaccs_cat_filt)
plt.bar([1], np.mean(all_wo_decodingaccs_cat_filt), label='within-cluster')
plt.scatter(np.ones(len(all_wo_decodingaccs_cat_filt)), all_wo_decodingaccs_cat_filt)
for i in range(len(all_wo_decodingaccs_cat_filt)):
    plt.plot([0,1], [all_wi_decodingaccs_cat_filt[i], all_wo_decodingaccs_cat_filt[i]],
             color='grey')
plt.ylabel('Decoding accuracy')
plt.ylim(bottom=0.4)
plt.xticks([0,1], ['Within-cluster', 'Without-cluster'])
plt.show()

ttest_rel(all_wi_decodingaccs_cat_filt, all_wo_decodingaccs_cat_filt)

#%% Plot results of neuron clusters across all FOVs

catall_learning_wi_decodingaccs = cat(all_learning_wi_decodingaccs)
catall_learning_wo_decodingaccs = cat(all_learning_wo_decodingaccs)

plt.bar([0,1], [np.mean(catall_learning_wi_decodingaccs), np.mean(catall_learning_wo_decodingaccs)])
plt.scatter(np.zeros(len(catall_learning_wo_decodingaccs)), catall_learning_wi_decodingaccs)
plt.scatter(np.ones(len(catall_learning_wo_decodingaccs)), catall_learning_wo_decodingaccs)
for i in range(len(catall_learning_wo_decodingaccs)):
    plt.plot([0,1], [catall_learning_wi_decodingaccs[i], catall_learning_wo_decodingaccs[i]],
             color='grey')
plt.ylabel('Decoding accuracy')
plt.xticks([0,1], ['Within-cluster' , 'Without-cluster'])
plt.title('Learning clusters')
plt.show()
t_stat, p_value = ttest_rel(catall_learning_wi_decodingaccs,
                            catall_learning_wo_decodingaccs) # Paired t-test
print(t_stat, p_value)

catall_expert_wi_decodingaccs = cat(all_expert_wi_decodingaccs)
catall_expert_wo_decodingaccs = cat(all_expert_wo_decodingaccs)

plt.bar([0,1], [np.mean(catall_expert_wi_decodingaccs), np.mean(catall_expert_wo_decodingaccs)])
plt.scatter(np.zeros(len(catall_expert_wi_decodingaccs)), catall_expert_wi_decodingaccs)
plt.scatter(np.ones(len(catall_expert_wo_decodingaccs)), catall_expert_wo_decodingaccs)
for i in range(len(catall_expert_wo_decodingaccs)):
    plt.plot([0,1], [catall_expert_wi_decodingaccs[i], catall_expert_wo_decodingaccs[i]],
             color='grey')
plt.ylabel('Decoding accuracy')
plt.xticks([0,1], ['Within-cluster' , 'Without-cluster'])
plt.title('Expert clusters')
plt.show()
t_stat, p_value = ttest_rel(catall_expert_wi_decodingaccs,
                            catall_expert_wo_decodingaccs) # Paired t-test
print(t_stat, p_value)




plt.bar([0,1], [np.mean(catall_learning_wi_decodingaccs), np.mean(catall_expert_wi_decodingaccs)])
plt.scatter(np.zeros(len(catall_learning_wi_decodingaccs)), catall_learning_wi_decodingaccs)
plt.scatter(np.ones(len(catall_expert_wi_decodingaccs)), catall_expert_wi_decodingaccs)
for i in range(len(catall_learning_wi_decodingaccs)):
    plt.plot([0,1], [catall_learning_wi_decodingaccs[i], catall_expert_wi_decodingaccs[i]],
             color='grey')

plt.ylabel('Decoding accuracy')
plt.xticks([0,1], ['Learning' , 'Expert'])
plt.title('Within cluster decoding')
plt.show()
t_stat, p_value = ttest_rel(catall_learning_wi_decodingaccs,
                            catall_expert_wi_decodingaccs) # Paired t-test

print(t_stat, p_value)

plt.bar([0,1], [np.mean(catall_learning_wo_decodingaccs), np.mean(catall_expert_wo_decodingaccs)])
plt.scatter(np.zeros(len(catall_learning_wo_decodingaccs)), catall_learning_wo_decodingaccs)
plt.scatter(np.ones(len(catall_expert_wo_decodingaccs)), catall_expert_wo_decodingaccs)
for i in range(len(catall_learning_wo_decodingaccs)):
    plt.plot([0,1], [catall_learning_wo_decodingaccs[i], catall_expert_wo_decodingaccs[i]],
             color='grey')

plt.ylabel('Decoding accuracy')
plt.xticks([0,1], ['Learning' , 'Expert'])
plt.title('Without cluster decoding')
plt.show()
t_stat, p_value = ttest_rel(catall_learning_wo_decodingaccs,
                            catall_expert_wo_decodingaccs) # Paired t-test

print(t_stat, p_value)

#%% Plot neuron clusters from a per FOV perspective
learning_deltas = []
expert_deltas = []
for fov in range(len(all_learning_wi_decodingaccs)):
    learning_deltas += [np.array(all_learning_wi_decodingaccs[fov]) - np.array(all_learning_wo_decodingaccs[fov])]
    expert_deltas += [np.array(all_expert_wi_decodingaccs[fov]) - np.array(all_expert_wo_decodingaccs[fov])]

f, ax = plt.subplots(1,2, figsize=(10,6), sharey='row')
# Plot
for i in range(len(learning_deltas)):
    ax[0].scatter(np.ones(len(learning_deltas[i])) * i, learning_deltas[i]) #, label='fov {}'.format(i))



ax[0].axhline(0, ls='--', color='black')
ax[0].set_title('Learning FOVs changes')
ax[0].set_ylabel('Within cluster decoding - without cluster decoding')
ax[0].set_xlabel('FOV')


# Plot expert
for i in range(len(expert_deltas)):
    ax[1].scatter(np.ones(len(expert_deltas[i])) * i, expert_deltas[i]) #, label='fov {}'.format(i))

ax[1].axhline(0, ls='--', color='black')
ax[1].set_title('Expert FOVs changes')
ax[1].set_ylabel('Within cluster decoding - without cluster decoding')
ax[1].set_xlabel('FOV')
plt.show()

# Plot a matched version per FOV
f = plt.figure(figsize=(10,5))

for i in range(len(learning_deltas)):
    plt.scatter(np.ones(len(learning_deltas[i])) * (i - 0.2), learning_deltas[i])
    plt.scatter(np.ones(len(expert_deltas[i])) * (i + 0.2), expert_deltas[i]) #, label='fov {}'.format(i))
    for j in range(len(learning_deltas[i])):
        plt.plot([i-0.2, i+0.2], [learning_deltas[i][j], expert_deltas[i][j]],
                 color='grey')
plt.axhline(0, ls = '--', color='black')
plt.ylabel('Within cluster decoding - without cluster decoding')
plt.xlabel('FOV')


#%% Get index of clusters for each FOV that outperforms without cluster neurons
cluster_idx = []
for fov in range(len(learning_deltas)):
    cluster_idx += [np.where(learning_deltas[fov] > 0)[0]]

# Plot the selectivity (tstat) of neurons in clusters that outperform vs that don't
outperform_sel, other_sel = [], []

for fov in range(len(learning_deltas)):
    for i in cluster_idx[fov]:
        outperform_sel += [all_learning_tstat[fov][i]]
    other_idx = [j for j in range(len(all_learning_tstat[fov])) if j not in cluster_idx[fov]]
    for i in other_idx:
        other_sel += [all_learning_tstat[fov][i]]


# plt.bar([0,1],[np.median(outperform_sel), np.median(other_sel)])
plt.boxplot([outperform_sel,other_sel])
# plt.scatter(np.zeros(len(outperform_sel)), outperform_sel)
# plt.scatter(np.ones(len(other_sel)), other_sel)
# plt.ylim(top=0.6)
plt.yscale("log")
plt.ylabel('Selectivity (t-statistic)')
plt.xticks([1,2], ['Outperforming clusters',
                    'Other clusters'])

t_stat, p_value = ttest_ind(outperform_sel,
                            other_sel) # unpaired t-test

print(t_stat, p_value)


# Plot the size of the neuron clusters that outperform vs not
for fov in range(len(learning_deltas)):
    paths = agg_mice_paths[fov]
    clusters = pd.read_pickle(paths[3])
    trialparams = np.mean(clusters.trial_params.to_numpy())
    num_clusters = len(trialparams.columns)

    learning = trialparams.loc[idx['learning', :]]
    learning_normalized = pd.DataFrame(normalize(learning, norm='l1'), columns=learning.columns)#, index=learning.index)
    
    expert = trialparams.loc[idx['expert', :]]
    expert_normalized = pd.DataFrame(normalize(expert, norm='l1'), columns=expert.columns)#, index=expert.index)
    
    neuronparams = np.mean(clusters.components.to_numpy()).T
    neurons_norm = pd.DataFrame(normalize(neuronparams, norm='l1'), columns=neuronparams.columns, index=neuronparams.index)
    neurons_norm_arr = neurons_norm.to_numpy()
    max_neurons_norm_arr = np.where(np.argmax(neurons_norm_arr, axis=1) == cl)[0]

