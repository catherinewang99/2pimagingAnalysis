# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 14:59:48 2025

Use this script to properly select variables for LDA clusters by trial and neurons
------

Sanity checks for adjusting thresholds of probability of inclusion for neurons/trials

1. Between neuron covariances on within vs without cluster trials
      - there should be higher covariance on within trials (neurons doing similar things)

2. Is the firing rate (dF/F0) higher in clustered trials for within neurons vs without?
      - the neurons should be firing more for both 1. within cluster trials and 2. than without cluster neurons


Start by using the strictest inclusion threshold for neurons/trials

Plot the gradient over looser thresholds


@author: catherinewang

"""

#%% import funcs


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

# %% Paths
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

naivepath, learningpath, expertpath, clusterpath = [
                    r'H:\data\BAYLORCW046\python\2024_05_31',
                    r'H:\data\BAYLORCW046\python\2024_06_11',
                    r'H:\data\BAYLORCW046\python\2024_06_26',
                    r'H:\data\matched_topic_params\CW46_FOV3_table']

#%% Analysis

clusters = pd.read_pickle(clusterpath + '_last_1')
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

path = learningpath
s1 = Mode(path, use_reg = True, triple=True, baseline_normalization="median_zscore",
          filter_good_neurons=neurons_norm.index) # only read in selective neurons
orthonormal_basis, mean_exp = s1.plot_CD(ctl=True)

#%% Build sanity check analysis

cluster = 0
num_clusters = len(trialparams.columns)

# Max method
cluster_trials_all_idx = np.where(np.argmax(learning_normalized, axis=1) == cluster)[0]
max_neurons_norm_arr = np.where(np.argmax(neurons_norm_arr, axis=1) == cluster)[0]

# Threshold method
threshold = 1/num_clusters - 0
cluster_trials_all_idx = learning_normalized.index[learning_normalized['topic_{}'.format(cluster)] > threshold].to_numpy()
max_neurons_norm_arr = np.where(neurons_norm_arr[:, cluster] > threshold)[0]

non_cluster_idx = [i for i in range(len(neurons_norm_arr)) if i not in max_neurons_norm_arr]

s1 = Mode(path, use_reg = True, triple=True, baseline_normalization="median_zscore",
          filter_good_neurons=neurons_norm.index) # only read in selective neurons

# s1 = Mode(path, use_reg = True, triple=True, baseline_normalization="median_zscore",
#             filter_good_neurons=neurons_norm.index,
#             cluster_neurons=max_neurons_norm_arr,
#             i_good = cluster_trials_all_idx,
#             lda_cluster=True)


# Check if within cluster neurons covary more than without cluster neurons on within cluster trials
within_cluster_neurons = s1.good_neurons[max_neurons_norm_arr]
within_cluster_trials = cluster_trials_all_idx
without_cluster_trials = np.array([t for t in s1.i_good_trials if t not in within_cluster_trials])
without_cluster_neurons = s1.good_neurons[non_cluster_idx]
delay = np.arange(s1.delay, s1.response)

def plot_covariance_matrix(obj, neurons, trials, period):
    """
    Plot the covariance between neuron FR during the given period for a set of trials
    
    get variance between neurons on a per trial basis

    Parameters
    ----------
    neurons : TYPE
        DESCRIPTION.
    trials : TYPE
        DESCRIPTION.
    period : TYPE
        DESCRIPTION.

    Returns
    -------
    Covariance matrix?.

    """
    
    # for n in neurons:
    #     dff, _ = obj.get_trace_matrix(n, rtrials=trials)
    all_v = []
    
    for t in trials:
        
        dff = obj.dff[0,t][neurons, :obj.time_cutoff]
        
        cov_matrix = np.cov(dff[:, period])
        upper_triangle_values = cov_matrix[np.triu_indices_from(cov_matrix, k=1)]
        all_v += [np.mean(np.abs(upper_triangle_values))]

    return all_v
    

def plot_correlation_matrix(obj, neurons, trials, period):
    """
    Plot the correlation between neuron FR during the given period for a set of trials
    
    get variance between neurons on a per trial basis

    Parameters
    ----------
    neurons : TYPE
        DESCRIPTION.
    trials : TYPE
        DESCRIPTION.
    period : TYPE
        DESCRIPTION.

    Returns
    -------
    Covariance matrix?.

    """
    
    # for n in neurons:
    #     dff, _ = obj.get_trace_matrix(n, rtrials=trials)
    all_v = []
    
    for t in trials:
        
        dff = obj.dff[0,t][neurons, :obj.time_cutoff]
        
        cov_matrix = np.corrcoef(dff[:, period])
        upper_triangle_values = cov_matrix[np.triu_indices_from(cov_matrix, k=1)]
        all_v += [np.mean(upper_triangle_values)]

    return all_v


cov_within = plot_covariance_matrix(s1, within_cluster_neurons, within_cluster_trials, delay)
cov_without = plot_covariance_matrix(s1, without_cluster_neurons, within_cluster_trials, delay)



#%%  Look at covariance
plt.scatter(np.zeros(len(cov_within)), cov_within)
plt.scatter(np.ones(len(cov_without)), cov_without)
plt.title('Covariance for neurons_wi vs neurons_wo on within cluster trials')
plt.xticks([0,1], ['Neurons_wi', 'Neurons_wo'])
plt.show()

print(ttest_rel(cov_within, cov_without))
# Check between neuron covariances for within vs without cluster (build function?)

    
cov_within = plot_covariance_matrix(s1, within_cluster_neurons, within_cluster_trials, delay)
cov_without = plot_covariance_matrix(s1, within_cluster_neurons, without_cluster_trials, delay)

plt.scatter(np.zeros(len(cov_within)), cov_within)
plt.scatter(np.ones(len(cov_without)), cov_without)
plt.title('Covariance for neurons_wi on within vs without cluster trials')
plt.xticks([0,1], ['trials_wi', 'trials_wo'])
plt.show()

print(ttest_ind(cov_within, cov_without))

#%% Look at correlation

corr_within = plot_correlation_matrix(s1, within_cluster_neurons, within_cluster_trials, delay)
corr_without = plot_correlation_matrix(s1, without_cluster_neurons, within_cluster_trials, delay)

plt.scatter(np.zeros(len(corr_within)), corr_within)
plt.scatter(np.ones(len(corr_without)), corr_without)
plt.title('Correlation for neurons_wi vs neurons_wo on within cluster trials')
plt.xticks([0,1], ['Neurons_wi', 'Neurons_wo'])
plt.show()

print(ttest_rel(corr_within, corr_without))
# Check between neuron covariances for within vs without cluster (build function?)

    
corr_within = plot_correlation_matrix(s1, within_cluster_neurons, within_cluster_trials, delay)
corr_without = plot_correlation_matrix(s1, within_cluster_neurons, without_cluster_trials, delay)

plt.scatter(np.zeros(len(corr_within)), corr_within)
plt.scatter(np.ones(len(corr_without)), corr_without)
plt.title('Correlation for neurons_wi on within vs without cluster trials')
plt.xticks([0,1], ['trials_wi', 'trials_wo'])

print(ttest_ind(corr_within, corr_without))



#%% Look over multiple clusters as sanity check

all_ttest_corr = []
all_ttest_cov = []

all_mean_corr = []
all_mean_cov = [] 
    
for paths in agg_mice_paths:
    clusterpath = paths[3]
    path = paths[1]
    
    clusters = pd.read_pickle(clusterpath + '_last_1')
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
    
    s1 = Mode(path, use_reg = True, triple=True, baseline_normalization="median_zscore",
              filter_good_neurons=neurons_norm.index) # only read in selective neurons
    
    ttest_corr = []
    ttest_cov = []
    
    mean_corr = []
    mean_cov = [] 
    
    for cluster in range(num_clusters):
        num_clusters = len(trialparams.columns)
        
        # Max method
        cluster_trials_all_idx = np.where(np.argmax(learning_normalized, axis=1) == cluster)[0]
        max_neurons_norm_arr = np.where(np.argmax(neurons_norm_arr, axis=1) == cluster)[0]
        
        # Threshold method
        threshold = 1/num_clusters + 0.1
        cluster_trials_all_idx = learning_normalized.index[learning_normalized['topic_{}'.format(cluster)] > threshold].to_numpy()
        max_neurons_norm_arr = np.where(neurons_norm_arr[:, cluster] > threshold)[0]
        
        non_cluster_idx = [i for i in range(len(neurons_norm_arr)) if i not in max_neurons_norm_arr]
            
        # Check if within cluster neurons covary more than without cluster neurons on within cluster trials
        within_cluster_neurons = s1.good_neurons[max_neurons_norm_arr]
        within_cluster_trials = cluster_trials_all_idx
        without_cluster_trials = np.array([t for t in s1.i_good_trials if t not in within_cluster_trials])
        without_cluster_neurons = s1.good_neurons[non_cluster_idx]
        delay = np.arange(s1.delay, s1.response)
    
        cov_within = plot_covariance_matrix(s1, within_cluster_neurons, within_cluster_trials, delay)
        cov_without = plot_covariance_matrix(s1, without_cluster_neurons, within_cluster_trials, delay)
        corr_within = plot_correlation_matrix(s1, within_cluster_neurons, within_cluster_trials, delay)
        corr_without = plot_correlation_matrix(s1, without_cluster_neurons, within_cluster_trials, delay)
        
        mean_corr += [(np.mean(corr_within), np.mean(corr_without))]
        mean_cov += [(np.mean(cov_within), np.mean(cov_without))]
        
        tstat, p_val = ttest_rel(corr_within, corr_without)
        ttest_corr += [(tstat, p_val)]
        tstat, p_val = ttest_rel(cov_within, cov_without)
        ttest_cov += [(tstat, p_val)]
        
    
    all_ttest_corr += [ttest_corr]
    all_ttest_cov += [ttest_cov]

    all_mean_corr += [mean_corr]
    all_mean_cov += [mean_cov] 

#%% Plot results
# Correlation
for fov in range(len(all_ttest_corr)):

    plt.scatter(np.ones(len(all_ttest_corr[fov])) * fov, np.array(all_ttest_corr[fov])[:,0])
plt.axhline(0, ls='--')
plt.xlabel('FOVs')
plt.ylabel('t-statistics')
plt.show()
# For each FOV, how is the cluster quality

for fov in range(len(all_ttest_corr)):
    for cl in range(len(all_ttest_corr[fov])):
        if all_ttest_corr[fov][cl][1] > 0.01:
    
            plt.scatter([fov], [all_ttest_corr[fov][cl][0]], marker='x')
        else:
            plt.scatter([fov], [all_ttest_corr[fov][cl][0]], marker='o')

plt.xlabel('FOVs')
plt.ylabel('t-statistics')
plt.axhline(0, ls='--')
plt.show()

# Covariance
for fov in range(len(all_ttest_cov)):

    plt.scatter(np.ones(len(all_ttest_cov[fov])) * fov, np.array(all_ttest_cov[fov])[:,0])
plt.axhline(0, ls='--')
plt.xlabel('FOVs')
plt.ylabel('t-statistics')
plt.show()
# For each FOV, how is the cluster quality

for fov in range(len(all_ttest_cov)):
    for cl in range(len(all_ttest_cov[fov])):
        if all_ttest_corr[fov][cl][1] > 0.01:
    
            plt.scatter([fov], [all_ttest_cov[fov][cl][0]], marker='x')
        else:
            plt.scatter([fov], [all_ttest_cov[fov][cl][0]], marker='o')

plt.xlabel('FOVs')
plt.ylabel('t-statistics')
plt.axhline(0, ls='--')
plt.show()

# Size of dot is p-value. plot t-statistic

# Plot mean cov and mean corr






