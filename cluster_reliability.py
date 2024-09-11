# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 16:03:49 2024

@author: catherinewang


Code for cluster correlation matrix from https://wil.yegelwel.com/cluster-correlation-matrix/
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
import seaborn as sns
from collections import Counter
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score

cat = np.concatenate
plt.rcParams['pdf.fonttype'] = '42' 

#%% Functions
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

# Heatmap function
def plot_heatmap_across_sess(sess, neuron, return_arr=False):
    r, l = sess.get_trace_matrix(neuron)
    r, l = np.array(r), np.array(l)
        
    df = pd.DataFrame(r[:,range(sess.delay, sess.response)].T)  
    corrs = df.corr()
    
    df = pd.DataFrame(l[:,range(sess.delay, sess.response)].T)  
    l_corrs = df.corr()
    
    if return_arr:
        return corrs, l_corrs
    
    f = plt.figure(figsize = (5,5))
    plt.imshow(corrs)
    plt.xlabel('R trials')
    plt.title('Correlation of delay activity in R trials')
    plt.colorbar()   
    
    f = plt.figure(figsize = (5,5))
    plt.imshow(l_corrs)
    plt.xlabel('L trials')
    plt.title('Correlation of delay activity in L trials')
    plt.colorbar() 
    
    
import scipy
import scipy.cluster.hierarchy as sch

def cluster_corr(corr_array, method = 'complete', inplace=False, both = False):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly 
    correlated variables are next to eachother 
    
    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix 
        
    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    # linkage = sch.linkage(pairwise_distances, method=method)
    linkage = sch.linkage(corr_array, method=method)
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, 
                                        criterion='distance')
    # print(cluster_distance_threshold)
    idx = np.argsort(idx_to_cluster_array)
    if len(set(idx_to_cluster_array)) == corr_array.shape[0]:
        sil_score, sil_score1 = 0, 2
    else:
        sil_score = silhouette_score(corr_array, idx_to_cluster_array)
    
        sil_score1 = davies_bouldin_score(corr_array, idx_to_cluster_array)
    
    # sil_score = inconsistent(linkage)
    
    if not inplace:
        corr_array = corr_array.copy()
    
    if isinstance(corr_array, pd.DataFrame):
        if both:
            return corr_array.iloc[idx, :].T.iloc[idx, :], idx_to_cluster_array, (sil_score, sil_score1)
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx]


#%% Get clusters CW46 FOV 3
naivepath, learningpath, expertpath = [r'H:\data\BAYLORCW046\python\2024_05_31',
                    r'H:\data\BAYLORCW046\python\2024_06_11',
                  r'H:\data\BAYLORCW046\python\2024_06_26',]

num_clusters_l = []
num_clusters_r = []

max_clus_size_l = []
max_clus_size_r = []

av_clus_size_l = []
av_clus_size_r = []

all_sil_r = []
all_sil_l = []

path = learningpath
n=15
l1 = Mode(path, use_reg = True, triple=True, baseline_normalization="median_zscore")

for idx in range(len(l1.good_neurons)):

    rcorr, lcorr = plot_heatmap_across_sess(l1, l1.good_neurons[idx], return_arr=True)
    _, idmap_r, sil_r = cluster_corr(rcorr, both=True)
    _, idmap_l, sil_l = cluster_corr(lcorr, both=True)

    num_clusters_r += [len(set(idmap_r))]
    num_clusters_l += [len(set(idmap_l))]
    
    av_clus_size_r += [np.average(list(Counter(list(idmap_r)).values()))]
    av_clus_size_l += [np.average(list(Counter(list(idmap_l)).values()))]
    
    max_clus_size_r += [max(list(Counter(list(idmap_r)).values()))]
    max_clus_size_l += [max(list(Counter(list(idmap_l)).values()))]
    
    all_sil_r += [sil_r]
    all_sil_l += [sil_l]

all_sil_r = np.array(all_sil_r)
all_sil_l = np.array(all_sil_l)


#%% Example neuron old vis
idx=20
method = 'complete'
l1.plot_rasterPSTH_sidebyside(l1.good_neurons[idx])
plot_heatmap_across_sess(l1, l1.good_neurons[idx])
rcorr, lcorr = plot_heatmap_across_sess(l1, l1.good_neurons[idx], return_arr=True)
_, idmap_r, sil_r = cluster_corr(rcorr, method=method, both=True)
_, idmap_l, sil_l = cluster_corr(lcorr, method=method, both=True)
f = plt.figure(figsize = (6,5))
sns.heatmap(cluster_corr(rcorr, method=method))
f = plt.figure(figsize = (6,5))
sns.heatmap(cluster_corr(lcorr, method=method))
print(sil_r, sil_l)

#%% Example neuron binary view
idmap = idmap_r
n = len(idmap)
idx = np.argsort(idmap)
sorted_idmap = idmap[idx]
matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if sorted_idmap[i] == sorted_idmap[j]:
            matrix[i,j] = 1
            
plt.imshow(matrix)
plt.colorbar()
plt.show()

#%% Amount of overlap - show by sorting the same way
idx=10
idx=101
idx = 89
l1.plot_rasterPSTH_sidebyside(l1.good_neurons[idx])
rcorr, lcorr = plot_heatmap_across_sess(l1, l1.good_neurons[idx], return_arr=True)

method = 'complete'
corr = rcorr
_, idmap_l, sil_l = cluster_corr(corr, method=method, both=True)

idmap = idmap_l
n = len(idmap)
idx = np.argsort(idmap)
sorted_idmap = idmap[idx]
matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if sorted_idmap[i] == sorted_idmap[j]:
            matrix[i,j] = 1
            
plt.imshow(matrix)
plt.colorbar()
plt.show()

method = 'ward'
_, idmap_l_new, sil_l = cluster_corr(corr, method=method, both=True)

# idmap = idmap_l
# idx = np.argsort(idmap)
sorted_idmap = idmap_l_new[idx]
matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if sorted_idmap[i] == sorted_idmap[j]:
            matrix[i,j] = 1
            
plt.imshow(matrix)
plt.colorbar()
plt.show()
#%%  Measure overlap - count number of neurons in the same cluster regardless of how many clusters there are
idx = 89
idx=101
all_proportions_l = []
for good_n in l1.good_neurons:
    rcorr, lcorr = plot_heatmap_across_sess(l1, good_n, return_arr=True)
    
    corr = lcorr
    _, idmap, _ = cluster_corr(corr, method='complete', both=True)
    n = len(idmap)
    idx = np.argsort(idmap)
    sorted_idmap = idmap[idx]
    
    _, idmap_new, _ = cluster_corr(corr, method='ward', both=True)
    sorted_idmap_new = idmap_new[idx]
    
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if sorted_idmap[i] == sorted_idmap[j]:
                if sorted_idmap_new[i] == sorted_idmap_new[j]:
                    matrix[i,j] = 1
    
    # plt.imshow(matrix)
    # plt.colorbar()
    # plt.show()
    
    score = (np.sum(matrix) - n) / 2
    total_possible = n * n / 2
    
    proportion_clustered = score / total_possible
    all_proportions_l += [proportion_clustered]
#%% Look at distribution of proportions

f = plt.figure(figsize = (5,5))
# plt.hist(all_proportions)
plt.scatter(np.log(allr), np.log(all_proportions), color = 'b')
plt.scatter(np.log(allr), np.log(all_proportions_l), color='r')
plt.ylabel('Reliability of clusters')
plt.xlabel('Variance of weights (log)')
print(scipy.stats.pearsonr(np.log(allr), np.log(all_proportions)))
print(scipy.stats.pearsonr(np.log(allr), np.log(all_proportions_l)))

## Right vs left
f = plt.figure(figsize = (5,5))
plt.scatter(np.log(all_proportions_l), np.log(all_proportions), color = 'grey')
plt.ylabel('Left trial reliability')
plt.xlabel('Right trial reliability')
print(scipy.stats.pearsonr(np.log(all_proportions_l), np.log(all_proportions)))


# np.argsort(all_proportions)
# idx=55
# method = 'complete'
# l1.plot_rasterPSTH_sidebyside(l1.good_neurons[idx])
# plot_heatmap_across_sess(l1, l1.good_neurons[idx])
# rcorr, lcorr = plot_heatmap_across_sess(l1, l1.good_neurons[idx], return_arr=True)
# _, idmap_r, sil_r = cluster_corr(rcorr, method=method, both=True)
# _, idmap_l, sil_l = cluster_corr(lcorr, method=method, both=True)
# f = plt.figure(figsize = (6,5))
# sns.heatmap(cluster_corr(rcorr, method=method))
# f = plt.figure(figsize = (6,5))
# sns.heatmap(cluster_corr(lcorr, method=method))
# print(sil_r, sil_l)



