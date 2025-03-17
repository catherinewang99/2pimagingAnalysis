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

def reliability_score(corr):
    """
    Return the reliability score for a neuron across r OR l trials

    Returns
    -------
    Single number.

    """
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
    return proportion_clustered

def filter_idmap(idmap, minsize=15):
    """
    Take out all the cluster numbers that have fewer than 15 trials in a cluster,
    return shortened version of idmap

    Parameters
    ----------
    idmap : TYPE
        DESCRIPTION.
    minsize : TYPE, optional
        DESCRIPTION. The default is 15.

    Returns
    -------
    idmap_filtered : TYPE
        DESCRIPTION.

    """
    all_clusters = list(set(idmap))
    idmap_filtered = idmap
    for c in all_clusters:
        indices = np.where(idmap_filtered == c)[0]
        if len(indices) < minsize or len(indices) > len(idmap)/2: # Too big or too small
            
            idmap_filtered = np.delete(idmap_filtered, indices)   
            

    return idmap_filtered


#%% Plot example neurons for COSYNE
naivepath, learningpath, expertpath = [r'H:\data\BAYLORCW046\python\2024_05_31',
                    r'H:\data\BAYLORCW046\python\2024_06_11',
                  r'H:\data\BAYLORCW046\python\2024_06_26',]

naivepath, learningpath, expertpath =  [r'H:\data\BAYLORCW044\python\2024_05_22',
                  r'H:\data\BAYLORCW044\python\2024_06_06',
                r'H:\data\BAYLORCW044\python\2024_06_19']
l1 = Mode(learningpath, use_reg = True, triple=True, baseline_normalization="median_zscore")

_, l = l1.get_trace_matrix(62) # or 4

# Get just the delay activity
l_delay = [ltrial[l1.delay:l1.response] for ltrial in l]

l_delay_mean = np.reshape(np.mean(l_delay, axis=1), (1,-1))
l_delay_cat = np.reshape(cat(l_delay), (1, -1))

ax = plt.imshow(l_delay_mean, cmap='gray', interpolation='nearest', aspect='auto')
plt.colorbar(ax)

l2 = Mode(expertpath, use_reg = True, triple=True, baseline_normalization="median_zscore")
n = l2.good_neurons[np.where(l1.good_neurons == 62)[0][0]]
_, l = l2.get_trace_matrix(n) # or 4

# Get just the delay activity
l_delay = [ltrial[l2.delay:l2.response] for ltrial in l]

l_delay_mean_exp = np.reshape(np.mean(l_delay, axis=1), (1,-1))
l_delay_cat_exp = np.reshape(cat(l_delay), (1,-1))

ax = plt.imshow(l_delay_mean_exp, cmap='gray', interpolation='nearest', aspect='auto')
plt.colorbar(ax)

f = plt.figure(figsize=(15,1))
ax = plt.imshow(np.hstack((l_delay_mean,l_delay_mean_exp)), cmap='gray', interpolation='nearest', aspect='auto', vmin=1.5, vmax=10)
plt.colorbar(ax)
plt.axvline(l_delay_mean.shape[1], color='red')
plt.title('Mean delay activity')
# plt.savefig(r'H:\COSYNE 2025\CW46_singleneuronunreliable_n4_meanactivity.pdf')

f = plt.figure(figsize=(15,1))
ax = plt.imshow(np.hstack((l_delay_cat,l_delay_cat_exp)), cmap='gray', interpolation='nearest', aspect='auto')
plt.colorbar(ax)
plt.axvline(l_delay_cat.shape[1], color='red')
plt.title('All delay activity')
# plt.savefig(r'H:\COSYNE 2025\CW46_singleneuronunreliable_n4_alldelayactivity.pdf')

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

path = expertpath
sizelim = 15
l1 = Mode(path, use_reg = True, triple=True, baseline_normalization="median_zscore")

for idx in range(len(l1.good_neurons)):

    rcorr, lcorr = l1.plot_heatmap_across_sess(l1.good_neurons[idx], return_arr=True)
    _, idmap_r, sil_r = cluster_corr(rcorr, both=True)
    _, idmap_l, sil_l = cluster_corr(lcorr, both=True)
    
    idmap_r = filter_idmap(idmap_r, minsize=sizelim)
    idmap_l = filter_idmap(idmap_l, minsize=sizelim)

    num_clusters_r += [len(set(idmap_r))]
    num_clusters_l += [len(set(idmap_l))]
    
    if len(idmap_r) != 0:

        av_clus_size_r += [np.average(list(Counter(list(idmap_r)).values()))]
        max_clus_size_r += [max(list(Counter(list(idmap_r)).values()))]

    if len(idmap_l) != 0:
        av_clus_size_l += [np.average(list(Counter(list(idmap_l)).values()))]
        
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
rcorr, lcorr = l1.plot_heatmap_across_sess(l1.good_neurons[idx], return_arr=True)

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
    rcorr, lcorr = l1.plot_heatmap_across_sess(good_n, return_arr=True)
    
    proportion_clustered = reliability_score(rcorr)
    all_proportions_l += [proportion_clustered]
#%% Look at distribution of proportions in one session

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




#%% Run over ALL sessions

allnums_r = []
allavgs_r = []
allmaxs_r = []
allrel_r = []

allnums_l = []
allavgs_l = []
allmaxs_l = []
allrel_l = []

for i in range(3):
    
    nums_r = []
    avgs_r =[]
    maxs_r = []
    rel_r = []
    
    nums_l = []
    avgs_l =[]
    maxs_l = []
    rel_l = []
    
    for path in allpaths[i]:

        l1 = Session(path, use_reg = True, triple=True)
        
        # per neuron measurements
        num_clusters_l = []
        num_clusters_r = []

        max_clus_size_l = []
        max_clus_size_r = []

        av_clus_size_l = []
        av_clus_size_r = []
        
        all_rel_l = []
        all_rel_r = []

        for idx in range(len(l1.good_neurons)):
            
            rcorr, lcorr = l1.plot_heatmap_across_sess(l1.good_neurons[idx], return_arr=True)

            _, idmap_r, sil_r = cluster_corr(rcorr, both=True)
            _, idmap_l, sil_l = cluster_corr(lcorr, both=True)

            num_clusters_r += [len(set(idmap_r))]
            num_clusters_l += [len(set(idmap_l))]
            
            av_clus_size_r += [np.average(list(Counter(list(idmap_r)).values()))]
            av_clus_size_l += [np.average(list(Counter(list(idmap_l)).values()))]
            
            max_clus_size_r += [max(list(Counter(list(idmap_r)).values()))]
            max_clus_size_l += [max(list(Counter(list(idmap_l)).values()))]
            
            all_rel_r += [reliability_score(rcorr)]
            all_rel_l += [reliability_score(lcorr)]
            
            
        nums_r += [np.mean(num_clusters_r)]
        avgs_r += [np.mean(av_clus_size_r)]
        maxs_r += [np.median(max_clus_size_r)]
        rel_r += [all_rel_r]
        
        nums_l += [np.mean(num_clusters_l)]
        avgs_l += [np.mean(av_clus_size_l)]
        maxs_l += [np.median(max_clus_size_l)]
        rel_l += [all_rel_l]

        
    allnums_r += [nums_r]
    allavgs_r += [avgs_r]
    allmaxs_r += [maxs_r]
    allrel_r += [rel_r]
    
    allnums_l += [nums_l]
    allavgs_l += [avgs_l]
    allmaxs_l += [maxs_l]
    allrel_l += [rel_l]

#%% Violin plot

# Make df object to plot

df = pd.DataFrame()
df['score'] = cat(allrel_l[0])
df['Stage'] = 'Naive'
df['Trial'] = 'Left'

df1 = pd.DataFrame()
df1['score'] = cat(allrel_l[1])
df1['Stage'] = 'Learning'
df1['Trial'] = 'Left'

df2 = pd.DataFrame()
df2['score'] = cat(allrel_l[2])
df2['Stage'] = 'Expert'
df2['Trial'] = 'Left'

all_df = pd.concat((df,df1,df2))

df = pd.DataFrame()
df['score'] = cat(allrel_r[0])
df['Stage'] = 'Naive'
df['Trial'] = 'Right'

df1 = pd.DataFrame()
df1['score'] = cat(allrel_r[1])
df1['Stage'] = 'Learning'
df1['Trial'] = 'Right'

df2 = pd.DataFrame()
df2['score'] = cat(allrel_r[2])
df2['Stage'] = 'Expert'
df2['Trial'] = 'Right'

all_df = pd.concat((all_df, df, df1, df2))

# sns.violinplot(data=all_df, x='Stage', y='score', hue='Trial', split=True, inner="quart")
# plt.ylim(top=0.25)

sns.violinplot(data=all_df, x='Stage', y='score', hue='Trial', fill=False, inner="quart")
# plt.ylim(top=0.25)

#%% Median reliability score
f = plt.figure(figsize = (5,5))
plt.bar(np.arange(3)-0.2, [np.median(cat(s)) for s in allrel_r], 0.4, color='b', alpha=0.5)
plt.bar(np.arange(3)+0.2, [np.median(cat(s)) for s in allrel_l], 0.4, color='r', alpha=0.5)
for i in range(3):
    plt.scatter(np.ones(len(allrel_r[i])) * i - 0.2, [np.median((s)) for s in allrel_r[i]], color = 'b')
    plt.scatter(np.ones(len(allrel_l[i])) * i + 0.2, [np.median((s)) for s in allrel_l[i]], color = 'r')
    
plt.xticks(range(3), ['Naive', 'Learning', 'Expert'])
plt.ylabel('Med. reliability score')
plt.title('Median reliability score per FOV over learning')

#%% Get modularity and sample ampl from learning sess

mod = []
rob = []
seln_idx = []
sample_ampl = []
for path in allpaths[1]:

    l1 = Mode(path, use_reg = True, triple=True)
    m, _ = l1.modularity_proportion(p=0.01, period = range(l1.delay, l1.delay + int(1.5 * 1/l1.fs)))
    mod += [m]
    r, _ = l1.modularity_proportion(p=0.01, period = range(l1.response - int(1.5 * 1/l1.fs), l1.response))
    rob += [r]
    idx = [np.where(l1.good_neurons == n)[0][0] for n in l1.selective_neurons]
    seln_idx += [idx]
    
    orthonormal_basis, mean, db, acc_learning_sample = l1.decision_boundary(mode_input='stimulus', persistence=False)
    lea_sample = np.mean(acc_learning_sample)
    lea_sample = lea_sample if lea_sample > 0.5 else 1-lea_sample
    sample_ampl += [lea_sample]

    
#%% Learning sessions only - modularity vs reliability score
# silh_prop = []
# sil_cutoff = 0.35

# for i in range(len(allsils_l[1])):
#     left = np.array(allsils_l[1][i]) > sil_cutoff
#     right = np.array(allsils_r[1][i]) > sil_cutoff
#     total_alln = np.array([left[i] or right[i] for i in range(len(left))])
#     total = total_alln[seln_idx[i]]
#     silh_prop += [sum(total) / len(seln_idx[i]) * 100]

f = plt.figure(figsize = (5,5))
plt.scatter([np.median((s)) for s in allrel_r[1]], mod, marker='x', color='blue')
plt.scatter([np.median((s)) for s in allrel_l[1]], mod, marker='x', color='red')
plt.xlabel('Med. reliability score')
plt.ylabel('Modularity')
print(scipy.stats.pearsonr([np.median((s)) for s in allrel_l[1]], mod))

f = plt.figure(figsize = (5,5))
plt.scatter([np.median((s)) for s in allrel_r[1]], rob, marker='x', color='blue')
plt.scatter([np.median((s)) for s in allrel_l[1]], rob, marker='x', color='red')
plt.xlabel('Med. reliability score')
plt.ylabel('Robustness')
print(scipy.stats.pearsonr([np.median((s)) for s in allrel_l[1]], rob))

f = plt.figure(figsize = (5,5))
plt.scatter([np.median((s)) for s in allrel_r[1]], sample_ampl, marker='x', color='blue')
plt.scatter([np.median((s)) for s in allrel_l[1]], sample_ampl, marker='x', color='red')
plt.xlabel('Med. reliability score')
plt.ylabel('Sample amplitude')
print(scipy.stats.pearsonr([np.median((s)) for s in allrel_l[1]], sample_ampl))
print(scipy.stats.pearsonr([np.median((s)) for s in allrel_r[1]], sample_ampl))


#%% Learning sessions only - sample ampl vs cluster score
silh_prop = []
sil_cutoff = 0.35
for i in range(len(allsils_l[1])):
    left = np.array(allsils_l[1][i]) > sil_cutoff
    right = np.array(allsils_r[1][i]) > sil_cutoff
    total_alln = np.array([left[i] or right[i] for i in range(len(left))])
    total = total_alln[seln_idx[i]]
    # total = total_alln
    silh_prop += [sum(total) / len(seln_idx[i]) * 100]

f = plt.figure(figsize = (5,5))
plt.scatter(silh_prop, sample_ampl, marker='x')
plt.xlabel('% of well clustered neurons')
plt.ylabel('Sample amplitude')
print(scipy.stats.pearsonr(silh_prop, sample_ampl))