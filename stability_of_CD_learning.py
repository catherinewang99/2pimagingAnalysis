# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 11:25:20 2024

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

def cluster_corr(corr_array, inplace=False, both = False):
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
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, 
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)
    
    if not inplace:
        corr_array = corr_array.copy()
    
    if isinstance(corr_array, pd.DataFrame):
        if both:
            return corr_array.iloc[idx, :].T.iloc[idx, :], idx_to_cluster_array
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx]


#%% PATHS

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

#%% CW46 sess 3 load data

paths = [naivepath, learningpath, expertpath]

path = learningpath
n=15
l1 = Mode(path, use_reg = True, triple=True, baseline_normalization="median_zscore")
orthonormal_basis_initial, mean = l1.plot_CD()
maxval = max(orthonormal_basis_initial)
maxn = np.where(orthonormal_basis_initial == maxval)[0][0]

for _ in range(n-1):

    l1 = Mode(path, use_reg = True, triple=True, baseline_normalization="median_zscore")
    orthonormal_basis, mean = l1.plot_CD()
    sign = np.sign(orthonormal_basis[maxn])
    print("sign: ", sign)
    orthonormal_basis_initial = np.vstack((orthonormal_basis_initial, orthonormal_basis * sign))

allr = np.var(orthonormal_basis_initial, axis=0)
avg_weights = np.mean(orthonormal_basis_initial, axis=0)

#%% Single neuron analysis

    
# Histogram of weight variances
f = plt.figure(figsize = (5,5))
plt.hist(allr, bins=25)
plt.xlabel('Variance of weights')
plt.ylabel('Num neurons')

f = plt.figure(figsize = (5,5))
plt.hist(avg_weights, bins=25)
plt.xlabel('Weights')
plt.ylabel('Num neurons')

# Weights vs variance

f = plt.figure(figsize = (5,5))
plt.scatter(np.abs(avg_weights),allr)
plt.ylabel('Variance')
plt.xlabel('Weights(abs)')

f = plt.figure(figsize = (5,5))
plt.scatter(np.abs(avg_weights),np.log(allr))
plt.ylabel('Variance (log)')
plt.xlabel('Weights(abs)')



f = plt.figure(figsize = (5,5))
plt.scatter((avg_weights),allr)
plt.ylabel('Variance')
plt.xlabel('Weights')

f = plt.figure(figsize = (5,5))
weighted_var = [allr[i]/np.abs(avg_weights)[i] for i in range(len(allr))]
# plt.scatter(weighted_var, np.abs(avg_weights))
plt.scatter((avg_weights), np.log(weighted_var))
plt.ylabel('Log variance, norm by weight (var/abs(weight))')
plt.xlabel('Weights')

f = plt.figure(figsize = (5,5))
weighted_var = [allr[i]/np.abs(avg_weights)[i] for i in range(len(allr))]
# plt.scatter(weighted_var, np.abs(avg_weights))
plt.scatter(np.abs(avg_weights), np.log(weighted_var))
plt.ylabel('Log variance, norm by weight (var/abs(weight))')
plt.xlabel('Weights (abs)')


f = plt.figure(figsize = (5,5))
plt.hist(np.log(weighted_var), bins=25)
plt.xlabel('Log variance, norm by weight (var/abs(weight))')
plt.ylabel('Num neurons')

#%% Separate by quantile (buckets of 0.1)
f = plt.figure(figsize = (5,5))
plt.hist(np.log(weighted_var), bins=25)
plt.xlabel('Log variance, norm by weight (var/abs(weight))')
plt.ylabel('Num neurons')
for i in range(10,100,10):
    print(i)
    plt.axvline(np.percentile(np.log(weighted_var), i), color='r', ls='--')
    
    
f = plt.figure(figsize = (5,5))
weighted_var = [allr[i]/np.abs(avg_weights)[i] for i in range(len(allr))]
# plt.scatter(weighted_var, np.abs(avg_weights))
plt.scatter(np.abs(avg_weights), np.log(weighted_var))
plt.ylabel('Log variance, norm by weight (var/abs(weight))')
plt.xlabel('Weights (abs)')
for i in range(10,100,10):
    print(i)
    plt.axhline(np.percentile(np.log(weighted_var), i), color='r', ls='--')
    
#%% Look at decoding accuracy by quantile subtraction
all_accs = []
l1 = Mode(path, use_reg = True, triple=True, baseline_normalization="median_zscore")

for i in range(90,9,-10):
    
    # perc = np.percentile(np.log(weighted_var), i)
    # remove_n = np.where(np.log(weighted_var) > perc)[0]
    
    perc = np.percentile(allr, i)
    remove_n = np.where(allr > perc)[0]

    accs = []
    
    l1.plot_CD(mode_input='choice', remove_n = remove_n)
    print(len(remove_n))
    
    # for _ in range(1):
    orthonormal_basis, mean, db, acc_learning = l1.decision_boundary(mode_input='choice', remove_n=remove_n)
    lea = np.mean(acc_learning)
    # lea = lea if lea > 0.3 else 1-lea
    accs += [lea]
    
    all_accs += [lea]

f = plt.figure(figsize = (5,5))
plt.plot(range(9), all_accs, marker='x')

#%% Look at some random neurons

for idx in range(3):
    l1.plot_rasterPSTH_sidebyside(l1.good_neurons[idx])
    rcorr, lcorr = plot_heatmap_across_sess(l1, l1.good_neurons[idx], return_arr=True)
    f = plt.figure(figsize = (5,5))
    sns.heatmap(cluster_corr(rcorr))
    f = plt.figure(figsize = (5,5))
    sns.heatmap(cluster_corr(lcorr))

#%% Look at the high var neurons: non norm
idx_highvar = np.where(np.array(allr) > 0.006)[0]

for idx in idx_highvar[0:1]:
    l1.plot_rasterPSTH_sidebyside(l1.good_neurons[idx])
    rcorr, lcorr = plot_heatmap_across_sess(l1, l1.good_neurons[idx], return_arr=True)
    f = plt.figure(figsize = (6,5))
    sns.heatmap(cluster_corr(rcorr))
    f = plt.figure(figsize = (6,5))
    sns.heatmap(cluster_corr(lcorr))


# Look at these same neurons in expert session
# l2 = Mode(expertpath, use_reg=True, triple=True)
for idx in idx_highvar[0:1]:
    l2.plot_rasterPSTH_sidebyside(l2.good_neurons[idx])
    rcorr, lcorr = plot_heatmap_across_sess(l2, l2.good_neurons[idx], return_arr=True)
    f = plt.figure(figsize = (6,5))
    sns.heatmap(cluster_corr(rcorr))
    f = plt.figure(figsize = (6,5))
    sns.heatmap(cluster_corr(lcorr))
    
#%% Investiage the cluster trial averaged activity:
    
rcorr, lcorr = plot_heatmap_across_sess(l1, l1.good_neurons[idx_highvar[0]], return_arr=True)
_, idmap_r = cluster_corr(rcorr, both=True)
_, idmap_l = cluster_corr(lcorr, both=True)

dict_idmap_r =  dict(Counter(list(idmap_r)))
dict_idmap_l =  dict(Counter(list(idmap_l)))

l_corr_ctl_trials = np.array([l for l in l1.lick_correct_direction('l') if not l1.stim_ON[l]])
r_corr_ctl_trials = np.array([l for l in l1.lick_correct_direction('r') if not l1.stim_ON[l]])
# sort by size of cluster:
    
for cl in sorted(dict_idmap_l, key=dict_idmap_l.get, reverse=True):
    
    trialidx = np.where(idmap_l == cl)[0]
    trials = l_corr_ctl_trials[trialidx]
    l1.plot_single_neuron_multi_trial(l1.good_neurons[idx_highvar[0]], trials)

# look at the time course of the clusters arising
f = plt.figure(figsize = (8,5))

counter = 5
for cl in sorted(dict_idmap_l, key=dict_idmap_l.get, reverse=True):

    trialidx = np.where(idmap_l == cl)[0]
    plt.bar(trialidx, np.ones(len(trialidx)) * counter, alpha=0.5)
    counter -= 1


#%% Run measures of clusters over FOV:


num_clusters_l = []
num_clusters_r = []

max_clus_size_l = []
max_clus_size_r = []

av_clus_size_l = []
av_clus_size_r = []

all_sil_r = []
all_sil_l = []

# path = learningpath
# n=15
# l1 = Mode(path, use_reg = True, triple=True, baseline_normalization="median_zscore")

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



#%% Characterizing the number and size of clusters

f = plt.figure(figsize = (5,5))
plt.hist(num_clusters_r, bins=25, alpha=0.5, color='b', label='R trials')
plt.hist(num_clusters_l, bins=25, alpha=0.5, color='r', label='L trials')
plt.ylabel('Num')
plt.xlabel('Number of clusters')
plt.title('Number of clusters for R/L trials')
plt.legend()


f = plt.figure(figsize = (5,5))
plt.hist(av_clus_size_r, bins=25, alpha=0.5, color='b', label='R trials')
plt.hist(av_clus_size_l, bins=25, alpha=0.5, color='r', label='L trials')
plt.ylabel('Num')
plt.xlabel('Size of cluster')
plt.title('Av. size of clusters for R/L trials')
plt.legend()

f = plt.figure(figsize = (5,5))
plt.hist(max_clus_size_r, bins=25, alpha=0.5, color='b', label='R trials')
plt.hist(max_clus_size_l, bins=25, alpha=0.5, color='r', label='L trials')
plt.ylabel('Num')
plt.xlabel('Max size of clusters')
plt.title('Max size of clusters for R/L trials')
plt.legend()

f = plt.figure(figsize = (5,5))
plt.scatter(np.log(num_clusters_r),np.log(av_clus_size_r),color='b', label='R trials')
plt.scatter(np.log(num_clusters_l),np.log(av_clus_size_l),color='r', label='L trials')
plt.ylabel('Average cluster size (log)')
plt.xlabel('Number of clusters (log)')
plt.title('Average cluster size vs number of clusters')
plt.legend()

f = plt.figure(figsize = (5,5))
plt.scatter((num_clusters_r),(max_clus_size_r),color='b', label='R trials')
plt.scatter((num_clusters_l),(max_clus_size_l),color='r', label='L trials')
plt.ylabel('Max cluster size (log)')
plt.xlabel('Number of clusters (log)')
plt.title('Max cluster size vs number of clusters')
plt.legend()


f = plt.figure(figsize = (5,5))
plt.scatter(abs(avg_weights),(num_clusters_r),color='b')
plt.scatter(abs(avg_weights),num_clusters_l,color='r')
plt.ylabel('Number of clusters')
plt.xlabel('Weight')

#%% Relating the size and number of clusters to the variance measures

f = plt.figure(figsize = (5,5))
plt.scatter(np.log(allr),(num_clusters_r),color='b')
plt.scatter(np.log(allr),num_clusters_l,color='r')
plt.ylabel('Number of clusters')
plt.xlabel('Variance of weights (log)')
print(scipy.stats.pearsonr(np.log(allr), num_clusters_r))
print(scipy.stats.pearsonr(np.log(allr), num_clusters_l))



f = plt.figure(figsize = (5,5))
plt.scatter(np.log(allr),(max_clus_size_r),color='b')
plt.scatter(np.log(allr),(max_clus_size_l),color='r')
plt.ylabel('Max size of clusters')
plt.xlabel('Variance of weights (log)')
print(scipy.stats.pearsonr(np.log(allr), max_clus_size_r))
print(scipy.stats.pearsonr(np.log(allr), max_clus_size_l))


f = plt.figure(figsize = (5,5))
plt.scatter(np.log(allr),(av_clus_size_r),color='b')
plt.scatter(np.log(allr),(av_clus_size_l),color='r')
plt.ylabel('Avg size of clusters')
plt.xlabel('Variance of weights (log)')
print(scipy.stats.pearsonr(np.log(allr), av_clus_size_r))
print(scipy.stats.pearsonr(np.log(allr), av_clus_size_l))



f = plt.figure(figsize = (5,5))
plt.scatter(np.log(weighted_var),(max_clus_size_r),color='b')
plt.scatter(np.log(weighted_var),(max_clus_size_l),color='r')
plt.ylabel('Max size of clusters')
plt.xlabel('Variance of weights norm (log)')
print(scipy.stats.pearsonr(np.log(allr), max_clus_size_r))
print(scipy.stats.pearsonr(np.log(allr), max_clus_size_l))


f = plt.figure(figsize = (5,5))
plt.scatter(np.log(allr),(num_clusters_r),color='b')
plt.scatter(np.log(allr),num_clusters_l,color='r')
plt.ylabel('Number of clusters')
plt.xlabel('Variance of weights (log)')
print(scipy.stats.pearsonr(np.log(allr), num_clusters_r))
print(scipy.stats.pearsonr(np.log(allr), num_clusters_l))



# Number of clusters vs the size of the weight

f = plt.figure(figsize = (5,5))
plt.scatter(np.abs(avg_weights),(num_clusters_r),color='b')
plt.scatter(np.abs(avg_weights),num_clusters_l,color='r')
plt.ylabel('Number of clusters')
plt.xlabel('Weight value')


f = plt.figure(figsize = (5,5))
plt.scatter(np.abs(avg_weights),(max_clus_size_r),color='b')
plt.scatter(np.abs(avg_weights),max_clus_size_l,color='r')
plt.ylabel('Max size of clusters')
plt.xlabel('Weight value')


f = plt.figure(figsize = (5,5))
plt.scatter(np.abs(avg_weights),(av_clus_size_r),color='b')
plt.scatter(np.abs(avg_weights),av_clus_size_l,color='r')
plt.ylabel('Avg size of clusters')
plt.xlabel('Weight value')

#%% Look at time course of clusters for neurons

#%% Look at scores across the whole population of neurons

f = plt.figure(figsize = (5,5))
plt.hist(all_sil_r[:,0], bins=25, alpha=0.5, color='b', label='right trials')
plt.hist(all_sil_l[:,0], bins=25, alpha=0.5, color='r', label='left trials')
plt.legend()
plt.axvline(np.median(all_sil_r[:,0]), color='b')
plt.axvline(np.mean(all_sil_r[:,0]), color='b', ls='--')
plt.axvline(np.median(all_sil_l[:,0]), color='r')
plt.axvline(np.mean(all_sil_l[:,0]), color='r', ls='--')
# plt.xlabel('Davies-Bouldin score')
plt.xlabel('Silhouette score')
plt.title('Distribution of Silhouette scores for clusters across neurons')

print(diptest.diptest(np.array(all_sil_r)))
print(diptest.diptest(np.array(all_sil_l)))

#%% Look at scores of individual neurons
idx=225
l1.plot_rasterPSTH_sidebyside(l1.good_neurons[idx])
plot_heatmap_across_sess(l1, l1.good_neurons[idx])
rcorr, lcorr = plot_heatmap_across_sess(l1, l1.good_neurons[idx], return_arr=True)
_, idmap_r, sil_r = cluster_corr(rcorr, both=True)
_, idmap_l, sil_l = cluster_corr(lcorr, both=True)
f = plt.figure(figsize = (6,5))
sns.heatmap(cluster_corr(rcorr))
f = plt.figure(figsize = (6,5))
sns.heatmap(cluster_corr(lcorr))
print(sil_r, sil_l)

#%% Compare the scores across other measures - number of clusters vs score
f = plt.figure(figsize = (5,5))
plt.scatter(all_sil_l[:, 0], num_clusters_l, color='r')
plt.scatter(all_sil_r[:, 0], num_clusters_r, color='b')
plt.xlabel('Silhouette scores')
plt.ylabel('Number of clusters')

f = plt.figure(figsize = (5,5))
plt.scatter(all_sil_l[:, 0], av_clus_size_l, color='r')
plt.scatter(all_sil_r[:, 0], av_clus_size_r, color='b')
plt.xlabel('Silhouette scores')
plt.ylabel('Avg cluster size')


f = plt.figure(figsize = (5,5))
plt.scatter(all_sil_l[:, 0], max_clus_size_l, color='r')
plt.scatter(all_sil_r[:, 0], max_clus_size_r, color='b')
plt.xlabel('Silhouette scores')
plt.ylabel('Max cluster size')


f = plt.figure(figsize = (5,5))
plt.scatter(all_sil_l[:, 1], num_clusters_l, color='r')
plt.scatter(all_sil_r[:, 1], num_clusters_r, color='b')
plt.xlabel('Davies-Bouldin scores')
plt.ylabel('Number of clusters')

f = plt.figure(figsize = (5,5))
plt.scatter(all_sil_l[:, 1], av_clus_size_l, color='r')
plt.scatter(all_sil_r[:, 1], av_clus_size_r, color='b')
plt.xlabel('Davies-Bouldin scores')
plt.ylabel('Avg cluster size')


f = plt.figure(figsize = (5,5))
plt.scatter(all_sil_l[:, 1], max_clus_size_l, color='r')
plt.scatter(all_sil_r[:, 1], max_clus_size_r, color='b')
plt.xlabel('Davies-Bouldin scores')
plt.ylabel('Max cluster size')


#%% Compare the two scores with each other
f = plt.figure(figsize = (5,5))

all_sil_r = np.array(all_sil_r)
all_sil_l = np.array(all_sil_l)
plt.scatter(all_sil_r[:, 0], all_sil_r[:, 1],color='b')
plt.scatter(all_sil_l[:, 0], all_sil_l[:, 1],color='r')
plt.xlabel('Silhouette scores')
plt.ylabel('Davies-Bouldin scores')

#%% Now compare silh score with variance and weight

f = plt.figure(figsize = (5,5))

plt.scatter(np.log(allr), all_sil_r[:, 0],color='b')
plt.scatter(np.log(allr), all_sil_l[:, 0],color='r')
plt.ylabel('Goodness of cluster (silhouette score)')
plt.xlabel('Variance of weights (log)')
plt.title('Goodness of trial clustering and variance of CD_choice weights')
print(scipy.stats.pearsonr(np.log(allr), all_sil_r[:, 0]))
print(scipy.stats.pearsonr(np.log(allr), all_sil_l[:, 0]))


f = plt.figure(figsize = (5,5))

plt.scatter(np.abs(avg_weights), all_sil_r[:, 0],color='b')
plt.scatter(np.abs(avg_weights), all_sil_l[:, 0],color='r')
plt.ylabel('Goodness of cluster (silhouette score)')
plt.xlabel('Weight (abs)')
plt.title('Goodness of trial clustering and CD_choice weights')
print(scipy.stats.pearsonr(avg_weights, all_sil_r[:, 0]))
print(scipy.stats.pearsonr(avg_weights, all_sil_l[:, 0]))
#%% Correlate clusters with response to perturbation only sel neurons

# susc = l1.susceptibility()

# Susceptibility
f = plt.figure(figsize = (5,5))
plt.scatter(susc, all_sil_r[:,0], color='b')
plt.scatter(susc, all_sil_l[:, 0], color='r')
plt.xlabel('Susceptibility')
plt.ylabel('Silhouette score')
plt.title('Susceptibility vs silhouette score: all neurons')
print(scipy.stats.pearsonr(susc, all_sil_r[:, 0]))
print(scipy.stats.pearsonr(susc, all_sil_l[:, 0]))

# Robustness
# rob = l1.modularity_proportion_per_neuron()
f = plt.figure(figsize = (5,5))
# plt.scatter(np.log(np.abs(rob)), all_sil_r[:,0], color='b')
# plt.scatter(np.log(np.abs(rob)), all_sil_l[:, 0], color='r')
plt.scatter((rob), all_sil_r[:,0], color='b')
plt.scatter((rob), all_sil_l[:, 0], color='r')
plt.xlabel('Log(Abs(Robustness))')
plt.ylabel('Silhouette score')
plt.title('Robustness vs silhouette score: all neurons')
print(scipy.stats.pearsonr(np.log(np.abs(rob)), all_sil_r[:, 0]))
print(scipy.stats.pearsonr(np.log(np.abs(rob)), all_sil_l[:, 0]))
# Modularity
# rob = l1.modularity_proportion_per_neuron(period=range(l1.delay, l1.delay + int(1/l1.fs)))
f = plt.figure(figsize = (5,5))
plt.scatter(np.log(np.abs(rob)), all_sil_r[:,0], color='b')
plt.scatter(np.log(np.abs(rob)), all_sil_l[:, 0], color='r')
plt.xlabel('Modularity (log(abs))')
plt.ylabel('Silhouette score')
plt.title('Robustness vs silhouette score: all neurons')
print(scipy.stats.pearsonr(np.log(np.abs(rob)), all_sil_r[:, 0]))
print(scipy.stats.pearsonr(np.log(np.abs(rob)), all_sil_l[:, 0]))



#%% Correlate clusters with response to perturbation ALL NEURONS

# susc = l1.susceptibility()

# Susceptibility
f = plt.figure(figsize = (5,5))
plt.scatter(susc, all_sil_r[:,0], color='b')
plt.scatter(susc, all_sil_l[:, 0], color='r')
plt.xlabel('Susceptibility')
plt.ylabel('Silhouette score')
plt.title('Susceptibility vs silhouette score: all neurons')
print(scipy.stats.pearsonr(susc, all_sil_r[:, 0]))
print(scipy.stats.pearsonr(susc, all_sil_l[:, 0]))

# Robustness
# rob = l1.modularity_proportion_per_neuron()
f = plt.figure(figsize = (5,5))
# plt.scatter(np.log(np.abs(rob)), all_sil_r[:,0], color='b')
# plt.scatter(np.log(np.abs(rob)), all_sil_l[:, 0], color='r')
plt.scatter((rob), all_sil_r[:,0], color='b')
plt.scatter((rob), all_sil_l[:, 0], color='r')
plt.xlabel('Log(Abs(Robustness))')
plt.ylabel('Silhouette score')
plt.title('Robustness vs silhouette score: all neurons')
print(scipy.stats.pearsonr(np.log(np.abs(rob)), all_sil_r[:, 0]))
print(scipy.stats.pearsonr(np.log(np.abs(rob)), all_sil_l[:, 0]))
# Modularity
# rob = l1.modularity_proportion_per_neuron(period=range(l1.delay, l1.delay + int(1/l1.fs)))
f = plt.figure(figsize = (5,5))
plt.scatter(np.log(np.abs(rob)), all_sil_r[:,0], color='b')
plt.scatter(np.log(np.abs(rob)), all_sil_l[:, 0], color='r')
plt.xlabel('Modularity (log(abs))')
plt.ylabel('Silhouette score')
plt.title('Robustness vs silhouette score: all neurons')
print(scipy.stats.pearsonr(np.log(np.abs(rob)), all_sil_r[:, 0]))
print(scipy.stats.pearsonr(np.log(np.abs(rob)), all_sil_l[:, 0]))





#%%
# Investigate high norm var neurons
idx_highvar = np.where(np.array(weighted_var) > 2)[0]
for idx in idx_highvar:
    
    l1.plot_rasterPSTH_sidebyside(l1.good_neurons[idx])
    plot_heatmap_across_sess(l1.good_neurons[idx])
#%%

# Compare to low var high weight neurons
idx_lowvar = np.where(np.array(weighted_var) < 4)[0]
idx_highweight = np.where(np.array(avg_weights) < -0.13)[0]
for idx in idx_highweight:
    
    l1.plot_rasterPSTH_sidebyside(l1.good_neurons[idx])
    plot_heatmap_across_sess(l1.good_neurons[idx])
    

    
    
#%% Plot violin plots of weights
f = plt.figure(figsize = (50,5))
order = np.argsort(avg_weights)
ordered_orthonormal_basis_initial = np.take(orthonormal_basis_initial, order, axis=1)
plt.violinplot(ordered_orthonormal_basis_initial, showmeans=True, showextrema=True, showmedians=False)

plt.violinplot(ordered_orthonormal_basis_initial[:, -10:], showmeans=True, showextrema=True, showmedians=False)
plt.axhline(y=0, color='r', ls='--')

#%% Look at individual neurons
sorted_good_n = np.take(l1.good_neurons, order)
l1.plot_rasterPSTH_sidebyside(sorted_good_n[3])


#%% Weights vs selectivity (delay) - ignore, sel method is broken?

sel, _, _ = l1.get_epoch_selectivity(range(l1.delay, l1.response),l1.good_neurons,lickdir=True)
f = plt.figure(figsize = (5,5))
# plt.scatter(np.abs(avg_weights),np.abs(sel))
# plt.ylabel('Selectivity (abs)')
# plt.xlabel('Weights (abs)')
plt.scatter(avg_weights,sel)
plt.ylabel('Selectivity')
plt.xlabel('Weights')

#%% Look at individual "highly selective" neurons
idx_sel = np.where(np.array(sel) >0.3)[0]
l1.plot_rasterPSTH_sidebyside(l1.good_neurons[idx_sel])

#%% 