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
def plot_heatmap_across_sess(neuron):
    r, l = l1.get_trace_matrix(neuron)
    r, l = np.array(r), np.array(l)
    
    f = plt.figure(figsize = (5,5))
    df = pd.DataFrame(r[:,range(l1.delay, l1.response)].T)  
    corrs = df.corr()
    plt.imshow(corrs)
    plt.xlabel('R trials')
    plt.title('Correlation of delay activity in R trials')
    plt.colorbar()    
    
    f = plt.figure(figsize = (5,5))
    df = pd.DataFrame(l[:,range(l1.delay, l1.response)].T)  
    corrs = df.corr()
    plt.imshow(corrs)
    plt.xlabel('L trials')
    plt.title('Correlation of delay activity in L trials')
    plt.colorbar()   

import scipy
import scipy.cluster.hierarchy as sch

def cluster_corr(corr_array, inplace=False):
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
n=25
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