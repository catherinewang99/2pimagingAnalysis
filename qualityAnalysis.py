# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 12:31:52 2023

@author: Catherine Wang
"""

import sys
sys.path.append("C:\scripts\Imaging analysis")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import session
from matplotlib.pyplot import figure
from numpy import concatenate as cat
from sklearn.preprocessing import normalize
import quality


path = r'F:\data\BAYLORCW021\python\2023_05_03'

# path = r'F:\data\BAYLORCW027\python\2023_05_25'
# path = r'F:\data\BAYLORCW021\python\2023_04_27'

# # path = r'F:\data\BAYLORCW022\python\2023_03_06'
path = r'F:\data\BAYLORCW034\python\2023_10_24'
# path = r'F:\data\BAYLORCW021\python\2023_02_15'


l1 = quality.QC(path)
# l1.plot_pearsons_correlation()
# var = l1.plot_variance_spread()
### TOTAL NUMBER OF NEURONS: ###

# total_n = 0
# for i in range(6):
#     l1 = session.Session(path, i+1)
#     # l1.crop_trials(245, end = 330)
#     total_n += l1.num_neurons

# print(total_n)


### Total number of selective neurons per category ###

# path = r'F:\data\BAYLORCW021\python\2023_04_27'
# trained = session.Session(path)
# path = r'F:\data\BAYLORCW021\python\2023_02_08'
# naive = session.Session(path)

# trained_num = []
# naive_num = []

# epochs = [range(naive.time_cutoff), range(8,14), range(19,28), range(29,naive.time_cutoff)]
# titles = ['Whole-trial', 'Sample', 'Delay', 'Response']
        
# for e in epochs:
#     trained_num += [len(trained.get_epoch_selective(e)) / trained.num_neurons]        
#     naive_num += [len(naive.get_epoch_selective(e)) / naive.num_neurons]        

# plt.plot(titles, trained_num, label='Trained', marker = 'o')
# plt.plot(titles, naive_num, label = 'Naive', marker = 'o')
# plt.ylabel('Proportion of selective ROIs')
# plt.legend()

### EFFECT OF OPTO INHIBITION ###


# l1.crop_trials(108,end=111)


### Histogram of average dff during stim period

# delay_stim = cat(cat([stim_dff[f][:, 13:28] for f in range(stim_dff.shape[0])]))
# delay_nostim = cat(cat([non_stim_dff[f][:, 13:28] for f in range(non_stim_dff.shape[0])]))

# plt.hist(delay_stim, alpha = 0.7, bins = 5000, label = 'Stim')
# plt.hist(delay_nostim, alpha = 0.7, bins = 5000, label = 'Control')
# plt.xlim(-5, 5)
# plt.legend()

### Average dff during stim period
# delay_stim = cat([stim_dff[f][:, 13:28] for f in range(stim_dff.shape[0])])
# delay_nostim = cat([non_stim_dff[f][:, 13:28] for f in range(non_stim_dff.shape[0])])
# x=range(15)
# stim = np.mean(delay_stim, axis = 0)
# nostim = np.mean(delay_nostim, axis = 0)

# stim_err = np.std(delay_stim, axis=0) / np.sqrt(len(delay_stim)) 
# nostim_err = np.std(delay_nostim, axis=0) / np.sqrt(len(delay_nostim))

# plt.plot(stim, 'r-')
# plt.plot(nostim, 'b-')

            
# plt.fill_between(x, stim - stim_err, 
#           stim + stim_err,
#           color=['#ffaeb1'])
# plt.fill_between(x, nostim - nostim_err, 
#           nostim + nostim_err,
#           color=['#b4b2dc'])

### Heat map of neurons during stim vs. control

l1.all_neurons_heatmap()
# l1.all_neurons_heatmap_stimlevels()
# control_neuron_dff, ratio = l1.stim_activity_proportion()
## Histogram of F values before finding F0

# n0 = [l1.dff[0,t][0, :] for t in range(l1.num_trials)]
# plt.hist(cat(n0), bins = 'auto')

# plt.axvline(x=np.quantile(cat(n0), q=0.10), color='r')

# f0 = [np.mean(l1.dff[0,t][0, :7]) for t in range(l1.num_trials)]
# f01 = [np.quantile(l1.dff[0,t][0, :], q=0.10) for t in range(l1.num_trials)]

# for f in f0:
#     # print(f)
#     plt.axvline(x=f, color='g', linewidth=0.2)
    
# plt.show()

# plt.hist(cat(n0), bins = 'auto')

# for i in range(len(f0)):
    
#     plt.axvline(x=f0[i], color='g', linewidth=0.2)
#     plt.axvline(x=f01[i], color='r', linewidth=0.2)


# plt.show()
# plt.hist(f0, alpha = 0.5, color = 'r')
# plt.hist(f01, alpha = 0.5, color='g')
















