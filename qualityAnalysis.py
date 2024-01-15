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
from scipy import stats
plt.rcParams['pdf.fonttype'] = '42' 


path = r'F:\data\BAYLORCW021\python\2023_05_03'

# path = r'F:\data\BAYLORCW027\python\2023_05_25'
# path = r'F:\data\BAYLORCW021\python\2023_04_27'

# # path = r'F:\data\BAYLORCW022\python\2023_03_06'
path = r'F:\data\BAYLORCW037\python\2023_11_22'
# path = r'F:\data\BAYLORCW021\python\2023_02_15'
# path = r'F:\data\BAYLORCW032\python\2023_10_08'

path = r'F:\data\BAYLORCW034\python\2023_10_24'

path = 'F:\\data\\BAYLORCW037\\python\\2023_11_21'
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


#%% Heatmap AGG over 5 mice

paths = [r'F:\data\BAYLORCW032\python\2023_10_23',
         r'F:\data\BAYLORCW036\python\2023_10_20',
         r'F:\data\BAYLORCW034\python\2023_10_24',
         r'F:\data\BAYLORCW035\python\2023_12_06',
         r'F:\data\BAYLORCW037\python\2023_11_22'
         ]

allstack, allstimstack = np.zeros(61), np.zeros(61)
for path in paths:
    
    l1 = quality.QC(path)

    stack, stimstack = l1.all_neurons_heatmap(return_traces=True)

    allstack = np.vstack((allstack, stack))
    allstimstack = np.vstack((allstimstack, stimstack))
    
allstack = allstack[1:,12:38]
allstimstack = allstimstack[1:,12:38]
# allstack = normalize(allstack[1:,12:38])
# allstimstack = normalize(allstimstack[1:,12:38])

f, axarr = plt.subplots(2,2)#, sharex='col')
x = np.arange(-4.97,4,l1.fs)[:26]
# x = np.arange(-6.97,4,self.fs)[:self.time_cutoff]

 
axarr[0,0].matshow(allstimstack, cmap='gray', interpolation='nearest', aspect='auto')
axarr[0,0].axis('off')
axarr[0,0].set_title('Opto')
axarr[0,0].axvline(x=l1.delay-12, c='b', linewidth = 0.5)
# axarr[0,0].axvline(x=-3, c='b', linewidth = 0.5)

axarr[1,0].plot(x, np.mean(allstimstack, axis = 0))
axarr[1,0].fill_between(x, np.mean(allstimstack, axis = 0) - stats.sem(allstimstack, axis=0), 
          np.mean(allstimstack, axis = 0) + stats.sem(allstimstack, axis=0),
          color='lightblue')   
# axarr[1,0].set_ylim(top=self.fs)
# axarr[1,0].axvline(x=l1.delay-6, c='b', linewidth = 0.5)
axarr[1,0].axvline(x=-3, c='b', linewidth = 0.5)
# axarr[1,0].set_xticks(range(0,allstack.shape[1], 10), [int(d) for d in x[::10]])

axarr[0,1].matshow(allstack, cmap='gray', interpolation='nearest', aspect='auto')
axarr[0,1].axis('off')
axarr[0,1].set_title('Control')

axarr[1,1].plot(x, np.mean(allstack, axis = 0))

axarr[1,1].fill_between(x, np.mean(allstack, axis = 0) - stats.sem(allstack, axis=0), 
          np.mean(allstack, axis = 0) + stats.sem(allstack, axis=0),
          color='lightblue')   
# axarr[1,1].set_ylim(top=0.2)
axarr[1,0].set_ylabel('dF/F0')
# axarr[1,1].set_xticks(range(0,allstack.shape[1], 10), [int(d) for d in x[::10]])
axarr[1,0].set_xlabel('Time from Go cue (s)')

plt.suptitle('n=3431 neurons')

# plt.savefig(r'F:\data\Fig 3\opto_effect.pdf')
plt.show()

#%% Overlay plots

plt.plot(x, np.mean(allstimstack, axis = 0), color = 'red', label='Optogenetic stimulation trials')
plt.fill_between(x, np.mean(allstimstack, axis = 0) - stats.sem(allstimstack, axis=0), 
          np.mean(allstimstack, axis = 0) + stats.sem(allstimstack, axis=0),
          color='lightcoral')   
plt.plot(x, np.mean(allstack, axis = 0) + 0.02, color = 'grey', label='Control trials')
plt.axvline(x=-3, c='b', linewidth = 0.5)

plt.fill_between(x, np.mean(allstack, axis = 0) + 0.02 - stats.sem(allstack, axis=0), 
          np.mean(allstack, axis = 0) + 0.02 + stats.sem(allstack, axis=0),
          color='silver')   
# axarr[1,1].set_ylim(top=0.2)
plt.ylabel('dF/F0')
# axarr[1,1].set_xticks(range(0,allstack.shape[1], 10), [int(d) for d in x[::10]])
plt.xlabel('Time from Go cue (s)')
plt.legend()
plt.savefig(r'F:\data\Fig 3\opto_effect_overlay.pdf')

plt.show()


