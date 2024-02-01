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
from session import Session
from matplotlib.pyplot import figure
from numpy import concatenate as cat
from sklearn.preprocessing import normalize
import quality
from scipy import stats
from scipy.stats import zscore

plt.rcParams['pdf.fonttype'] = '42' 
#%%

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

paths = [
            # r'F:\data\BAYLORCW032\python\2023_10_24',
            # r'F:\data\BAYLORCW034\python\2023_10_27',
            # r'F:\data\BAYLORCW036\python\2023_10_30',
            r'F:\data\BAYLORCW035\python\2023_12_15',
            # r'F:\data\BAYLORCW037\python\2023_12_15',
            ]
# allstack, allstimstack = np.zeros(61), np.zeros(61)
# allstack, allcontrastimstack = np.zeros(61), np.zeros(61)
allstack, allcontrastimstack = np.zeros(26), np.zeros(26)

for path in paths:
    
    l1 = quality.QC(path, use_background_sub=False)

    stack, stimstack = l1.all_neurons_heatmap(return_traces=True)

    # allstack = np.vstack((allstack, stack))
    # allcontrastimstack = np.vstack((allcontrastimstack, stimstack))
    
    allstack = np.vstack((allstack, normalize(stack[:, 12:38])))
    allcontrastimstack = np.vstack((allcontrastimstack, normalize(stimstack[:, 12:38])))
    
    # allstack = np.vstack((allstack, zscore(stack ,axis=0)))
    # allcontrastimstack = np.vstack((allcontrastimstack, zscore(stimstack, axis=0)))
    
# allstack = allstack[1:,12:38]
# allcontrastimstack = allcontrastimstack[1:,12:38]
allstack = allstack[1:]
allcontrastimstack = allcontrastimstack[1:]

# allstack = normalize(allstack[1:,12:38])
# allstimstack = normalize(allstimstack[1:,12:38])
# allcontrastimstack = normalize(allcontrastimstack[1:,12:38])




#%% Plot all opto trials as heatmap 

x = np.arange(-7.97,4,l1.fs)[12:38]
# x = np.arange(-6.97,4,self.fs)[:self.time_cutoff]
# allstack = normalize(allstack)
# allstimstack = normalize(allstimstack)
plt.figure(figsize=(8,5))
plt.matshow(allcontrastimstack, cmap='gray', fignum=1, aspect='auto')
plt.xticks(range(0,38-12, 6), [int(d) for d in x[::6]])
plt.axvline(x=l1.delay-10, c='b', linewidth = 0.5)
# plt.savefig(r'F:\data\Fig 3\contra_opto_effect_backgroundsubtract.pdf')

#%%
f, axarr = plt.subplots(2,2)#, sharex='col')

axarr[0,0].matshow(allstimstack, cmap='gray', interpolation='nearest', aspect='auto')
# axarr[0,0].axis('off')
axarr[0,0].set_xticks(range(0,38-12, 6), [int(d) for d in x[::6]])
# 
axarr[0,0].set_title('Opto')
axarr[0,0].axvline(x=l1.delay-10, c='b', linewidth = 0.5)
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
# axarr[0,1].axis('off')
axarr[0,1].set_xticks(range(0,38-12, 6), [int(d) for d in x[::6]])

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

# plt.savefig(r'F:\data\Fig 3\ipsi_opto_effect.pdf')
plt.show()

#%% Overlay plots
x = np.arange(-6.97,4,l1.fs)[12:38]

plt.plot(x, np.mean(allcontrastimstack, axis = 0), color = 'red', label='Optogenetic stimulation trials')
plt.fill_between(x, np.mean(allcontrastimstack, axis = 0) - stats.sem(allcontrastimstack, axis=0), 
          np.mean(allcontrastimstack, axis = 0) + stats.sem(allcontrastimstack, axis=0),
          color='lightcoral')   

plt.plot(x, np.mean(allstack, axis = 0), color = 'grey', label='Control trials')
plt.axvline(x=-3, c='b', linewidth = 0.5)
plt.fill_between(x, np.mean(allstack, axis = 0) - stats.sem(allstack, axis=0), 
          np.mean(allstack, axis = 0) + stats.sem(allstack, axis=0),
          color='silver')   
# axarr[1,1].set_ylim(top=0.2)
plt.ylabel('dF/F0')
# axarr[1,1].set_xticks(range(0,allstack.shape[1], 10), [int(d) for d in x[::10]])
plt.xlabel('Time from Go cue (s)')
plt.legend()
# plt.savefig(r'F:\data\Fig 3\contra_opto_effect_overlay_subtractbackground.pdf')

plt.show()

#%% Count number of neurons for summarizing:

agg_mice_paths = [
    
        [r'F:\data\BAYLORCW032\python\2023_10_08',
          r'F:\data\BAYLORCW032\python\2023_10_16',
          r'F:\data\BAYLORCW032\python\2023_10_25',],

        [ r'F:\data\BAYLORCW034\python\2023_10_12',
              r'F:\data\BAYLORCW034\python\2023_10_22',
              r'F:\data\BAYLORCW034\python\2023_10_27'],
         
        [r'F:\data\BAYLORCW036\python\2023_10_09',
            r'F:\data\BAYLORCW036\python\2023_10_19',
            r'F:\data\BAYLORCW036\python\2023_10_30'],
    
        [r'F:\data\BAYLORCW035\python\2023_10_26',
            r'F:\data\BAYLORCW035\python\2023_12_07',
            r'F:\data\BAYLORCW035\python\2023_12_15',],
     
        [r'F:\data\BAYLORCW037\python\2023_11_21',
             r'F:\data\BAYLORCW037\python\2023_12_08',
             r'F:\data\BAYLORCW037\python\2023_12_15',]

        ]

for paths in agg_mice_paths:
    for path in paths:
        print(path)
        
        # all neurons
        l1 = Session(path)
        print(len(l1.good_neurons))

        # all matched 
        l1 = Session(path, use_reg=True, triple=True, filter_reg = False)
        print(len(l1.good_neurons))

        # all matched filtered
        l1 = Session(path, use_reg=True, triple=True)
        print(len(l1.good_neurons))
        
#%% Compare the background extracted fluorescence


paths = [r'F:\data\BAYLORCW032\python\2023_10_24',
         r'F:\data\BAYLORCW032\python\2023_10_24to',]
             # r'F:\data\BAYLORCW034\python\2023_10_27',
            # r'F:\data\BAYLORCW036\python\2023_10_30',
            #   r'F:\data\BAYLORCW035\python\2023_12_15',
            # r'F:\data\BAYLORCW037\python\2023_12_15',]
# allstack, allstimstack = np.zeros(61), np.zeros(61)

for path in paths:
    allstack, allcontrastimstack = np.zeros(61), np.zeros(61)

    
    l1 = quality.QC(path)

    stack, stimstack = l1.all_neurons_heatmap(return_traces=True)

    allstack = np.vstack((allstack, stack))
    # allstimstack = np.vstack((allstimstack, stimstack))
    allcontrastimstack = np.vstack((allcontrastimstack, stimstack))
    
    allstack = allstack[1:,12:38]
    # allstimstack = allstimstack[1:,12:38]
    allcontrastimstack = allcontrastimstack[1:,12:38]
    
    x = np.arange(-6.97,4,l1.fs)[12:38]
    
    plt.plot(x, np.mean(allcontrastimstack, axis = 0), color = 'red', label='Optogenetic stimulation trials')
    plt.fill_between(x, np.mean(allcontrastimstack, axis = 0) - stats.sem(allcontrastimstack, axis=0), 
              np.mean(allcontrastimstack, axis = 0) + stats.sem(allcontrastimstack, axis=0),
              color='lightcoral')   
    plt.plot(x, np.mean(allstack, axis = 0), color = 'grey', label='Control trials')
    plt.axvline(x=-3, c='b', linewidth = 0.5)
    
    plt.fill_between(x, np.mean(allstack, axis = 0) - stats.sem(allstack, axis=0), 
              np.mean(allstack, axis = 0) + stats.sem(allstack, axis=0),
              color='silver')   
    # axarr[1,1].set_ylim(top=0.2)
    plt.ylabel('dF/F0')
    # axarr[1,1].set_xticks(range(0,allstack.shape[1], 10), [int(d) for d in x[::10]])
    plt.xlabel('Time from Go cue (s)')
    plt.legend()
    # plt.savefig(r'F:\data\Fig 3\contra_opto_effect_overlay.pdf')
    
    plt.show()
    
#%% Plot the background fluorescence 
path = r'F:\data\BAYLORCW032\python\2023_10_24to'

l1 = quality.QC(path)
l1.plot_background()

f, axarr = plt.subplots(5,1, sharex='col', figsize=(10, 10))
# Plot with neuropil and dFF trace
for layer in range(5):
    path = r'F:\data\BAYLORCW032\python\2023_10_24to'
    l1 = quality.QC(path, layer_num=layer+1)
    background, npil, _ = l1.plot_background_and_traces(single_layer = True, return_traces=True)
    
    path = r'F:\data\BAYLORCW032\python\2023_10_24'
    l1 = quality.QC(path, layer_num=layer+1)
    _, _, f = l1.plot_background_and_traces(single_layer = True, return_traces=True, only_f=True)
    
    axarr[layer].plot(background, label = 'F_background')
    axarr[layer].plot(npil, label = 'F_npil')
    axarr[layer].plot(f, label = 'F')
    
    axarr[layer].axvline(l1.sample-12, ls = '--', color='grey')
    axarr[layer].axvline(l1.delay-12, ls = '--', color='red')
    axarr[layer].axvline(l1.delay-12+6, ls = '--', color='red')
    
    axarr[layer].set_title("Layer {}".format(layer+1))
plt.suptitle("Traces on opto stim trials")
plt.legend()
plt.show()
    

#%% Plot background trace on a trial by trial basis per layer 
path = r'F:\data\BAYLORCW032\python\2023_10_24'
path = r'F:\data\BAYLORCW035\python\2023_12_15'

l1 = Session(path, use_background_sub=True)

# F background
for layer in range(5):
    
    window = range(12, 42)
    # window = range(l1.time_cutoff)
    x = np.arange(-6.97,4,l1.fs)[window]

    stim_trials = np.where(l1.stim_ON)[0]
    f, ax = plt.subplots(len(stim_trials), figsize=(7,14))
    
    
    for i in range(len(stim_trials)):
        ax[i].plot(x, l1.background[0,stim_trials[i]][layer, window])
        ax[i].axis('off')
        ax[i].axvline(x=-3, c='red', ls = '--', linewidth = 0.5)
        ax[i].axvline(x=-2, c='red', ls = '--', linewidth = 0.5)
        # ax[i].axvline(x=-1, c='red', ls = '--', linewidth = 0.5)
        # ax[i].axvline(x=-0, c='red', ls = '--', linewidth = 0.5)
    ax[0].set_title("Layer {} background".format(layer+1))
    plt.show()

    
#%% Plot background trace on a trial by trial basis averaged over layers
path = r'F:\data\BAYLORCW032\python\2023_10_24'
path = r'F:\data\BAYLORCW035\python\2023_12_15'
l1 = Session(path, use_background_sub=True)

window = range(12, 42)
# window = range(l1.time_cutoff)
x = np.arange(-6.97,4,l1.fs)[window]

stim_trials = np.where(l1.stim_ON)[0]
f, ax = plt.subplots(len(stim_trials), figsize=(7,15))


for i in range(len(stim_trials)):
    ax[i].plot(x, np.mean([l1.background[0,stim_trials[i]][layer, window] for layer in range(5)], axis=0))
    ax[i].axis('off')
    ax[i].axvline(x=-3, c='red', ls = '--', linewidth = 0.5)
    ax[i].axvline(x=-2, c='red', ls = '--', linewidth = 0.5)
    # ax[i].axvline(x=-1, c='red', ls = '--', linewidth = 0.5)
    # ax[i].axvline(x=-0, c='red', ls = '--', linewidth = 0.5)
ax[0].set_title("F background (av.)")
plt.show()

# %%
path = r'F:\\data\\BAYLORCW035\\python\\2023_12_15'
# # path = r'F:\data\BAYLORCW032\python\2023_10_24'
l1 = Session(path, use_background_sub=True)

window = range(23, 32)
# window = range(l1.time_cutoff)
x = np.arange(-6.97,4,l1.fs)[window]
neuron = 11


control_trials = np.where(~l1.stim_ON)[0]

for i in range(len(control_trials)):
    plt.plot(x, l1.dff[0,control_trials[i]][neuron, window], color='grey', linewidth = 0.5, alpha=0.5)
    # plt.plot(x, l1.background[0,control_trials[i]][0, window], color='grey', linewidth = 0.5, alpha=0.5)

stim_trials = np.where(l1.stim_ON)[0]

for i in range(len(stim_trials)):
    plt.plot(x, l1.dff[0,stim_trials[i]][neuron, window], color='red', linewidth = 0.5, alpha=0.5)
    # plt.plot(x, l1.background[0,stim_trials[i]][0, window], color='red', linewidth = 0.5, alpha=0.5)
    
    
plt.plot(x, np.mean([l1.dff[0,control_trials[i]][neuron, window] for i in range(len(control_trials))], axis=0), color='black')
  
plt.plot(x, np.mean([l1.dff[0,stim_trials[i]][neuron, window] for i in range(len(stim_trials))], axis=0), color='red')


# plt.plot(x, np.mean([l1.background[0,control_trials[i]][0, window] for i in range(len(control_trials))], axis=0), color='black')
  
# plt.plot(x, np.mean([l1.background[0,stim_trials[i]][0, window] for i in range(len(stim_trials))], axis=0), color='red')


plt.axvline(x=-3, c='red', ls = '--', linewidth = 0.5)
plt.axvline(x=-2, c='red', ls = '--', linewidth = 0.5)
# plt.ylim((-1, 1))
plt.ylim((-0.02, 0.02))
# ax[i].axvline(x=-1, c='red', ls = '--', linewidth = 0.5)
# ax[i].axvline(x=-0, c='red', ls = '--', linewidth = 0.5)
    
plt.title("F background (av.)")
plt.show()
#%% Plot overlay traces of f-background:
    
path = r'F:\data\BAYLORCW032\python\2023_10_24'
paths = [
            # r'F:\data\BAYLORCW032\python\2023_10_24',
            # r'F:\data\BAYLORCW034\python\2023_10_27',
            # r'F:\data\BAYLORCW036\python\2023_10_30',
            r'F:\data\BAYLORCW035\python\2023_12_15',
            # r'F:\data\BAYLORCW037\python\2023_12_15',
            ]
allcontrastimstack, allstack = np.zeros(26), np.zeros(26)
window = range(12, 38)

for path in paths:
    l1 = Session(path, use_background_sub=True)
    
    control_trials = np.where(~l1.stim_ON)[0]
    
    for i in range(len(control_trials)):
        stack = l1.background[0,control_trials[i]][:, window]
        allstack = np.vstack((allstack, stack))
    
    
    stim_trials = np.where(l1.stim_ON)[0]
    
    for i in range(len(stim_trials)):
        stack = l1.background[0,stim_trials[i]][:, window]
        allcontrastimstack = np.vstack((allcontrastimstack, stack))

    
allstack = normalize(allstack[1:])
allcontrastimstack = normalize(allcontrastimstack[1:])

x = np.arange(-6.97,4,l1.fs)[12:38]

plt.plot(x, np.mean(allcontrastimstack, axis = 0), color = 'red', label='Optogenetic stimulation trials')
plt.fill_between(x, np.mean(allcontrastimstack, axis = 0) - stats.sem(allcontrastimstack, axis=0), 
          np.mean(allcontrastimstack, axis = 0) + stats.sem(allcontrastimstack, axis=0),
          color='lightcoral')   

plt.plot(x, np.mean(allstack, axis = 0), color = 'grey', label='Control trials')
plt.axvline(x=-3, c='b', linewidth = 0.5)
plt.fill_between(x, np.mean(allstack, axis = 0) - stats.sem(allstack, axis=0), 
          np.mean(allstack, axis = 0) + stats.sem(allstack, axis=0),
          color='silver')   
# axarr[1,1].set_ylim(top=0.2)
plt.ylabel('dF/F0')
# axarr[1,1].set_xticks(range(0,allstack.shape[1], 10), [int(d) for d in x[::10]])
plt.xlabel('Time from Go cue (s)')
plt.legend()
# plt.savefig(r'F:\data\Fig 3\background_effect_overlay.pdf')

plt.show()

