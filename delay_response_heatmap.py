# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:30:14 2025

For Nuo pres 2/2025 heatmaps of delay and response neurons in naive --> expert

@author: catherinewang
"""


import sys
sys.path.append("C:\scripts\Imaging analysis")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from alm_2p.session import Session
from matplotlib.pyplot import figure

from scipy.stats import chisquare
import pandas as pd
from sklearn.preprocessing import normalize

def z_score_normalize(data):
    return (data - np.mean(data)) / np.std(data)

#%%
naivepath, expertpath, _ = [r'H:\data\BAYLORCW046\python\2024_05_29',
             r'H:\data\BAYLORCW046\python\2024_06_24',
             r'H:\data\BAYLORCW046\python\2024_06_28']

naivepath, expertpath, _ = [r'H:\data\BAYLORCW046\python\2024_05_31',
             r'H:\data\BAYLORCW046\python\2024_06_26',
             r'H:\data\BAYLORCW046\python\2024_06_28']

# naivepath, expertpath, _ = [r'H:\data\BAYLORCW046\python\2024_05_30',
#              r'H:\data\BAYLORCW046\python\2024_06_27',
#              r'H:\data\BAYLORCW046\python\2024_06_28']


# naivepath, expertpath, _ = [r'H:\data\BAYLORCW044\python\2024_05_22',
#              r'H:\data\BAYLORCW044\python\2024_06_19',
#              r'H:\data\BAYLORCW046\python\2024_06_28']


matchpath = r'H:\data\BAYLORCW046\python\cellreg\layer0\0529_0624pairs.mat'

#%% Get delay neurons and sort them
s1 = Session(expertpath, use_reg=True, triple=True)
neurons, tstat = s1.get_epoch_selective(epoch = range(s1.delay-1, s1.response), p=0.01, return_stat=True)
neurons_excl_response, _ = s1.get_epoch_selective(epoch = range(s1.response, s1.time_cutoff), p=0.05, return_stat=True)
include_idx = [np.where(neurons == n)[0][0] for n in neurons if n not in neurons_excl_response]

neurons = np.array(neurons)[include_idx]
tstat = np.array(tstat)[include_idx]

tstatidx = np.where(np.array(tstat) > 0)[0]
neurons = np.array(neurons)[tstatidx]
L_traces, _ = s1.get_trace_matrix_multiple(neurons) 

# add a normalizing step for every neuron
L_traces = np.array([z_score_normalize(L_traces[i]) for i in range(L_traces.shape[0])])
#
L_traces_delay = L_traces[:, ] # range(s1.delay, s1.response)] # only consider delay
max_timestep = [np.argmax(i) for i in L_traces_delay]
neuron_idx = np.argsort(max_timestep)
max_timestep_sorted = np.array(max_timestep)[neuron_idx]
threshold_response = np.where(max_timestep_sorted > s1.response)[0][0] + 2
neuron_idx = neuron_idx[:threshold_response]
#
sorted_neurons = neurons[neuron_idx]
good_idx = [np.where(s1.good_neurons == n)[0][0] for n in sorted_neurons]
#%%
s2 = Session(naivepath, use_reg=True, triple=True)

neurons = s2.good_neurons[good_idx]
L_traces_naive, _ = s2.get_trace_matrix_multiple(neurons)
L_traces_naive = np.array([z_score_normalize(L_traces_naive[i]) for i in range(L_traces_naive.shape[0])])

#%% Save to file
savepath = r'H:\data\BAYLORCW046\python'
scio.savemat(savepath+r'\expert_delay.mat', {'expert_stack': L_traces[neuron_idx, int(2*1/s1.fs):]})
scio.savemat(savepath+r'\naive_delay.mat', {'naive_stack': L_traces_naive[:, int(2*1/s1.fs):]})



#%% plot
plt.imshow(L_traces[neuron_idx, int(2*1/s1.fs):], cmap='viridis', vmin=-0.2, vmax=1.3)
plt.axvline(s1.sample - int(2*1/s1.fs), ls = '--', color='white')
plt.axvline(s1.delay - int(2*1/s1.fs), ls = '--', color='white')
plt.axvline(s1.response - int(2*1/s1.fs), ls = '--', color='white')

plt.show()

plt.imshow(L_traces_naive[:, int(2*1/s1.fs):], cmap='viridis', vmin=-0.2, vmax=1.3)
plt.axvline(s1.sample - int(2*1/s1.fs), ls = '--', color='white')
plt.axvline(s1.delay - int(2*1/s1.fs), ls = '--', color='white')
plt.axvline(s1.response - int(2*1/s1.fs), ls = '--', color='white')

plt.show()
# if save: L_traces[neuron_idx, int(2*1/s1.fs):]

#%% Get reseponse neurons and sort them
s1 = Session(expertpath, use_reg=True, triple=True)
neurons, tstat = s1.get_epoch_selective(epoch = range(s1.response, s1.time_cutoff), p=0.01, return_stat=True)
neurons_excl_sample, _ = s1.get_epoch_selective(epoch = range(s1.sample, s1.delay), p=0.05, return_stat=True)
neurons_excl_delay, _ = s1.get_epoch_selective(epoch = range(s1.delay, s1.response), p=0.05, return_stat=True)
include_idx = [np.where(neurons == n)[0][0] for n in neurons if n not in neurons_excl_sample and n not in neurons_excl_delay]

neurons = np.array(neurons)[include_idx]
tstat = np.array(tstat)[include_idx]
# neurons = [n for n in neurons if n not in neurons_excl_sample and n not in neurons_excl_delay]

tstatidx = np.where(np.array(tstat) < 0)[0]
neurons = np.array(neurons)[tstatidx]
L_traces, _ = s1.get_trace_matrix_multiple(neurons) 

# add a normalizing step for every neuron
L_traces = np.array([z_score_normalize(L_traces[i]) for i in range(L_traces.shape[0])])
#
L_traces_delay = L_traces[:, range(s1.response, s1.time_cutoff)] # only consider delay
# max_timestep = [np.argmax(i) for i in L_traces_delay]
# neuron_idx = np.argsort(max_timestep)
neuron_idx = range(len(L_traces_delay)) # no sorting

#
sorted_neurons = neurons[neuron_idx]
good_idx = [np.where(s1.good_neurons == n)[0][0] for n in sorted_neurons]
s2 = Session(naivepath, use_reg=True, triple=True)

neurons = s2.good_neurons[good_idx]
L_traces_naive, _ = s2.get_trace_matrix_multiple(neurons)
L_traces_naive = np.array([z_score_normalize(L_traces_naive[i]) for i in range(L_traces_naive.shape[0])])

#%% Save to file
savepath = r'H:\data\BAYLORCW046\python'
scio.savemat(savepath+r'\expert_response.mat', {'expert_response_stack': L_traces[neuron_idx, int(2*1/s1.fs):]})
scio.savemat(savepath+r'\naive_response.mat', {'naive_response_stack': L_traces_naive[neuron_idx, int(2*1/s1.fs):]})

#%% Normalize both delay and response heatmaps to each other


