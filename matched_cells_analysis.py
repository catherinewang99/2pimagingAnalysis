#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 16:42:58 2023

@author: catherinewang
"""

import sys
sys.path.append("C:\scripts\Imaging analysis")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import session
from matplotlib.pyplot import figure
import decon
from scipy.stats import chisquare
import pandas as pd
from sklearn.preprocessing import normalize



 #%% ### Single neuron PSTHs over sessions ###
 
for l_num in range(1,6):
    path = r'F:\data\BAYLORCW032\python\2023_10_05'
    s1 = session.Session(path, layer_num = l_num)
    path1 = r'F:\data\BAYLORCW032\python\2023_10_24'
    s2 = session.Session(path1, layer_num = l_num)
    match_n_path = r'F:\data\BAYLORCW032\python\cellreg\layer{}\1005_1024translationspairs_proc.npy'
    epoch = range(s1.delay, s1.response)
    matched_neurons=np.load(match_n_path.format(l_num-1))
    
    for n in s2.get_epoch_selective(range(s2.delay, s2.response)):
        idx = np.where(matched_neurons[:, 1] == n)[0]
        if len(idx) == 0:
            print('Neuron {} not matched'.format(n))
            continue
        
        n1 = matched_neurons[idx[0], 0]   
        
        savepath = r'F:\data\SFN 2023\matched_neuron_{}_{}.pdf'
        s1.plot_rasterPSTH_sidebyside(n1,save=savepath.format(n1, 'naive'))
        # n2 = matched_neurons[i, 1]   
        s2.plot_rasterPSTH_sidebyside(n,save=savepath.format(n1, 'trained'))



 #%% ### Heatmap analysis over sessions ###


 #%% Set paths
# path = r'F:\data\BAYLORCW032\python\2023_10_05'
# s1 = session.Session(path, layer_num =1)
# path1 = r'F:\data\BAYLORCW032\python\2023_10_24'
# s2 = session.Session(path1, layer_num =1)
# match_n_path = r'F:\data\BAYLORCW032\python\cellreg\layer{}\1005_1024translationspairs_proc.npy'
# epoch = range(s1.delay, s1.response)


# path = r'F:\data\BAYLORCW034\python\2023_10_10'
# s1 = session.Session(path, layer_num =1)
# path1 = r'F:\data\BAYLORCW034\python\2023_10_24'
# s2 = session.Session(path1, layer_num =1)
# match_n_path = r'F:\data\BAYLORCW034\python\cellreg\layer{}\1010_1024translationspairs_proc.npy'
# epoch = range(s1.delay, s1.response)


path = r'F:\data\BAYLORCW036\python\2023_10_09'
s1 = session.Session(path, layer_num =1, triple=True)
path1 = r'F:\data\BAYLORCW036\python\2023_10_30'
s2 = session.Session(path1, layer_num =1, triple=True)
match_n_path = r'F:\data\BAYLORCW036\python\cellreg\layer{}\1009_1019_1030pairs_proc.npy'
epoch = range(s1.delay, s1.response)
triple=True
#%% DEBUGGING SECTION
path1 = r'F:\data\BAYLORCW032\python\2023_10_24'
s2 = session.Session(path1, layer_num =1)
epoch = range(s2.delay+9, s2.response)
neurons_ranked, selectivity = s2.ranked_cells_by_selectivity(p=0.05)
counter =0

right_stack_post = np.zeros(s2.time_cutoff) 
left_stack_post = np.zeros(s2.time_cutoff) 

for nnum in range(len(neurons_ranked)):
    # s2.plot_rasterPSTH_sidebyside(i)
    # counter +=1
    
    neuron = neurons_ranked[nnum]
    r,l = s2.get_trace_matrix(neuron)
    
    right_trace = np.mean(r, axis=0) #/ np.mean(np.mean(r, axis=0)[epoch])
    left_trace = np.mean(l, axis=0) #/ np.mean(np.mean(l, axis=0)[epoch])
    
    right_stack_post = np.vstack((right_stack_post, right_trace))
    left_stack_post = np.vstack((left_stack_post, left_trace))

f, axarr = plt.subplots(2, sharex='col', figsize=(15,10))
vmin, vmax= -0,0.5

right_stack_post = normalize(right_stack_post)
# right_stack_post = (right_stack_post[1:])
right_im = axarr[0].imshow(right_stack_post, cmap='viridis', interpolation='nearest', aspect='auto', vmin=vmin, vmax=vmax)
axarr[0].axis('off')
f.colorbar(right_im, shrink = 0.2)
axarr[0].set_title("Session 2")
# Left trials
left_stack_post = normalize(left_stack_post)
# left_stack_post = (left_stack_post[1:])
leftim = axarr[1].imshow(left_stack_post, cmap='viridis', interpolation='nearest', aspect='auto',vmin=vmin, vmax=vmax)
axarr[1].axis('off')
f.colorbar(leftim, shrink = 0.2)

#%%
# NAIVE --> TRAINED
right_stack = np.zeros(s1.time_cutoff) 
left_stack = np.zeros(s1.time_cutoff) 

right_stack_post = np.zeros(s2.time_cutoff) 
left_stack_post = np.zeros(s2.time_cutoff) 
overallsel = []

for lnum in range(1,6):
    s1 = session.Session(path, layer_num=lnum, use_reg=True, triple=triple)
    s2 = session.Session(path1, layer_num=lnum, use_reg=True, triple=triple)
    
    
    neurons_ranked, selectivity, trials = s1.ranked_cells_by_selectivity(p=0.05)
    rt,lt = trials
    matched_neurons=np.load(match_n_path.format(lnum-1))
    if triple:
        matched_neurons = matched_neurons[:,[0,2]]
    
    overallsel = np.append(overallsel, selectivity)

    for nnum in range(len(neurons_ranked)):
        
        if neurons_ranked[nnum] not in matched_neurons[:,0]:
            print(neurons_ranked[nnum], 'not matched.')
            continue
        
        ## FIRST SESS
        neuron = neurons_ranked[nnum]
        r,l = s1.get_trace_matrix(neuron, rtrials = rt, ltrials=lt)
        
        right_trace = np.mean(r, axis=0) #/ np.mean(np.mean(r, axis=0)[epoch])
        left_trace = np.mean(l, axis=0) #/ np.mean(np.mean(l, axis=0)[epoch])
        
        right_stack = np.vstack((right_stack, right_trace))
        left_stack = np.vstack((left_stack, left_trace))
        
        ## SECOND SESS
        nind = np.where(matched_neurons[:,0] == neuron)[0]
        neuron = matched_neurons[nind, 1] # Grab the ranked neuron
        r,l = s2.get_trace_matrix(neuron)
        
        right_trace = np.mean(r, axis=0) #/ np.mean(np.mean(r, axis=0)[0,epoch])
        left_trace = np.mean(l, axis=0) #/ np.mean(np.mean(l, axis=0)[0,epoch])
        
        right_stack_post = np.vstack((right_stack_post, right_trace))
        left_stack_post = np.vstack((left_stack_post, left_trace))
        
right_stack = np.take(right_stack, np.argsort(overallsel), axis = 0)
left_stack = np.take(left_stack, np.argsort(overallsel), axis = 0)

right_stack_post = np.take(right_stack_post, np.argsort(overallsel), axis = 0)
left_stack_post = np.take(left_stack_post, np.argsort(overallsel), axis = 0)
#%%
# TRAINED --> NAIVE
right_stack = np.zeros(s1.time_cutoff) 
left_stack = np.zeros(s1.time_cutoff) 

right_stack_post = np.zeros(s2.time_cutoff) 
left_stack_post = np.zeros(s2.time_cutoff) 

overallsel = []
for lnum in range(1,6):
    s1 = session.Session(path, layer_num=lnum, use_reg=True,  triple=triple)
    s2 = session.Session(path1, layer_num=lnum, use_reg=True,  triple=triple)
    
    neurons_ranked, selectivity, trials = s2.ranked_cells_by_selectivity(p=0.05)
    rt,lt = trials
    matched_neurons=np.load(match_n_path.format(lnum-1))
    if triple:
        matched_neurons = matched_neurons[:,[0,2]]
        
    overallsel = np.append(overallsel, selectivity)
    
    for nnum in range(len(neurons_ranked)):
        
        if neurons_ranked[nnum] not in matched_neurons[:,1]:
            print(neurons_ranked[nnum], 'not matched.')
            continue
        neuron = neurons_ranked[nnum]

        ## FIRST SESS
        nind = np.where(matched_neurons[:,1] == neuron)[0]
        neuron = matched_neurons[nind, 0] # Grab the ranked neuron
        r,l = s1.get_trace_matrix(neuron)
        
        right_trace = np.mean(r, axis=0)# / np.mean(np.mean(r, axis=0)[0,epoch])
        left_trace = np.mean(l, axis=0)# / np.mean(np.mean(l, axis=0)[0,epoch])
        
        right_stack = np.vstack((right_stack, right_trace))
        left_stack = np.vstack((left_stack, left_trace))
        
        ## SECOND SESS
        neuron = neurons_ranked[nnum]
        r,l = s2.get_trace_matrix(neuron, rtrials = rt, ltrials=lt)
        
        right_trace = np.mean(r, axis=0)# / np.mean(np.mean(r, axis=0)[epoch])
        left_trace = np.mean(l, axis=0) #/ np.mean(np.mean(l, axis=0)[epoch])
        
        right_stack_post = np.vstack((right_stack_post, right_trace))
        left_stack_post = np.vstack((left_stack_post, left_trace))

right_stack = np.take(right_stack, np.argsort(overallsel), axis = 0)
left_stack = np.take(left_stack, np.argsort(overallsel), axis = 0)

right_stack_post = np.take(right_stack_post, np.argsort(overallsel), axis = 0)
left_stack_post = np.take(left_stack_post, np.argsort(overallsel), axis = 0)

#%% SAVE TO MATLAB

savepath = r'F:\data\SFN 2023\to matlab'
right_stack = normalize(right_stack[1:, 6:])
left_stack = normalize(left_stack[1:, 6:])
right_stack_post = normalize(right_stack_post[1:, 6:])
left_stack_post = normalize(left_stack_post[1:, 6:])

scio.savemat(savepath + r'\right_stack.mat', {'right_stack': right_stack})
scio.savemat(savepath + r'\left_stack.mat', {'left_stack' : left_stack})
scio.savemat(savepath + r'\right_stack_post.mat', {'right_stack_post' : right_stack_post})
scio.savemat(savepath + r'\left_stack_post.mat', {'left_stack_post': left_stack_post})
#%% PLOT 
        
f, axarr = plt.subplots(2,2, sharex='col', figsize=(15,10))
vmin, vmax= 0,0.32
## FIRST SESS
# Right trials first
right_stack = normalize(right_stack[1:, 6:])
# right_stack = (right_stack[1:])
right_im = axarr[1,0].imshow(right_stack, cmap='viridis', interpolation='nearest', aspect='auto', vmin=vmin, vmax=vmax)
axarr[1,0].axis('off')
f.colorbar(right_im, shrink = 0.2)
axarr[0,0].set_title("Session 1")
axarr[1,0].set_ylabel("Right trials")

# Left trials
left_stack = normalize(left_stack[1:, 6:])
# left_stack = (left_stack[1:])
leftim = axarr[0,0].imshow(left_stack, cmap='viridis', interpolation='nearest', aspect='auto',vmin=vmin, vmax=vmax)
axarr[0,0].axis('off')
axarr[0,0].set_ylabel("Left trials")

f.colorbar(leftim, shrink = 0.2)


## SECOND SESS
# Right trials first
right_stack_post = normalize(right_stack_post[1:, 6:])
# right_stack_post = (right_stack_post[1:])
right_im = axarr[1,1].imshow(right_stack_post, cmap='viridis', interpolation='nearest', aspect='auto', vmin=vmin, vmax=vmax)
axarr[1,1].axis('off')
f.colorbar(right_im, shrink = 0.2)
axarr[0,1].set_title("Session 2")
# Left trials
left_stack_post = normalize(left_stack_post[1:, 6:])
# left_stack_post = (left_stack_post[1:])
leftim = axarr[0,1].imshow(left_stack_post, cmap='viridis', interpolation='nearest', aspect='auto',vmin=vmin, vmax=vmax)
axarr[0,1].axis('off')
f.colorbar(leftim, shrink = 0.2)

plt.savefig(r'F:\data\SFN 2023\trained_matched_pop.pdf')