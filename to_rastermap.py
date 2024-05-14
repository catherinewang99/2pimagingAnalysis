# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 13:05:18 2023

@author: Catherine Wang

Prepare neural data for Rastermap viewing
Use matched cells, grouped by trial type
or concatenated over trials to test states
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
import random

### Concatenating over all trials and combine with behavior state
#%% Format data for rastermap (concatenate)
path = r'F:\data\BAYLORCW036\python\2023_10_19'

l1 = session.Session(path, use_reg=True, triple=True)

F = None


# concatenate all trials together
for t in l1.i_good_trials:
    if F is None:
        F = l1.dff[0,t][:, 9:l1.time_cutoff] # take out first 1.5 seconds of baseline
    else:
        
        F = np.hstack((F, l1.dff[0,t][:, 9:l1.time_cutoff]))


F = F[l1.good_neurons]

np.save(l1.path + '\\F.npy', F)
### BELOW: grouping by trial type to view neuron sorting
#%% Format data for rastermap

paths = [r'F:\data\BAYLORCW032\python\2023_10_05',
          r'F:\data\BAYLORCW032\python\2023_10_19',
          r'F:\data\BAYLORCW032\python\2023_10_24',]

path= r'F:\data\BAYLORCW032\python\2023_10_24'

l1 = session.Session(path, use_reg=True, triple=True)

F = np.zeros(l1.time_cutoff)

rtrials = l1.lick_correct_direction('r')
random.shuffle(rtrials)
rtrials_train = rtrials[:50]
rtrials_test = rtrials[50:]

for n in l1.good_neurons:
    
    # Get all correct lick right trials
    R, _ = l1.get_trace_matrix(n, rtrials = rtrials_train)
    
    F = np.vstack((F, np.mean(R,axis=0)))
    
np.save(l1.path + '\\F.npy', F[1:, 9:])
#%% Plot rastermap results after run across train and test set

path= r'F:\data\BAYLORCW032\python\2023_10_24'
l1 = session.Session(path, use_reg=True, triple=True)

input_traces = np.load(path + '\\F.npy') # old non-CV traces
    
rmap = np.load(path + '\\F_embedding.npy', allow_pickle=True)
rmap = rmap.item()

vmin, vmax= -0.3,0.8
stack = np.zeros(input_traces.shape[1]) 

# Plot right traces first
for n in rmap['isort']:
    stack = np.vstack((stack, input_traces[n]))

f, axarr = plt.subplots(1, 2, sharex='col', figsize=(21,10))
axarr[0].imshow(stack, cmap='viridis', interpolation='nearest', aspect='auto', vmin=vmin, vmax=vmax)

input_traces = np.zeros(l1.time_cutoff)
for n in l1.good_neurons:
    
    # Get all correct lick right trials
    R, _ = l1.get_trace_matrix(n, rtrials = rtrials_test)
    
    input_traces = np.vstack((input_traces, np.mean(R,axis=0)))
input_traces = input_traces[1:, 9:]

stack = np.zeros(input_traces.shape[1]) 

# Plot right traces first
for n in rmap['isort']:
    stack = np.vstack((stack, input_traces[n]))
axarr[1].imshow(stack, cmap='viridis', interpolation='nearest', aspect='auto', vmin=vmin, vmax=vmax)

#%% Plot rastermap results after run across trial type
path= r'F:\data\BAYLORCW032\python\2023_10_24'
l1 = session.Session(path, use_reg=True, triple=True)

# input_traces = np.load(path + '\\F.npy') # old non-CV traces
# New cross vali traces
input_traces = np.zeros(l1.time_cutoff)
for n in l1.good_neurons:
    
    # Get all correct lick right trials
    R, _ = l1.get_trace_matrix(n, rtrials = rtrials_test)
    
    input_traces = np.vstack((input_traces, np.mean(R,axis=0)))
input_traces = input_traces[1:, 9:]
    
rmap = np.load(path + '\\F_embedding.npy', allow_pickle=True)
rmap = rmap.item()

vmin, vmax= -0.3,0.8
stack = np.zeros(input_traces.shape[1]) 

# Plot right traces first
for n in rmap['isort']:
    stack = np.vstack((stack, input_traces[n]))

f, axarr = plt.subplots(1, 2, sharex='col', figsize=(21,10))
axarr[0].imshow(stack, cmap='viridis', interpolation='nearest', aspect='auto', vmin=vmin, vmax=vmax)

stack = np.zeros(input_traces.shape[1]) 
for idx in rmap['isort']:
    n = l1.good_neurons[idx]
    _, L = l1.get_trace_matrix(n)
    stack = np.vstack((stack, np.mean(L,axis=0)[9:]))
    
axarr[1].imshow(stack[1:], cmap='viridis', interpolation='nearest', aspect='auto', vmin=vmin, vmax=vmax)


#%% Plot rastermap results after run across training stages
vmin, vmax= -0.3,0.8

path= r'F:\data\BAYLORCW032\python\2023_10_24'

# input_traces = np.load(path + '\\F.npy') # Old non-cv
input_traces = np.zeros(l1.time_cutoff)
for n in l1.good_neurons:
    
    # Get all correct lick right trials
    R, _ = l1.get_trace_matrix(n, rtrials = rtrials_test)
    
    input_traces = np.vstack((input_traces, np.mean(R,axis=0)))
input_traces = input_traces[1:, 9:]

rmap = np.load(path + '\\F_embedding.npy', allow_pickle=True)
rmap = rmap.item()

stack = np.zeros(input_traces.shape[1]) 

for n in rmap['isort']:
    stack = np.vstack((stack, input_traces[n]))

f, axarr = plt.subplots(1, 3, sharex='col', figsize=(21,10))
axarr[0].imshow(stack, cmap='viridis', interpolation='nearest', aspect='auto', vmin=vmin, vmax=vmax)

# Apply backwards

path= r'F:\data\BAYLORCW032\python\2023_10_05'

l1 = session.Session(path, use_reg=True, triple=True)

F = np.zeros(l1.time_cutoff)
for n in l1.good_neurons:
    # Get all correct lick right trials
    R, _ = l1.get_trace_matrix(n)
    F = np.vstack((F, np.mean(R,axis=0)))
F = F[1:, 9:]

stack = np.zeros(input_traces.shape[1]) 
for n in rmap['isort']:
    stack = np.vstack((stack, F[n]))
axarr[2].imshow(stack, cmap='viridis', interpolation='nearest', aspect='auto', vmin=vmin, vmax=vmax)

# Apply to middle section


path= r'F:\data\BAYLORCW032\python\2023_10_19'

l1 = session.Session(path, use_reg=True, triple=True)

F = np.zeros(l1.time_cutoff)
for n in l1.good_neurons:
    # Get all correct lick right trials
    R, _ = l1.get_trace_matrix(n)
    F = np.vstack((F, np.mean(R,axis=0)))
F = F[1:, 9:]

stack = np.zeros(input_traces.shape[1]) 
for n in rmap['isort']:
    stack = np.vstack((stack, F[n]))
axarr[1].imshow(stack, cmap='viridis', interpolation='nearest', aspect='auto', vmin=vmin, vmax=vmax)