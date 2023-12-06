# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 13:05:18 2023

@author: Catherine Wang

Prepare neural data for Rastermap viewing
Use matched cells, grouped by trial type
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

#%% Format data for rastermap

paths = [r'F:\data\BAYLORCW032\python\2023_10_05',
          r'F:\data\BAYLORCW032\python\2023_10_19',
          r'F:\data\BAYLORCW032\python\2023_10_24',]

path= r'F:\data\BAYLORCW032\python\2023_10_24'

l1 = session.Session(path, use_reg=True, triple=True)

F = np.zeros(l1.time_cutoff)

for n in l1.good_neurons:
    
    # Get all correct lick right trials
    R, _ = l1.get_trace_matrix(n)
    
    F = np.vstack((F, np.mean(R,axis=0)))
    
np.save(l1.path + '\\F.npy', F[1:, 9:])
#%% Plot rastermap results after run across trial type
path= r'F:\data\BAYLORCW032\python\2023_10_05'
l1 = session.Session(path, use_reg=True, triple=True)

input_traces = np.load(path + '\\F.npy')
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

path= r'F:\data\BAYLORCW032\python\2023_10_05'

input_traces = np.load(path + '\\F.npy')
rmap = np.load(path + '\\F_embedding.npy', allow_pickle=True)
rmap = rmap.item()

stack = np.zeros(input_traces.shape[1]) 

for n in rmap['isort']:
    stack = np.vstack((stack, input_traces[n]))

f, axarr = plt.subplots(1, 3, sharex='col', figsize=(21,10))
axarr[0].imshow(stack, cmap='viridis', interpolation='nearest', aspect='auto', vmin=vmin, vmax=vmax)

# Apply backwards

path= r'F:\data\BAYLORCW032\python\2023_10_24'

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