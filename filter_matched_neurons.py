# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 10:35:55 2023

@author: Catherine Wang

Script designed to go through matched neurons and filter out noisy neurons
"""


import sys
sys.path.append("C:\scripts\Imaging analysis")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from session import Session
from matplotlib.pyplot import figure
import decon
from scipy.stats import chisquare
import pandas as pd
from activityMode import Mode


paths = [r'F:\data\BAYLORCW032\python\2023_10_05',
           r'F:\data\BAYLORCW032\python\2023_10_19',
          r'F:\data\BAYLORCW032\python\2023_10_24',]

for path in paths:
    total_neurons = 0

    for layer_num in range(1,6):
        
        # path = r'F:\data\BAYLORCW032\python\2023_10_19'
        
        l1 = Session(path, layer_num = layer_num, use_reg=True, triple=True)
        reg = np.load(path + r'\layer{}_triple_registered_neurons.npy'.format(layer_num-1))
        neurons, _ = l1.get_pearsonscorr_neuron(cutoff=0.5)
        
        print("Proportion: ", len(neurons)/l1.num_neurons)
        print("Num neurons: ", len(neurons))
        
        total_neurons += len(neurons)
    
    print("TOTAL NEURONS ", total_neurons)
    
    
#%%

total_neurons = 0
earliercorr = []
for layer_num in range(1,6):
    
    path = r'F:\data\BAYLORCW032\python\2023_10_05'
    
    l1 = Session(path, layer_num = layer_num, use_reg=True, triple=True)
    reg = np.load(path + r'\layer{}_triple_registered_neurons.npy'.format(layer_num-1))
    _, corr = l1.get_pearsonscorr_neuron(cutoff=0.5, postreg = True)

    earliercorr += corr

plt.hist(earliercorr)    
#%%
total_neurons = 0
latercorr = []
for layer_num in range(1,6):
    
    path = r'F:\data\BAYLORCW032\python\2023_10_24'
    
    l1 = Session(path, layer_num = layer_num, use_reg=True, triple=True)
    reg = np.load(path + r'\layer{}_triple_registered_neurons.npy'.format(layer_num-1))
    _, corr = l1.get_pearsonscorr_neuron(cutoff=0.5, postreg = True)

    latercorr += corr

plt.hist(latercorr)    

#%%
plt.scatter(earliercorr, latercorr)
plt.xlabel('Early')
plt.ylabel('Late')
plt.axhline(0.5, ls = '--', color = 'grey')
plt.axvline(0.5, ls = '--', color = 'grey')
plt.axhline(0, ls = '--', color = 'grey')
plt.axvline(0, ls = '--', color = 'grey')
