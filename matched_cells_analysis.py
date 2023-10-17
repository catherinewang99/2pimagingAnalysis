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


### Single neuron PSTHs over sessions ###



### Heatmap analysis over sessions ###

path = r'F:\data\BAYLORCW036\python\2023_10_07'
s1 = session.Session(path)

path = r'F:\data\BAYLORCW036\python\2023_10_09'
s2 = session.Session(path)

f, axarr = plt.subplots(1,2, sharex='col', figsize=(20,7))

neurons_ranked, selectivity = s1.ranked_cells_by_selectivity(p=0.05)

right_stack = np.zeros(s1.time_cutoff) 
left_stack = np.zeros(s1.time_cutoff) 
for neuron in neurons_ranked:
    r,l = s1.get_trace_matrix(neuron)
    
    right_trace = np.mean(r, axis=0)
    left_trace = np.mean(l, axis=0)
    
    right_stack = np.vstack((right_stack, right_trace))
    left_stack = np.vstack((left_stack, left_trace))
                            
# Right trials first
right_stack = normalize(right_stack[1:])
right_im = axarr[0].matshow(right_stack, cmap='YlGnBu', interpolation='nearest', aspect='auto')
axarr[0].axis('off')
f.colorbar(right_im, shrink = 0.2)
# Left trials
left_stack = normalize(left_stack[1:])
leftim = axarr[1].matshow(left_stack, cmap='YlGnBu', interpolation='nearest', aspect='auto')
axarr[1].axis('off')
f.colorbar(leftim, shrink = 0.2)
