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

path = r'F:\data\BAYLORCW021\python\2023_02_08'

### TOTAL NUMBER OF NEURONS: ###

# total_n = 0
# for i in range(6):
#     l1 = session.Session(path, i+1)
#     # l1.crop_trials(245, end = 330)
#     total_n += l1.num_neurons

# print(total_n)

### EFFECT OF OPTO INHIBITION ###

l1 = session.Session(path, 6)

stim_dff = l1.dff[0][l1.stim_ON]
non_stim_dff = l1.dff[0][~l1.stim_ON]

### Histogram of average dff during stim period
# delay_stim = cat(cat([stim_dff[f][:, 13:28] for f in range(stim_dff.shape[0])]))
# delay_nostim = cat(cat([non_stim_dff[f][:, 13:28] for f in range(non_stim_dff.shape[0])]))

# plt.hist(delay_stim, alpha = 0.7, bins = 5000, label = 'Stim')
# plt.hist(delay_nostim, alpha = 0.7, bins = 5000, label = 'Control')
# plt.xlim(-5, 5)
# plt.legend()

### Average dff during stim period
delay_stim = cat([stim_dff[f][:, 13:28] for f in range(stim_dff.shape[0])])
delay_nostim = cat([non_stim_dff[f][:, 13:28] for f in range(non_stim_dff.shape[0])])
x=range(15)
stim = np.mean(delay_stim, axis = 0)
nostim = np.mean(delay_nostim, axis = 0)

stim_err = np.std(delay_stim, axis=0) / np.sqrt(len(delay_stim)) 
nostim_err = np.std(delay_nostim, axis=0) / np.sqrt(len(delay_nostim))

plt.plot(stim, 'r-')
plt.plot(nostim, 'b-')

            
plt.fill_between(x, stim - stim_err, 
          stim + stim_err,
          color=['#ffaeb1'])
plt.fill_between(x, nostim - nostim_err, 
          nostim + nostim_err,
          color=['#b4b2dc'])