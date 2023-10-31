# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 14:50:02 2023

@author: Catherine Wang
"""

import sys
sys.path.append("C:\scripts\Imaging analysis")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import session
from matplotlib.pyplot import figure
# import decon
from scipy.stats import chisquare
import pandas as pd
plt.rcParams['pdf.fonttype'] = '42' 

##Proportion of selective contra ipsi ###

# Aggregate plot NAIVE ##
allcontra, allipsi = [], []
paths = [r'F:\data\BAYLORCW032\python\2023_10_05',
        # r'F:\data\BAYLORCW034\python\2023_10_12',
        r'F:\data\BAYLORCW036\python\2023_10_09']

for path in paths:
    l1 = session.Session(path)
    contra, ipsi = l1.plot_number_of_sig_neurons(return_nums=True)
    allcontra += [contra]
    allipsi += [ipsi]



x = np.arange(-6.97,6,l1.fs)[:l1.time_cutoff]

plt.bar(x, np.sum(allcontra, axis=0), color = 'b', edgecolor = 'white', width = 0.17, label = 'contra')
plt.bar(x, -np.sum(allipsi, axis=0), color = 'r',edgecolor = 'white', width = 0.17, label = 'ipsi')
plt.axvline(-4.3)
plt.axvline(-3)
plt.axvline(0)
plt.ylim(bottom = -140)
plt.ylim(top = 160)
plt.ylabel('Number of sig sel neurons')
plt.xlabel('Time from Go cue (s)')
plt.legend()
plt.savefig(r'F:\data\SFN 2023\naive_num_sel_neurons.pdf')

# Aggregate plot TRAINED ##
# allcontra, allipsi = [], []
# paths = [r'F:\data\BAYLORCW032\python\2023_10_24',
#         # r'F:\data\BAYLORCW034\python\2023_10_27',
#         r'F:\data\BAYLORCW036\python\2023_10_28']

# for path in paths:
#     l1 = session.Session(path)
#     contra, ipsi = l1.plot_number_of_sig_neurons(return_nums=True)
#     allcontra += [contra]
#     allipsi += [ipsi]



# x = np.arange(-6.97,6,l1.fs)[:l1.time_cutoff]

# plt.bar(x, np.sum(allcontra, axis=0), color = 'b', edgecolor = 'white', width = 0.17, label = 'contra')
# plt.bar(x, -np.sum(allipsi, axis=0), color = 'r',edgecolor = 'white', width = 0.17, label = 'ipsi')
# plt.axvline(-4.3)
# plt.axvline(-3)
# plt.axvline(0)
# # if len(y_axis) != 0:
# plt.ylim(bottom = -140)
# plt.ylim(top = 160)
# plt.ylabel('Number of sig sel neurons')
# plt.xlabel('Time from Go cue (s)')
# plt.legend()
# plt.savefig(r'F:\data\SFN 2023\trained_num_sel_neurons.pdf')