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
import scipy

#%% Proportion selective no contra ipsi info

# Aggregate plot NAIVE ##
allcontra, allipsi = [], []
# new sessions


paths = [
        r'H:\data\BAYLORCW044\python\2024_05_22',
        r'H:\data\BAYLORCW044\python\2024_05_23',
        
        r'H:\data\BAYLORCW046\python\2024_05_29',
        r'H:\data\BAYLORCW046\python\2024_05_30',
        r'H:\data\BAYLORCW046\python\2024_05_31',

        ]
for path in paths:
    l1 = session.Session(path, use_reg=True, triple=True)
    contra, ipsi = l1.plot_number_of_sig_neurons(return_nums=True)

    contra = contra + ipsi
    
    x = np.arange(-6.97,6,1/30)[:l1.time_cutoff*2]
    nums = np.interp(x, np.arange(-6.97,6,1/15)[:l1.time_cutoff], contra)
    contra = scipy.signal.decimate(nums, 5)

    allcontra += [contra/len(l1.good_neurons)]

paths = [
        r'F:\data\BAYLORCW032\python\2023_10_08',
        # r'F:\data\BAYLORCW034\python\2023_10_12',
        r'F:\data\BAYLORCW036\python\2023_10_09',
        r'F:\data\BAYLORCW035\python\2023_10_26',
        r'F:\data\BAYLORCW037\python\2023_11_21',
        ]
for path in paths:
    l1 = session.Session(path, use_reg=True, triple=True)
    contra, ipsi = l1.plot_number_of_sig_neurons(return_nums=True)
    contra = contra + ipsi
    allcontra = np.vstack((allcontra, contra/len(l1.good_neurons)))
    
    
x = np.arange(-6.97,6,1/6)[:61] # Downsample everything to this

plt.plot(x, np.mean(allcontra, axis=0), color = 'b', marker = 'o', markersize=5)

for i in range(9):
    plt.plot(x, allcontra[i], color = 'b', alpha = 0.5, linewidth = 0.5)

plt.axvline(-4.3, ls='--', color='grey')
plt.axvline(-3, ls='--', color='grey')
plt.axvline(0, ls='--', color='grey')
# plt.ylim(bottom = -350)
plt.ylim(top = 0.35)
plt.ylabel('Proportion of sig sel neurons')
plt.xlabel('Time from Go cue (s)')
plt.legend()
plt.savefig(r'F:\data\Fig 1\naive_numproportionagg_sel_neuronsALL.pdf')

plt.show()
#%%
# Aggregate plot LEARNING ##


allcontra, allipsi = [], []
# new sessions


paths = [
        r'H:\data\BAYLORCW044\python\2024_06_06',
        r'H:\data\BAYLORCW044\python\2024_06_04',
        
        r'H:\data\BAYLORCW046\python\2024_06_07',
        r'H:\data\BAYLORCW046\python\2024_06_10',
        r'H:\data\BAYLORCW046\python\2024_06_11',

        ]
for path in paths:
    l1 = session.Session(path, use_reg=True, triple=True)
    contra, ipsi = l1.plot_number_of_sig_neurons(return_nums=True)
    contra = contra + ipsi

    x = np.arange(-6.97,6,1/30)[:l1.time_cutoff*2]
    nums = np.interp(x, np.arange(-6.97,6,1/15)[:l1.time_cutoff], contra)
    contra = scipy.signal.decimate(nums, 5)

    allcontra += [contra/len(l1.good_neurons)]

paths = [
        r'F:\data\BAYLORCW032\python\2023_10_16',
        # r'F:\data\BAYLORCW034\python\2023_10_22',
        r'F:\data\BAYLORCW036\python\2023_10_19',
        r'F:\data\BAYLORCW035\python\2023_12_07',
        r'F:\data\BAYLORCW037\python\2023_12_08',
        ]
for path in paths:
    l1 = session.Session(path, use_reg=True, triple=True)
    contra, ipsi = l1.plot_number_of_sig_neurons(return_nums=True)
    contra = contra + ipsi

    allcontra = np.vstack((allcontra, contra/len(l1.good_neurons)))    
    
x = np.arange(-6.97,6,1/6)[:61] # Downsample everything to this

plt.plot(x, np.mean(allcontra, axis=0), color = 'b', marker = 'o', markersize=5)

for i in range(9):
    plt.plot(x, allcontra[i], color = 'b', alpha = 0.5, linewidth = 0.5)

plt.axvline(-4.3, ls='--', color='grey')
plt.axvline(-3, ls='--', color='grey')
plt.axvline(0, ls='--', color='grey')
# plt.ylim(bottom = -350)
plt.ylim(top = 0.35)
plt.ylabel('Proportion of sig sel neurons')
plt.xlabel('Time from Go cue (s)')
plt.legend()
plt.savefig(r'F:\data\Fig 1\lea_numproportionagg_sel_neuronsALL.pdf')

plt.show()


#%% Aggregate plot EXPERT ##
allcontra, allipsi = [], []
#original sessions
paths = [r'F:\data\BAYLORCW032\python\2023_10_24',
        # r'F:\data\BAYLORCW034\python\2023_10_27',
        r'F:\data\BAYLORCW036\python\2023_10_28']



paths = [
        r'H:\data\BAYLORCW044\python\2024_06_19',
        r'H:\data\BAYLORCW044\python\2024_06_18',
        
        r'H:\data\BAYLORCW046\python\2024_06_24',
        r'H:\data\BAYLORCW046\python\2024_06_27',
        r'H:\data\BAYLORCW046\python\2024_06_26',

        ]
for path in paths:
    l1 = session.Session(path, use_reg=True, triple=True)
    contra, ipsi = l1.plot_number_of_sig_neurons(return_nums=True)
    contra = contra + ipsi
    contra = contra[:151]

    x = np.arange(-6.97,6,1/30)[:l1.time_cutoff*2]
    nums = np.interp(x, np.arange(-6.97,6,1/15)[:151], contra)
    contra = scipy.signal.decimate(nums, 5)
    
    allcontra += [contra/len(l1.good_neurons)]
    
#testing new sessions
paths = [
        r'F:\data\BAYLORCW032\python\2023_10_25',
        # r'F:\data\BAYLORCW034\python\2023_10_27',
        r'F:\data\BAYLORCW036\python\2023_10_30',
        r'F:\data\BAYLORCW035\python\2023_12_15',
        r'F:\data\BAYLORCW037\python\2023_12_15',
        ]
for path in paths:
    l1 = session.Session(path, use_reg=True, triple=True)
    contra, ipsi = l1.plot_number_of_sig_neurons(return_nums=True)
    contra = contra + ipsi
    
    allcontra = np.vstack((allcontra, contra/len(l1.good_neurons)))
    
    
x = np.arange(-6.97,6,1/6)[:61] # Downsample everything to this

plt.plot(x, np.mean(allcontra, axis=0), color = 'b', marker = 'o', markersize=5)

for i in range(9):
    plt.plot(x, allcontra[i], color = 'b', alpha = 0.5, linewidth = 0.5)

plt.axvline(-4.3, ls='--', color='grey')
plt.axvline(-3, ls='--', color='grey')
plt.axvline(0, ls='--', color='grey')
# plt.ylim(bottom = -350)
plt.ylim(top = 0.35)
plt.ylabel('Proportion of sig sel neurons')
plt.xlabel('Time from Go cue (s)')
plt.legend()
plt.savefig(r'F:\data\Fig 1\exp_numproportionagg_sel_neuronsALL.pdf')
# 
plt.show()

#%%
##Proportion of selective contra ipsi ###
#%% ALL AGGREGATED CONTRA IPSI PROPORTION
# Aggregate plot NAIVE ##
allcontra, allipsi = [], []
# new sessions


paths = [
        r'H:\data\BAYLORCW044\python\2024_05_22',
        r'H:\data\BAYLORCW044\python\2024_05_23',
        
        r'H:\data\BAYLORCW046\python\2024_05_29',
        r'H:\data\BAYLORCW046\python\2024_05_30',
        r'H:\data\BAYLORCW046\python\2024_05_31',

        ]
for path in paths:
    l1 = session.Session(path, use_reg=True, triple=True)
    contra, ipsi = l1.plot_number_of_sig_neurons(return_nums=True)

    x = np.arange(-6.97,6,1/30)[:l1.time_cutoff*2]
    nums = np.interp(x, np.arange(-6.97,6,1/15)[:l1.time_cutoff], contra)
    contra = scipy.signal.decimate(nums, 5)
    
    nums = np.interp(x, np.arange(-6.97,6,1/15)[:l1.time_cutoff], ipsi)
    ipsi = scipy.signal.decimate(nums, 5)
    
    allcontra += [contra/len(l1.good_neurons)]
    allipsi += [ipsi/len(l1.good_neurons)]    

paths = [
        r'F:\data\BAYLORCW032\python\2023_10_08',
        # r'F:\data\BAYLORCW034\python\2023_10_12',
        r'F:\data\BAYLORCW036\python\2023_10_09',
        r'F:\data\BAYLORCW035\python\2023_10_26',
        r'F:\data\BAYLORCW037\python\2023_11_21',
        ]
for path in paths:
    l1 = session.Session(path, use_reg=True, triple=True)
    contra, ipsi = l1.plot_number_of_sig_neurons(return_nums=True)
    allcontra = np.vstack((allcontra, contra/len(l1.good_neurons)))
    allipsi = np.vstack((allipsi, ipsi/len(l1.good_neurons)))
    
    
x = np.arange(-6.97,6,1/6)[:61] # Downsample everything to this

plt.plot(x, np.mean(allcontra, axis=0), color = 'b', marker = 'o', markersize=5)
plt.plot(x, -np.mean(allipsi, axis=0), color = 'r', marker = 'o', markersize=5)

for i in range(9):
    plt.plot(x, allcontra[i], color = 'b', alpha = 0.5, linewidth = 0.5)
    plt.plot(x, -allipsi[i], color = 'r', alpha = 0.5, linewidth = 0.5)

plt.axvline(-4.3)
plt.axvline(-3)
plt.axvline(0)
# plt.ylim(bottom = -350)
# plt.ylim(top = 350)
plt.ylabel('Proportion of sig sel neurons')
plt.xlabel('Time from Go cue (s)')
plt.legend()
plt.savefig(r'F:\data\Fig 1\naive_numproportion_sel_neuronsALL.pdf')

plt.show()
#%%
# Aggregate plot LEARNING ##


allcontra, allipsi = [], []
# new sessions


paths = [
        r'H:\data\BAYLORCW044\python\2024_06_06',
        r'H:\data\BAYLORCW044\python\2024_06_04',
        
        r'H:\data\BAYLORCW046\python\2024_06_07',
        r'H:\data\BAYLORCW046\python\2024_06_10',
        r'H:\data\BAYLORCW046\python\2024_06_11',

        ]
for path in paths:
    l1 = session.Session(path, use_reg=True, triple=True)
    contra, ipsi = l1.plot_number_of_sig_neurons(return_nums=True)

    x = np.arange(-6.97,6,1/30)[:l1.time_cutoff*2]
    nums = np.interp(x, np.arange(-6.97,6,1/15)[:l1.time_cutoff], contra)
    contra = scipy.signal.decimate(nums, 5)
    
    nums = np.interp(x, np.arange(-6.97,6,1/15)[:l1.time_cutoff], ipsi)
    ipsi = scipy.signal.decimate(nums, 5)
    
    allcontra += [contra/len(l1.good_neurons)]
    allipsi += [ipsi/len(l1.good_neurons)]    

paths = [
        r'F:\data\BAYLORCW032\python\2023_10_16',
        # r'F:\data\BAYLORCW034\python\2023_10_22',
        r'F:\data\BAYLORCW036\python\2023_10_19',
        r'F:\data\BAYLORCW035\python\2023_12_07',
        r'F:\data\BAYLORCW037\python\2023_12_08',
        ]
for path in paths:
    l1 = session.Session(path, use_reg=True, triple=True)
    contra, ipsi = l1.plot_number_of_sig_neurons(return_nums=True)
    allcontra = np.vstack((allcontra, contra/len(l1.good_neurons)))
    allipsi = np.vstack((allipsi, ipsi/len(l1.good_neurons)))
    
    
x = np.arange(-6.97,6,1/6)[:61] # Downsample everything to this

plt.plot(x, np.mean(allcontra, axis=0), color = 'b', marker = 'o', markersize=5)
plt.plot(x, -np.mean(allipsi, axis=0), color = 'r', marker = 'o', markersize=5)

for i in range(9):
    plt.plot(x, allcontra[i], color = 'b', alpha = 0.5, linewidth = 0.5)
    plt.plot(x, -allipsi[i], color = 'r', alpha = 0.5, linewidth = 0.5)

plt.axvline(-4.3)
plt.axvline(-3)
plt.axvline(0)
# plt.ylim(bottom = -350)
# plt.ylim(top = 350)
plt.ylabel('Proportion of sig sel neurons')
plt.xlabel('Time from Go cue (s)')
plt.legend()
# plt.savefig(r'F:\data\Fig 1\naive_numproportion_sel_neuronsALL.pdf')

plt.show()


#%% Aggregate plot EXPERT ##
allcontra, allipsi = [], []
#original sessions
paths = [r'F:\data\BAYLORCW032\python\2023_10_24',
        # r'F:\data\BAYLORCW034\python\2023_10_27',
        r'F:\data\BAYLORCW036\python\2023_10_28']



paths = [
        r'H:\data\BAYLORCW044\python\2024_06_19',
        r'H:\data\BAYLORCW044\python\2024_06_18',
        
        r'H:\data\BAYLORCW046\python\2024_06_24',
        r'H:\data\BAYLORCW046\python\2024_06_27',
        r'H:\data\BAYLORCW046\python\2024_06_26',

        ]
for path in paths:
    l1 = session.Session(path, use_reg=True, triple=True)
    contra, ipsi = l1.plot_number_of_sig_neurons(return_nums=True)

    x = np.arange(-6.97,6,1/30)[:l1.time_cutoff*2]
    nums = np.interp(x, np.arange(-6.97,6,1/15)[:151], contra)
    contra = scipy.signal.decimate(nums, 5)
    
    nums = np.interp(x, np.arange(-6.97,6,1/15)[:151], ipsi)
    ipsi = scipy.signal.decimate(nums, 5)
    
    allcontra += [contra/len(l1.good_neurons)]
    allipsi += [ipsi/len(l1.good_neurons)]   
    
#testing new sessions
paths = [
        r'F:\data\BAYLORCW032\python\2023_10_25',
        # r'F:\data\BAYLORCW034\python\2023_10_27',
        r'F:\data\BAYLORCW036\python\2023_10_30',
        r'F:\data\BAYLORCW035\python\2023_12_15',
        r'F:\data\BAYLORCW037\python\2023_12_15',
        ]
for path in paths:
    l1 = session.Session(path, use_reg=True, triple=True)
    contra, ipsi = l1.plot_number_of_sig_neurons(return_nums=True)
    allcontra = np.vstack((allcontra, contra/len(l1.good_neurons)))
    allipsi = np.vstack((allipsi, ipsi/len(l1.good_neurons)))
    
    
x = np.arange(-6.97,6,1/6)[:61] # Downsample everything to this

plt.plot(x, np.mean(allcontra, axis=0), color = 'b', marker = 'o', markersize=5)
plt.plot(x, -np.mean(allipsi, axis=0), color = 'r', marker = 'o', markersize=5)

for i in range(9):
    plt.plot(x, allcontra[i], color = 'b', alpha = 0.5, linewidth = 0.5)
    plt.plot(x, -allipsi[i], color = 'r', alpha = 0.5, linewidth = 0.5)

plt.axvline(-4.3)
plt.axvline(-3)
plt.axvline(0)
# plt.ylim(bottom = -350)
# plt.ylim(top = 350)
plt.ylabel('Proportion of sig sel neurons')
plt.xlabel('Time from Go cue (s)')
plt.legend()
# plt.savefig(r'F:\data\Fig 1\naive_numproportion_sel_neuronsALL.pdf')

plt.show()



#%% ALL AGGREGATED CONTRA IPSI POPULATION
# Aggregate plot NAIVE ##
allcontra, allipsi = [], []
# new sessions


paths = [
        r'H:\data\BAYLORCW044\python\2024_05_22',
        r'H:\data\BAYLORCW044\python\2024_05_23',
        
        r'H:\data\BAYLORCW046\python\2024_05_29',
        r'H:\data\BAYLORCW046\python\2024_05_30',
        r'H:\data\BAYLORCW046\python\2024_05_31',

        ]
for path in paths:
    l1 = session.Session(path, use_reg=True, triple=True)
    contra, ipsi = l1.plot_number_of_sig_neurons(return_nums=True)
    allcontra += [contra]
    allipsi += [ipsi]


x = np.arange(-6.97,6,1/30)[:l1.time_cutoff*2]
nums = np.interp(x, np.arange(-6.97,6,1/15)[:l1.time_cutoff], np.sum(allcontra, axis=0))
allcontra = scipy.signal.decimate(nums, 5)

x = np.arange(-6.97,6,1/30)[:l1.time_cutoff*2]
nums = np.interp(x, np.arange(-6.97,6,1/15)[:l1.time_cutoff], np.sum(allipsi, axis=0))
allipsi = scipy.signal.decimate(nums, 5)

paths = [
        r'F:\data\BAYLORCW032\python\2023_10_08',
        # r'F:\data\BAYLORCW034\python\2023_10_12',
        r'F:\data\BAYLORCW036\python\2023_10_09',
        r'F:\data\BAYLORCW035\python\2023_10_26',
        r'F:\data\BAYLORCW037\python\2023_11_21',
        ]
for path in paths:
    l1 = session.Session(path, use_reg=True, triple=True)
    contra, ipsi = l1.plot_number_of_sig_neurons(return_nums=True)
    allcontra = np.vstack((allcontra, contra))
    allipsi = np.vstack((allipsi, ipsi))
    
    
x = np.arange(-6.97,6,1/6)[:61] # Downsample everything to this

plt.bar(x, np.sum(allcontra, axis=0), color = 'b', edgecolor = 'white', width = 0.17, label = 'contra')
plt.bar(x, -np.sum(allipsi, axis=0), color = 'r',edgecolor = 'white', width = 0.17, label = 'ipsi')
plt.axvline(-4.3)
plt.axvline(-3)
plt.axvline(0)
# plt.ylim(bottom = -350)
# plt.ylim(top = 350)
plt.ylabel('Number of sig sel neurons')
plt.xlabel('Time from Go cue (s)')
plt.legend()
# plt.savefig(r'F:\data\Fig 1\naive_numproportion_sel_neuronsALL.pdf')

plt.show()
#%%
# Aggregate plot LEARNING ##


allcontra, allipsi = [], []
# new sessions


paths = [
        r'H:\data\BAYLORCW044\python\2024_06_06',
        r'H:\data\BAYLORCW044\python\2024_06_04',
        
        r'H:\data\BAYLORCW046\python\2024_06_07',
        r'H:\data\BAYLORCW046\python\2024_06_10',
        r'H:\data\BAYLORCW046\python\2024_06_11',

        ]
for path in paths:
    l1 = session.Session(path, use_reg=True, triple=True)
    contra, ipsi = l1.plot_number_of_sig_neurons(return_nums=True)
    allcontra += [contra]
    allipsi += [ipsi]


x = np.arange(-6.97,6,1/30)[:l1.time_cutoff*2]
nums = np.interp(x, np.arange(-6.97,6,1/15)[:l1.time_cutoff], np.sum(allcontra, axis=0))
allcontra = scipy.signal.decimate(nums, 5)

x = np.arange(-6.97,6,1/30)[:l1.time_cutoff*2]
nums = np.interp(x, np.arange(-6.97,6,1/15)[:l1.time_cutoff], np.sum(allipsi, axis=0))
allipsi = scipy.signal.decimate(nums, 5)

paths = [
        r'F:\data\BAYLORCW032\python\2023_10_16',
        # r'F:\data\BAYLORCW034\python\2023_10_22',
        r'F:\data\BAYLORCW036\python\2023_10_19',
        r'F:\data\BAYLORCW035\python\2023_12_07',
        r'F:\data\BAYLORCW037\python\2023_12_08',
        ]
for path in paths:
    l1 = session.Session(path, use_reg=True, triple=True)
    contra, ipsi = l1.plot_number_of_sig_neurons(return_nums=True)
    allcontra = np.vstack((allcontra, contra))
    allipsi = np.vstack((allipsi, ipsi))
    
    
x = np.arange(-6.97,6,1/6)[:61] # Downsample everything to this

plt.bar(x, np.sum(allcontra, axis=0), color = 'b', edgecolor = 'white', width = 0.17, label = 'contra')
plt.bar(x, -np.sum(allipsi, axis=0), color = 'r',edgecolor = 'white', width = 0.17, label = 'ipsi')
plt.axvline(-4.3)
plt.axvline(-3)
plt.axvline(0)
# plt.ylim(bottom = -350)
# plt.ylim(top = 350)
plt.ylabel('Number of sig sel neurons')
plt.xlabel('Time from Go cue (s)')
plt.legend()

plt.savefig(r'F:\data\Fig 1\learning_numproportion_sel_neuronsALL.pdf')
plt.show()



#%% Aggregate plot EXPERT ##
allcontra, allipsi = [], []
#original sessions
paths = [r'F:\data\BAYLORCW032\python\2023_10_24',
        # r'F:\data\BAYLORCW034\python\2023_10_27',
        r'F:\data\BAYLORCW036\python\2023_10_28']



paths = [
        r'H:\data\BAYLORCW044\python\2024_06_19',
        r'H:\data\BAYLORCW044\python\2024_06_18',
        
        r'H:\data\BAYLORCW046\python\2024_06_24',
        r'H:\data\BAYLORCW046\python\2024_06_27',
        r'H:\data\BAYLORCW046\python\2024_06_26',

        ]
for path in paths:
    l1 = session.Session(path, use_reg=True, triple=True)
    contra, ipsi = l1.plot_number_of_sig_neurons(return_nums=True)
    allcontra += [contra]
    allipsi += [ipsi]


x = np.arange(-6.97,6,1/30)[:l1.time_cutoff*2]
nums = np.interp(x, np.arange(-6.97,6,1/15)[:l1.time_cutoff], np.sum(allcontra, axis=0))
allcontra = scipy.signal.decimate(nums, 5)

x = np.arange(-6.97,6,1/30)[:l1.time_cutoff*2]
nums = np.interp(x, np.arange(-6.97,6,1/15)[:l1.time_cutoff], np.sum(allipsi, axis=0))
allipsi = scipy.signal.decimate(nums, 5)

#testing new sessions
paths = [
        r'F:\data\BAYLORCW032\python\2023_10_25',
        # r'F:\data\BAYLORCW034\python\2023_10_27',
        r'F:\data\BAYLORCW036\python\2023_10_30',
        r'F:\data\BAYLORCW035\python\2023_12_15',
        r'F:\data\BAYLORCW037\python\2023_12_15',
        ]
for path in paths:
    l1 = session.Session(path, use_reg=True, triple=True)
    contra, ipsi = l1.plot_number_of_sig_neurons(return_nums=True)
    allcontra = np.vstack((allcontra, contra))
    allipsi = np.vstack((allipsi, ipsi))
    
    
x = np.arange(-6.97,6,1/6)[:61] # Downsample everything to this

plt.bar(x, np.sum(allcontra, axis=0), color = 'b', edgecolor = 'white', width = 0.17, label = 'contra')
plt.bar(x, -np.sum(allipsi, axis=0), color = 'r',edgecolor = 'white', width = 0.17, label = 'ipsi')
plt.axvline(-4.3)
plt.axvline(-3)
plt.axvline(0)
# plt.ylim(bottom = -350)
# plt.ylim(top = 350)
plt.ylabel('Number of sig sel neurons')
plt.xlabel('Time from Go cue (s)')
plt.legend()
plt.savefig(r'F:\data\Fig 1\expert_numproportion_sel_neuronsALL.pdf')


#%% 
### TOTAL NUMBER OF SELECTIVE CONTRA IPSI ### without new sessions CW44/46
#%%
# Aggregate plot NAIVE ##
allcontra, allipsi = [], []

# new sessions
paths = [
        r'F:\data\BAYLORCW032\python\2023_10_08',
        # r'F:\data\BAYLORCW034\python\2023_10_12',
        r'F:\data\BAYLORCW036\python\2023_10_09',
        r'F:\data\BAYLORCW035\python\2023_10_26',
        r'F:\data\BAYLORCW037\python\2023_11_21',
        ]
paths = [
        r'H:\data\BAYLORCW044\python\2024_05_22',
        r'H:\data\BAYLORCW044\python\2024_05_23',
        
        r'H:\data\BAYLORCW046\python\2024_05_29',
        r'H:\data\BAYLORCW046\python\2024_05_30',
        r'H:\data\BAYLORCW046\python\2024_05_31',

        ]
for path in paths:
    l1 = session.Session(path, use_reg=True, triple=True)
    contra, ipsi = l1.plot_number_of_sig_neurons(return_nums=True)
    allcontra += [contra]
    allipsi += [ipsi]



x = np.arange(-6.97,6,l1.fs)[:l1.time_cutoff]

plt.bar(x, np.sum(allcontra, axis=0), color = 'b', edgecolor = 'white', width = 0.17, label = 'contra')
plt.bar(x, -np.sum(allipsi, axis=0), color = 'r',edgecolor = 'white', width = 0.17, label = 'ipsi')
plt.axvline(-4.3)
plt.axvline(-3)
plt.axvline(0)
# plt.ylim(bottom = -300)
# plt.ylim(top = 400)
plt.ylim(bottom = -200)
plt.ylim(top = 250)
plt.ylabel('Number of sig sel neurons')
plt.xlabel('Time from Go cue (s)')
plt.legend()
# plt.savefig(r'F:\data\Fig 1\naive_numproportion_sel_neuronsALL.pdf')

plt.show()
#%%
# Aggregate plot LEARNING ##
allcontra, allipsi = [], []

# new sessions
paths = [
        r'F:\data\BAYLORCW032\python\2023_10_16',
        # r'F:\data\BAYLORCW034\python\2023_10_22',
        r'F:\data\BAYLORCW036\python\2023_10_19',
        r'F:\data\BAYLORCW035\python\2023_12_07',
        r'F:\data\BAYLORCW037\python\2023_12_08',
        ]
for path in paths:
    l1 = session.Session(path, use_reg=True, triple=True)
    contra, ipsi = l1.plot_number_of_sig_neurons(return_nums=True)
    allcontra += [contra]
    allipsi += [ipsi]



x = np.arange(-6.97,6,l1.fs)[:l1.time_cutoff]

plt.bar(x, np.sum(allcontra, axis=0), color = 'b', edgecolor = 'white', width = 0.17, label = 'contra')
plt.bar(x, -np.sum(allipsi, axis=0), color = 'r',edgecolor = 'white', width = 0.17, label = 'ipsi')
plt.axvline(-4.3)
plt.axvline(-3)
plt.axvline(0)
# plt.ylim(bottom = -300)
# plt.ylim(top = 400)
plt.ylim(bottom = -200)
plt.ylim(top = 250)
plt.ylabel('Number of sig sel neurons')
plt.xlabel('Time from Go cue (s)')
plt.legend()
plt.savefig(r'F:\data\Fig 1\learning_num_sel_neuronsALL.pdf')
plt.show()



#%%
# Aggregate plot TRAINED ##
allcontra, allipsi = [], []
#original sessions
paths = [r'F:\data\BAYLORCW032\python\2023_10_24',
        # r'F:\data\BAYLORCW034\python\2023_10_27',
        r'F:\data\BAYLORCW036\python\2023_10_28']

#testing new sessions
paths = [
        r'F:\data\BAYLORCW032\python\2023_10_25',
        # r'F:\data\BAYLORCW034\python\2023_10_27',
        r'F:\data\BAYLORCW036\python\2023_10_30',
        r'F:\data\BAYLORCW035\python\2023_12_15',
        r'F:\data\BAYLORCW037\python\2023_12_15',
        ]
for path in paths:
    l1 = session.Session(path, use_reg=True, triple=True)
    contra, ipsi = l1.plot_number_of_sig_neurons(return_nums=True)
    allcontra += [contra]
    allipsi += [ipsi]



x = np.arange(-6.97,6,l1.fs)[:l1.time_cutoff]

plt.bar(x, np.sum(allcontra, axis=0), color = 'b', edgecolor = 'white', width = 0.17, label = 'contra')
plt.bar(x, -np.sum(allipsi, axis=0), color = 'r',edgecolor = 'white', width = 0.17, label = 'ipsi')
plt.axvline(-4.3)
plt.axvline(-3)
plt.axvline(0)
# plt.ylim(bottom = -300)
# plt.ylim(top = 400)
plt.ylim(bottom = -200)
plt.ylim(top = 250)
plt.ylabel('Number of sig sel neurons')
plt.xlabel('Time from Go cue (s)')
plt.legend()
plt.savefig(r'F:\data\Fig 1\trained_num_sel_neuronsALL.pdf')