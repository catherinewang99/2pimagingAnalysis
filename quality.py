# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 10:12:39 2023

@author: Catherine Wang
"""
import numpy as np
from numpy import concatenate as cat
import matplotlib.pyplot as plt
from scipy import stats
import copy
import scipy.io as scio
from sklearn.preprocessing import normalize
from matplotlib.pyplot import figure
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import normalize
import os
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
from scipy.stats import mannwhitneyu
from session import Session
 


class QC(Session):

    def __init__(self, path, layer_num='all', guang=False, passive=False):
        
        super().__init__(path, layer_num, guang, passive)
        self.sample = 7
        self.delay = 13
        self.response = 28
        if 'CW030' in path:
            self.sample += 5
            self.delay += 5
            self.response += 5
            
        
    ### Quality analysis section ###
            
    def all_neurons_heatmap(self, save=False):
        
        f, axarr = plt.subplots(2,2, sharex='col')
        # x = np.arange(-5.97,4,0.2)[:self.time_cutoff]

        stim_dff = self.dff[0][self.stim_ON]
        non_stim_dff = self.dff[0][~self.stim_ON]

        stack = np.zeros(self.time_cutoff)

        for neuron in range(stim_dff[0].shape[0]):
            dfftrial = []
            for trial in range(stim_dff.shape[0]):
                dfftrial += [stim_dff[trial][neuron, :self.time_cutoff]]

            stack = np.vstack((stack, np.mean(np.array(dfftrial), axis=0)))

        stack = normalize(stack[1:])
        axarr[0,0].matshow(stack, cmap='gray', interpolation='nearest', aspect='auto')
        axarr[0,0].axis('off')
        axarr[0,0].set_title('Opto')
        axarr[0,0].axvline(x=self.delay, c='b', linewidth = 0.5)
        axarr[1,0].plot(np.mean(stack, axis = 0))
        axarr[1,0].set_ylim(top=0.2)
        axarr[1,0].axvline(x=self.delay, c='b', linewidth = 0.5)

        stack = np.zeros(self.time_cutoff)

        for neuron in range(non_stim_dff[0].shape[0]):
            dfftrial = []
            for trial in range(non_stim_dff.shape[0]):
                dfftrial += [non_stim_dff[trial][neuron, :self.time_cutoff]]

            stack = np.vstack((stack, np.mean(np.array(dfftrial), axis=0)))

        stack = normalize(stack[1:])

        axarr[0,1].matshow(stack, cmap='gray', interpolation='nearest', aspect='auto')
        axarr[0,1].axis('off')
        axarr[0,1].set_title('Control')

        axarr[1,1].plot(np.mean(stack, axis = 0))
        axarr[1,1].set_ylim(top=0.2)
        axarr[1,0].set_ylabel('dF/F0')
        # axarr[1,0].set_xlabel('Time from Go cue (s)')

        if save:
            plt.savefig(self.path + r'dff_contra_stimall.jpg')

        plt.show()
    
        return None
    
    def all_neurons_heatmap_stimlevels(self, save=False):
        
        f, axarr = plt.subplots(2,6, sharex='col', figsize=(20,6))
        # x = np.arange(-5.97,4,0.2)[:self.time_cutoff]

        non_stim_dff = self.dff[0][self.stim_level == 0]
        
        for i in range(1, len(set(self.stim_level))):
            
            level = sorted(list(set(self.stim_level)))[i]
            stim_dff = self.dff[0][self.stim_level == level]
    
            stack = np.zeros(self.time_cutoff)
    
            for neuron in range(stim_dff[0].shape[0]):
                dfftrial = []
                for trial in range(stim_dff.shape[0]):
                    dfftrial += [stim_dff[trial][neuron, :self.time_cutoff]]
    
                stack = np.vstack((stack, np.mean(np.array(dfftrial), axis=0)))
    
            stack = normalize(stack[1:])
            axarr[0,i].matshow(stack, cmap='gray', interpolation='nearest', aspect='auto')
            axarr[0,i].axis('off')
            axarr[0,i].set_title('Opto {} AOM'.format(level))
            axarr[0,i].axvline(x=self.delay, c='b', linewidth = 0.5)
            axarr[1,i].plot(np.mean(stack, axis = 0))
            axarr[1,i].set_ylim(top=0.2)
            axarr[1,i].axvline(x=self.delay, c='b', linewidth = 0.5)

        stack = np.zeros(self.time_cutoff)

        for neuron in range(non_stim_dff[0].shape[0]):
            dfftrial = []
            for trial in range(non_stim_dff.shape[0]):
                dfftrial += [non_stim_dff[trial][neuron, :self.time_cutoff]]

            stack = np.vstack((stack, np.mean(np.array(dfftrial), axis=0)))

        stack = normalize(stack[1:])

        axarr[0,0].matshow(stack, cmap='gray', interpolation='nearest', aspect='auto')
        axarr[0,0].axis('off')
        axarr[0,0].set_title('Control')

        axarr[1,0].plot(np.mean(stack, axis = 0))
        axarr[1,0].set_ylim(top=0.2)
        axarr[1,0].set_ylabel('dF/F0')
        # axarr[1,0].set_xlabel('Time from Go cue (s)')

        if save:
            plt.savefig(self.path + r'dff_contra_stimall.jpg')

        plt.show()
    
        return None
    
    def stim_activity_proportion(self, save=False):
        
        stim_period = range(16,19)
        
        f, axarr = plt.subplots(1,5, sharex='col', figsize = (20, 4))
        # x = np.arange(-5.97,4,0.2)[:self.time_cutoff]
        
        control_neuron_dff = []
        opto_neuron_dff = []

        non_stim_dff = self.dff[0][self.stim_level == 0]
        
        for n in range(self.num_neurons):
            av = []
            
            for t in range(non_stim_dff.shape[0]):
                av += [non_stim_dff[t][n, 16:19]]
                
            control_neuron_dff += [np.mean(av)]
            
        
        for i in range(1, len(set(self.stim_level))):
            stimlevel = sorted(list(set(self.stim_level)))[i]
            stim_dff = self.dff[0][self.stim_level == stimlevel]
            level = []
            
            for n in range(self.num_neurons):
                av = []
                
                for t in range(stim_dff.shape[0]):
                    av += [stim_dff[t][n, 16:19]]
                    
                level += [np.mean(av)]
            
            opto_neuron_dff += [level]
            
        # for i in range(len(set(self.stim_level))-1):
            
            # ratio = [opto_neuron_dff[i-1][j] / control_neuron_dff[j] for j in range(len(control_neuron_dff))]
            ratio = [level[j] / control_neuron_dff[j] for j in range(len(control_neuron_dff))]
            ratio = np.array(ratio)[np.array(ratio) > -100]
            ratio = np.array(ratio)[np.array(ratio) < 100]

            axarr[i-1].scatter(control_neuron_dff, level)
 
            axarr[i-1].set_title('{} AOM'.format(stimlevel))
            # axarr[i-1].hist(ratio, bins = 500)
            # axarr[i-1].set_xlim(-10,10)
        axarr[0].set_ylabel('Opto level')
        axarr[2].set_xlabel('Control Level')
        plt.show()
        return control_neuron_dff, ratio
    
    def plot_variance_spread(self):
    # Plot the variance of neurons as a histogram
    
        variance = []
        for n in range(self.num_neurons):
            
            unit = [self.dff[0, t][n, :self.time_cutoff] for t in range(self.num_trials)]
            
            variance += [np.var(unit)]
        variance = np.array(variance)
        plt.hist(variance[variance < 1.5], bins=100)
        
        return variance
            
            
            
        