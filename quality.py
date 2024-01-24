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

    def __init__(self, path, layer_num='all', guang=False, passive=False, quality=True):
        
        super().__init__(path, layer_num, guang, passive, quality=quality)
            
        
    ### Quality analysis section ###
            
    def all_neurons_heatmap(self, save=False, return_traces=False):
        
        # x = np.arange(-5.97,4,self.fs)[:self.time_cutoff] if 'CW03' not in self.path else np.arange(-6.97,4,self.fs)[:self.time_cutoff]


        f, axarr = plt.subplots(2,2, sharex='col')
        # x = np.arange(-5.97,4,self.fs)[:self.time_cutoff]
        delay = self.delay  # if 'CW03' not in self.path else self.delay-5
        # self.stim_ON = self.stim_ON[:-1]
        stimon, stimoff = [], []
        
        for i in range(len(self.stim_ON)):
            
            stimon += [self.stim_ON[i]] if i in self.i_good_trials else [False]
            stimoff += [~self.stim_ON[i]] if i in self.i_good_trials else [False]
        
        # stimon = [self.stim_ON[i] for i in range(len(self.stim_ON)) if i in self.i_good_trials else False]
        # stimoff = [~self.stim_ON[i] for i in range(len(self.stim_ON)) if i in self.i_good_trials else False]

        stim_dff = self.dff[0][stimon]
        non_stim_dff = self.dff[0][stimoff]

        stack = np.zeros(self.time_cutoff) # if 'CW03' not in self.path else np.zeros(self.time_cutoff-5)

        # for neuron in range(self.num_neurons):
        for neuron in self.good_neurons:
            dfftrial = []
            for trial in range(len(stim_dff)):

                dfftrial += [stim_dff[trial][neuron, :self.time_cutoff]]

            stack = np.vstack((stack, np.mean(np.array(dfftrial), axis=0)))

        stack = normalize(stack[1:])
        axarr[0,0].matshow(stack, cmap='gray', interpolation='nearest', aspect='auto')
        axarr[0,0].axis('off')
        axarr[0,0].set_title('Opto')
        axarr[0,0].axvline(x=delay, c='b', linewidth = 0.5)
        axarr[1,0].plot(np.mean(stack, axis = 0))
        # axarr[1,0].set_ylim(top=self.fs)
        axarr[1,0].axvline(x=delay, c='b', linewidth = 0.5)
        # axarr[1,0].set_xticks(range(0,stack.shape[1], 10), [int(d) for d in x[::10]])
        
        stack = np.zeros(self.time_cutoff) # if 'CW03' not in self.path else np.zeros(self.time_cutoff-5)

        # for neuron in range(self.num_neurons):
        for neuron in self.good_neurons:

            dfftrial = []
            for trial in range(len(non_stim_dff)):

                dfftrial += [non_stim_dff[trial][neuron, :self.time_cutoff]]

            stack = np.vstack((stack, np.mean(np.array(dfftrial), axis=0)))

        stack = normalize(stack[1:])
        axarr[0,1].matshow(stack, cmap='gray', interpolation='nearest', aspect='auto')
        axarr[0,1].axis('off')
        axarr[0,1].set_title('Control')

        axarr[1,1].plot(np.mean(stack, axis = 0))
        # axarr[1,1].set_ylim(top=0.2)
        axarr[1,0].set_ylabel('dF/F0')
        # axarr[1,1].set_xticks(range(0,stack.shape[1], 10), [int(d) for d in x[::10]])
        axarr[1,0].set_xlabel('Time from Go cue (s)')

        if save:
            plt.savefig(self.path + r'dff_contra_stimall.jpg')

        plt.show()
        
        
        # Second plot
        stack = np.zeros(self.time_cutoff) # if 'CW03' not in self.path else np.zeros(self.time_cutoff-5)
        stimstack = np.zeros(self.time_cutoff) # if 'CW03' not in self.path else np.zeros(self.time_cutoff-5)

        # for neuron in range(self.num_neurons):
        for neuron in self.good_neurons:

            dfftrial = []
            stimdfftrial = []
            for trial in range(len(non_stim_dff)):

                dfftrial += [non_stim_dff[trial][neuron, :self.time_cutoff]]
            for trial in range(len(stim_dff)):
                
                stimdfftrial += [stim_dff[trial][neuron, :self.time_cutoff]]

            stack = np.vstack((stack, np.mean(np.array(dfftrial), axis=0)))
            stimstack = np.vstack((stimstack, np.mean(np.array(stimdfftrial), axis=0)))

        # stack = normalize(stack[1:])
        # stimstack = normalize(stimstack[1:])

        plt.plot(np.mean(stimstack, axis = 0), 'r')
        # plt.set_ylim(top=self.fs)
        plt.axvline(x=delay, c='b', linewidth = 0.5)
        
        plt.plot(np.mean(stack, axis = 0), 'b')
        # plt.set_ylim(top=0.2)
        plt.ylabel('dF/F0')
        # axarr[1,1].set_xticks(range(0,stack.shape[1], 10), [int(d) for d in x[::10]])
        plt.xlabel('Time from Go cue (s)')
        plt.show()
        
        if return_traces:
            return stack, stimstack
    
    def all_neurons_traces(self, save = False):
        
        x = np.arange(-5.97,4,self.fs)[:self.time_cutoff] if 'CW03' not in self.path else np.arange(-5.97,4,self.fs)[:self.time_cutoff-5]


        f, axarr = plt.subplots(2,1, sharex='col',figsize=(20,10))
        # x = np.arange(-5.97,4,self.fs)[:self.time_cutoff]
        delay = self.delay if 'CW03' not in self.path else self.delay-5
        
        stimon, stimoff = [], []
        
        for i in range(len(self.stim_ON)):
            
            stimon += [self.stim_ON[i]] if i in self.i_good_trials else [False]
            stimoff += [~self.stim_ON[i]] if i in self.i_good_trials else [False]
        
        # stimon = [self.stim_ON[i] for i in range(len(self.stim_ON)) if i in self.i_good_trials else False]
        # stimoff = [~self.stim_ON[i] for i in range(len(self.stim_ON)) if i in self.i_good_trials else False]

        stim_dff = self.dff[0][stimon]
        non_stim_dff = self.dff[0][stimoff]

        stack = np.zeros(self.time_cutoff) if 'CW03' not in self.path else np.zeros(self.time_cutoff-5)

        # for neuron in range(self.num_neurons):
        for neuron in self.good_neurons:
            dfftrial = []
            for trial in range(len(stim_dff)):
                if 'CW03' in self.path:
                    dfftrial += [stim_dff[trial][neuron, 5:self.time_cutoff]]
                else:
                    dfftrial += [stim_dff[trial][neuron, :self.time_cutoff]]

            stack = np.vstack((stack, np.mean(np.array(dfftrial), axis=0)))

        stack = normalize(stack[1:])
        axarr[1].plot(stack.T, alpha=0.5)
        # axarr[0,0].axis('off')
        axarr[1].set_title('Opto')
        axarr[1].axvline(x=delay, c='b', linewidth = 0.5)
        # axarr[1,0].plot(np.mean(stack, axis = 0))
        # axarr[1,0].set_ylim(top=0.2)
        # axarr[1,0].axvline(x=delay, c='b', linewidth = 0.5)
        # axarr[1,0].set_xticks(range(0,stack.shape[1], 10), [int(d) for d in x[::10]])
        
        stack = np.zeros(self.time_cutoff) if 'CW03' not in self.path else np.zeros(self.time_cutoff-5)

        # for neuron in range(self.num_neurons):
        for neuron in self.good_neurons:

            dfftrial = []
            for trial in range(len(non_stim_dff)):
                if 'CW03' in self.path:
                    dfftrial += [non_stim_dff[trial][neuron, 5:self.time_cutoff]]
                else:
                    dfftrial += [non_stim_dff[trial][neuron, :self.time_cutoff]]

            stack = np.vstack((stack, np.mean(np.array(dfftrial), axis=0)))

        stack = normalize(stack[1:])

        axarr[0].plot(stack.T, alpha=0.5)
        # axarr[0,1].axis('off')
        axarr[0].set_title('Control')

        # axarr[1,1].plot(np.mean(stack, axis = 0))
        # axarr[1,1].set_ylim(top=0.2)
        axarr[0].set_ylabel('dF/F0')
        # axarr[1,1].set_xticks(range(0,stack.shape[1], 10), [int(d) for d in x[::10]])
        axarr[1].set_xlabel('Time (s)')

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
    
            for neuron in range(self.num_neurons):
                dfftrial = []
                for trial in self.i_good_trials:
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

        for neuron in range(self.num_neurons):
            dfftrial = []
            for trial in self.i_good_trials:
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
    
    def stim_activity_proportion(self, stim_period = range(16,19), save=False):
        
        powers = len(set(self.stim_level))
        f, axarr = plt.subplots(1, powers, sharex='col', figsize = ((powers)*5, 4))
        # x = np.arange(-5.97,4,self.fs)[:self.time_cutoff]
        
        control_neuron_dff = []
        opto_neuron_dff = []

        # nonstiminds = np.where(self.stim_level == 0)[0]
        # nonstiminds = [n for n in nonstiminds if n in self.i_good_trials]

        non_stim_dff = self.dff[0][self.stim_level == 0]
        # non_stim_dff = self.dff[0][nonstiminds]
        
        for n in range(self.num_neurons):
            av = []
            
            for t in range(non_stim_dff.shape[0]):
                av += [non_stim_dff[t][n, stim_period]]
                
            control_neuron_dff += [np.mean(av)]
            
        
        for i in range(1, len(set(self.stim_level))):
            stimlevel = sorted(list(set(self.stim_level)))[i]
            # stiminds = np.where(self.stim_level == stimlevel)[0]
            # stiminds = [n for n in stiminds if n in self.i_good_trials]

            # stim_dff = self.dff[0][stiminds]
            stim_dff = self.dff[0][self.stim_level == stimlevel]
            level = []
            
            for n in range(self.num_neurons):
                av = []
                
                # for t in self.i_good_trials:
                for t in range(len(stim_dff)):
                    av += [stim_dff[t][n, stim_period]]
                    
                level += [np.mean(av)]
            
            opto_neuron_dff += [level]
            
        # for i in range(len(set(self.stim_level))-1):
            
            # ratio = [opto_neuron_dff[i-1][j] / control_neuron_dff[j] for j in range(len(control_neuron_dff))]
            ratio = [level[j] / control_neuron_dff[j] for j in range(len(control_neuron_dff))]
            # ratio = np.array(ratio)[np.array(ratio) > -100]
            # ratio = np.array(ratio)[np.array(ratio) < 100]

            axarr[i-1].scatter(control_neuron_dff, level)
 
            axarr[i-1].set_title('{} AOM'.format(stimlevel))
            axarr[i-1].plot(range(-25,100), range(-25,100), 'r')
            # axarr[i-1].plot(range(-2,2), range(-2,2), 'r')
            # axarr[i-1].hist(ratio, bins = 500)
            axarr[i-1].set_xlim(min(control_neuron_dff)-0.1,max(control_neuron_dff)+0.1)
            axarr[i-1].set_ylim(min(level)-0.1,max(level)+0.1)
            
            
        axarr[0].set_ylabel('Opto level')
        axarr[0].set_xlabel('Control Level')
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
            
    def plot_pearsons_correlation(self):
        
        neurons, corrs = self.get_pearsonscorr_neuron()
        
        plt.hist(corrs)
        plt.show()
        
    def plot_background(self):
        
        """
        Plot the five background traces (one for each layer) average over trials
        separated by control vs opto trials, with the stim period highlighted
        
        """
        f, axarr = plt.subplots(5,1, sharex='col', figsize=(10, 10))
        x = np.arange(-6.97,4,self.fs)[:self.time_cutoff]
        window = range(12,38)
        for plane in range(5):
            plane_av_control = []
            for t in np.where(~self.stim_ON)[0]:
                plane_av_control += [self.background[0, t][plane, window]]
            
            plane_av_opto = []
            for t in np.where(self.stim_ON)[0]:
                plane_av_opto += [self.background[0, t][plane, window]]
                # axarr[plane].plot(self.background[0, t][plane, window], color='red', alpha = 0.3)

                
            axarr[plane].plot(np.mean(plane_av_control, axis=0), color='darkgrey', label='Control')
            axarr[plane].plot(np.mean(plane_av_opto, axis=0), color='red', label='Opto')
            axarr[plane].axvline(self.sample-12, ls = '--', color = 'grey')
            axarr[plane].axvline(self.delay-12, ls = '--', color = 'red')
            axarr[plane].axvline(self.delay-12+6, ls = '--', color = 'red')
            # axarr[plane].axvline(self.response-12, ls = '--', color = 'grey')
            axarr[plane].set_title('F_background (plane {})'.format(plane+1))
            
        plt.legend()
        plt.show()
        
        # return plane_av_control, plane_av_opto
    def plot_background_and_traces(self, return_traces=False,  single_layer=False, no_background=False):
        """
        Plots traces with ROI and neuropil and background

        Returns
        -------
        None.

        """
        if not no_background:
            overall_background = []
            if single_layer:
                for t in np.where(self.stim_ON)[0]:
                    overall_background += [self.background[0, t][0, :self.time_cutoff]]
            else:
                for plane in range(5):
                    for t in np.where(self.stim_ON)[0]:
                        overall_background += [self.background[0, t][plane, :self.time_cutoff]]
            overall_background = np.mean(overall_background, axis=0)
        else:
            overall_background = np.zeros(61)
        
        overall_npil = []
        for n in self.good_neurons:
            for t in np.where(self.stim_ON)[0]:
                overall_npil += [self.npil[0, t][n, :self.time_cutoff]]
        overall_npil = np.mean(overall_npil, axis=0)
        
        overall_F = []
        for n in self.good_neurons:
            for t in np.where(self.stim_ON)[0]:
                overall_F += [self.dff[0, t][n, :self.time_cutoff]]
        overall_F = np.mean(overall_F, axis=0)
        
        if return_traces:
            return ((overall_background[12:38] - np.mean(overall_background[12:38])) / np.std(overall_background[12:38]),
                    (overall_npil[12:38] - np.mean(overall_npil[12:38])) / np.std(overall_npil[12:38]),
                    (overall_F[12:38] - np.mean(overall_F[12:38])) / np.std(overall_F[12:38])
                    )
            
        plt.plot((overall_background[12:38] - np.mean(overall_background[12:38]))/np.std(overall_background[12:38]))
        plt.plot((overall_npil[12:38] - np.mean(overall_npil[12:38]))/np.std(overall_npil[12:38]))
        # plt.plot(overall_F[12:38])
        # plt.plot(overall_F)
        # plt.plot(overall_npil[12:38])

        plt.show()
        

        