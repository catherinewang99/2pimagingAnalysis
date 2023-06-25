# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 10:46:39 2023

@author: Catherine Wang
"""
# from neuralFuncs import plot_PSTH
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
from scipy.stats import mstats

class Session:
    
    def __init__(self, path, layer_num='all', guang=False, passive=False):
        
        if layer_num != 'all':
            
            layer_og = scio.loadmat(r'{}\layer_{}.mat'.format(path, layer_num))
            layer = copy.deepcopy(layer_og)
            self.dff = layer['dff']

        else:
            # Load all layers
            self.dff = None
            for layer in os.listdir(path):
                if 'layer' in layer:
                    layer_og = scio.loadmat(r'{}\{}'.format(path, layer))
                    layer = copy.deepcopy(layer_og)
                    
                    if self.dff == None:
                        
                        
                        self.dff = layer['dff']
                        self.num_trials = layer['dff'].shape[1] 
                    else:

                        for t in range(self.num_trials):
                            add = layer['dff'][0, t]
                            self.dff[0, t] = np.vstack((self.dff[0, t], add))
                        
        
        behavior = scio.loadmat(r'{}\behavior.mat'.format(path))
        self.path = path
        self.layer_num = layer_num
        self.passive = passive
        self.num_neurons = self.dff[0,0].shape[0]

        self.num_trials = self.dff.shape[1] 
        
        self.time_cutoff = self.determine_cutoff()
        
        self.recording_loc = 'l'
        # self.skew = layer['skew']
        
        # self.good_neurons = np.where(self.skew>=1)[1]
        
        if passive:
            self.i_good_trials = range(4, self.num_trials)
        else:
            self.i_good_trials = cat(behavior['i_good_trials']) - 1 # zero indexing in python
        
        self.L_correct = cat(behavior['L_hit_tmp'])
        self.R_correct = cat(behavior['R_hit_tmp'])
        
        self.early_lick = cat(behavior['LickEarly_tmp'])
        
        self.L_wrong = cat(behavior['L_miss_tmp'])
        self.R_wrong = cat(behavior['R_miss_tmp'])
        
        self.L_ignore = cat(behavior['L_ignore_tmp'])
        self.R_ignore = cat(behavior['R_ignore_tmp'])
                
        self.stim_ON = cat(behavior['StimDur_tmp']) > 0
        if 'StimLevel' in behavior.keys():
            self.stim_level = cat(behavior['StimLevel'])
            
        if self.i_good_trials[-1] > self.num_trials:
            
            print('More Bpod trials than 2 photon trials')
            self.i_good_trials = [i for i in self.i_good_trials if i < self.num_trials]
            self.stim_ON = self.stim_ON[:self.num_trials]
            
        self.sample = 7
        self.delay = 13
        self.response = 28
        if 'CW030' in path:
            self.sample += 5
            self.delay += 5
            self.response += 5
        
        # Measure that automatically crops out water leak trials before norming
        if not self.find_low_mean_F():

            self.plot_mean_F()

            if guang:
                # Guang's data
                self.num_neurons = layer['dff'][0,0].shape[1]  # Guang's data
    
                for t in range(self.num_trials):
                    self.dff[0, t] = self.dff[0, t].T
            else:
                self.normalize_all_by_neural_baseline()
                # self.normalize_by_histogram()
                # self.normalize_all_by_histogram()
                # self.normalize_all_by_baseline()
                self.normalize_z_score()    



        self.good_neurons, _ = self.get_pearsonscorr_neuron()
        # self.num_neurons = len(self.good_neurons)
        self.old_i_good_trials = copy.copy(self.i_good_trials)

        
    def crop_baseline(self):
        # BROKEN :()
        
        dff = copy.deepcopy(self.dff)
        dff_copy = np.array([]).reshape(1,-1)

        # for i in self.good_neurons:
            
            # nmean = np.mean([self.dff[0, t][i, :7] for t in range(self.num_trials)]).copy()
            
        for j in range(self.num_trials):
            
            trialdff = np.array([])
            
            for i in range(self.num_neurons):

            # for j in self.i_good_trials:
                newdff = dff[0, j][i, 5:] # later cutoff because of transient activation
                # dff[0, j][i] = newdff
                if i == 0:
                    trialdff = newdff
                else:
                    trialdff = np.vstack((trialdff, newdff))
            
            if j==0:
                
                dff_copy = trialdff
            else:
                dff_copy = np.hstack((dff_copy, trialdff))
            
        self.dff=dff_copy
        
    def determine_cutoff(self):
        
        cutoff = 1e10
        
        for t in range(self.num_trials):
            
            if self.dff[0, t].shape[1] < cutoff:
                
                cutoff = self.dff[0, t].shape[1]
        
        print("Minimum cutoff is {}".format(cutoff))
        
        return cutoff
    
    def find_low_mean_F(self, cutoff = 50):
        
        # Usual cutoff is 50
        # Reject outliers based on medians
        meanf = np.array([])
        for trial in range(self.num_trials):
            meanf = np.append(meanf, np.mean(cat(self.dff[0, trial])))
        
        med = np.median(meanf) # median approach
        
        trial_idx = np.where(meanf < cutoff)[0]
        
        if trial_idx.size == 0:
            
            return 0
        
        else:
            print('Water leak trials: {}'.format(trial_idx))

            self.crop_trials(0, singles=True, arr = trial_idx)
            return 1
        
    def reject_outliers(data, m = 2.):
        
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d/mdev if mdev else np.zero(len(d))
        
        return data[s<m]
    
    def get_pearsonscorr_neuron(self, cutoff = 0.7):
        
        # Only returns neurons that have a consistent trace over all trials
        
        neurons = []
        evens, odds = [], []
        corrs = []
        # inds = [i for i in range(self.num_trials) if self.stim_ON[i] == 0]
        for i in self.i_good_trials:
            if i%2 == 0: # Even trials
                evens += [i]
            else:
                odds += [i]
        filtert = len(evens) if len(evens) < len(odds) else len(odds)
        evens, odds = evens[:filtert], odds[:filtert] # Ensure number of trials is the same
        
        for neuron_num in range(self.num_neurons):
            R_av_dff = []
            for i in evens:
                # R_av_dff += [self.normalize_by_baseline(self.dff[0, i][neuron_num, :self.time_cutoff])]
                R_av_dff += [self.dff[0, i][neuron_num, :self.time_cutoff]]
    
            L_av_dff = []
            for i in odds:
                # L_av_dff += [self.normalize_by_baseline(self.dff[0, i][neuron_num, :self.time_cutoff])]
                L_av_dff += [self.dff[0, i][neuron_num, :self.time_cutoff]]
                
            corr, p = mstats.pearsonr(np.mean(L_av_dff, axis = 0), np.mean(R_av_dff,axis=0))
            corrs += [corr]
            if corr > cutoff:
                
                neurons += [neuron_num]
            
        return neurons, corrs
    
    def plot_mean_F(self):
        
        # Plots mean F for all neurons over trials in session
        meanf = list()
        # for trial in range(self.num_trials):
        for trial in self.i_good_trials:
            meanf.append(np.mean(cat(self.dff[0, trial])))
        
        plt.plot(self.i_good_trials, meanf, 'b-')
        plt.title("Mean F for layer {}".format(self.layer_num))
        plt.show()

    def crop_trials(self, trial_num, end=False, singles = False, arr = []):
        
        # If called, crops out all trials after given trial number
        # Can optionally crop from trial_num to end indices
        
        if not end and not singles:
            
            # self.L_correct = self.L_correct[:trial_num]
            # self.R_correct = self.R_correct[:trial_num]
            # self.L_wrong = self.L_wrong[:trial_num]
            # self.R_wrong = self.R_wrong[:trial_num]
            
            # self.dff = self.dff[:, :trial_num]
            self.i_good_trials = [i for i in self.i_good_trials if i < trial_num]
            self.num_trials = trial_num
            # self.stim_ON = self.stim_ON[:trial_num]
            if self.passive:
                self.stim_level = self.stim_level[:trial_num]
            # self.normalize_all_by_baseline()
            # self.normalize_z_score()    

            
            # self.plot_mean_F()
            
        
        elif singles:
            
            # self.L_correct = np.delete(self.L_correct, arr)
            # self.R_correct = np.delete(self.R_correct, arr)
            # self.L_wrong = np.delete(self.L_wrong, arr)
            # self.R_wrong = np.delete(self.R_wrong, arr)
            
            # self.dff = np.delete(self.dff, arr)
            # self.dff = np.reshape(self.dff, (1,-1))

            igoodremove = np.where(np.in1d(self.i_good_trials, arr))[0]
            self.i_good_trials = np.delete(self.i_good_trials, igoodremove)
            self.num_trials = self.num_trials - len(arr)            
            # self.stim_ON = np.delete(self.stim_ON, arr)
            if self.passive:
                self.stim_level = np.delete(self.stim_level, arr)
            # self.normalize_all_by_baseline()
            # self.normalize_z_score()   

            # self.plot_mean_F()
            
        else:
            
            arr = np.arange(trial_num, end)

            # self.L_correct = np.delete(self.L_correct, arr)
            # self.R_correct = np.delete(self.R_correct, arr)
            # self.L_wrong = np.delete(self.L_wrong, arr)
            # self.R_wrong = np.delete(self.R_wrong, arr)
            
            # self.dff = np.delete(self.dff, arr)
            # self.dff = np.reshape(self.dff, (1,-1))

            igoodremove = np.where(np.in1d(self.i_good_trials, arr))[0]
            self.i_good_trials = np.delete(self.i_good_trials, igoodremove)
            self.num_trials = self.num_trials - len(arr)            
            # self.stim_ON = np.delete(self.stim_ON, arr)
            if self.passive:
                self.stim_level = np.delete(self.stim_level, arr)

            # self.i_good_trials = [i for i in self.i_good_trials if i < trial_num or i > end]
            # self.num_trials = trial_num            
            # self.stim_ON = np.append(self.stim_ON[:trial_num], self.stim_ON[end:])

            # self.normalize_all_by_baseline()
            # self.normalize_z_score()   

            # self.plot_mean_F()

        self.plot_mean_F()

        # self.normalize_all_by_baseline()
        self.normalize_all_by_neural_baseline()

        self.normalize_z_score()    
        
        
        print('New number of good trials: {}'.format(len(self.i_good_trials)))
    
    def lick_correct_direction(self, direction):
        ## Returns list of indices of lick left correct trials
        
        if direction == 'l':
            idx = np.where(self.L_correct == 1)[0]
        elif direction == 'r':
            idx = np.where(self.R_correct == 1)[0]
        else:
            raise Exception("Sorry, only 'r' or 'l' input accepted!")
            
        early_idx = np.where(self.early_lick == 1)[0]
        
        idx = [i for i in idx if i not in early_idx]
        
        idx = [i for i in idx if i in self.i_good_trials]
        
        return idx
    
    def lick_incorrect_direction(self, direction):
        ## Returns list of indices of lick left correct trials
        
        if direction == 'l':
            idx = np.where(self.L_wrong == 1)[0]
        elif direction == 'r':
            idx = np.where(self.R_wrong == 1)[0]
        else:
            raise Exception("Sorry, only 'r' or 'l' input accepted!")
            
        early_idx = np.where(self.early_lick == 1)[0]
        
        idx = [i for i in idx if i not in early_idx]
        
        idx = [i for i in idx if i in self.i_good_trials]
        
        return idx
    
    def get_trace_matrix(self, neuron_num, error=False, bias_trials = [], non_bias=False, both=False):
        
        ## Returns matrix of all trial firing rates of a single neuron for lick left
        ## and lick right trials. Firing rates are normalized with individual trial
        ## baselines as well as overall firing rate z-score normalized.
        
        right_trials = self.lick_correct_direction('r')
        left_trials = self.lick_correct_direction('l')
        
        if error:
            right_trials = self.lick_incorrect_direction('r')
            left_trials = self.lick_incorrect_direction('l')
        
        if both:
            right_trials = cat((self.lick_correct_direction('r'), self.lick_incorrect_direction('r')))
            left_trials = cat((self.lick_correct_direction('l'), self.lick_incorrect_direction('l')))
        
        if len(bias_trials) != 0:
            right_trials = [b for b in bias_trials if self.instructed_side[b] == 0]
            left_trials = [b for b in bias_trials if self.instructed_side[b] == 1]
            # print(right_trials)
            # print(left_trials)
            if non_bias: # Get control trials - bias trials
            
                ctlright_trials = self.lick_correct_direction('r')
                ctlleft_trials = self.lick_correct_direction('l')
                right_trials = [b for b in ctlright_trials if b not in bias_trials]
                left_trials = [b for b in ctlleft_trials if b not in bias_trials]
                # print(right_trials)
                # print(left_trials)
            
        # Filter out opto trials
        right_trials = [r for r in right_trials if not self.stim_ON[r]]
        left_trials = [r for r in left_trials if not self.stim_ON[r]]
        
        R_av_dff = []
        for i in right_trials:
            # R_av_dff += [self.normalize_by_baseline(self.dff[0, i][neuron_num, :self.time_cutoff])]
            R_av_dff += [self.dff[0, i][neuron_num, :self.time_cutoff]]

        L_av_dff = []
        for i in left_trials:
            # L_av_dff += [self.normalize_by_baseline(self.dff[0, i][neuron_num, :self.time_cutoff])]
            L_av_dff += [self.dff[0, i][neuron_num, :self.time_cutoff]]
            
            
        return R_av_dff, L_av_dff
    
    def get_opto_trace_matrix(self, neuron_num, error=False):
        
        
        right_trials = self.lick_correct_direction('r')
        left_trials = self.lick_correct_direction('l')
        
        if error:
            right_trials = self.lick_incorrect_direction('r')
            left_trials = self.lick_incorrect_direction('l')
        
        # Filter for opto trials
        right_trials = [r for r in right_trials if self.stim_ON[r]]
        left_trials = [r for r in left_trials if self.stim_ON[r]]

        
        R_av_dff = []
        for i in right_trials:
            
            R_av_dff += [self.dff[0, i][neuron_num, :self.time_cutoff]]
        
        L_av_dff = []
        for i in left_trials:

            L_av_dff += [self.dff[0, i][neuron_num, :self.time_cutoff]]
            
        
            
        return R_av_dff, L_av_dff
    
    def get_trace_matrix_multiple(self, neuron_nums, opto=False, error=False, both=False, bias_trials = None, non_bias=False):
        
        ## Returns matrix of average firing rates of a list of neurons for lick left
        ## and lick right trials. Firing rates are normalized with individual trial
        ## baselines as well as overall firing rate z-score normalized.
        
        R, L = [], []
        
        for neuron_num in neuron_nums:
            if not opto:
                R_av_dff, L_av_dff = self.get_trace_matrix(neuron_num, error=error, bias_trials = bias_trials, non_bias=non_bias, both=both)
            else:
                R_av_dff, L_av_dff = self.get_opto_trace_matrix(neuron_num, error=error)
            # if both:
            #     right_trials = cat((self.lick_correct_direction('r'), self.lick_incorrect_direction('r')))
            #     left_trials = cat((self.lick_correct_direction('l'), self.lick_incorrect_direction('l')))
            
            # elif not error:
            #     right_trials = self.lick_correct_direction('r')
            #     left_trials = self.lick_correct_direction('l')
            # elif error:
            #     right_trials = self.lick_incorrect_direction('r')
            #     left_trials = self.lick_incorrect_direction('l')
                
            # # Filter out opto trials
            # if not opto:
            #     right_trials = [r for r in right_trials if not self.stim_ON[r]]
            #     left_trials = [r for r in left_trials if not self.stim_ON[r]]
            # elif opto:
            #     right_trials = [r for r in right_trials if self.stim_ON[r]]
            #     left_trials = [r for r in left_trials if self.stim_ON[r]]           
                
            
            # R_av_dff = []
            # for i in right_trials:
            #     # R_av_dff += [self.normalize_by_baseline(self.dff[0, i][neuron_num, :self.time_cutoff])]
            #     R_av_dff += [self.dff[0, i][neuron_num, :self.time_cutoff]]
    
            # L_av_dff = []
            # for i in left_trials:
            #     # L_av_dff += [self.normalize_by_baseline(self.dff[0, i][neuron_num, :self.time_cutoff])]
            #     L_av_dff += [self.dff[0, i][neuron_num, :self.time_cutoff]]
            
            R += [np.mean(R_av_dff, axis = 0)]
            L += [np.mean(L_av_dff, axis = 0)]
            
        return np.array(R), np.array(L)

    def plot_PSTH(self, neuron_num, opto = False):
        
        ## Plots single neuron PSTH for R/L trials
        
        if not opto:
            R, L = self.get_trace_matrix(neuron_num)
        else:
            R, L = self.get_opto_trace_matrix(neuron_num)

        
        R_av, L_av = np.mean(R, axis = 0), np.mean(L, axis = 0)
        
        left_err = np.std(L, axis=0) / np.sqrt(len(L)) 
        right_err = np.std(R, axis=0) / np.sqrt(len(R))
                    
        plt.plot(L_av, 'r-')
        plt.plot(R_av, 'b-')
        
        x = range(self.time_cutoff)

        plt.fill_between(x, L_av - left_err, 
                 L_av + left_err,
                 color=['#ffaeb1'])
        plt.fill_between(x, R_av - right_err, 
                 R_av + right_err,
                 color=['#b4b2dc'])
        plt.title("Neuron {}: R/L PSTH".format(neuron_num))
        plt.show()

    def plot_single_trial_PSTH(self, trial, neuron_num):
        
        plt.plot(self.dff[0, trial][neuron_num], 'b-')
        plt.title("Neuron {}: PSTH for trial {}".format(neuron_num, trial))
        plt.show()

    def plot_population_PSTH(self, neurons, opto = False):
        
        # Returns the population average activity for L/R trials of these neurons
        
        overall_R = []
        overall_L = []

        for neuron_num in neurons:
            
            if not opto:
                R, L = self.get_trace_matrix(neuron_num)
            else:
                R, L = self.get_opto_trace_matrix(neuron_num)
                
            overall_R += R
            overall_L += L
        
        R_av, L_av = np.mean(overall_R, axis = 0), np.mean(overall_L, axis = 0)
        
        left_err = np.std(overall_L, axis=0) / np.sqrt(len(overall_L)) 
        right_err = np.std(overall_R, axis=0) / np.sqrt(len(overall_R))
                    
        plt.plot(L_av, 'r-')
        plt.plot(R_av, 'b-')
        
        x = range(self.time_cutoff)

        plt.fill_between(x, L_av - left_err, 
                 L_av + left_err,
                 color=['#ffaeb1'])
        plt.fill_between(x, R_av - right_err, 
                 R_av + right_err,
                 color=['#b4b2dc'])
        plt.title("Neural population PSTH")
        plt.show()

    def normalize_by_baseline(self, trace):
        
        # Function to normalize by first 7 time points
        
        # return trace
        
        mean = np.mean(trace[:7])
        if mean == 0:
            raise Exception("Neuron has mean 0.")
            
        return (trace - mean) / mean # norm by F0
    
    def normalize_all_by_neural_baseline(self):
        
        # Normalize all neurons by neural trial-averaged F0
        
        for i in range(self.num_neurons):
        # for i in self.good_neurons:

            nmean = np.mean([self.dff[0, t][i, self.sample-3:self.sample] for t in range(self.num_trials)]).copy()
            # nmean = np.mean([self.dff[0, t][i, self.sample-3:self.sample] for t in self.i_good_trials]).copy()
            
            for j in range(self.num_trials):
            # for j in self.i_good_trials:
                
                # nmean = np.mean(self.dff[0, j][i, :7])
                self.dff[0, j][i] = (self.dff[0, j][i] - nmean) / nmean
        
        return None
    
    def normalize_all_by_baseline(self):
        
        # Normalize all neurons by individual trial-averaged F0
        
        # dff = copy.deepcopy(self.dff)
        dff = self.dff.copy()

        for i in range(self.num_neurons):
        # for i in self.good_neurons:
            
            # nmean = np.mean([self.dff[0, t][i, :7] for t in range(self.num_trials)]).copy()
            
            for j in range(self.num_trials):
            # for j in self.i_good_trials:

                nmean = np.mean(dff[0, j][i, self.sample-3:self.sample]) # later cutoff because of transient activation
                self.dff[0, j][i, :] = (self.dff[0, j][i] - nmean) / nmean
        # self.dff = dff
        return None

    def normalize_by_histogram(self):
        
        # Normalize all neurons by individual trial-averaged F0
        
        for i in range(self.num_neurons):
        # for i in self.good_neurons:
            
            # nmean = np.quantile(cat([self.dff[0,t][i, :] for t in range(self.num_trials)]), q=0.10)
            nmean = np.quantile(cat([self.dff[0,t][i, :] for t in self.i_good_trials]), q=0.10)
            
            # for j in range(self.num_trials):
            for j in self.i_good_trials:

                self.dff[0, j][i] = (self.dff[0, j][i] - nmean) / nmean
        
        return None
    
    def normalize_all_by_histogram(self):
        
        # Normalize all neurons by individual trial-averaged F0
        
        for i in range(self.num_neurons):
                        
            # for j in range(self.num_trials):
            for j in self.i_good_trials:
                nmean = np.quantile(self.dff[0, j][i, :], q=0.10)

                self.dff[0, j][i] = (self.dff[0, j][i] - nmean) / nmean
        
        return None
    
    def normalize_z_score(self):
        
        # Normalize by mean of all neurons in layer
        
        # overall_mean = np.mean(cat([cat(i) for i in self.dff[0]])).copy()
        # std = np.std(cat([cat(i) for i in self.dff[0]])).copy()

        overall_mean = np.mean(cat([cat(self.dff[0, i]) for i in self.i_good_trials])).copy()
        std = np.std(cat([cat(self.dff[0, i]) for i in self.i_good_trials])).copy()
        
        # for i in range(self.num_trials):
        for i in self.i_good_trials:
            for j in range(self.num_neurons):
                self.dff[0, i][j] = (self.dff[0, i][j] - overall_mean) / std
                
        # self.dff = normalize(self.dff)
        
        return None

    def get_epoch_selective(self, epoch, p = 0.01, bias=False):
        selective_neurons = []
        # for neuron in range(self.num_neurons):
        for neuron in self.good_neurons:
            right, left = self.get_trace_matrix(neuron)
            
            if bias:
                biasidx = self.find_bias_trials()
                right,left = self.get_trace_matrix(neuron, bias_trials= biasidx)
            
            left_ = [l[epoch] for l in left]
            right_ = [r[epoch] for r in right]
            tstat, p_val = stats.ttest_ind(np.mean(left_, axis = 1), np.mean(right_, axis = 1))
            # p = 0.01/self.num_neurons
            # p = 0.01
            # p = 0.0001
            if p_val < p:
                selective_neurons += [neuron]
        # print("Total delay selective neurons: ", len(selective_neurons))
        self.selective_neurons = selective_neurons
        return selective_neurons
   
    
    def screen_preference(self, neuron_num, epoch, samplesize = 10):

        # Input: neuron of interest
        # Output: (+) if left pref, (-) if right pref, then indices of trials to plot
        
        # All trials where the mouse licked left or right AND non stim
        
        R, L = self.get_trace_matrix(neuron_num)
        l_trials = range(len(L))  
        r_trials = range(len(R))
        
        # Skip neuron if less than 15
        if len(l_trials) < 15 or len(r_trials) < 15:
            raise Exception("Neuron {} has fewer than 15 trials in R or L lick trials".format(neuron_num))
            return 0
        
        # Pick 20 random trials as screen for left and right
        screen_l = np.random.choice(l_trials, size = samplesize, replace = False)
        screen_r = np.random.choice(r_trials, size = samplesize, replace = False)
    
        # Remainder of trials are left for plotting in left and right separately
        test_l = [t for t in l_trials if t not in screen_l]
        test_r = [t for t in r_trials if t not in screen_r]
        
        # Compare late delay epoch for preference
        avg_l = np.mean([np.mean(L[i][epoch]) for i in screen_l])
        avg_r = np.mean([np.mean(R[i][epoch]) for i in screen_r])
    
        return avg_l > avg_r, test_l, test_r

    def plot_selectivity(self, neuron_num, plot=True, epoch=range(21,28)):
        
        R, L = self.get_trace_matrix(neuron_num)
        pref, l, r = self.screen_preference(neuron_num, epoch)
        left_trace = [L[i] for i in l]
        right_trace = [R[i] for i in r]

        if pref: # prefers left
            sel = np.mean(left_trace, axis = 0) - np.mean(right_trace, axis=0)
        else:
            sel = np.mean(right_trace, axis = 0) - np.mean(left_trace, axis=0)
        
        if plot:
            direction = 'Left' if pref else 'Right'
            plt.plot(range(self.time_cutoff), sel, 'b-')
            plt.axhline(y=0)
            plt.title('Selectivity of neuron {}: {} selective'.format(neuron_num, direction))
            plt.show()
        
        return sel
    
    def contra_ipsi_pop(self, epoch):
        
        # Returns the neuron ids for contra and ipsi populations

        selective_neurons = self.get_epoch_selective(epoch)
        
        contra_neurons = []
        ipsi_neurons = []
        
        contra_LR, ipsi_LR = dict(), dict()
        contra_LR['l'], contra_LR['r'] = [], []
        ipsi_LR['l'], ipsi_LR['r'] = [], []
        
        
        for neuron_num in selective_neurons:
            
            # Skip sessions with fewer than 15 neurons
            if self.screen_preference(neuron_num, epoch) != 0:
                
                R, L = self.get_trace_matrix(neuron_num)

                pref, test_l, test_r = self.screen_preference(neuron_num, epoch) 
        
                if self.recording_loc == 'l':

                    if pref:
                        # print("Ipsi_preferring: {}".format(neuron_num))
                        ipsi_neurons += [neuron_num]
                        ipsi_LR['l'] += [[L[i] for i in test_l]]
                        ipsi_LR['r'] += [[R[i] for i in test_r]]
                    else:
                        # print("Contra preferring: {}".format(neuron_num))
                        contra_neurons += [neuron_num] 
                        contra_LR['l'] += [[L[i] for i in test_l]]
                        contra_LR['r'] += [[R[i] for i in test_r]]
                    
                elif self.recording_loc == 'r':

                    if not pref:
                        ipsi_neurons += [neuron_num]
                        ipsi_LR['l'] += [L[i] for i in test_l]
                        ipsi_LR['r'] += [R[i] for i in test_r]
                    else:
                        contra_neurons += [neuron_num] 
                        contra_LR['l'] += [L[i] for i in test_l]
                        contra_LR['r'] += [R[i] for i in test_r]
                        
        return contra_neurons, ipsi_neurons, contra_LR, ipsi_LR
    
    def plot_contra_ipsi_pop(self, e=False, bias=False):
        x = np.arange(-5.97,4,0.2)[:self.time_cutoff]

        epoch = e if e != False else range(self.delay, self.response)
        
        contra_neurons, ipsi_neurons, contra_trace, ipsi_trace = self.contra_ipsi_pop(epoch)
        
        if len(ipsi_neurons) != 0:
        
            overall_R, overall_L = ipsi_trace['r'], ipsi_trace['l']
            overall_R = [np.mean(overall_R[r], axis=0) for r in range(len(overall_R))]
            overall_L = [np.mean(overall_L[l], axis=0) for l in range(len(overall_L))]
            
            if bias:
                overall_R, overall_L = self.get_trace_matrix_multiple(ipsi_neurons, bias_trials=self.find_bias_trials())
            
            R_av, L_av = np.mean(overall_R, axis = 0), np.mean(overall_L, axis = 0)
            
            left_err = np.std(overall_L, axis=0) / np.sqrt(len(overall_L)) 
            right_err = np.std(overall_R, axis=0) / np.sqrt(len(overall_R))
                        
            plt.plot(x, L_av, 'r-')
            plt.plot(x, R_av, 'b-')
            
    
            plt.fill_between(x, L_av - left_err, 
                     L_av + left_err,
                     color=['#ffaeb1'])
            plt.fill_between(x, R_av - right_err, 
                     R_av + right_err,
                     color=['#b4b2dc'])
            plt.title("Ipsi-preferring neurons")
            plt.xlabel('Time from Go cue (s)')

            plt.show()
        
        else:
            print('No ipsi selective neurons')
    
        if len(contra_neurons) != 0:

            overall_R, overall_L = contra_trace['r'], contra_trace['l']
            overall_R = [np.mean(overall_R[r], axis=0) for r in range(len(overall_R))]
            overall_L = [np.mean(overall_L[l], axis=0) for l in range(len(overall_L))]
            
            if bias:
                overall_R, overall_L = self.get_trace_matrix_multiple(contra_neurons, bias_trials=self.find_bias_trials())
            
            R_av, L_av = np.mean(overall_R, axis = 0), np.mean(overall_L, axis = 0)
            
            left_err = np.std(overall_L, axis=0) / np.sqrt(len(overall_L)) 
            right_err = np.std(overall_R, axis=0) / np.sqrt(len(overall_R))
                        
            plt.plot(x, L_av, 'r-')
            plt.plot(x, R_av, 'b-')
            
    
            plt.fill_between(x, L_av - left_err, 
                      L_av + left_err,
                      color=['#ffaeb1'])
            plt.fill_between(x, R_av - right_err, 
                      R_av + right_err,
                      color=['#b4b2dc'])
            plt.title("Contra-preferring neurons")
            plt.xlabel('Time from Go cue (s)')
            plt.show()
        else:
            print('No contra selective neurons')
            
    def plot_prefer_nonprefer(self, e=False, bias=False):
        x = np.arange(-5.97,4,0.2)[:self.time_cutoff]

        epoch = e if e != False else range(self.delay, self.response)
        
        contra_neurons, ipsi_neurons, contra_trace, ipsi_trace = self.contra_ipsi_pop(epoch)
        
        pref, nonpref = [], []
        preferr, nonpreferr = [], []
        
        if len(ipsi_neurons) != 0:
        
            overall_R, overall_L = ipsi_trace['r'], ipsi_trace['l']
            overall_R = np.array([np.mean(overall_R[r], axis=0) for r in range(len(overall_R))])
            overall_L = np.array([np.mean(overall_L[l], axis=0) for l in range(len(overall_L))])
            
            if bias:
                overall_R, overall_L = self.get_trace_matrix_multiple(ipsi_neurons, bias_trials=self.find_bias_trials())
            
            pref, nonpref = overall_L, overall_R
            
        else:
            print('No ipsi selective neurons')
    
        if len(contra_neurons) != 0:

            overall_R, overall_L = contra_trace['r'], contra_trace['l']
            overall_R = np.array([np.mean(overall_R[r], axis=0) for r in range(len(overall_R))])
            overall_L = np.array([np.mean(overall_L[l], axis=0) for l in range(len(overall_L))])
            
            if bias:
                overall_R, overall_L = self.get_trace_matrix_multiple(contra_neurons, bias_trials=self.find_bias_trials())
            
            pref, nonpref = np.vstack((pref, overall_R)), np.vstack((nonpref, overall_L))

                        

        else:
            print('No contra selective neurons')
            
        
        nonpreferr = np.std(nonpref, axis=0) / np.sqrt(len(nonpref)) 
        preferr = np.std(pref, axis=0) / np.sqrt(len(pref))
                    
        pref, nonpref = np.mean(pref, axis = 0), np.mean(nonpref, axis = 0)

        plt.plot(x, pref, 'r-', label='Pref')
        plt.plot(x, nonpref, 'darkgrey', label='Non-pref')
        

        plt.fill_between(x, pref - preferr, 
                  pref + preferr,
                  color=['#ffaeb1'])
        plt.fill_between(x, nonpref - nonpreferr, 
                  nonpref + nonpreferr,
                  color='lightgrey')
        plt.title("Selective neurons")
        plt.xlabel('Time from Go cue (s)')
        plt.ylabel('Selectivity')
        plt.legend()
        plt.show()
            
    def plot_individual_raster(self, neuron_num):
        
                
        trace = [self.dff[0, t][neuron_num, :self.time_cutoff] for t in range(self.num_trials)]

        vmin, vmax = min(cat(trace)), max(cat(trace))
        trace = np.matrix(trace)
        
        plt.matshow(trace, cmap='gray', vmin=vmin, vmax=vmax)
        plt.axis('off')
        
        return trace
        
    def plot_left_right_raster(self, neuron_num, opto=False):
        
        r, l = self.get_trace_matrix(neuron_num)
        if opto:
            r, l = self.get_opto_trace_matrix(neuron_num)

        vmin, vmax = min(cat(cat((r,l)))), max(cat(cat((r,l))))
        
        r_trace, l_trace = np.matrix(r), np.matrix(l)
        
        stack = np.vstack((r_trace, l_trace))
        
        plt.matshow(stack, cmap='gray') #, norm ="log") #, vmin=vmin, vmax=vmax)
        plt.axis('off')
        # plt.figsize(10,01)
        return stack
        
    
    def filter_by_deltas(self, neuron_num):
        
        # Filters out neurons with low variance across trials
        
        r, l = self.get_trace_matrix(neuron_num)
        
        all_t = cat((r, l))
        ds = []
        for t in all_t:
            ds += [max(t) - min(t)]
            
        if np.median(ds) > 500:
            return True
        else:
            return False
        
    def plot_raster_and_PSTH(self, neuron_num, opto=False, bias=False):

        if not opto:
            R, L = self.get_trace_matrix(neuron_num)
            r, l = self.get_trace_matrix(neuron_num)
            title = "Neuron {}: Raster and PSTH".format(neuron_num)
        elif bias:
            
            bias_idx = self.find_bias_trials()
            R, L = self.get_trace_matrix(neuron_num, bias_trials = bias_idx)
            r, l = self.get_trace_matrix(neuron_num, bias_trials = bias_idx)
            
        else:
            R, L = self.get_opto_trace_matrix(neuron_num)
            r, l = self.get_opto_trace_matrix(neuron_num)
            title = "Neuron {}: Opto Raster and PSTH".format(neuron_num)
        
        r_trace, l_trace = np.matrix(r), np.matrix(l)
        
        # r_trace, l_trace = r_trace[:, 3:], l_trace[:, 3:]
        
        # stack = np.vstack((r_trace, np.ones(self.time_cutoff), l_trace))
        stack = np.vstack((r_trace, l_trace))


        
        R_av, L_av = np.mean(R, axis = 0), np.mean(L, axis = 0)
        
        left_err = np.std(L, axis=0) / np.sqrt(len(L)) 
        right_err = np.std(R, axis=0) / np.sqrt(len(R))
        
        # R_av, L_av, left_err, right_err = R_av[3:], L_av[3:], left_err[3:], right_err[3:]
                    

        f, axarr = plt.subplots(2, sharex=True, figsize=(10,10))

        axarr[0].matshow(stack, cmap='gray', interpolation='nearest', aspect='auto')
        axarr[0].axis('off')
        
        axarr[1].plot(L_av, 'r-')
        axarr[1].plot(R_av, 'b-')
        
        x = range(self.time_cutoff)
        # x = x[3:]
        
        axarr[1].fill_between(x, L_av - left_err, 
                 L_av + left_err,
                 color=['#ffaeb1'])
        axarr[1].fill_between(x, R_av - right_err, 
                 R_av + right_err,
                 color=['#b4b2dc'])
        axarr[1].axvline(self.sample, linestyle = '--')
        axarr[1].axvline(self.delay, linestyle = '--')
        axarr[1].axvline(self.response, linestyle = '--')
        axarr[0].set_title(title)
        plt.show()
        

    def plot_rasterPSTH_sidebyside(self, neuron_num, bias=False):
        
        if bias:
            bias_trials = self.find_bias_trials()
            R, L = self.get_trace_matrix(neuron_num, bias_trials=bias_trials, non_bias=True)
            r, l = self.get_trace_matrix(neuron_num, bias_trials=bias_trials, non_bias=True)
        else:
            R, L = self.get_trace_matrix(neuron_num)
            r, l = self.get_trace_matrix(neuron_num)
        title = "Neuron {}: Control".format(neuron_num)
        


        # f, axarr = plt.subplots(2,2, sharex='col', sharey = 'row')
        f, axarr = plt.subplots(2,2, sharex='col')
        
        r_trace, l_trace = np.matrix(r), np.matrix(l)
        
        stack = np.vstack((r_trace, np.ones(self.time_cutoff), l_trace))
        stack = np.vstack((r_trace, l_trace))


        
        R_av, L_av = np.mean(R, axis = 0), np.mean(L, axis = 0)
        
        left_err = np.std(L, axis=0) / np.sqrt(len(L)) 
        right_err = np.std(R, axis=0) / np.sqrt(len(R))
                    

        axarr[0, 0].matshow(stack, cmap='gray', interpolation='nearest', aspect='auto')
        axarr[0, 0].axis('off')
        
        axarr[1, 0].plot(L_av, 'r-')
        axarr[1, 0].plot(R_av, 'b-')
        axarr[1, 0].axvline(self.sample, linestyle = '--')
        axarr[1, 0].axvline(self.delay, linestyle = '--')
        axarr[1, 0].axvline(self.response, linestyle = '--')
        
        x = range(self.time_cutoff)

        axarr[1, 0].fill_between(x, L_av - left_err, 
                 L_av + left_err,
                 color=['#ffaeb1'])
        axarr[1, 0].fill_between(x, R_av - right_err, 
                 R_av + right_err,
                 color=['#b4b2dc'])
        
        axarr[0,0].set_title(title)
        
        if bias:
            bias_trials = self.find_bias_trials()
            R, L = self.get_trace_matrix(neuron_num, bias_trials=bias_trials)
            r, l = self.get_trace_matrix(neuron_num, bias_trials=bias_trials)
            title = "Neuron {}: Bias".format(neuron_num)
            
        else:
    
            R, L = self.get_opto_trace_matrix(neuron_num)
            r, l = self.get_opto_trace_matrix(neuron_num)
            title = "Neuron {}: Opto".format(neuron_num)

                
        r_trace, l_trace = np.matrix(r), np.matrix(l)
        
        stack = np.vstack((r_trace, np.ones(self.time_cutoff), l_trace))
        stack = np.vstack((r_trace, l_trace))

        R_av, L_av = np.mean(R, axis = 0), np.mean(L, axis = 0)
        vmax = max(cat([R_av, L_av]))

        left_err = np.std(L, axis=0) / np.sqrt(len(L)) 
        right_err = np.std(R, axis=0) / np.sqrt(len(R))
                    

        axarr[0, 1].matshow(stack, cmap='gray', interpolation='nearest', aspect='auto')
        axarr[0, 1].axis('off')
        
        axarr[1, 1].plot(L_av, 'r-')
        axarr[1, 1].plot(R_av, 'b-')
        axarr[1, 1].axvline(self.sample, linestyle = '--')
        axarr[1, 1].axvline(self.delay, linestyle = '--')
        axarr[1, 1].axvline(self.response, linestyle = '--')
        if not bias:
            axarr[1, 1].hlines(y=vmax, xmin=self.delay, xmax=self.delay + 5, linewidth=10, color='red')
        
        x = range(self.time_cutoff)

        axarr[1, 1].fill_between(x, L_av - left_err, 
                 L_av + left_err,
                 color=['#ffaeb1'])
        axarr[1, 1].fill_between(x, R_av - right_err, 
                 R_av + right_err,
                 color=['#b4b2dc'])
        
        axarr[0,1].set_title(title)
        axarr[1,0].set_ylabel('dF/F0')
        
        plt.show()
        

### EPHYS PLOTS TO MY DATA ###

    def plot_number_of_sig_neurons(self, save=False):
        
        contra = np.zeros(self.time_cutoff)
        ipsi = np.zeros(self.time_cutoff)
        x = np.arange(-5.97,4,0.2)[:self.time_cutoff]
        steps = range(self.time_cutoff)
        
        if 'CW030' in self.path:
            contra = np.zeros(self.time_cutoff-5)
            ipsi = np.zeros(self.time_cutoff-5)
            x = np.arange(-5.97,4,0.2)[:self.time_cutoff-5]
            steps = range(5, self.time_cutoff)

        for t in steps:
            
            sig_neurons = []

            # for n in range(self.num_neurons):
            for n in self.good_neurons:
                
                r, l = self.get_trace_matrix(n)
                r, l = np.matrix(r), np.matrix(l)
                t_val, p = stats.ttest_ind(r[:, t], l[:, t])
                
                if p < 0.01:
                     
                    if np.mean(r[:, t]) < np.mean(l[:, t]):
                        sig_neurons += [1]  # ipsi
                        
                    elif np.mean(r[:, t]) > np.mean(l[:, t]):
                        sig_neurons += [-1]  # contra
                    
                    else:
                        print("Error on neuron {} at time {}".format(n,t))

                else:
                    
                    sig_neurons += [0]
            
            contra[t-5] = sum(np.array(sig_neurons) == -1)
            ipsi[t-5] = sum(np.array(sig_neurons) == 1)

        plt.bar(x, contra, color = 'b', edgecolor = 'white', width = 0.2, label = 'contra')
        plt.bar(x, -ipsi, color = 'r',edgecolor = 'white', width = 0.2, label = 'ipsi')
        plt.axvline(-4.3)
        plt.axvline(-3)
        plt.axvline(0)
        
        plt.ylabel('Number of sig sel neurons')
        plt.xlabel('Time from Go cue (s)')
        plt.legend()
        
        if save:
            plt.savefig(self.path + r'number_sig_neurons.png')
        
        plt.show()
        
    def selectivity_table_by_epoch(self, save=False):
        
        # Plots fractions of contra/ipsi neurons and their overall trace

        f, axarr = plt.subplots(4,3, sharex='col', figsize=(14, 12))
        epochs = [range(self.time_cutoff), range(8,14), range(19,28), range(29,self.time_cutoff)]
        x = np.arange(-5.97,6,0.2)[:self.time_cutoff]
        if 'CW030' in self.path:
            x = np.arange(-5.97,6,0.2)[:self.time_cutoff-5]

        titles = ['Whole-trial', 'Sample', 'Delay', 'Response']
        
        for i in range(4):
            
            contra_neurons, ipsi_neurons, contra_trace, ipsi_trace = self.contra_ipsi_pop(epochs[i])

            # Bar plot
            axarr[i, 0].bar(['Contra', 'Ipsi'], [len(contra_neurons)/len(self.selective_neurons),
                                    len(ipsi_neurons)/len(self.selective_neurons)], 
                            color = ['b', 'r'])
            
            axarr[i, 0].set_ylim(0,1)
            axarr[i, 0].set_title(titles[i])
            
            if len(ipsi_neurons) != 0:
            
                overall_R, overall_L = ipsi_trace['r'], ipsi_trace['l']
                overall_R = [np.mean(overall_R[r], axis=0) for r in range(len(overall_R))]
                overall_L = [np.mean(overall_L[l], axis=0) for l in range(len(overall_L))]
                
                R_av = np.mean(overall_R, axis = 0) 
                L_av = np.mean(overall_L, axis = 0)
                
                left_err = np.std(overall_L, axis=0) / np.sqrt(len(overall_L)) 
                right_err = np.std(overall_R, axis=0) / np.sqrt(len(overall_R))
                            
                if 'CW030' in self.path:
                    L_av = L_av[5:]
                    R_av = R_av[5:]
                    left_err = left_err[5:]
                    right_err = right_err[5:]
                    
                    
                axarr[i, 2].plot(x, L_av, 'r-')
                axarr[i, 2].plot(x, R_av, 'b-')
                        
                axarr[i, 2].fill_between(x, L_av - left_err, 
                         L_av + left_err,
                         color=['#ffaeb1'])
                axarr[i, 2].fill_between(x, R_av - right_err, 
                         R_av + right_err,
                         color=['#b4b2dc'])
                axarr[i, 2].set_title("Ipsi-preferring neurons")
            
            else:
                print('No ipsi selective neurons')
        
            if len(contra_neurons) != 0:
    
                overall_R, overall_L = contra_trace['r'], contra_trace['l']
                overall_R = [np.mean(overall_R[r], axis=0) for r in range(len(overall_R))]
                overall_L = [np.mean(overall_L[l], axis=0) for l in range(len(overall_L))]
                
                R_av = np.mean(overall_R, axis = 0) 
                L_av = np.mean(overall_L, axis = 0)
                
                left_err = np.std(overall_L, axis=0) / np.sqrt(len(overall_L)) 
                right_err = np.std(overall_R, axis=0) / np.sqrt(len(overall_R))
                            
                if 'CW030' in self.path:
                    L_av = L_av[5:]
                    R_av = R_av[5:]
                    left_err = left_err[5:]
                    right_err = right_err[5:]
                    
                axarr[i, 1].plot(x, L_av, 'r-')
                axarr[i, 1].plot(x, R_av, 'b-')
                        
                axarr[i, 1].fill_between(x, L_av - left_err, 
                          L_av + left_err,
                          color=['#ffaeb1'])
                axarr[i, 1].fill_between(x, R_av - right_err, 
                          R_av + right_err,
                          color=['#b4b2dc'])
                axarr[i, 1].set_title("Contra-preferring neurons")

            else:
                print('No contra selective neurons')
                
        axarr[0,0].set_ylabel('Proportion of neurons')
        axarr[0,1].set_ylabel('dF/F0')
        axarr[3,1].set_xlabel('Time from Go cue (s)')
        axarr[3,2].set_xlabel('Time from Go cue (s)')
        
        if save:
            plt.savefig(self.path + r'contra_ipsi_SDR_table.png')
        
        plt.show()

    def plot_three_selectivity(self,save=False):
        
        f, axarr = plt.subplots(1,5, sharex='col', figsize=(21,5))
        
        epochs = [range(self.time_cutoff), range(8,14), range(19,28), range(29,self.time_cutoff)]
        x = np.arange(-5.97,4,0.2)[:self.time_cutoff]
        titles = ['Whole-trial', 'Sample', 'Delay', 'Response']
        
        num_epochs = []
        
        for i in range(4):
            
            contra_neurons, ipsi_neurons, contra_trace, ipsi_trace = self.contra_ipsi_pop(epochs[i])
            
            if len(contra_neurons) == 0:
                
                nonpref, pref = ipsi_trace['r'], ipsi_trace['l']
                
            elif len(ipsi_neurons) == 0:
                nonpref, pref = contra_trace['l'], contra_trace['r']

            else:
                nonpref, pref = cat((ipsi_trace['r'], contra_trace['l'])), cat((ipsi_trace['l'], contra_trace['r']))
                
                
            sel = np.mean(pref, axis = 0) - np.mean(nonpref, axis = 0)
            
            err = np.std(pref, axis=0) / np.sqrt(len(pref)) 
            err += np.std(nonpref, axis=0) / np.sqrt(len(nonpref))
                        
            axarr[i + 1].plot(x, sel, 'b-')
                    
            axarr[i + 1].fill_between(x, sel - err, 
                      sel + err,
                      color=['#b4b2dc'])

            axarr[i + 1].set_title(titles[i])
            
            num_epochs += [len(contra_neurons) + len(ipsi_neurons)]

        # Bar plot
        axarr[0].bar(['S', 'D', 'R'], np.array(num_epochs[1:]) / sum(num_epochs[1:]), color = ['dimgray', 'darkgray', 'gainsboro'])
        
        axarr[0].set_ylim(0,1)
        axarr[0].set_title('Among all ALM neurons')
        
        axarr[0].set_ylabel('Proportion of neurons')
        axarr[1].set_ylabel('Selectivity')
        axarr[2].set_xlabel('Time from Go cue (s)')
        
        
        plt.show()
        
    def population_sel_timecourse(self, save=True):
        
        f, axarr = plt.subplots(2, 1, sharex='col', figsize=(20,15))
        epochs = [range(14,28), range(21,self.time_cutoff), range(29,self.time_cutoff)]
        x = np.arange(-5.97,4,0.2)[:self.time_cutoff]
        titles = ['Preparatory', 'Prep + response', 'Response']
        
        sig_n = dict()
        sig_n['c'] = []
        sig_n['i'] = []
        contra_mat = np.zeros(self.time_cutoff)
        ipsi_mat = np.zeros(self.time_cutoff)
        
        for i in range(3):
            
            # contra_neurons, ipsi_neurons, contra_trace, ipsi_trace = self.contra_ipsi_pop(epochs[i])
            
            for n in self.get_epoch_selective(epochs[i]):
                                
                r, l = self.get_trace_matrix(n)
                r, l = np.array(r), np.array(l)
                side = 'c' if np.mean(r[:, epochs[i]]) > np.mean(l[:,epochs[i]]) else 'i'
                
                r, l = np.mean(r,axis=0), np.mean(l,axis=0)
                
                if side == 'c' and n not in sig_n['c']:
                    
                    sig_n['c'] += [n]
    
                    contra_mat = np.vstack((contra_mat, r - l))

                if side == 'i' and n not in sig_n['i']:
                    
                    sig_n['i'] += [n]

                    ipsi_mat = np.vstack((ipsi_mat, l - r))

        axarr[0].matshow((ipsi_mat[1:]), aspect="auto", cmap='jet')
        axarr[0].set_title('Ipsi-preferring neurons')
        
        axarr[1].matshow(-(contra_mat[1:]), aspect="auto", cmap='jet')
        axarr[1].set_title('Contra-preferring neurons')
        
        if save:
            plt.savefig(self.path + r'population_selectivity_overtime.jpg')
        
        plt.show()


    def selectivity_optogenetics(self, save=False):
        
        f, axarr = plt.subplots(1,1, sharex='col', figsize=(5,5))
        
        x = np.arange(-5.97,4,0.2)[:self.time_cutoff]

        # Get delay selective neurons
        contra_neurons, ipsi_neurons, contra_trace, ipsi_trace = self.contra_ipsi_pop(range(self.response-9,self.response)) 
        
        if len(contra_neurons) == 0:
            
            nonpref, pref = cat(ipsi_trace['r']), cat(ipsi_trace['l'])
            optonp, optop = self.get_trace_matrix_multiple(ipsi_neurons, opto=True, both=True)
            # errnp, errpref = self.get_trace_matrix_multiple(ipsi_neurons, opto=True, error=True)
            
        elif len(ipsi_neurons) == 0:
            
            nonpref, pref = cat(contra_trace['l']), cat(contra_trace['r'])
            optop, optonp = self.get_trace_matrix_multiple(contra_neurons, opto=True, both=True)
            # errpref, errnp = self.get_trace_matrix_multiple(contra_neurons, opto=True, error=True)

        else:
            
            nonpref, pref = cat((cat(ipsi_trace['r']), cat(contra_trace['l']))), cat((cat(ipsi_trace['l']), cat(contra_trace['r'])))
            optonp, optop = self.get_trace_matrix_multiple(ipsi_neurons, opto=True, both=True)
            optop1, optonp1 = self.get_trace_matrix_multiple(contra_neurons, opto = True, both=True)
            optonp, optop = cat((optonp, optonp1)), cat((optop, optop1))
            
            # errnp, errpref = self.get_trace_matrix_multiple(ipsi_neurons, opto=True, error=True)
            # errpref1, errnp1 = self.get_trace_matrix_multiple(contra_neurons, opto=True, error=True)
            # errpref, errnp = cat((errpref, errpref1)), cat((errnp, errnp1))

            
        sel = np.mean(pref, axis = 0) - np.mean(nonpref, axis = 0)
        err = np.std(pref, axis=0) / np.sqrt(len(pref)) 
        err += np.std(nonpref, axis=0) / np.sqrt(len(nonpref))
        
        selo = np.mean(optop, axis = 0) - np.mean(optonp, axis = 0)
        erro = np.std(optop, axis=0) / np.sqrt(len(optop)) 
        erro += np.std(optonp, axis=0) / np.sqrt(len(optonp))  

        if 'CW030' in self.path:
            
            sel = sel[5:]
            selo = selo[5:]
            err = err[5:]
            erro = erro[5:]
            x = np.arange(-5.97,4,0.2)[:self.time_cutoff-5]

        axarr.plot(x, sel, 'black')
                
        axarr.fill_between(x, sel - err, 
                  sel + err,
                  color=['darkgray'])
        
        axarr.plot(x, selo, 'b-')
                
        axarr.fill_between(x, selo - erro, 
                  selo + erro,
                  color=['#b4b2dc'])       

        axarr.set_title('Optogenetic effect on selectivity')                  
        axarr.set_xlabel('Time from Go cue (s)')
        axarr.set_ylabel('Selectivity')
        # axarr[0].plot(x, sel, 'black')
                
        # axarr[0].fill_between(x, sel - err, 
        #           sel + err,
        #           color=['darkgray'])
        
        # axarr[0].plot(x, selo, 'b-')
                
        # axarr[0].fill_between(x, selo - erro, 
        #           selo + erro,
        #           color=['#b4b2dc'])       

        # axarr[0].set_title('Optogenetic effect on selectivity')
        
        # selo = np.mean(errpref, axis = 0) - np.mean(errnp, axis = 0)
        # erro = np.std(errpref, axis=0) / np.sqrt(len(errpref)) 
        # erro += np.std(errnp, axis=0) / np.sqrt(len(errnp)) 
        
        # axarr[1].plot(x, sel, 'black')
                
        # axarr[1].fill_between(x, sel - err, 
        #           sel + err,
        #           color=['darkgray'])
        
        # axarr[1].plot(x, selo, 'b-')
                
        # axarr[1].fill_between(x, selo - erro, 
        #           selo + erro,
        #           color=['#b4b2dc'])   

        # axarr[1].set_title('Incorrect trials')
        
        if save:
            plt.savefig(self.path + r'opto_effect_on_selectivity.png')

        
        plt.show()
        
        
    def single_neuron_sel(self, type):
        
        def mean_count(XX, timebin):
            
            coeff = 1/(len(XX) * len(timebin))
            
            # numerator = sum([sum(XX[t][timebin]) for t in range(len(XX))])
            numerator = np.mean([XX[t][timebin] for t in range(len(XX))], axis=0)
            
            return numerator 
        
        if type == 'Chen 2017':
            
            stim = []
            lick = []
            reward = []
            mixed = []
                
            for t in range(self.time_cutoff):
                
                s,l,r,m = 0,0,0,0
                
                # for n in range(self.num_neurons):
                for n in self.good_neurons:
                    dff = [self.dff[0, trial][n, t] for trial in self.i_good_trials]
                    
                    df = pd.DataFrame({'stim': self.R_correct + self.R_wrong,
                                       'lick': self.R_correct + self.L_wrong,
                                       'reward': self.R_correct + self.L_correct,
                                       'dff': dff})
                    
                    model = ols("""dff ~ C(stim) + C(lick) + C(reward) +
                                    C(stim):C(lick) + C(stim):C(reward) + C(lick):C(reward) +
                                    C(stim):C(lick):C(reward)""", data = df).fit()
                    
                    table = sm.stats.anova_lm(model, type=2)
                    
                    sig = np.where(np.array(table['PR(>F)'] < 0.01) == True)[0]
                    if len(sig) == 0:
                        continue
                    elif 0 in sig:
                        s+=1
                    elif 1 in sig:
                        l+=1
                    elif 2 in sig:
                        r+=1
                    elif len(sig) >= 1:
                        if len(sig) == 1:
                            if 7 in sig:
                                continue
                        m+=1
                
                stim += [s]
                lick += [l]
                reward += [r]
                mixed += [m]
            
            f, axarr = plt.subplots(1,4, sharey='row', figsize=(20,5))
            x = np.arange(-5.97,4,0.2)[:self.time_cutoff]

            axarr[0].plot(x, np.array(stim)/self.num_neurons, color='magenta')
            axarr[0].set_title('Lick direction cell')
            axarr[1].plot(x, np.array(lick)/self.num_neurons, color='lime')
            axarr[1].set_title('Object location cell')
            axarr[2].plot(x, np.array(reward)/self.num_neurons, color='cyan')
            axarr[2].set_title('Outcome cell')
            axarr[3].plot(x, np.array(mixed)/self.num_neurons, color='gold')
            axarr[3].set_title('Mixed cell')

            for i in range(4):
                
                axarr[i].axvline(0, color = 'grey', alpha=0.5, ls = '--')
                axarr[i].axvline(-4.3, color = 'grey', alpha=0.5, ls = '--')
                axarr[i].axvline(-3, color = 'grey', alpha=0.5, ls = '--')
                
            plt.show()
            
            return stim, lick, reward, mixed
            
        if type == 'Susu method':
            
            stim, choice, action, outcome = 0,0,0,0
            
            stim_neurons, choice_neurons, action_neurons, outcome_neurons = [],[],[],[]
            
            # for n in range(self.num_neurons):
            for n in self.good_neurons:
            
                RR, LL = self.get_trace_matrix(n)
                
                RL, LR = self.get_trace_matrix(n, error=True)
                
                # stim = (mean_count(RR, range(7,13)) + mean_count(RL, range(7,13))) - (mean_count(LL, range(7,13)) + mean_count(LR, range(7,13)))
                # choice = (mean_count(RR, range(21,28)) + mean_count(LR, range(21,28))) - (mean_count(LL, range(21,28)) + mean_count(RL, range(21,28)))
                # action = (mean_count(RR, range(28,34)) + mean_count(LR, range(28,34))) - (mean_count(LL, range(28,34)) + mean_count(RL, range(28,34)))
                # outcome = (mean_count(LL, range(34,40)) + mean_count(RR, range(34,40))) - (mean_count(LR, range(34,40)) + mean_count(RL, range(34,40)))
                
                _, stimp = mannwhitneyu(cat((mean_count(RR, range(7,13)), mean_count(RL, range(7,13)))),
                                        cat((mean_count(LL, range(7,13)), mean_count(LR, range(7,13)))))
                _, choicep = mannwhitneyu(cat((mean_count(RR, range(21,28)), mean_count(LR, range(21,28)))),
                                          cat((mean_count(LL, range(21,28)), mean_count(RL, range(21,28)))))
                _, actionp = mannwhitneyu(cat((mean_count(RR, range(28,40)), mean_count(LR, range(28,40)))),
                                          cat((mean_count(LL, range(28,40)), mean_count(RL, range(28,40)))))
                _, outcomep = mannwhitneyu(cat((mean_count(LL, range(34,40)), mean_count(RR, range(34,40)))),
                                           cat((mean_count(LR, range(34,40)), mean_count(RL, range(34,40)))))
                
                # stim += [stimp]
                stim += stimp<0.05
                choice += choicep<0.05
                action += actionp<0.05
                outcome += outcomep<0.05
                
                stim_neurons += [n] if stimp<0.05 else []
                choice_neurons += [n] if choicep<0.05 else []
                action_neurons += [n] if actionp<0.05 else []
                outcome_neurons += [n] if outcomep<0.05 else []
                
                
            plt.bar(['stim', 'choice', 'action', 'outcome'], [stim/len(self.good_neurons), 
                                                              choice/len(self.good_neurons), 
                                                              action/len(self.good_neurons),
                                                              outcome/len(self.good_neurons)])
            plt.xlabel('Epoch selective')
            plt.ylabel('Proportion of neurons')
            # plt.ylim(0,0.5)
            plt.show()
                
            return stim_neurons, choice_neurons, action_neurons, outcome_neurons

    def stim_choice_outcome_selectivity(self):
        
        stim_neurons, choice_neurons, _, outcome_neurons = self.single_neuron_sel('Susu method')
        
        stim_sel, outcome_sel, choice_sel = [], [], []
        
        f, axarr = plt.subplots(1,3, sharey='row', figsize=(15,5))
        
        epochs = [range(7,13), range(21,28), range(34,self.time_cutoff)]
        x = np.arange(-5.97,4,0.2)[:self.time_cutoff]
        titles = ['Stimulus selective', 'Choice selective', 'Outcome selective']
        
        
            
        contra_neurons, ipsi_neurons, contra_trace, ipsi_trace = self.contra_ipsi_pop(epochs[0])
        pref, nonpref = [], []

        if len(contra_neurons) == 0:
            for j in range(len(ipsi_neurons)):
                
                if ipsi_neurons[j] not in stim_neurons:
                    continue
                
                nonpref += ipsi_trace['r'][j]
                pref += ipsi_trace['l'][j]
                
                stim_sel += [np.mean(np.mean(pref, axis=0)[epochs[0]] - np.mean(nonpref, axis=0)[epochs[0]])]
            
        elif len(ipsi_neurons) == 0:
            
            for j in range(len(contra_neurons)):
                
                if contra_neurons[j] not in stim_neurons:
                    continue
                
                nonpref += contra_trace['l'][j]
                pref += contra_trace['r'][j]
                stim_sel += [np.mean(np.mean(pref, axis=0)[epochs[0]] - np.mean(nonpref, axis=0)[epochs[0]])]


        else:
            
            for j in range(len(ipsi_neurons)):
                
                if ipsi_neurons[j] not in stim_neurons:
                    continue
                
                nonpref += ipsi_trace['r'][j]
                pref += ipsi_trace['l'][j]
                stim_sel += [np.mean(np.mean(pref, axis=0)[epochs[0]] - np.mean(nonpref, axis=0)[epochs[0]])]

            for j in range(len(contra_neurons)):
                
                if contra_neurons[j] not in stim_neurons:
                    continue
                
                nonpref += contra_trace['l'][j]
                pref += contra_trace['r'][j]
                
                stim_sel += [np.mean(np.mean(pref, axis=0)[epochs[0]] - np.mean(nonpref, axis=0)[epochs[0]])]

        # pref, nonpref = [], []
        
        # for n in stim_neurons:
            
        #     r, l = self.get_trace_matrix(n)
        #     timebin=range(7,13)
        #     if np.mean([r[t][timebin] for t in range(len(r))]) - np.mean([l[t][timebin] for t in range(len(l))]):
                
        #         pref += [np.mean(r,axis=0)]
        #         nonpref += [np.mean(l,axis=0)]
            
        #     else:
        #         pref += [np.mean(l,axis=0)]
        #         nonpref += [np.mean(r,axis=0)]
                                
        pref, nonpref = np.array(pref), np.array(nonpref)

        
        sel = np.mean(pref, axis = 0) - np.mean(nonpref, axis = 0)
        
        err = np.std(pref, axis=0) / np.sqrt(len(pref)) 
        err += np.std(nonpref, axis=0) / np.sqrt(len(nonpref))
                    
        axarr[0].plot(x, sel, color='green')
                
        axarr[0].fill_between(x, sel - err, 
                  sel + err,
                  color='lightgreen')

        axarr[0].set_title(titles[0])
        #############################
        pref, nonpref = [], []
        
        if len(contra_neurons) == 0:
            for j in range(len(ipsi_neurons)):
                
                if ipsi_neurons[j] not in choice_neurons:
                    continue
                
                nonpref += ipsi_trace['r'][j]
                pref += ipsi_trace['l'][j]
                choice_sel += [np.mean(np.mean(pref, axis=0)[epochs[1]] - np.mean(nonpref, axis=0)[epochs[1]])]

            
        elif len(ipsi_neurons) == 0:
            
            for j in range(len(contra_neurons)):
                
                if contra_neurons[j] not in choice_neurons:
                    continue
                
                nonpref += contra_trace['l'][j]
                pref += contra_trace['r'][j]
                choice_sel += [np.mean(np.mean(pref, axis=0)[epochs[1]] - np.mean(nonpref, axis=0)[epochs[1]])]
                

        else:
            
            for j in range(len(ipsi_neurons)):
                
                if ipsi_neurons[j] not in choice_neurons:
                    continue
                
                nonpref += ipsi_trace['r'][j]
                pref += ipsi_trace['l'][j]
                choice_sel += [np.mean(np.mean(pref, axis=0)[epochs[1]] - np.mean(nonpref, axis=0)[epochs[1]])]

            for j in range(len(contra_neurons)):
                
                if contra_neurons[j] not in choice_neurons:
                    continue
                
                nonpref += contra_trace['l'][j]
                pref += contra_trace['r'][j]
                choice_sel += [np.mean(np.mean(pref, axis=0)[epochs[1]] - np.mean(nonpref, axis=0)[epochs[1]])]

                
        pref, nonpref = np.array(pref), np.array(nonpref)
        
        sel = np.mean(pref, axis = 0) - np.mean(nonpref, axis = 0)
        
        err = np.std(pref, axis=0) / np.sqrt(len(pref)) 
        err += np.std(nonpref, axis=0) / np.sqrt(len(nonpref))
                    
        axarr[1].plot(x, sel, color='purple')
                
        axarr[1].fill_between(x, sel - err, 
                  sel + err,
                  color='violet')
        axarr[1].set_title(titles[1])

        ####################################
        
        pref, nonpref = [], []
        
        # for n in outcome_neurons:
            
        #     r, l = self.get_trace_matrix(n)
        #     timebin=range(34,self.time_cutoff)
        #     if np.mean([r[t][timebin] for t in range(len(r))]) - np.mean([l[t][timebin] for t in range(len(l))]):
                
        #         pref += [np.mean(r,axis=0)]
        #         nonpref += [np.mean(l,axis=0)]
            
        #     else:
        #         pref += [np.mean(l,axis=0)]
        #         nonpref += [np.mean(r,axis=0)]
        
        
        contra_neurons, ipsi_neurons, contra_trace, ipsi_trace = self.contra_ipsi_pop(epochs[2])
        pref, nonpref = [], []

        if len(contra_neurons) == 0:
            for j in range(len(ipsi_neurons)):
                
                if ipsi_neurons[j] not in outcome_neurons:
                    continue
                
                nonpref += ipsi_trace['r'][j]
                pref += ipsi_trace['l'][j]
                outcome_sel += [np.mean(np.mean(pref, axis=0)[epochs[2]] - np.mean(nonpref, axis=0)[epochs[2]])]

        elif len(ipsi_neurons) == 0:
            
            for j in range(len(contra_neurons)):
                
                if contra_neurons[j] not in outcome_neurons:
                    continue
                
                nonpref += contra_trace['l'][j]
                pref += contra_trace['r'][j]
                
                outcome_sel += [np.mean(np.mean(pref, axis=0)[epochs[2]] - np.mean(nonpref, axis=0)[epochs[2]])]

        else:
            
            for j in range(len(ipsi_neurons)):
                
                if ipsi_neurons[j] not in outcome_neurons:
                    continue
                
                nonpref += ipsi_trace['r'][j]
                pref += ipsi_trace['l'][j]
                outcome_sel += [np.mean(np.mean(pref, axis=0)[epochs[2]] - np.mean(nonpref, axis=0)[epochs[2]])]

            for j in range(len(contra_neurons)):
                
                if contra_neurons[j] not in outcome_neurons:
                    continue
                
                nonpref += contra_trace['l'][j]
                pref += contra_trace['r'][j]
                outcome_sel += [np.mean(np.mean(pref, axis=0)[epochs[2]] - np.mean(nonpref, axis=0)[epochs[2]])]

        pref, nonpref = np.array(pref), np.array(nonpref)

        
        sel = np.mean(pref, axis = 0) - np.mean(nonpref, axis = 0)
        
        err = np.std(pref, axis=0) / np.sqrt(len(pref)) 
        err += np.std(nonpref, axis=0) / np.sqrt(len(nonpref))
                    
        axarr[2].plot(x, sel, color='dodgerblue')
                
        axarr[2].fill_between(x, sel - err, 
                  sel + err,
                  color='lightskyblue')

        axarr[2].set_title(titles[2])
        
        #####################################################
        
        axarr[0].set_ylabel('Selectivity')
        axarr[1].set_xlabel('Time from Go cue (s)')
        
        for i in range(3):
            
            axarr[i].axvline(0, color = 'grey', alpha=0.5, ls = '--')
            axarr[i].axvline(-4.3, color = 'grey', alpha=0.5, ls = '--')
            axarr[i].axvline(-3, color = 'grey', alpha=0.5, ls = '--')        
            axarr[i].axhline(0, color = 'grey', alpha=0.5, ls = '--')        

        plt.show()
        return stim_neurons, choice_neurons, outcome_neurons, stim_sel, choice_sel, outcome_sel


    def find_bias_trials(self, glmhmm=True, sampling='confidence'):
        
        self.correct_trials = self.L_correct + self.R_correct
        self.instructed_side = self.L_correct + self.L_wrong # 1 if left, 0 if right trial
        
        bias_trials = []
        
        for i in range(20, len(self.correct_trials)): # only start after autolearn is turned off
            
            if self.correct_trials[i] == 0: # error trials
                
                if self.correct_trials[i-1] == 1 and self.instructed_side[i] != self.instructed_side[i-1]:
                    
                    bias_trials += [i]
                
                elif self.correct_trials[i-1] == 0 and self.instructed_side[i] == self.instructed_side[i-1]:
                    
                    bias_trials += [i]
        bias_trials = [b for b in bias_trials if b in self.i_good_trials]
        
        # Pre-bias trials:
        prebias_trials = [b-1 for b in bias_trials if b in self.i_good_trials]
        prebias_trials = [b for b in prebias_trials if b not in bias_trials]

        if glmhmm:
            # CW: TODO
            # Add ability to grab glmhmm trials in bias states
            # Biased state will be first always
            states = np.load(r'{}\states.npy'.format(self.path))
            
            st = []
            # Two different sampling methods
            if sampling == 'confidence':
                for i in range(states.shape[0]):
        
                    top_state = np.argmax(states[i])
                    if states[i][top_state] > 0.75:
                        st += [top_state]
            else:
                for i in range(states.shape[0]):
                    st += [np.random.choice([0, 1, 2], p=states[i])]
            
            inds = np.where(np.array(st) == 0)[0]
            bias_trials = self.old_i_good_trials[inds]
            bias_trials = [b for b in bias_trials if b in self.i_good_trials] #Filter out water leak trials

        # return prebias_trials
        return bias_trials
            
    














