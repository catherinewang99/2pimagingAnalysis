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

class Session:
    
    def __init__(self, layer_og, layer_num, behavior):
        
        layer = copy.deepcopy(layer_og)
        
        self.layer_num = layer_num
        self.dff = layer['dff']
        self.num_neurons = layer['dff'][0,0].shape[0]
        self.num_trials = layer['dff'].shape[1]
        self.time_cutoff = 40
        self.recording_loc = 'l'
        
        self.i_good_trials = cat(behavior['i_good_trials']) - 1 # zero indexing in python
        
        self.L_correct = cat(behavior['L_hit_tmp'])
        self.R_correct = cat(behavior['R_hit_tmp'])
        
        self.early_lick = cat(behavior['LickEarly_tmp'])
        
        self.L_wrong = cat(behavior['L_miss_tmp'])
        self.R_wrong = cat(behavior['R_miss_tmp'])
        
        self.L_ignore = cat(behavior['L_ignore_tmp'])
        self.R_ignore = cat(behavior['R_ignore_tmp'])
        
        self.stim_ON = cat(behavior['StimDur_tmp']) == 1

        self.plot_mean_F()

        self.normalize_all_by_baseline()
        self.normalize_z_score()        

        
    def plot_mean_F(self):
        
        # Plots mean F for all neurons over trials in session
        meanf = list()
        for trial in range(self.num_trials):
            meanf.append(np.mean(cat(self.dff[0, trial])))
        
        plt.plot(range(self.num_trials), meanf, 'b-')
        plt.title("Mean F for layer {}".format(self.layer_num))
        plt.show()

    def crop_trials(self, trial_num, end=True):
        
        # If called, crops out all trials after given trial number
        # Can optionally crop from trial_num to end indices
        
        if end:
            
            self.L_correct = self.L_correct[:trial_num]
            self.R_correct = self.R_correct[:trial_num]
            
            self.dff = self.dff[:, :trial_num]
            
            self.i_good_trials = [i for i in self.i_good_trials if i < trial_num]
            self.num_trials = trial_num
            self.plot_mean_F()
            
        else:
            
            self.L_correct = np.append(self.L_correct[:trial_num], self.L_correct[end:])
            self.R_correct = np.append(self.R_correct[:trial_num], self.R_correct[end:])
            
            self.dff = np.append(self.dff[:, :trial_num], self.dff[:, end:])
            
            self.i_good_trials = [i for i in self.i_good_trials if i < trial_num or i > end]
            self.num_trials = trial_num
            self.plot_mean_F()

        
        # self.normalize_all_by_baseline()
        # self.normalize_z_score()    
        
        # self.plot_mean_F()
        
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
    
    def get_trace_matrix(self, neuron_num):
        
        ## Returns matrix of all trial firing rates of a single neuron for lick left
        ## and lick right trials. Firing rates are normalized with individual trial
        ## baselines as well as overall firing rate z-score normalized.
        
        right_trials = self.lick_correct_direction('r')
        left_trials = self.lick_correct_direction('l')
        
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
    
    def get_opto_trace_matrix(self, neuron_num):
        
        
        right_trials = self.lick_correct_direction('r')
        left_trials = self.lick_correct_direction('l')
        
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
    
    def normalize_all_by_baseline(self):
        
        # Normalize all neurons by individual trial-averaged F0
        
        for i in range(self.num_neurons):
            
            nmean = np.mean([self.dff[0, t][i, :7] for t in range(self.num_trials)]).copy()
            
            for j in range(self.num_trials):
                self.dff[0, j][i] = (self.dff[0, j][i] - nmean) / nmean
        
        return None
    
    def normalize_z_score(self):
        
        # Normalize by mean of all neurons in layer
        overall_mean = np.mean(cat([cat(i) for i in self.dff[0]])).copy()
        std = np.std(cat([cat(i) for i in self.dff[0]])).copy()
        
        for i in range(self.num_trials):
            for j in range(self.num_neurons):
                self.dff[0, i][j] = (self.dff[0, i][j] - overall_mean) / std
        
        return None

    def get_delay_selective(self):
        selective_neurons = []
        for neuron in range(self.num_neurons):
            right, left = self.get_trace_matrix(neuron)
            tstat, p_val = stats.ttest_ind(np.mean(left, axis = 0)[21:28], np.mean(right, axis = 0)[21:28])
            p_measure = 0.01/self.num_neurons
            # p_measure = 0.05
            # p_measure = 0.0001
            if p_val < p_measure:
                selective_neurons += [neuron]
        print("Total delay selective neurons: ", len(selective_neurons))
        self.selective_neurons = selective_neurons
        return selective_neurons
   
    def get_response_selective(self):
        selective_neurons = []
        for neuron in range(self.num_neurons):
            right, left = self.get_trace_matrix(neuron)
            tstat, p_val = stats.ttest_ind(np.mean(left, axis = 0)[29:38], np.mean(right, axis = 0)[29:38])
            p_measure = 0.01/self.num_neurons
            # p_measure = 0.01
            # p_measure = 0.0001
            if p_val < p_measure:
                selective_neurons += [neuron]
        print("Total response selective neurons: ", len(selective_neurons))
        self.selective_neurons = selective_neurons
        return selective_neurons
 
    def get_sample_selective(self):
        selective_neurons = []
        for neuron in range(self.num_neurons):
            right, left = self.get_trace_matrix(neuron)
            tstat, p_val = stats.ttest_ind(np.mean(left, axis = 0)[7:13], np.mean(right, axis = 0)[7:13])
            p_measure = 0.01/self.num_neurons
            # p_measure = 0.01
            # p_measure = 0.0001
            if p_val < p_measure:
                selective_neurons += [neuron]
        print("Total sample selective neurons: ", len(selective_neurons))
        self.selective_neurons = selective_neurons
        return selective_neurons
    
    def screen_preference(self, neuron_num, samplesize = 10):

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
        avg_l = np.mean([np.mean(L[i][21:28]) for i in screen_l])
        avg_r = np.mean([np.mean(R[i][21:28]) for i in screen_r])
    
        return avg_l > avg_r, test_l, test_r

    def plot_selectivity(self, neuron_num):
        
        R, L = self.get_trace_matrix(neuron_num)
        pref, l, r = self.screen_preference(neuron_num)
        left_trace = [L[i] for i in l]
        right_trace = [R[i] for i in r]

        if pref: # prefers left
            sel = np.mean(left_trace, axis = 0) - np.mean(right_trace, axis=0)
        else:
            sel = np.mean(right_trace, axis = 0) - np.mean(left_trace, axis=0)
            
        direction = 'Left' if pref else 'Right'
        plt.plot(range(self.time_cutoff), sel, 'b-')
        plt.axhline(y=0)
        plt.title('Selectivity of neuron {}: {} selective'.format(neuron_num, direction))
        plt.show()
    
    def contra_ipsi_pop(self):
        
        # Returns the neuron ids for contra and ipsi populations
        
        selective_neurons = self.get_delay_selective()
        
        contra_neurons = []
        ipsi_neurons = []
        
        contra_LR, ipsi_LR = dict(), dict()
        contra_LR['l'], contra_LR['r'] = [], []
        ipsi_LR['l'], ipsi_LR['r'] = [], []
        
        
        for neuron_num in selective_neurons:
            
            # Skip sessions with fewer than 15 neurons
            if self.screen_preference(neuron_num) != 0:
                
                R, L = self.get_trace_matrix(neuron_num)

                pref, test_l, test_r = self.screen_preference(neuron_num) 
        
                if self.recording_loc == 'l':

                    if pref:
                        # print(pref)
                        print("Ipsi_preferring: {}".format(neuron_num))
                        ipsi_neurons += [neuron_num]
                        ipsi_LR['l'] += [L[i] for i in test_l]
                        ipsi_LR['r'] += [R[i] for i in test_r]
                    else:
                        print("Contra preferring: {}".format(neuron_num))
                        contra_neurons += [neuron_num] 
                        contra_LR['l'] += [L[i] for i in test_l]
                        contra_LR['r'] += [R[i] for i in test_r]
                    
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
    
    def plot_contra_ipsi_pop(self):
        
        contra_neurons, ipsi_neurons, contra_trace, ipsi_trace = self.contra_ipsi_pop()
        
        overall_R, overall_L = ipsi_trace['r'], ipsi_trace['l']
        
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
        plt.title("Ipsi-preferring neurons")
        plt.show()
    
    
        overall_R, overall_L = contra_trace['r'], contra_trace['l']
        
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
        plt.title("Contra-preferring neurons")
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
        
    def plot_selective_raster(self, neuron_num):
        
        return None
    
    def filter_by_deltas(self, neuron_num):
        
        # Filters out neurons without significant changes in trials
        
        r, l = self.get_trace_matrix(neuron_num)
        
        all_t = cat((r, l))
        ds = []
        for t in all_t:
            ds += [max(t) - min(t)]
            
        if np.median(ds) > 500:
            return True
        else:
            return False
        
    def plot_raster_and_PSTH(self, neuron_num, opto=False):

        if not opto:
            R, L = self.get_trace_matrix(neuron_num)
            r, l = self.get_trace_matrix(neuron_num)
            title = "Neuron {}: Raster and PSTH".format(neuron_num)

        else:
            R, L = self.get_opto_trace_matrix(neuron_num)
            r, l = self.get_opto_trace_matrix(neuron_num)
            title = "Neuron {}: Opto Raster and PSTH".format(neuron_num)


        vmin, vmax = min(cat(cat((r,l)))), max(cat(cat((r,l))))
        
        r_trace, l_trace = np.matrix(r), np.matrix(l)
        
        stack = np.vstack((r_trace, np.ones(self.time_cutoff), l_trace))
        stack = np.vstack((r_trace, l_trace))

        


        
        R_av, L_av = np.mean(R, axis = 0), np.mean(L, axis = 0)
        
        left_err = np.std(L, axis=0) / np.sqrt(len(L)) 
        right_err = np.std(R, axis=0) / np.sqrt(len(R))
                    

        f, axarr = plt.subplots(2, sharex=True)

        axarr[0].matshow(stack, cmap='gray', interpolation='nearest', aspect='auto')
        axarr[0].axis('off')
        
        axarr[1].plot(L_av, 'r-')
        axarr[1].plot(R_av, 'b-')
        
        x = range(self.time_cutoff)

        axarr[1].fill_between(x, L_av - left_err, 
                 L_av + left_err,
                 color=['#ffaeb1'])
        axarr[1].fill_between(x, R_av - right_err, 
                 R_av + right_err,
                 color=['#b4b2dc'])
        
        axarr[0].set_title(title)
        plt.show()
        

        
        
        
        
        
        
        
        
        
        
        
        
        
    