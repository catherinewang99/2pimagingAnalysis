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
from LinRegpval import LinearRegression
plt.rcParams['pdf.fonttype'] = 42 
import time 
import random

class Session:
    """
    A class used to store and process two photon imaging data alongside
    behavior data
 
    ...
 
    Attributes
    ----------

 
    Methods
    -------

    """
    
    
    def __init__(self, path, layer_num='all', use_reg = False, triple = False, sess_reg = False, guang=False, passive=False, quality=False):
        
        """
        Parameters
        ----------
        path : str
            Path to the folder containing layers.mat and behavior.mat files
        layer_num : str or int, optional
            Layer number to analyze (default is all the layers)
        use_reg : bool, optional
            Contains neurons that are matched only, for all layers
        sess_reg : bool, optional
            Reads in .npy file containing the registered neurons only. 
            Usually, this means only one layer. TBC. (default False)
        guang : bool, optional
            Boolean indicating is Guang's data is being used (default False)
        passive : bool, optional
            If dataset is from passive experiment (default False)
        quality : bool, optional
            If parent class is quality
        """        
        
        if layer_num != 'all':
            
            layer_og = scio.loadmat(r'{}\layer_{}.mat'.format(path, layer_num))
            layer = copy.deepcopy(layer_og)
            self.dff = layer['dff']
            self.fs = 1/6
            if use_reg:
                if triple:
                    self.good_neurons = np.load(path + r'\layer{}_triple_registered_neurons.npy'.format(layer_num-1))
                else:
                    self.good_neurons = np.load(path + r'\layer{}_registered_neurons.npy'.format(layer_num-1))


            
        else:
            # Load all layers

            self.dff = None
            counter = 0
            for layer_pth in os.listdir(path):
                if 'layer' in layer_pth and '.mat' in layer_pth:
                    layer_og = scio.loadmat(r'{}\{}'.format(path, layer_pth))
                    layer = copy.deepcopy(layer_og)
                    
                    if self.dff == None:
                        
                        if use_reg:
                            if triple:
                                self.good_neurons = np.load(path + r'\layer{}_triple_registered_neurons.npy'.format(counter))

                            else:
                                self.good_neurons = np.load(path + r'\layer{}_registered_neurons.npy'.format(counter))

                            # self.good_neurons = layer['dff'][:, :][neurons]
                        self.dff = layer['dff']
                        self.num_trials = layer['dff'].shape[1] 
                        
                    else:
                        if use_reg:
                            # raise NotImplementedError("Multi plane reg not implemented!")
                            if triple:
                                neurons = np.load(path + r'\layer{}_triple_registered_neurons.npy'.format(counter))
                                self.good_neurons = np.append(self.good_neurons, neurons + self.dff[0,0].shape[0])
                            else:
                                neurons = np.load(path + r'\layer{}_registered_neurons.npy'.format(counter))
                                self.good_neurons = np.append(self.good_neurons, neurons + self.dff[0,0].shape[0])
                            
                        for t in range(self.num_trials):

                            add = layer['dff'][0, t]
                            self.dff[0, t] = np.vstack((self.dff[0, t], add))
                    
                    counter += 1
            self.fs = 1/(30/counter)

                            

                                        
        
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
        
        # if self.path == 'F:\\data\\BAYLORCW03\\python\\2023_06_26':
        #     self.i_good_trials = self.i_good_trials[:100]
        
        
        self.L_correct = cat(behavior['L_hit_tmp'])
        self.R_correct = cat(behavior['R_hit_tmp'])
        
        self.early_lick = cat(behavior['LickEarly_tmp'])
        
        self.L_wrong = cat(behavior['L_miss_tmp'])
        self.R_wrong = cat(behavior['R_miss_tmp'])
        
        self.L_ignore = cat(behavior['L_ignore_tmp'])
        self.R_ignore = cat(behavior['R_ignore_tmp'])
                
        self.L_trials = np.sort(cat((np.where(self.L_correct)[0], 
                                     np.where(self.L_wrong)[0],
                                     np.where(self.L_ignore)[0])))
        
        self.R_trials = np.sort(cat((np.where(self.R_correct)[0], 
                                     np.where(self.R_wrong)[0],
                                     np.where(self.R_ignore)[0]))) 
        
        self.stim_ON = cat(behavior['StimDur_tmp']) > 0
        if 'StimLevel' in behavior.keys():
            self.stim_level = cat(behavior['StimLevel'])
            
        if self.i_good_trials[-1] > self.num_trials:
            
            print('More Bpod trials than 2 photon trials')
            self.i_good_trials = [i for i in self.i_good_trials if i < self.num_trials]
            self.stim_ON = self.stim_ON[:self.num_trials]
            
        self.sample = int(2.5*(1/self.fs))
        self.delay = self.sample + int(1.3*(1/self.fs))
        self.response = self.delay + int(3*(1/self.fs))
        # if 'CW03' in path:
        #     self.sample += 5
        #     self.delay += 5
        #     self.response += 5
        self.old_i_good_trials = copy.copy(self.i_good_trials)

        # Measure that automatically crops out water leak trials before norming
        if not self.find_low_mean_F():

            if quality:
                self.plot_mean_F()
            print("No water leak!")
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

        self.i_good_non_stim_trials = [t for t in self.i_good_trials if not self.stim_ON[t]]

        if not sess_reg and not use_reg:
            print("Using Pearsons corr to filter neurons.")
            self.good_neurons, _ = self.get_pearsonscorr_neuron(cutoff=0.5)
        elif sess_reg:
            print(sess_reg)
            self.good_neurons = np.load(path + r'\registered_neurons.npy')
        
    def crop_baseline(self):
        """Crops baseline out of trial data

        To deal with transient activation problem, this function will crop out 
        the extra long baseline period (2.5 seconds)

        Raises
        ------
        NotImplementedError
            If function isn't implemented
        
        """        
        broken = True
        if broken:
            raise NotImplementedError("Baseline cropping not implemented!")
        
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
        
        """Finds the shortest trial length out of all trials
    
        Returns
        -------
        int
            int corresponding to the length of shortest trial in whole session
        """
        
        cutoff = 1e10
        
        for t in range(self.num_trials):
            
            if self.dff[0, t].shape[1] < cutoff:
                
                cutoff = self.dff[0, t].shape[1]
        
        print("Minimum cutoff is {}".format(cutoff))
        
        return cutoff
    
    def find_low_mean_F(self, cutoff = 50):
        """Finds and crop low F trials that correspond to water leaks
        
        Calls crop_trials if there are water leak trials
        
        Parameters
        ----------
        cutoff : int, optional
            Limit for F value to be cropped (default 50)
        
        Returns
        -------
        int
            0/1 for no water leak or has water leak
        """
        # Usual cutoff is 50
        # Reject outliers based on medians
        meanf = np.array([])
        for trial in range(self.num_trials):
            meanf = np.append(meanf, np.mean(cat(self.dff[0, trial])))
        
        
        trial_idx = np.where(meanf < cutoff)[0]
        
        if trial_idx.size == 0:
            
            return 0
        
        else:
            print('Water leak trials: {}'.format(trial_idx))

            self.crop_trials(0, singles=True, arr = trial_idx)
            return 1
        
    def reject_outliers(data, m = 2.):
        """Rejects outliers below some standard deviation
        
        Parameters
        ----------
        data : array
            Array of numbers from which to crop out outliers
        m : int, optional
            Standard deviation cutoff (default 2)
        
        Returns
        -------
        array
            Cropped array from input variable data
        """
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d/mdev if mdev else np.zero(len(d))
        
        return data[s<m]
    
    def get_pearsonscorr_neuron(self, cutoff = 0.7, postreg= False):
        """Filters neurons based on the consistency of their signal
        
        Only returns neurons that have a consistent trace over all trials
        comparing across even/odd and first/last 100 trials with a Pearson's 
        correlation test

        Parameters
        ----------
        cutoff : int, optional
            p-value cutoff to use in Pearson's correlation test
        postreg : bool, optional
            whether or not to use good neurons or all neurons
        
        Returns
        -------
        neurons : array
            List of neurons to keep based on their high signal consistency
        corr : array
            List of correlation values for all neurons
        """
        
        
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
        
        first, last = self.i_good_trials[:100], self.i_good_trials[-100:]
        
        neuronlist = range(self.num_neurons) if not postreg else self.good_neurons
        
        for neuron_num in neuronlist:
            R_av_dff = []
            for i in evens:
                # R_av_dff += [self.normalize_by_baseline(self.dff[0, i][neuron_num, :self.time_cutoff])]
                R_av_dff += [self.dff[0, i][neuron_num, 5:self.time_cutoff]]
    
            L_av_dff = []
            for i in odds:
                # L_av_dff += [self.normalize_by_baseline(self.dff[0, i][neuron_num, :self.time_cutoff])]
                L_av_dff += [self.dff[0, i][neuron_num, 5:self.time_cutoff]]
                
            corr, p = mstats.pearsonr(np.mean(L_av_dff, axis = 0), np.mean(R_av_dff,axis=0))
            corrs += [corr]
            if corr > cutoff:
                
                # Do pearsons for first half, last half for leaky neurons
                R_av_dff = []
                for i in first:
                    # R_av_dff += [self.normalize_by_baseline(self.dff[0, i][neuron_num, :self.time_cutoff])]
                    R_av_dff += [self.dff[0, i][neuron_num, 5:self.time_cutoff]]
        
                L_av_dff = []
                for i in last:
                    # L_av_dff += [self.normalize_by_baseline(self.dff[0, i][neuron_num, :self.time_cutoff])]
                    L_av_dff += [self.dff[0, i][neuron_num, 5:self.time_cutoff]]
                    
                corr1, p1 = mstats.pearsonr(np.mean(L_av_dff, axis = 0), np.mean(R_av_dff,axis=0))
                
                if corr1 > cutoff:

                
                    neurons += [neuron_num]
            
        return neurons, corrs
    
    def plot_mean_F(self):
        """Plots mean F for all neurons over trials in session

        Averaged over all neurons and plotted over all trials
        """
        meanf = list()
        # for trial in range(self.num_trials):
        for trial in self.i_good_trials:
            meanf.append(np.mean(cat(self.dff[0, trial])))
        
        plt.plot(self.i_good_trials, meanf, 'b-')
        plt.title("Mean F for layer {}".format(self.layer_num))
        plt.show()

    def crop_trials(self, trial_num, end=False, singles = False, arr = []):
        """Removes trials from i_good_trials based on inputs
        
        After cropping, calls plot_mean_F and two normalizing functions

        Parameters
        ----------
        trial_num : int or list
            Trial number or numbers to crop out
        end : bool or int, optional
            If provided, together with trial_num form the start and end num of
            trials to be cropped out (default False)
        singles : bool, optional
            if True, crop trials provided in arr variable (default False)
        arr : list, optional
            if singles is True, arr is a list of trial numbers to be cropped
            and is usually disjointed (default empty list)

        """
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
        
    def performance_in_trials(self, trials):
        """
        Get the performance as a percentage correct for the given trials numbers

        Parameters
        ----------
        trials : list
            List of trials to calculate correctness.

        Returns
        -------
        A single number corresponding to proportion correct in left and right trials.

        """
        
        proportion_correct_left = np.sum(self.L_correct[trials]) / np.sum(self.L_correct[trials] + self.L_wrong[trials] + self.L_ignore[trials])
        proportion_correct_right = np.sum(self.R_correct[trials]) /  np.sum(self.R_correct[trials] + self.R_wrong[trials] + self.R_ignore[trials])
        proportion_correct = np.sum(self.L_correct[trials] + self.R_correct[trials]) / np.sum(self.L_correct[trials] + 
                                                                                              self.L_wrong[trials] + 
                                                                                              self.L_ignore[trials] +
                                                                                              self.R_correct[trials] + 
                                                                                              self.R_wrong[trials] + 
                                                                                              self.R_ignore[trials])
                                                                                          
    
        return proportion_correct_right, proportion_correct_left, proportion_correct
    
    def lick_correct_direction(self, direction):
        """Finds trial numbers corresponding to correct lick in specified direction

        Parameters
        ----------
        direction : str
            'r' or 'l' indicating desired lick direction
        
        Returns
        -------
        idx : array
            list of correct, no early lick, i_good trials licking in specified
            direction
        """
        
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
        """Finds trial numbers corresponding to incorrect lick in specified direction

        Parameters
        ----------
        direction : str
            'r' or 'l' indicating desired lick direction
        
        Returns
        -------
        idx : array
            list of incorrect, no early lick, i_good trials licking in specified
            direction
        """
        
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
    
    def lick_actual_direction(self, direction):
        """Finds trial numbers corresponding to an actual lick direction
        
        Filters out early lick and non i_good trials but includes correct and 
        error trials

        Parameters
        ----------
        direction : str
            'r' or 'l' indicating desired lick direction
        
        Returns
        -------
        idx : array
            list of trials corresponding to specified lick direction
        """
        
        ## Returns list of indices of actual lick left/right trials
        
        if direction == 'l':
            idx = np.where((self.L_correct + self.L_wrong) == 1)[0]
        elif direction == 'r':
            idx = np.where((self.R_correct + self.R_wrong) == 1)[0]
        else:
            raise Exception("Sorry, only 'r' or 'l' input accepted!")
            
        early_idx = np.where(self.early_lick == 1)[0]
        
        idx = [i for i in idx if i not in early_idx]
        
        idx = [i for i in idx if i in self.i_good_trials]
        
        return idx
    
    def trial_type_direction(self, direction):
        """Finds trial numbers corresponding to trial type direction
        
        Filters out early lick and non i_good trials but includes correct and 
        error trials

        Parameters
        ----------
        direction : str
            'r' or 'l' indicating desired trial type
        
        Returns
        -------
        idx : array
            list of trials corresponding to specified lick direction
        """
        
        ## Returns list of indices of actual lick left/right trials
        
        if direction == 'l':
            idx = np.where((self.L_correct + self.L_wrong) == 1)[0]
        elif direction == 'r':
            idx = np.where((self.R_correct + self.R_wrong) == 1)[0]
        else:
            raise Exception("Sorry, only 'r' or 'l' input accepted!")
            
        early_idx = np.where(self.early_lick == 1)[0]
        
        idx = [i for i in idx if i not in early_idx]
        
        idx = [i for i in idx if i in self.i_good_trials]
        
        return idx
    
    def get_trace_matrix(self, neuron_num, error=False, bias_trials = [], rtrials=[],ltrials=[], non_bias=False, both=False, lickdir=False, trialtype = False, opto=False):
        
        """Returns matrices of dF/F0 traces over right/left trials of a single neuron
        
        Two matrices for right and left trials respectively. Opto trials are
        filtered out by default.

        Parameters
        ----------
        neuron_num : int
            Neuron number to get matrix from 
        error : bool, optional
            Indicates if correct or incorrect trials wanted (default False)
        bias_trials :  list, optional
            List of bias trials that are used to build matrix (default empty list)
        rtrials and ltrials : lists, optional
            Which specific right and left trials to grab
        non_bias : bool, optional
            If True, returns all trials NOT in previous bias_trials variable
            (default False)
        both : bool, optional,
            If True, returns both correct and incorrect trials, sorted by 
            instructed lick direction (default False)
        lickdir : bool, optional
            If True, returns matrix R/L based on actual lick direction instead
            of instructed (default False)
        trialtype : bool, optional
            If True, returns matrix R/L based on trial type direction 
        opto : bool, optional
            If True, returns only the optogenetic stimulation trials. Otherwise,
            only returns the control trials. (default False)
            
        Returns
        -------
        R_av_dff, L_av_dff : list of lists
            Two lists of dF/F0 traces over all right and left trials
        """
        

        if lickdir:
            R,L = self.lick_actual_direction('r'), self.lick_actual_direction('l')
        elif trialtype or opto: # always group opto by trial type
            R,L = self.trial_type_direction('r'), self.trial_type_direction('l')
        else:
            R,L = self.lick_correct_direction('r'), self.lick_correct_direction('l')
        
        right_trials = R
        left_trials = L
        
        if error:
            right_trials = self.lick_incorrect_direction('r')
            left_trials = self.lick_incorrect_direction('l')
        
        if both:
            right_trials = cat((R, self.lick_incorrect_direction('r')))
            left_trials = cat((L, self.lick_incorrect_direction('l')))
        
        if len(bias_trials) != 0:
            right_trials = [b for b in bias_trials if self.instructed_side[b] == 0]
            left_trials = [b for b in bias_trials if self.instructed_side[b] == 1]
            
            if lickdir:
                right_trials = [b for b in bias_trials if b in R]
                left_trials = [b for b in bias_trials if b in L]
                

            if non_bias: # Get control trials - bias trials
            
                ctlright_trials = R
                ctlleft_trials = L
                right_trials = [b for b in ctlright_trials if b not in bias_trials]
                left_trials = [b for b in ctlleft_trials if b not in bias_trials]

        if len(rtrials) > 0:
            right_trials = rtrials

        if len(ltrials) > 0:
            left_trials = ltrials
            
        # Filter out opto trials
        if not opto:
            right_trials = [r for r in right_trials if not self.stim_ON[r]]
            left_trials = [r for r in left_trials if not self.stim_ON[r]]
        elif opto:
            right_trials = [r for r in right_trials if self.stim_ON[r]]
            left_trials = [r for r in left_trials if self.stim_ON[r]]
            
        R_av_dff = []
        for i in right_trials:
            # R_av_dff += [self.normalize_by_baseline(self.dff[0, i][neuron_num, :self.time_cutoff])]
            R_av_dff += [self.dff[0, i][neuron_num, :self.time_cutoff]]

        L_av_dff = []
        for i in left_trials:
            # L_av_dff += [self.normalize_by_baseline(self.dff[0, i][neuron_num, :self.time_cutoff])]
            L_av_dff += [self.dff[0, i][neuron_num, :self.time_cutoff]]
            
            
        return R_av_dff, L_av_dff
    

    
    def get_trace_matrix_multiple(self, neuron_nums, error=False, bias_trials = [], rtrials=[],ltrials=[], non_bias=False, both=False, lickdir=False, trialtype = False, opto=False):
        
       
        """Returns matrices of dF/F0 traces averaged over right/left trials of multiple neurons
        
        Two matrices for right and left trials respectively. Opto trials are
        filtered out by default. Different than get_trace_matrix because it grabs
        multiple neurons at once.

        Parameters
        ----------
        neuron_nums : list
            List of neuron numbers to get matrix from 
        error : bool, optional
            Indicates if correct or incorrect trials wanted (default False)
        bias_trials :  list, optional
            List of bias trials that are used to build matrix (default empty list)
        rtrials and ltrials : lists, optional
            Which specific right and left trials to grab
        non_bias : bool, optional
            If True, returns all trials NOT in previous bias_trials variable
            (default False)
        both : bool, optional,
            If True, returns both correct and incorrect trials, sorted by 
            instructed lick direction (default False)
        lickdir : bool, optional
            If True, returns matrix R/L based on actual lick direction instead
            of instructed (default False)
        trialtype : bool, optional
            If True, returns matrix R/L based on trial type direction 
        opto : bool, optional
            If True, returns only the optogenetic stimulation trials. Otherwise,
            only returns the control trials. (default False)
            
        Returns
        -------
        R, L : array of lists
            Two array of dF/F0 traces of length = number of neurons
            over all right and left trials
        """
                
        R, L = [], []
        
        for neuron_num in neuron_nums:
            R_av_dff, L_av_dff = self.get_trace_matrix(neuron_num, 
                                                       error=error, 
                                                       bias_trials = bias_trials, 
                                                       rtrials=[],
                                                       ltrials=[],
                                                       non_bias=non_bias, 
                                                       both=both, 
                                                       lickdir=lickdir,
                                                       trialtype=trialtype,
                                                       opto=opto)
 
            
            R += [np.mean(R_av_dff, axis = 0)]
            L += [np.mean(L_av_dff, axis = 0)]
            
        return np.array(R), np.array(L)

    def plot_PSTH(self, neuron_num, opto = False):
        
        """Plots single neuron PSTH over R/L trials
        
        Right trials plotted in blue, left in red.

        Parameters
        ----------
        neuron_num : int
            Neuron number to plot 
        opto : bool, optional
            Plotting opto trials or not (default False)

        """
        
        if not opto:
            R, L = self.get_trace_matrix(neuron_num)
        else:
            R, L = self.get_trace_matrix(neuron_num, opto=True)

        
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
        """Plots single neuron PSTH on a single trial

        Parameters
        ----------
        trial : int
            Trial number to plot
        neuron_num : int
            Neuron number to plot 
        """
        plt.plot(self.dff[0, trial][neuron_num], 'b-')
        plt.title("Neuron {}: PSTH for trial {}".format(neuron_num, trial))
        plt.show()
        

    def plot_population_PSTH(self, neurons, opto = False):
        """Plots many neurons PSTH over R/L trials

        Parameters
        ----------
        neurons : list
            Neuron numbers to plot
        opto : int, optional
            If plotting opto trials (default False) 
        """
        
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
        """Normalize a list by its first 7 timesteps
        
        Parameters
        ----------
        trace : list
            Non descript list of more than 7 length 
            
        Returns
        -------
        list
            Original trace after 'normalizing' step
        """

        mean = np.mean(trace[:7])
        if mean == 0:
            raise Exception("Neuron has mean 0.")
            
        return (trace - mean) / mean # norm by F0
    
    def normalize_all_by_neural_baseline(self):
        """Normalize all neurons by each neuron's trial-averaged F0
        
        Calculates F0 separately for each neuron, defined by its trial-averaged
        F for the 3 timebins preceding the sample period. F0 is then subtracted
        and divided from each trace on each trial. Modifies self.dff directly.
            
        """
        # 
        
        for i in range(self.num_neurons):
        # for i in self.good_neurons:

            nmean = np.mean([self.dff[0, t][i, self.sample-3:self.sample] for t in range(self.num_trials)]).copy()
            # nmean = np.mean([self.dff[0, t][i, self.sample-3:self.sample] for t in self.i_good_trials]).copy()
            
            for j in range(self.num_trials):
            # for j in self.i_good_trials:
                
                # nmean = np.mean(self.dff[0, j][i, :7])
                self.dff[0, j][i] = (self.dff[0, j][i] - nmean) / nmean
        
    
    def normalize_all_by_baseline(self):
        """Normalize all neurons by each neuron's F0 on each trial
        
        Calculates F0 separately for each neuron, defined by its F for the 3 
        timebins preceding the sample period every trial. F0 is then subtracted
        and divided from each trace on each trial. Modifies self.dff directly.
            
        """
        
        dff = self.dff.copy()

        for i in range(self.num_neurons):

            
            for j in range(self.num_trials):
            # for j in self.i_good_trials:

                nmean = np.mean(dff[0, j][i, self.sample-3:self.sample]) # later cutoff because of transient activation
                self.dff[0, j][i, :] = (self.dff[0, j][i] - nmean) / nmean

    def normalize_by_histogram(self):
        """Normalize all neurons by each neuron's F0 based on bottom quantile over all trials
        
        Calculates F0 separately for each neuron, defined by the bottom 10% 
        quantile over all timebins on all trials. F0 is then subtracted
        and divided from each trace on each trial. Modifies self.dff directly.
            
        """
        
        for i in range(self.num_neurons):
        # for i in self.good_neurons:
            
            # nmean = np.quantile(cat([self.dff[0,t][i, :] for t in range(self.num_trials)]), q=0.10)
            nmean = np.quantile(cat([self.dff[0,t][i, :] for t in self.i_good_trials]), q=0.10)
            
            # for j in range(self.num_trials):
            for j in self.i_good_trials:

                self.dff[0, j][i] = (self.dff[0, j][i] - nmean) / nmean
        
    
    def normalize_all_by_histogram(self):
        """Normalize all neurons by each neuron's F0 based on bottom quantile for each trial
        
        Calculates F0 separately for each neuron, defined by the bottom 10% 
        quantile over all timebins for each trial. F0 is then subtracted
        and divided from each trace on each trial. Modifies self.dff directly.
            
        """
        
        for i in range(self.num_neurons):
                        
            # for j in range(self.num_trials):
            for j in self.i_good_trials:
                nmean = np.quantile(self.dff[0, j][i, :], q=0.10)

                self.dff[0, j][i] = (self.dff[0, j][i] - nmean) / nmean
        
    
    def normalize_z_score(self):
        """Z-score normalizes all neurons traces
        
        Overall z-score normalization of all neurons over all traces.
            
        """
        # Normalize by mean of all neurons in layer
        
        # overall_mean = np.mean(cat([cat(i) for i in self.dff[0]])).copy()
        # std = np.std(cat([cat(i) for i in self.dff[0]])).copy()

        overall_mean = np.mean(cat([cat(self.dff[0, i]) for i in self.i_good_trials])).copy()
        std = np.std(cat([cat(self.dff[0, i]) for i in self.i_good_trials])).copy()
        
        # for i in range(self.num_trials):
        for i in self.i_good_trials:
            for j in range(self.num_neurons):
                self.dff[0, i][j] = (self.dff[0, i][j] - overall_mean) / std
                
    def is_selective(self, neuron, epoch, p = 0.0001, bias=False, lickdir = False):
        right, left = self.get_trace_matrix(neuron)
        if lickdir:
            right, left = self.get_trace_matrix(neuron, lickdir=True)
            
            
        if bias:
            biasidx = self.find_bias_trials()
            right,left = self.get_trace_matrix(neuron, bias_trials= biasidx)
        
        left_ = [l[epoch] for l in left]
        right_ = [r[epoch] for r in right]
        tstat, p_val = stats.ttest_ind(np.mean(left_, axis = 1), np.mean(right_, axis = 1))
        # p = 0.01/self.num_neurons
        # p = 0.01
        # p = 0.0001
        return p_val < p
        
        

    def get_epoch_selective(self, epoch, p = 0.0001, bias=False, rtrials=[], ltrials=[], return_stat = False, lickdir = False):
        """Identifies neurons that are selective in a given epoch
        
        Saves neuron list in self.selective_neurons as well.
        
        Parameters
        ----------
        epoch : list
            Range of timesteps to evaluate selectivity over
        p : int, optional
            P-value cutoff to be deemed selectivity (default 0.0001)
        bias : bool, optional
            If true, only use the bias trials to evaluate (default False)
        rtrials, ltrials: list, optional
            If provided, use these trials to evaluate selectivty
        return_stat : bool, optional
            If true, returns the t-statistic to use for ranking
        lickdir : bool, optional
            If True, use the lick direction instaed of correct only
            
        Returns
        -------
        list
            List of neurons that are selective
        list, optional
            T-statistic associated with neuron, positive if left selective, 
            negative if right selective
        """
        selective_neurons = []
        all_tstat = []
        # for neuron in range(self.num_neurons):
        for neuron in self.good_neurons: # Only look at non-noise neurons
            right, left = self.get_trace_matrix(neuron, rtrials=rtrials, ltrials=ltrials)
            if lickdir:
                right, left = self.get_trace_matrix(neuron, lickdir=True, rtrials=rtrials, ltrials=ltrials)
                
                
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
                all_tstat += [tstat] # Positive if L selective, negative if R selective
        # print("Total delay selective neurons: ", len(selective_neurons))
        self.selective_neurons = selective_neurons
        
        if return_stat:
            return selective_neurons, all_tstat
        
        return selective_neurons
    
    def get_epoch_tstat(self, epoch, neurons, bias=False, rtrials=[], ltrials=[], lickdir = False):
        """
        Get tstat of provided neurons during provided epoch

        Parameters
        ----------
        epoch : list
            DESCRIPTION.
        neurons : list
            DESCRIPTION.
        rtrials, ltrials: list, optional
            If provided, use these trials to evaluate selectivty

        Returns
        -------
        List of t-statistics associated with input neurons.

        """
        
        all_tstat = []
        poststat, negtstat = [],[]
        # for neuron in range(self.num_neurons):
        for neuron in neurons: # Only look at provided neurons
            right, left = self.get_trace_matrix(neuron, rtrials=rtrials, ltrials=ltrials)
            if lickdir:
                right, left = self.get_trace_matrix(neuron, lickdir=True, rtrials=rtrials, ltrials=ltrials)
                
                
            if bias:
                biasidx = self.find_bias_trials()
                right,left = self.get_trace_matrix(neuron, bias_trials= biasidx)
            
            left_ = [l[epoch] for l in left]
            right_ = [r[epoch] for r in right]
            tstat, p_val = stats.ttest_ind(np.mean(left_, axis = 1), np.mean(right_, axis = 1))


            all_tstat += [tstat] # Positive if L selective, negative if R selective
            if tstat > 0:
                poststat += [tstat]
            else:
                negtstat += [tstat]


        return all_tstat, poststat, negtstat

    def get_epoch_selectivity(self, epoch, neurons, bias=False, rtrials=[], ltrials=[], lickdir = False):
        """
        Get adjusted L/R difference of provided neurons during provided epoch

        Parameters
        ----------
        epoch : list
            DESCRIPTION.
        neurons : list
            DESCRIPTION.
        rtrials, ltrials: list, optional
            If provided, use these trials to evaluate selectivty

        Returns
        -------
        List of diff associated with input neurons.

        """
        
        all_tstat = []
        poststat, negtstat = [],[]
        # for neuron in range(self.num_neurons):
        for neuron in neurons: # Only look at provided neurons
            right, left = self.get_trace_matrix(neuron, rtrials=rtrials, ltrials=ltrials)
            if lickdir:
                right, left = self.get_trace_matrix(neuron, lickdir=True, rtrials=rtrials, ltrials=ltrials)
                
                
            if bias:
                biasidx = self.find_bias_trials()
                right,left = self.get_trace_matrix(neuron, bias_trials= biasidx)
            
            left_ = [l[epoch] for l in left]
            right_ = [r[epoch] for r in right]
            diff = np.sum(np.mean(left_, axis = 0) - np.mean(right_, axis = 0))

            tstat = diff / np.sum(np.mean(left_, axis = 0) + np.mean(right_, axis = 0))
            all_tstat += [tstat] # Positive if L selective, negative if R selective
            if tstat > 0:
                poststat += [tstat]
            else:
                negtstat += [tstat]
                
        return all_tstat, poststat, negtstat
    
    
    def get_epoch_mean_diff(self, epoch, trials):
        """Identifies neurons that are selective in a given epoch using mean 
        difference during selected epoch
        
        Saves neuron list in self.selective_neurons as well.
        
        Parameters
        ----------
        epoch : list
            Range of timesteps to evaluate selectivity over
        trials : tuple
            List of r and l trials to use

            
        Returns
        -------
        list
            Mean differences of neurons
            

        """
        diffs = []
        r,l=trials
        # for neuron in range(self.num_neurons):
        for neuron in self.good_neurons: # Only look at non-noise neurons
            right, left = self.get_trace_matrix(neuron, rtrials=r, ltrials =l)

            
            left_ = [l[epoch] for l in left]
            right_ = [r[epoch] for r in right]
            
            #average across trials
            # d = np.mean(np.mean(right_,axis=0) - np.mean(left_,axis=0)) / np.mean(cat((np.mean(left_, axis = 1), np.mean(right_, axis = 1))))
            # d = (np.mean(right_) - np.mean(left_)) / np.mean(cat((np.mean(left_, axis = 1), np.mean(right_, axis = 1))))
            d = (np.mean(right_) - np.mean(left_))
            
            diffs += [d]
            
            # p = 0.01/self.num_neurons
            # p = 0.01
            # p = 0.0001
            # Positive if L selective, negative if R selective
        # print("Total delay selective neurons: ", len(selective_neurons))
        
        return diffs
        
     
    
    def screen_preference(self, neuron_num, epoch, samplesize = 10):
        """Determine if a neuron is left or right preferring
                
        Iterate 30 times over different test batches to get a high confidence
        estimation of neuron preference.
        
        Parameters
        ----------
        neuron_num : int
            Neuron to screen in function
        epoch : list
            Timesteps over which to evaluate the preference
        samplesize : int, optional
            Number of trials to use in the test batch (default 10)
            
        Returns
        -------
        choice : bool
            True if left preferring, False if right preferring
        l_trials, r_trials : list
            All left and right trials        
        """
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
        
        pref = 0
        for _ in range(30): # Perform 30 times
            
            # Pick 20 random trials as screen for left and right
            screen_l = np.random.choice(l_trials, size = samplesize, replace = False)
            screen_r = np.random.choice(r_trials, size = samplesize, replace = False)
        
            # Remainder of trials are left for plotting in left and right separately
            test_l = [t for t in l_trials if t not in screen_l]
            test_r = [t for t in r_trials if t not in screen_r]
            
            # Compare late delay epoch for preference
            avg_l = np.mean([np.mean(L[i][epoch]) for i in screen_l])
            avg_r = np.mean([np.mean(R[i][epoch]) for i in screen_r])
    
            pref += avg_l > avg_r
        
        choice = True if pref/30 > 0.5 else False
        # return avg_l > avg_r, test_l, test_r
        return choice, l_trials, r_trials

    def plot_selectivity(self, neuron_num, plot=True, epoch=[]):
        
        """Plots a single line representing selectivity of given neuron over all trials
        
        Evaluates the selectivity using preference in delay epoch
        
        Parameters
        ----------
        neuron_num : int
            Neuron to plot
        plot : bool, optional
            Whether to plot or not (default True)
        epoch : list, optional
            Timesteps to evaluate preference and selectivity over (default empty list)
            
        Returns
        -------
        list
            Selectivity calculated and plotted
        """
        if len(epoch) == 0:
            epoch = range(self.delay, self.response)
        
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
    
    def contra_ipsi_pop(self, epoch, return_sel = False, selective_n = [], p=0.0001):
        
        """Finds neurons that are left and right preferring 
        
        Returns as contra / ipsi but imaging hemisphere is always left side
        
        Parameters
        ----------
        epoch : list
            Timesteps over which to identify selective neurons
        return_sel : bool, optional
            Whether to return selectivity array at end (default False)
        selective_n : list, optional
            List of pre-selected selective neurons (default empty list)
        p : int, optional
            p-value to use to evaluate selectivity (default 0.0001)
            
        Returns
        -------
        contra_neurons, ipsi_neurons : list
            List of contra and ipsi selective neurons
        contra_LR, ipsi_LR : dict
            Dictionary of left/right trials traces stored as list of lists for 
            each neuron included
                
            OR
            
        list 
            Selectivity trace
        error : list
            Error margin
        """
        # Returns the neuron ids for contra and ipsi populations
        n = self.get_epoch_selective(epoch, p=0.01) if self.sample in epoch else self.get_epoch_selective(epoch, p=p)
        selective_neurons = n if len(selective_n) == 0 else selective_n
        
        contra_neurons = []
        ipsi_neurons = []
        
        contra_LR, ipsi_LR = dict(), dict()
        contra_LR['l'], contra_LR['r'] = [], []
        ipsi_LR['l'], ipsi_LR['r'] = [], []
        pref, nonpref = [], []
        
        for neuron_num in selective_neurons:
            
            # Skip sessions with fewer than 15 neurons
            if self.screen_preference(neuron_num, epoch) != 0:
                
                R, L = self.get_trace_matrix(neuron_num)

                pref_choice, test_l, test_r = self.screen_preference(neuron_num, epoch) 
        
                if self.recording_loc == 'l':

                    if pref_choice:
                        # print("Ipsi_preferring: {}".format(neuron_num))
                        ipsi_neurons += [neuron_num]
                        ipsi_LR['l'] += [[L[i] for i in test_l]]
                        ipsi_LR['r'] += [[R[i] for i in test_r]]
                        if return_sel:
                            # pref += [np.mean([L[i] for i in test_l], axis=0)]
                            # nonpref += [np.mean([R[i] for i in test_r], axis=0)]
                            pref += [L[i] for i in test_l]
                            nonpref += [R[i] for i in test_r]                   
                    else:
                        # print("Contra preferring: {}".format(neuron_num))
                        contra_neurons += [neuron_num] 
                        contra_LR['l'] += [[L[i] for i in test_l]]
                        contra_LR['r'] += [[R[i] for i in test_r]]
                        
                        if return_sel:
                            # nonpref += [np.mean([L[i] for i in test_l], axis=0)]
                            # pref += [np.mean([R[i] for i in test_r], axis=0)]
                            
                            nonpref += [L[i] for i in test_l]
                            pref += [R[i] for i in test_r]         
                            
                elif self.recording_loc == 'r':

                    if not pref:
                        ipsi_neurons += [neuron_num]
                        ipsi_LR['l'] += [L[i] for i in test_l]
                        ipsi_LR['r'] += [R[i] for i in test_r]
                    else:
                        contra_neurons += [neuron_num] 
                        contra_LR['l'] += [L[i] for i in test_l]
                        contra_LR['r'] += [R[i] for i in test_r]
                        

        if return_sel:
            # err = np.std(pref, axis=0) / np.sqrt(len(pref)) 
            # err += np.std(nonpref, axis=0) / np.sqrt(len(nonpref))
            # return np.mean(pref, axis=0) - np.mean(nonpref, axis=0), err
            return nonpref, pref
        else:
            return contra_neurons, ipsi_neurons, contra_LR, ipsi_LR
    
    def plot_contra_ipsi_pop(self, e=False, bias=False, filter_dendrites = False):
        """Plots contra and ipsi preferring neurons' traces in two plots
        
                
        Parameters
        ----------
        e : list, optional
            Timesteps over which to identify selective neurons (default False)
            Default is to use delay period if False
        bias : bool, optional
            Whether to use biased trials or not (default False)
        filter_dendrites : numpy array or bool, optional
            If provided, list of neuron IDs corresponding to somas only 
            (default False)

        """
        
        x = np.arange(-5.97,4,self.fs)[:self.time_cutoff]
        x = np.arange(-5.97,4,self.fs)[2:self.time_cutoff] if 'CW03' not in self.path else np.arange(-6.97,4,self.fs)[2:self.time_cutoff+2]

        epoch = e if e != False else range(self.delay, self.response)
        
        contra_neurons, ipsi_neurons, contra_trace, ipsi_trace = self.contra_ipsi_pop(epoch)
        
        if filter_dendrites != False:
            
            contra_neurons = [c for c in contra_neurons if c in filter_dendrites[self.layer_num - 1]]
            ipsi_neurons = [c for c in ipsi_neurons if c in filter_dendrites[self.layer_num - 1]]

            contra_trace['l'] = [contra_trace['l'][c] for c in range(len(contra_neurons)) if contra_neurons[c] in filter_dendrites[self.layer_num - 1]]
            contra_trace['r'] = [contra_trace['r'][c] for c in range(len(contra_neurons)) if contra_neurons[c] in filter_dendrites[self.layer_num - 1]]
            ipsi_trace['l'] = [ipsi_trace['l'][c] for c in range(len(ipsi_neurons)) if ipsi_neurons[c] in filter_dendrites[self.layer_num - 1]]
            ipsi_trace['r'] = [ipsi_trace['r'][c] for c in range(len(ipsi_neurons)) if ipsi_neurons[c] in filter_dendrites[self.layer_num - 1]]


                    
        if len(ipsi_neurons) != 0:
        
            overall_R, overall_L = ipsi_trace['r'], ipsi_trace['l']
            print(len(ipsi_trace['l']))
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
        """Plots preferred and nonpreferred traces for all selective neurons in one graph
        
        TODO: add filter_dendrites variable in
                
        Parameters
        ----------
        e : list or bool, optional
            Timesteps over which to identify selective neurons (default False)
            Default is to use delay period if False
        bias : bool, optional
            Whether to use biased trials or not (default False)
        filter_dendrites : numpy array or bool, optional
            If provided, list of neuron IDs corresponding to somas only 
            (default False)

        """
        
        x = np.arange(-5.97,4,self.fs)[:self.time_cutoff]

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

    def plot_prefer_nonprefer_sidebyside(self, e=False):
        
        """Plots preferred and nonpreferred traces for all selective neurons in control vs bias trials
        
        TODO: add filter_dendrites and bias_trials variable in
                
        Parameters
        ----------
        e : list or bool, optional
            Timesteps over which to identify selective neurons (default False)
            Default is to use delay period if False
        filter_dendrites : numpy array or bool, optional
            If provided, list of neuron IDs corresponding to somas only 
            (default False)

        """
        
        x = np.arange(-5.97,4,self.fs)[:self.time_cutoff]
        f, axarr = plt.subplots(1,2, sharex=True, sharey=True, figsize=(20,7))

        epoch = e if e != False else range(self.delay, self.response)
        
        contra_neurons, ipsi_neurons, contra_trace, ipsi_trace = self.contra_ipsi_pop(epoch)
        
        pref, nonpref = [], []
        preferr, nonpreferr = [], []

        for i in range(2):       
            
            if len(ipsi_neurons) != 0:
            
                overall_R, overall_L = ipsi_trace['r'], ipsi_trace['l']
                overall_R = np.array([np.mean(overall_R[r], axis=0) for r in range(len(overall_R))])
                overall_L = np.array([np.mean(overall_L[l], axis=0) for l in range(len(overall_L))])
                
                if i:
                    overall_R, overall_L = self.get_trace_matrix_multiple(ipsi_neurons, bias_trials=self.find_bias_trials(), lickdir=True)
                
                else:
                    overall_R, overall_L = self.get_trace_matrix_multiple(ipsi_neurons, bias_trials=self.find_bias_trials(), non_bias=True)
                    
                
                pref, nonpref = overall_L, overall_R
                
            else:
                print('No ipsi selective neurons')
        
            if len(contra_neurons) != 0:
    
                overall_R, overall_L = contra_trace['r'], contra_trace['l']
                overall_R = np.array([np.mean(overall_R[r], axis=0) for r in range(len(overall_R))])
                overall_L = np.array([np.mean(overall_L[l], axis=0) for l in range(len(overall_L))])
                
                if i:
                    overall_R, overall_L = self.get_trace_matrix_multiple(contra_neurons, bias_trials=self.find_bias_trials(), lickdir=True)
                else:
                    
                    overall_R, overall_L = self.get_trace_matrix_multiple(contra_neurons, bias_trials=self.find_bias_trials(), non_bias=True)

                
                pref, nonpref = np.vstack((pref, overall_R)), np.vstack((nonpref, overall_L))
    
                            
    
            else:
                print('No contra selective neurons')
                
            
            nonpreferr = np.std(nonpref, axis=0) / np.sqrt(len(nonpref)) 
            preferr = np.std(pref, axis=0) / np.sqrt(len(pref))
                        
            pref, nonpref = np.mean(pref, axis = 0), np.mean(nonpref, axis = 0)
    
            axarr[i].plot(x, pref, 'r-', label='Pref')
            axarr[i].plot(x, nonpref, 'darkgrey', label='Non-pref')
            
    
            axarr[i].fill_between(x, pref - preferr, 
                      pref + preferr,
                      color=['#ffaeb1'])
            axarr[i].fill_between(x, nonpref - nonpreferr, 
                      nonpref + nonpreferr,
                      color='lightgrey')
            axarr[i].legend()

        axarr[0].set_title("Control selectivity")
        axarr[1].set_title("Bias trial selectivity")

        axarr[0].set_xlabel('Time from Go cue (s)')
        axarr[0].set_ylabel('Population trace')

    def plot_pref_overstates(self, e=False, opto=False, lickdir=True):
        
        """Plots preferred and nonpreferred traces for all selective neurons across 3 behavioral states and control
        
        Non states defined as trials not above a certain probability for any of 3 states
                        
        Parameters
        ----------
        e : list or bool, optional
            Timesteps over which to identify selective neurons (default False)
            Default is to use delay period if False
        opto : bool, optional
            Whether to plot opto trials (default False)
        lickdir : bool, optional
            Whether to use actual lick direction or correct trials only

        """
        
        x = np.arange(-5.97,4,self.fs)[2:self.time_cutoff] if 'CW03' not in self.path else np.arange(-6.97,6,self.fs)[2:self.time_cutoff]
        titles = ['Non state selectivity', 'State 1 selectivity', 'State 2 selectivity', 'State 3 selectivity', 'State 4 selectivity']
        epoch = e if e != False else range(self.delay, self.response)
        states = np.load(r'{}\states.npy'.format(self.path))
        num_state = states.shape[1]
        
        f, axarr = plt.subplots(1,num_state + 1, sharex=True, sharey=True, figsize=(20,5))

        contra_neurons, ipsi_neurons, contra_trace, ipsi_trace = self.contra_ipsi_pop(epoch)
        

        for i in range(num_state + 1):       
            
            if len(ipsi_neurons) == 0 and len(contra_neurons) == 0:
                print("No selective neurons in state {}".format('nonstate' if i == 0 else i + 1))
                continue
                
            pref, nonpref = np.zeros(self.time_cutoff), np.zeros(self.time_cutoff)
            
            if len(ipsi_neurons) != 0:
            
                # overall_R, overall_L = ipsi_trace['r'], ipsi_trace['l']
                # overall_R = np.array([np.mean(overall_R[r], axis=0) for r in range(len(overall_R))])
                # overall_L = np.array([np.mean(overall_L[l], axis=0) for l in range(len(overall_L))])
                
                if i:
                    overall_R, overall_L = self.get_trace_matrix_multiple(ipsi_neurons, bias_trials=self.find_bias_trials(state = i-1), opto=opto, lickdir=lickdir)
                
                else:
                    _ = self.find_bias_trials()
                    overall_R, overall_L = self.get_trace_matrix_multiple(ipsi_neurons, bias_trials=self.nonstate_trials, opto=opto, lickdir=lickdir)
                
                pref, nonpref = np.vstack((pref, overall_L)), np.vstack((nonpref, overall_R))

                # pref, nonpref = overall_L, overall_R
                
            else:
                print('No ipsi selective neurons')
        
            if len(contra_neurons) != 0:
    
                # overall_R, overall_L = contra_trace['r'], contra_trace['l']
                # overall_R = np.array([np.mean(overall_R[r], axis=0) for r in range(len(overall_R))])
                # overall_L = np.array([np.mean(overall_L[l], axis=0) for l in range(len(overall_L))])
                
                if i:
                    overall_R, overall_L = self.get_trace_matrix_multiple(contra_neurons, bias_trials=self.find_bias_trials(state = i-1), opto=opto, lickdir=lickdir)
                else:
                    _ = self.find_bias_trials()
                    overall_R, overall_L = self.get_trace_matrix_multiple(contra_neurons, bias_trials=self.nonstate_trials, opto=opto, lickdir=lickdir)
                
                pref, nonpref = np.vstack((pref, overall_R)), np.vstack((nonpref, overall_L))
    
                            
    
            else:
                print('No contra selective neurons')
            
            
            pref, nonpref = pref[1:], nonpref[1:]
            
            nonpreferr = np.std(nonpref, axis=0) / np.sqrt(len(nonpref)) 
            preferr = np.std(pref, axis=0) / np.sqrt(len(pref))
                        
            pref, nonpref = np.mean(pref, axis = 0), np.mean(nonpref, axis = 0)
    
            pref, nonpref, nonpreferr, preferr = pref[2:], nonpref[2:], nonpreferr[2:], preferr[2:]
    
            axarr[i].plot(x, pref, 'r-', label='Pref')
            axarr[i].plot(x, nonpref, 'darkgrey', label='Non-pref')
            
    
            axarr[i].fill_between(x, pref - preferr, 
                      pref + preferr,
                      color=['#ffaeb1'])
            axarr[i].fill_between(x, nonpref - nonpreferr, 
                      nonpref + nonpreferr,
                      color='lightgrey')
            
            axarr[i].legend()
            axarr[i].axvline(-4.3, linestyle = '--')
            axarr[i].axvline(-3, linestyle = '--')
            axarr[i].axvline(0, linestyle = '--')
            axarr[i].set_title(titles[i])

        axarr[0].set_xlabel('Time from Go cue (s)')
        axarr[0].set_ylabel('Population trace')
        
    def plot_selectivity_overstates(self, e=False, opto=False, lickdir=True):
        """Plots selectivity traces for all selective neurons across 3 behavioral states and control in one graph
        
        Non states defined as trials not above a certain probability for any of 3 states
                        
        Parameters
        ----------
        e : list or bool, optional
            Timesteps over which to identify selective neurons (default False)
            Default is to use delay period if False
        opto : bool, optional
            Whether to plot opto trials (default False)
        lickdir : bool, optional
            Whether to use actual lick direction or correct trials only

        """
        
        x = np.arange(-5.97,4,self.fs)[2:self.time_cutoff] if 'CW03' not in self.path else np.arange(-6.97,4,self.fs)[2:self.time_cutoff]
        # f, axarr = plt.subplots(1,4, sharex=True, sharey=True, figsize=(20,5))
        states = np.load(r'{}\states.npy'.format(self.path))
        num_state = states.shape[1]
        titles = ['Non state selectivity', 'State 1 selectivity', 'State 2 selectivity', 'State 3 selectivity', 'State 4 selectivity']
        colors = ['grey', 'green', 'blue', 'salmon', 'yellow']
        if e != False:
            epoch = e
            contra_neurons, ipsi_neurons, contra_trace, ipsi_trace = self.contra_ipsi_pop(epoch, p=0.01)

        else:
            epoch = range(self.delay, self.response)
            contra_neurons, ipsi_neurons, contra_trace, ipsi_trace = self.contra_ipsi_pop(epoch)

        
        pref, nonpref = [], []
        pref, nonpref = np.zeros(self.time_cutoff), np.zeros(self.time_cutoff)

        for i in range(num_state + 1):       
            
            if len(ipsi_neurons) == 0 and len(contra_neurons) == 0:
                print("No selective neurons in state {}".format('nonstate' if i == 0 else i + 1))
                continue
                
            if len(ipsi_neurons) != 0:
            
                # overall_R, overall_L = ipsi_trace['r'], ipsi_trace['l']
                # overall_R = np.array([np.mean(overall_R[r], axis=0) for r in range(len(overall_R))])
                # overall_L = np.array([np.mean(overall_L[l], axis=0) for l in range(len(overall_L))])
                
                if i:
                    overall_R, overall_L = self.get_trace_matrix_multiple(ipsi_neurons, bias_trials=self.find_bias_trials(state = i-1), opto=opto, lickdir=lickdir)
                
                else:
                    _ = self.find_bias_trials()
                    overall_R, overall_L = self.get_trace_matrix_multiple(ipsi_neurons, bias_trials=self.nonstate_trials, opto=opto, lickdir=lickdir)
                    
                pref, nonpref = np.vstack((pref, overall_L)), np.vstack((nonpref, overall_R))

                # pref, nonpref = overall_L, overall_R
                
            else:
                print('No ipsi selective neurons')
        
            if len(contra_neurons) != 0:
    
                # overall_R, overall_L = contra_trace['r'], contra_trace['l']
                # overall_R = np.array([np.mean(overall_R[r], axis=0) for r in range(len(overall_R))])
                # overall_L = np.array([np.mean(overall_L[l], axis=0) for l in range(len(overall_L))])
                
                if i:
                    overall_R, overall_L = self.get_trace_matrix_multiple(contra_neurons, bias_trials=self.find_bias_trials(state = i-1), opto=opto, lickdir=lickdir)
                else:
                    _ = self.find_bias_trials()
                    overall_R, overall_L = self.get_trace_matrix_multiple(contra_neurons, bias_trials=self.nonstate_trials, opto=opto, lickdir=lickdir)
                
                # print(i)
                # print(pref, nonpref)
                # print()
                pref, nonpref = np.vstack((pref, overall_R)), np.vstack((nonpref, overall_L))
    
                            
    
            else:
                print('No contra selective neurons')
                
            pref, nonpref = pref[1:], nonpref[1:]
            
            selerr = np.std(np.vstack((nonpref, pref)), axis=0) / np.sqrt(len(np.vstack((nonpref, pref))))
                        
            sel = np.mean(pref, axis = 0) - np.mean(nonpref, axis = 0)
    
            sel, selerr = sel[2:], selerr[2:]
    
            plt.plot(x, sel, color = colors[i], label=titles[i])
            
    
            plt.fill_between(x, sel - selerr, 
                      sel + selerr,
                      color='light' + colors[i])
       
            print(i)
        plt.legend()
        plt.axvline(-4.3, linestyle = '--')
        plt.axvline(-3, linestyle = '--')
        plt.axvline(0, linestyle = '--')

        plt.xlabel('Time from Go cue (s)')
        plt.ylabel('Selectivity')
        plt.show()
        
    def plot_individual_raster(self, neuron_num):
        """Plots greyscale heatmap-style graph of a single neuron across all trials
                                
        Parameters
        ----------
        neuron_num : int
            Neuron number to be plotted
        
        Returns
        -------
        numpy matrix
            Traces from all trials for given neuron

        """
                
        trace = [self.dff[0, t][neuron_num, :self.time_cutoff] for t in range(self.num_trials)]

        vmin, vmax = min(cat(trace)), max(cat(trace))
        trace = np.matrix(trace)
        
        plt.matshow(trace, cmap='gray', vmin=vmin, vmax=vmax)
        plt.axis('off')
        
        return trace
        
    def plot_left_right_raster(self, neuron_num, opto=False):
        """Plots greyscale heatmap-style graph of a single neuron right trials then left trials
                                
        Parameters
        ----------
        neuron_num : int
            Neuron number to be plotted
        opto : bool, optional
            Whether to plot optogenetic trials or not
        
        Returns
        -------
        numpy matrix
            Traces from all trials for given neuron organized right then left trials

        """
        
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
        """Filters out neurons with low variance across trials
                                
        Parameters
        ----------
        neuron_num : int
            Neuron number to be evaluated
        
        Returns
        -------
        bool
            True if keep, False if discard

        """
        
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
        """Plot heatmap then averaged L/R trace for a single neuron
                                
        Parameters
        ----------
        neuron_num : int
            Neuron number to be evaluated
        opto : bool, optional
            Whether to plot optogenetic trials or not (default False)
        bias: bool, optional
            Whether to plot bias trials (default False)
            
        """
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
        

    def plot_rasterPSTH_sidebyside(self, neuron_num, bias=False,save=[]):
        """Plot heatmap then averaged L/R trace for a single neuron comparing control and opto trials
                                
        Parameters
        ----------
        neuron_num : int
            Neuron number to be evaluated
        opto : bool, optional
            Whether to plot optogenetic trials or not (default False)
        bias: bool, optional
            Whether to plot bias trials (default False)
        """
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
    
            R, L = self.get_trace_matrix(neuron_num, opto=True)
            r, l = self.get_trace_matrix(neuron_num, opto=True)
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
        
        if len(save) != 0:
            plt.savefig(save)
        plt.show()
        

### EPHYS PLOTS TO MY DATA ###

    def plot_number_of_sig_neurons(self, return_nums=False, save=False, y_axis = []):
        """Plots number of contra / ipsi neurons over course of trial
                                
        Parameters
        ----------
        return_nums : bool, optional
            return number of contra ispi neurons to do an aggregate plot
        
        save : bool, optional
            Whether to save fig to file (default False)
            
        y_axis : list, optional
            set top and bottom ylim
        """
        
        contra = np.zeros(self.time_cutoff)
        ipsi = np.zeros(self.time_cutoff)
        x = np.arange(-6.97,6,self.fs)[:self.time_cutoff]
        steps = range(self.time_cutoff)
        
        # if 'CW03' in self.path:
        #     contra = np.zeros(self.time_cutoff-5)
        #     ipsi = np.zeros(self.time_cutoff-5)
        #     x = np.arange(-5.97,4,self.fs)[:self.time_cutoff-5]
        #     steps = range(5, self.time_cutoff)

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
            
            contra[t] = sum(np.array(sig_neurons) == -1)
            ipsi[t] = sum(np.array(sig_neurons) == 1)
        
        if return_nums:
            return contra, ipsi

        plt.bar(x, contra, color = 'b', edgecolor = 'white', width = 0.17, label = 'contra')
        plt.bar(x, -ipsi, color = 'r',edgecolor = 'white', width = 0.17, label = 'ipsi')
        plt.axvline(-4.3)
        plt.axvline(-3)
        plt.axvline(0)
        if len(y_axis) != 0:
            plt.ylim(bottom = y_axis[0])
            plt.ylim(top = y_axis[1])
        plt.ylabel('Number of sig sel neurons')
        plt.xlabel('Time from Go cue (s)')
        plt.legend()
        
        if save:
            plt.savefig(self.path + r'number_sig_neurons.pdf')
        
        plt.show()
        
    def selectivity_table_by_epoch(self, save=False):
        """Plots table of L/R traces of selective neurons over three epochs and contra/ipsi population proportions
                                
        Parameters
        ----------
        save : bool, optional
            Whether to save fig to file (default False)
        """

        f, axarr = plt.subplots(4,3, sharex='col', figsize=(14, 12))
        epochs = [range(self.time_cutoff), range(self.sample, self.delay), range(self.delay, self.response), range(self.response, self.time_cutoff)]

        x = np.arange(-5.97,6,self.fs)[:self.time_cutoff]
        if 'CW03' in self.path:
            x = np.arange(-6.97,6,self.fs)[:self.time_cutoff]

        titles = ['Whole-trial', 'Sample', 'Delay', 'Response']
        
        for i in range(4):
            
            contra_neurons, ipsi_neurons, contra_trace, ipsi_trace = self.contra_ipsi_pop(epochs[i])

            # Bar plot
            contraratio = len(contra_neurons)/len(self.selective_neurons) if len(self.selective_neurons) > 0 else 0
            ipsiratio = len(ipsi_neurons)/len(self.selective_neurons) if len(self.selective_neurons) > 0 else 0
            
            axarr[i, 0].bar(['Contra', 'Ipsi'], [contraratio, ipsiratio], 
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
                            
                # if 'CW03' in self.path:
                #     L_av = L_av[5:]
                #     R_av = R_av[5:]
                #     left_err = left_err[5:]
                #     right_err = right_err[5:]
                    
                    
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
                            
                # if 'CW03' in self.path:
                #     L_av = L_av[5:]
                #     R_av = R_av[5:]
                #     left_err = left_err[5:]
                #     right_err = right_err[5:]
                    
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
        """Plots selectivity traces over three epochs and number of neurons in each population
                                
        Parameters
        ----------
        save : bool, optional
            Whether to save fig to file (default False)
        """
        
        f, axarr = plt.subplots(1,5, sharex='col', figsize=(21,5))
        
        epochs = [range(self.time_cutoff), range(8,14), range(19,28), range(29,self.time_cutoff)]
        x = np.arange(-5.97,4,self.fs)[:self.time_cutoff]
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
        
    def population_sel_timecourse(self, save=False):
        """Plots selectivity traces over three periods and number of neurons in each population
                                
        Parameters
        ----------
        save : bool, optional
            Whether to save fig to file (default False)
        """
        
        f, axarr = plt.subplots(2, 1, sharex='col', figsize=(20,15))
        epochs = [range(14,28), range(21,self.time_cutoff), range(29,self.time_cutoff)]
        x = np.arange(-5.97,4,self.fs)[:self.time_cutoff]
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


    def selectivity_optogenetics(self, save=False, p = 0.0001, return_traces = False):
        """Plots overall selectivity trace across opto vs control trials
        
        Uses late delay epoch to calculate selectivity
                                
        Parameters
        ----------
        save : bool, optional
            Whether to save fig to file (default False)
        p : int, optional
            P-value to use in the selectivity calculations
        """
        
        f, axarr = plt.subplots(1,1, sharex='col', figsize=(5,5))
        
        x = np.arange(-5.97,4,self.fs)[:self.time_cutoff] if 'CW03' not in self.path else np.arange(-6.97,4,self.fs)[:self.time_cutoff]

        # Get late delay selective neurons
        contra_neurons, ipsi_neurons, contra_trace, ipsi_trace = self.contra_ipsi_pop(range(self.response-9,self.response), p=p) 
        
        if len(contra_neurons) == 0 and len(ipsi_neurons) == 0:
            
            raise Exception("No selective neurons :^(") 
            
        elif len(contra_neurons) == 0:
            
            
            nonpref, pref = cat(ipsi_trace['r']), cat(ipsi_trace['l'])
            optonp, optop = self.get_trace_matrix_multiple(ipsi_neurons, opto=True, both=False)
            # errnp, errpref = self.get_trace_matrix_multiple(ipsi_neurons, opto=True, error=True)
            
        elif len(ipsi_neurons) == 0:
            
            nonpref, pref = cat(contra_trace['l']), cat(contra_trace['r'])
            optop, optonp = self.get_trace_matrix_multiple(contra_neurons, opto=True, both=False)
            # errpref, errnp = self.get_trace_matrix_multiple(contra_neurons, opto=True, error=True)

        else:
            
            nonpref, pref = cat((cat(ipsi_trace['r']), cat(contra_trace['l']))), cat((cat(ipsi_trace['l']), cat(contra_trace['r'])))
            optonp, optop = self.get_trace_matrix_multiple(ipsi_neurons, opto=True, both=False)
            optop1, optonp1 = self.get_trace_matrix_multiple(contra_neurons, opto = True, both=False)
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

        if 'CW03' in self.path:
            
            # sel = sel[5:]
            # selo = selo[5:]
            # err = err[5:]
            # erro = erro[5:]
            x = np.arange(-6.97,4,self.fs)[:self.time_cutoff]

        axarr.plot(x, sel, 'black')
                
        axarr.fill_between(x, sel - err, 
                  sel + err,
                  color=['darkgray'])
        
        axarr.plot(x, selo, 'r-')
                
        axarr.fill_between(x, selo - erro, 
                  selo + erro,
                  color=['#ffaeb1'])       
        
        axarr.axvline(-4.3, color = 'grey', alpha=0.5, ls = '--')
        axarr.axvline(-3, color = 'grey', alpha=0.5, ls = '--')
        axarr.axvline(0, color = 'grey', alpha=0.5, ls = '--')
        axarr.hlines(y=max(cat((selo, sel))), xmin=-3, xmax=-2, linewidth=10, color='red')

        axarr.set_title('Optogenetic effect on selectivity (n = {} neurons)'.format(len(self.selective_neurons)))                  
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
        
        if return_traces:
            return pref, nonpref, optop, optonp
            
    def single_neuron_sel(self, type, p=0.01, save=False):
        """Plots proportion of stim/lick/reward/mixed cells over trial using two different methods
        
        Inputs are 'Chen 2017' or 'Susu method'
                                
        Parameters
        ----------
        type : str
            'Chen 2017' or 'Susu method' to indicate which paper figure to replicate
        save : bool, optional
            Whether to save as pdf
            
        Returns 
        -------
        list (4)
            Neurons belong to stim/lick/reward/mixed categories

        """
        
        def mean_count(XX, timebin):
            
            coeff = 1/(len(XX) * len(timebin))
            
            # numerator = sum([sum(XX[t][timebin]) for t in range(len(XX))])
            numerator = np.mean([XX[t][timebin] for t in range(len(XX))], axis=1)
            
            # numerator = np.sum([XX[t][timebin] for t in range(len(XX))])
            # return numerator * coeff
            return numerator
        if type == 'Chen 2017':
            
            stim = []
            lick = []
            reward = []
            mixed = []
                
            for t in range(self.time_cutoff):
                start_time = time.time()

                s,l,r,m = 0,0,0,0
                
                # for n in range(self.num_neurons):
                for n in self.good_neurons:
                    dff = [self.dff[0, trial][n, t] for trial in self.i_good_non_stim_trials]
                    
                    df = pd.DataFrame({'stim': self.R_correct[self.i_good_non_stim_trials] + self.R_wrong[self.i_good_non_stim_trials],
                                       'lick': self.R_correct[self.i_good_non_stim_trials] + self.L_wrong[self.i_good_non_stim_trials],
                                       'reward': self.R_correct[self.i_good_non_stim_trials] + self.L_correct[self.i_good_non_stim_trials],
                                       'constant' : np.ones(len(self.i_good_non_stim_trials)),
                                       'dff': dff})
                    
                    # model = ols("""dff ~ C(stim) + C(lick) + C(reward) +
                    #                 C(stim):C(lick) + C(stim):C(reward) + C(lick):C(reward) +
                    #                 C(stim):C(lick):C(reward)""", data = df).fit()
                                    
                    model = ols("""dff ~ C(stim) + C(lick) + C(reward) + C(constant)""", data = df).fit()

                    # table = sm.stats.anova_lm(model)
                    # sig = np.where(np.array(table['PR(>F)'] < 0.01) == True)[0]
                    
                    results = (model.summary2().tables[1]['P>|t|'] < 0.01).to_numpy()[1:]
                    # h=False
                    
                    # if sum(results) > 1:
                    #     m += 1
                        
                    if results[0]:
                        s += 1
                    
                    elif results[1]:
                        l += 1
                        
                    elif results[2]:
                        r += 1
                        
                print("Runtime timestep {} : {} secs".format(t, time.time() - start_time))

                stim += [s]
                lick += [l]
                reward += [r]
                mixed += [m]
            
            f, axarr = plt.subplots(1,4, sharey='row', figsize=(20,5))
            x = np.arange(-5.97,4,self.fs)[:self.time_cutoff] if 'CW03' not in self.path else np.arange(-6.97,4,self.fs)[:self.time_cutoff]

            num_neurons = len(self.good_neurons)
            axarr[0].plot(x, np.array(lick)/num_neurons, color='magenta')
            axarr[0].set_title('Lick direction cell')
            axarr[1].plot(x, np.array(stim)/num_neurons, color='lime')
            axarr[1].set_title('Object location cell')
            axarr[2].plot(x, np.array(reward)/num_neurons, color='cyan')
            axarr[2].set_title('Outcome cell')
            axarr[3].plot(x, np.array(mixed)/num_neurons, color='gold')
            axarr[3].set_title('Mixed cell')

            for i in range(4):
                
                axarr[i].axvline(0, color = 'grey', alpha=0.5, ls = '--')
                axarr[i].axvline(-4.3, color = 'grey', alpha=0.5, ls = '--')
                axarr[i].axvline(-3, color = 'grey', alpha=0.5, ls = '--')
            if save:
                plt.savefig(self.path + r'single_neuron_sel.pdf')
                
            plt.show()
            
            return stim, lick, reward, mixed
            
        if type == 'Chen proportions':
            
            
            stim, lick, reward = [],[],[]
            
            stimtime, delaytime, outcometime = range(self.sample, self.delay), range(self.response-6, self.response), range(self.time_cutoff - 6, self.time_cutoff)

            for t in range(self.time_cutoff):
                
                if t not in cat((stimtime, delaytime, outcometime)):
                    
                    continue
                    
                start_time = time.time()

                s,l,r,m = 0,0,0,0
                
                # for n in range(self.num_neurons):
                for n in self.good_neurons:
                    dff = [self.dff[0, trial][n, t] for trial in self.i_good_non_stim_trials]
                    
                    df = pd.DataFrame({'stim': self.R_correct[self.i_good_non_stim_trials] + self.R_wrong[self.i_good_non_stim_trials],
                                       'lick': self.R_correct[self.i_good_non_stim_trials] + self.L_wrong[self.i_good_non_stim_trials],
                                       'reward': self.R_correct[self.i_good_non_stim_trials] + self.L_correct[self.i_good_non_stim_trials],
                                       'constant' : np.ones(len(self.i_good_non_stim_trials)),
                                       'dff': dff})
                    
                    # model = ols("""dff ~ C(stim) + C(lick) + C(reward) +
                    #                 C(stim):C(lick) + C(stim):C(reward) + C(lick):C(reward) +
                    #                 C(stim):C(lick):C(reward)""", data = df).fit()
                                    
                    model = ols("""dff ~ C(stim) + C(lick) + C(reward) + C(constant)""", data = df).fit()

                    # table = sm.stats.anova_lm(model)
                    # sig = np.where(np.array(table['PR(>F)'] < 0.01) == True)[0]
                    
                    results = (model.summary2().tables[1]['P>|t|'] < p).to_numpy()[1:]
                    # h=False
                    
                    # if sum(results) > 1:
                    #     m += 1
                    
                        
                    if results[0]:
                        s += 1
                    
                    elif results[1]:
                        l += 1
                        
                    elif results[2]:
                        r += 1
                        
                print("Runtime timestep {} : {} secs".format(t, time.time() - start_time))

                if t in stimtime:
                    stim += [s]
                elif t in delaytime:
                    lick += [l]
                elif t in outcometime:
                    reward += [r]
            
            return np.mean(stim), np.mean(lick), 0, np.mean(reward)

        if type == 'Susu method':
            
            stim, choice, action, outcome = 0,0,0,0
            
            stim_neurons, choice_neurons, action_neurons, outcome_neurons = [],[],[],[]
            
            # for n in range(self.num_neurons):
            for n in self.good_neurons:
            
                RR, LL = self.get_trace_matrix(n)
                
                RL, LR = self.get_trace_matrix(n, error=True)
                
                # Match trial numbers
                length = min([len(RR), len(LL), len(RL), len(LR)])
                RR = np.array(RR)[np.random.choice(len(RR), length, replace=False)]
                LL = np.array(LL)[np.random.choice(len(LL), length, replace=False)]
                RL = np.array(RL)[np.random.choice(len(RL), length, replace=False)]
                LR = np.array(LR)[np.random.choice(len(LR), length, replace=False)]
                
                # stim = (mean_count(RR, range(7,13)) + mean_count(RL, range(7,13))) - (mean_count(LL, range(7,13)) + mean_count(LR, range(7,13)))
                # choice = (mean_count(RR, range(21,28)) + mean_count(LR, range(21,28))) - (mean_count(LL, range(21,28)) + mean_count(RL, range(21,28)))
                # action = (mean_count(RR, range(28,34)) + mean_count(LR, range(28,34))) - (mean_count(LL, range(28,34)) + mean_count(RL, range(28,34)))
                # outcome = (mean_count(LL, range(34,40)) + mean_count(RR, range(34,40))) - (mean_count(LR, range(34,40)) + mean_count(RL, range(34,40)))
                
                _, stimp = mannwhitneyu(cat((mean_count(RR, range(self.sample,self.delay)), mean_count(RL, range(self.sample,self.delay)))),
                                        cat((mean_count(LL, range(self.sample,self.delay)), mean_count(LR, range(self.sample,self.delay)))))
                _, choicep = mannwhitneyu(cat((mean_count(RR, range(self.response-6,self.response)), mean_count(LR, range(self.response-6,self.response)))),
                                          cat((mean_count(LL, range(self.response-6,self.response)), mean_count(RL, range(self.response-6,self.response)))))
                _, actionp = mannwhitneyu(cat((mean_count(RR, range(self.response,self.time_cutoff)), mean_count(LR, range(self.response,self.time_cutoff)))),
                                          cat((mean_count(LL, range(self.response,self.time_cutoff)), mean_count(RL, range(self.response,self.time_cutoff)))))
                _, outcomep = mannwhitneyu(cat((mean_count(LL, range(self.time_cutoff - 6,self.time_cutoff)), mean_count(RR, range(self.time_cutoff - 6,self.time_cutoff)))),
                                            cat((mean_count(LR, range(self.time_cutoff - 6,self.time_cutoff)), mean_count(RL, range(self.time_cutoff - 6,self.time_cutoff)))))
                
                # _, stimp = mannwhitneyu(mean_count(RR, range(self.sample,self.delay)) + mean_count(RL, range(self.sample,self.delay)),
                #                         mean_count(LL, range(self.sample,self.delay)) + mean_count(LR, range(self.sample,self.delay)))
                # _, choicep = mannwhitneyu(mean_count(RR, range(self.response-6,self.response)) + mean_count(LR, range(self.response-6,self.response)),
                #                           mean_count(LL, range(self.response-6,self.response))+ mean_count(RL, range(self.response-6,self.response)))
                # _, actionp = mannwhitneyu(mean_count(RR, range(self.response,self.time_cutoff)) + mean_count(LR, range(self.response,self.time_cutoff)),
                #                           mean_count(LL, range(self.response,self.time_cutoff)) + mean_count(RL, range(self.response,self.time_cutoff)))
                # _, outcomep = mannwhitneyu(mean_count(LL, range(self.time_cutoff - 6,self.time_cutoff)) + mean_count(RR, range(self.time_cutoff - 6,self.time_cutoff)),
                #                            mean_count(LR, range(self.time_cutoff - 6,self.time_cutoff)) + mean_count(RL, range(self.time_cutoff - 6,self.time_cutoff)))

                
                
                # stim += [stimp]
                pval = p
                
                stim += stimp<pval
                choice += choicep<pval
                action += actionp<pval
                outcome += outcomep<pval
                
                stim_neurons += [n] if stimp<pval else []
                choice_neurons += [n] if choicep<pval else []
                action_neurons += [n] if actionp<pval else []
                outcome_neurons += [n] if outcomep<pval else []
                
                
            plt.bar(['stim', 'choice', 'action', 'outcome'], [stim/len(self.good_neurons), 
                                                              choice/len(self.good_neurons), 
                                                              action/len(self.good_neurons),
                                                              outcome/len(self.good_neurons)])
            plt.xlabel('Epoch selective')
            plt.ylabel('Proportion of neurons')
            # plt.ylim(0,0.5)
            plt.show()
                
            return stim_neurons, choice_neurons, action_neurons, outcome_neurons

    def stim_choice_outcome_selectivity(self, save=False, y_axis = 0):
        """Plots selectivity traces of stim/lick/reward/mixed cells using Susu's method
        
        Susu method called from single_neuron_sel method
        
        Parameters
        ----------
        save : bool, optional
            Whether to save fig as pdf
        y_axis : int
            0 if to leave unchanged, else int that is the ylim top limit
        

        """
        stim_neurons, choice_neurons, _, outcome_neurons = self.single_neuron_sel('Susu method')
        
        stim_sel, outcome_sel, choice_sel = [], [], []
        
        f, axarr = plt.subplots(1,3, sharey='row', figsize=(15,5))
        
        epochs = [range(self.sample,self.delay), range(self.delay,self.response), range(self.response,self.time_cutoff)]
        x = np.arange(-5.97,4,self.fs)[:self.time_cutoff] if 'CW03' not in self.path else np.arange(-6.97,4,self.fs)[:self.time_cutoff]
        titles = ['Stimulus selective', 'Choice selective', 'Outcome selective']
        
        
            
        nonpref, pref = self.contra_ipsi_pop(epochs[0], return_sel=True, selective_n = stim_neurons)
        
        err = np.std(pref, axis=0) / np.sqrt(len(pref)) 
        err += np.std(nonpref, axis=0) / np.sqrt(len(nonpref))
        sel = np.mean(pref, axis=0) - np.mean(nonpref, axis=0) 
        stim_sel = nonpref, pref
        
        if type(sel) != np.ndarray:
            print("Empty selectivity vec: {}".format(sel))
        else:
            axarr[0].plot(x, sel, color='green')
                    
            axarr[0].fill_between(x, sel - err, 
                      sel + err,
                      color='lightgreen')
    
            axarr[0].set_title(titles[0])
            

        #######################################
        

        nonpref, pref = self.contra_ipsi_pop(epochs[1], return_sel=True, selective_n = choice_neurons)
        err = np.std(pref, axis=0) / np.sqrt(len(pref)) 
        err += np.std(nonpref, axis=0) / np.sqrt(len(nonpref))
        sel = np.mean(pref, axis=0) - np.mean(nonpref, axis=0)
        choice_sel = nonpref, pref
        
        if type(sel) != np.ndarray:
            print("Empty selectivity vec: {}".format(sel))
        else:

            
            axarr[1].plot(x, sel, color='purple')
                    
            axarr[1].fill_between(x, sel - err, 
                      sel + err,
                      color='violet')
            axarr[1].set_title(titles[1])
        
        #######################################

        nonpref, pref = self.contra_ipsi_pop(epochs[2], return_sel=True, selective_n = outcome_neurons)
        err = np.std(pref, axis=0) / np.sqrt(len(pref)) 
        err += np.std(nonpref, axis=0) / np.sqrt(len(nonpref))
        sel = np.mean(pref, axis=0) - np.mean(nonpref, axis=0)
        outcome_sel = nonpref, pref
        
        axarr[2].plot(x, sel, color='dodgerblue')
                
        axarr[2].fill_between(x, sel - err, 
                  sel + err,
                  color='lightskyblue')

        axarr[2].set_title(titles[2])
        
        ###########################################
        
        axarr[0].set_ylabel('Selectivity')
        axarr[1].set_xlabel('Time from Go cue (s)')
        
        for i in range(3):
            
            axarr[i].axvline(0, color = 'grey', alpha=0.5, ls = '--')
            axarr[i].axvline(-4.3, color = 'grey', alpha=0.5, ls = '--')
            axarr[i].axvline(-3, color = 'grey', alpha=0.5, ls = '--')        
            axarr[i].axhline(0, color = 'grey', alpha=0.5, ls = '--')
            if y_axis != 0:
                axarr[i].set_ylim(top=y_axis)
        if save:
            plt.savefig(self.path + r'stim_choice_outcome_selectivity.pdf')
            
        plt.show()
        return stim_neurons, choice_neurons, outcome_neurons, stim_sel, choice_sel, outcome_sel

######### BEHAVIOR STATE FUNCTIONS #################

    def find_bias_trials(self, glmhmm=True, sampling='confidence', state=0):
        """Finds trials belonging to behavioral states calculated via the GLM-HMM or other method
        
        Two options to calculate the bias trials: in-house calculations based on
        correct vs incorrect trials (first code block), or using inputted GLM-HMM
        results saved in external file. 
        
        Parameters
        ----------
        glmhmm : bool, optional
            Whether to use GLM-HMM results to identify bias trials (default True)
        sampling : str, optional
            How to sample trials from GLM-HMM results; use only high confidence
            or high probability trials, or to use random sampling method
            (default 'confidence')
        state : int, optional
            Usually read in using built in path (default 0)
            
        Returns 
        -------
        list
            Bias trials as calculated in function

        """
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
            num_state = states.shape[1]

            st = []
            non = []
            # Two different sampling methods
            if sampling == 'confidence':
                for i in range(states.shape[0]):
        
                    top_state = np.argmax(states[i])
                    conf = (1/num_state) + 0.2
                    if states[i][top_state] > conf:
                        st += [top_state]
                    else:
                        non += [i] # Trials where top state is less than 60% confidence
            else:
                for i in range(states.shape[0]):
                    st += [np.random.choice([0, 1, 2], p=states[i])] # sample according to probability
            
            inds = np.where(np.array(st) == state)[0] # Grab specific state (0,1,2)
            bias_trials = self.old_i_good_trials[inds]
            bias_trials = [b for b in bias_trials if b in self.i_good_trials] #Filter out water leak trials
            self.bias_trials = bias_trials
            self.nonstate_trials = non
            if len(bias_trials) == 0:
                print("Error: no bias trials found for state {}".format(state))
            
            return bias_trials

        # return prebias_trials
        return bias_trials
            
    
    def ranked_cells_by_selectivity(self, epoch, p=0.0001):
        """Returns list of neurons based on trial type selectivity 
        
        Goes from most right preferring to most left preferring (rank 1 through -1)
        First, rank by half of trials, then return (ordered) selectivity of neurons
        using remaining half of trials. (half / half rule not implemented )
        
        Parameters
        ----------
        p : int, optional
            Probability for determining selectivity of neurons
            
        Returns 
        -------
        list
            Selective neurons, ranked
        list
            Selectivity of said ranked neurons
         """
         
        # Find all delay selective neurons first:
        # Selectivity returns positive if left pref, neg if right pref
        # delay_selective_neurons, selectivity = self.get_epoch_selective(range(self.delay+9, self.response),
        #                                                                 p = p,
        #                                                                 return_stat=True,
        #                                                                 lickdir=False)
        R,L = self.lick_correct_direction('r'), self.lick_correct_direction('l')
        random.shuffle(R)
        random.shuffle(L)
        train_r, test_r, train_l, test_l = np.array(R)[:int(len(R)/2)], np.array(R)[int(len(R)/2):], np.array(L)[:int(len(L)/2)], np.array(L)[int(len(L)/2):]
        
        selectivity = self.get_epoch_mean_diff(epoch, (train_r, train_l)) # sort by difference
        selectivity, _, _ = self.get_epoch_tstat(epoch, self.good_neurons, rtrials = train_r, ltrials = train_l)
        order = np.argsort(selectivity) # sorts from lowest to highest
         
        # Split trials into half, maintaining lick right and lick left proportions
        # Exclude opto trials
        neurons = []
        for n in order:
            neurons += [self.good_neurons[n]]
            # pref, l_trials, r_trials = self.screen_preference(neuron, range(self.delay, self.response))

        # return neurons, np.take(selectivity,order), (test_r, test_l) # Used to plot heatmap
        
        return order, np.take(selectivity,order), (test_r, test_l)
        











