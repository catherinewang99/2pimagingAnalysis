# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 15:21:58 2023

@author: Catherine Wang
"""

import sys
sys.path.append("C:\scripts\Imaging analysis")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from session import Session
import os
from numpy import concatenate as cat


class Behavior():
    def __init__(self, path, single=False, behavior_only=False, glmhmm=[]):
        
        # If not single: path is the folder "/.../python/" that contains all the sessions and python compatible mat data
        
        total_sessions = 0
        self.path = path
        self.sessions = []
        
        self.opto_trials = dict()
        
        self.i_good_trials = dict() # zero indexing in python

        self.L_correct = dict()
        self.R_correct = dict()
        
        self.early_lick = dict()
        
        self.L_wrong = dict()
        self.R_wrong = dict()
        
        self.L_ignore = dict()
        self.R_ignore = dict()
        
        self.stim_ON = dict()
        self.stim_level = dict()
        
        self.delay_duration = dict()
        self.protocol = dict()
        
        if not single:
        
            for i in os.listdir(path):
                
                if len(glmhmm) != 0:
                    if i not in glmhmm:
                        continue
                
                if os.path.isdir(os.path.join(path, i)):
                    for j in os.listdir(os.path.join(path, i)):
                        if 'behavior' in j:
                                                        
                            behavior_old = scio.loadmat(os.path.join(path, i, j))
                            behavior = behavior_old.copy()
                            self.sessions += [i]
    
                            self.L_correct[total_sessions] = cat(behavior['L_hit_tmp'])
                            self.R_correct[total_sessions] = cat(behavior['R_hit_tmp'])
                            
                            self.early_lick[total_sessions] = cat(behavior['LickEarly_tmp'])
                            
                            self.L_wrong[total_sessions] = cat(behavior['L_miss_tmp'])
                            self.R_wrong[total_sessions] = cat(behavior['R_miss_tmp'])
                            
                            self.L_ignore[total_sessions] = cat(behavior['L_ignore_tmp'])
                            self.R_ignore[total_sessions] = cat(behavior['R_ignore_tmp'])
                           
                            if behavior_only:
                                
                                self.delay_duration[total_sessions] = cat(behavior['delay_duration'])
                                self.protocol[total_sessions] = cat(cat(behavior['protocol']))
                                
                            elif not behavior_only:
                            
                                self.stim_ON[total_sessions] = np.where(cat(behavior['StimDur_tmp']) > 0)
                                # self.stim_level[total_sessions] = cat(behavior['StimLevel'])
                                self.i_good_trials[total_sessions] = cat(behavior['i_good_trials']) - 1 # zero indexing in python


                            total_sessions += 1
    
                            
            self.total_sessions = total_sessions
        
        elif single:
            behavior_old = scio.loadmat(os.path.join(path, 'behavior.mat'))
            behavior = behavior_old.copy()

            self.i_good_trials[total_sessions] = cat(behavior['i_good_trials']) - 1 # zero indexing in python

            self.L_correct[total_sessions] = cat(behavior['L_hit_tmp'])
            self.R_correct[total_sessions] = cat(behavior['R_hit_tmp'])
            
            self.early_lick[total_sessions] = cat(behavior['LickEarly_tmp'])
            
            self.L_wrong[total_sessions] = cat(behavior['L_miss_tmp'])
            self.R_wrong[total_sessions] = cat(behavior['R_miss_tmp'])
            
            self.L_ignore[total_sessions] = cat(behavior['L_ignore_tmp'])
            self.R_ignore[total_sessions] = cat(behavior['R_ignore_tmp'])
            
            self.stim_ON[total_sessions] = np.where(cat(behavior['StimDur_tmp']) > 0)
            if 'StimLevel' in behavior.keys():
                self.stim_level[total_sessions] = cat(behavior['StimLevel'])
            self.total_sessions = 1
            

    def plot_performance_over_sessions(self, all=False):
        
        reg = []
        opto_p = []

        for i in range(self.total_sessions):
            if all:
                correct = self.L_correct[i] + self.R_correct[i]

            else:

                igood = self.i_good_trials[i]
                opto = self.stim_ON[i][0]
                igood_opto = np.setdiff1d(self.i_good_trials[i], self.stim_ON[i])
                # igood_opto = np.setdiff1d(range(len(self.L_correct[i])), self.stim_ON[i])
    
                # Filter out early lick
                opto = [o for o in opto if not self.early_lick[i][o]]
                igood_opto = [j for j in igood_opto if not self.early_lick[i][j]]
    
                reg += [np.sum([(self.L_correct[i][t] + self.R_correct[i][t]) for t in igood_opto]) / len(igood_opto)]
                    
                opto_p += [np.sum([(self.L_correct[i][t] + self.R_correct[i][t]) for t in opto]) / len(opto)]

            
        plt.plot(reg, 'g-', label='control')
        plt.plot(opto_p, 'r-', label = 'opto')
        plt.title('Performance over time')
        plt.xticks(range(self.total_sessions), self.sessions, rotation = 45)
        plt.axhline(0.5)
        plt.legend()
        plt.show()
        
        
        return reg, opto_p
        
        
    def plot_LR_performance_over_sessions(self):
        
        Lreg = []
        Rreg = []
        
        Lopto = []
        Ropto = []
         
        for i in range(self.total_sessions):
            
            igood = self.i_good_trials[i]
            opto = self.stim_ON[i][0]
            igood_opto = np.setdiff1d(igood, opto)
            
            # Filter out early lick
            opto = [o for o in opto if not self.early_lick[i][o]]
            igood_opto = [j for j in igood_opto if not self.early_lick[i][j]]

            # if not only_opto:
            Lreg += [np.sum([self.L_correct[i][t] for t in igood_opto]) / 
                     np.sum([(self.L_correct[i][t] + self.L_wrong[i][t] + self.L_ignore[i][t]) for t in igood_opto])]

            Rreg += [np.sum([self.R_correct[i][t] for t in igood_opto]) / 
                     np.sum([(self.R_correct[i][t] + self.R_wrong[i][t] + self.R_ignore[i][t]) for t in igood_opto])]
                
            # if only_opto:
            # opto_p += [np.sum([(self.L_correct[i][t] + self.R_correct[i][t]) for t in opto]) / len(opto)]
            
            Lopto += [np.sum([self.L_correct[i][t] for t in opto]) / 
                     np.sum([(self.L_correct[i][t] + self.L_wrong[i][t] + self.L_ignore[i][t]) for t in opto])]
            
            Ropto += [np.sum([self.R_correct[i][t] for t in opto]) / 
                     np.sum([(self.R_correct[i][t] + self.R_wrong[i][t] + self.R_ignore[i][t]) for t in opto])]
            
        plt.plot(Lreg, 'r-')
        plt.plot(Lopto, 'r--')
        
        plt.plot(Rreg, 'b-')
        plt.plot(Ropto, 'b--')
        
        plt.title('Performance over time')
        plt.xticks(range(self.total_sessions), self.sessions, rotation = 45)
        plt.axhline(0.5)
        plt.show()
        
    def plot_early_lick(self):
        
        EL = list()
        
        for i in range(self.total_sessions):
            
            rate = sum(self.early_lick[i]) / len(self.early_lick[i])
            EL.append(rate)
            
        plt.plot(EL, 'b-')
        plt.title('Early lick rate over time')
        plt.xticks(range(self.total_sessions), self.sessions, rotation = 45)
        plt.show()
    
    def plot_single_session(self, save=False):
         
        Lreg = []
        Rreg = []
        
        Lopto = []
        Ropto = []
         
        i=0            
        
        igood = self.i_good_trials[i]
        opto = self.stim_ON[i][0]
        igood_opto = np.setdiff1d(igood, opto)
        
        # Filter out early lick
        opto = [o for o in opto if not self.early_lick[i][o]]
        igood_opto = [j for j in igood_opto if not self.early_lick[i][j]]

        # if not only_opto:
        Lreg += [np.sum([self.L_correct[i][t] for t in igood_opto]) / 
                 np.sum([(self.L_correct[i][t] + self.L_wrong[i][t] + self.L_ignore[i][t]) for t in igood_opto])]

        Rreg += [np.sum([self.R_correct[i][t] for t in igood_opto]) / 
                 np.sum([(self.R_correct[i][t] + self.R_wrong[i][t] + self.R_ignore[i][t]) for t in igood_opto])]
            
        # if only_opto:
        # opto_p += [np.sum([(self.L_correct[i][t] + self.R_correct[i][t]) for t in opto]) / len(opto)]
        
        Lopto += [np.sum([self.L_correct[i][t] for t in opto]) / 
                 np.sum([(self.L_correct[i][t] + self.L_wrong[i][t] + self.L_ignore[i][t]) for t in opto])]
        
        L_opto_num = len([(self.L_correct[i][t] + self.L_wrong[i][t] + self.L_ignore[i][t]) for t in opto])
        
        Ropto += [np.sum([self.R_correct[i][t] for t in opto]) / 
                 np.sum([(self.R_correct[i][t] + self.R_wrong[i][t] + self.R_ignore[i][t]) for t in opto])]
        
        R_opto_num =len([(self.R_correct[i][t] + self.R_wrong[i][t] + self.R_ignore[i][t]) for t in opto])
        
        plt.plot(cat((Lreg, Lopto)), 'r-', marker='o', label='Left')
        # plt.plot(Lopto, 'r--')
        
        plt.plot(cat((Rreg, Ropto)), 'b-', marker='o', label='Right')
        # plt.plot(Ropto, 'b--')
        
        plt.title('Late delay optogenetic effect on unilateral ALM')
        plt.xticks([0, 1], ['Control', 'Late Delay Epoch'])
        plt.ylim(0, 1)
        plt.xlabel('Proportion correct')
        plt.legend()
        
        if save:
            plt.savefig(self.path + 'stim_behavioral_effect.jpg')
        
        plt.show()
        
        return L_opto_num, R_opto_num
    
    
    
    def plot_single_session_multidose(self, save=False):
         
        Lreg = []
        Rreg = []
        
        Lopto = []
        Ropto = []
        
        L_opto_num, R_opto_num = 0, 0
         
        i=0            
        
        igood = self.i_good_trials[i]
        opto = self.stim_ON[i][0]
        igood_opto = np.setdiff1d(igood, opto)
        
        opto_levels = np.array(list(set(self.stim_level[0]))) 
        
        # Filter out early lick
        opto = [o for o in opto if not self.early_lick[i][o]]
        igood_opto = [j for j in igood_opto if not self.early_lick[i][j]]

        Lreg += [np.sum([self.L_correct[i][t] for t in igood_opto]) / 
                 np.sum([(self.L_correct[i][t] + self.L_wrong[i][t] + self.L_ignore[i][t]) for t in igood_opto])]

        Rreg += [np.sum([self.R_correct[i][t] for t in igood_opto]) / 
                 np.sum([(self.R_correct[i][t] + self.R_wrong[i][t] + self.R_ignore[i][t]) for t in igood_opto])]
            
        for level in opto_levels:
            if level == 0:
                continue
            
            opto = np.where(self.stim_level[0] == level)
            
            Lopto += [np.sum([self.L_correct[i][t] for t in opto]) / 
                     np.sum([(self.L_correct[i][t] + self.L_wrong[i][t] + self.L_ignore[i][t]) for t in opto])]
            
            L_opto_num += len([(self.L_correct[i][t] + self.L_wrong[i][t] + self.L_ignore[i][t]) for t in opto])
            
            Ropto += [np.sum([self.R_correct[i][t] for t in opto]) / 
                     np.sum([(self.R_correct[i][t] + self.R_wrong[i][t] + self.R_ignore[i][t]) for t in opto])]
            
            R_opto_num += len([(self.R_correct[i][t] + self.R_wrong[i][t] + self.R_ignore[i][t]) for t in opto])
        
        plt.plot(cat((Lreg, Lopto)), 'r-', marker='o', label='Left')
        # plt.plot(Lopto, 'r--')
        
        plt.plot(cat((Rreg, Ropto)), 'b-', marker='o', label='Right')
        # plt.plot(Ropto, 'b--')
        
        plt.title('Late delay optogenetic effect on unilateral ALM')
        ticks = ['{} AOM'.format(x) for x in opto_levels[1:]]
        plt.xticks(range(len(opto_levels)), ['Control'] + ticks)
        plt.ylim(0, 1)
        plt.xlabel('Proportion correct')
        plt.ylabel('Perturbation condition')
        plt.legend()
        
        if save:
            plt.savefig(self.path + 'stimDOSE_behavioral_effect.jpg')
        
        plt.show()       

        return L_opto_num, R_opto_num
    
    
    def plot_licks_single_sess(self):
        # JH Plot
        return None

    def learning_progression(self, window = 50, save=False, imaging=False, return_results=False):
        
        # Figures showing learning over protocol
        
        f, axarr = plt.subplots(3, 1, sharex='col', figsize=(16,12))
        
        # Concatenate all sessions
        delay_duration = np.array([])
        correctarr = np.array([])
        earlylicksarr = np.array([])
        num_trials = []
        
        for sess in range(self.total_sessions):
            

            # delay = np.convolve(self.delay_duration[sess], np.ones(window)/window, mode = 'same')
            delay = self.delay_duration[sess]
            
            if imaging:
                if 3 not in delay:
                    continue
                # if 'CW028' in self.path and sess ==1:
                #     continue
            
            delay_duration = np.append(delay_duration, delay[window:-window])

            # delay_duration = np.append(delay_duration, self.delay_duration[sess][window:-window])
            
            correct = self.L_correct[sess] + self.R_correct[sess]
            correct = np.convolve(correct, np.ones(window)/window, mode = 'same')
            correctarr = np.append(correctarr, correct[window:-window])
            
            earlylicks = np.convolve(self.early_lick[sess], np.ones(window)/window, mode = 'same')
            earlylicksarr = np.append(earlylicksarr, earlylicks[window:-window])
            
            num_trials += [len(self.L_correct[sess])-(window*2)]
        num_trials = np.cumsum(num_trials)
        
        # Protocol
        
        axarr[0].plot(delay_duration, 'r')
        axarr[0].set_ylabel('Delay duration (s)')

        
        # Performance
        
        axarr[1].plot(correctarr, 'g')        
        axarr[1].set_ylabel('% correct')
        axarr[1].axhline(y=0.7, alpha = 0.5, color='orange')
        axarr[1].axhline(y=0.5, alpha = 0.5, color='red', ls = '--')
        axarr[1].set_ylim(0, 1)
        
        # Early licking
        
        axarr[2].plot(earlylicksarr, 'b')        
        axarr[2].set_ylabel('% Early licks')
        axarr[2].set_xlabel('Trials')
        
        
        # Denote separate sessions
        
        for num in num_trials:
            axarr[0].axvline(num, color = 'grey', alpha=0.5, ls = '--')
            axarr[1].axvline(num, color = 'grey', alpha=0.5, ls = '--')
            axarr[2].axvline(num, color = 'grey', alpha=0.5, ls = '--')
        
        if save:
            plt.savefig(self.path + r'\learningcurve.png')
        plt.show()
        
        
        if return_results:
            
            return delay_duration, correctarr, cat(([0], num_trials))
        
        
        
        
        
        
        
        
        
        


                        
        

        