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
    def __init__(self, path):
        
        # Path is the folder "/.../python/" that contains all the sessions and python compatible mat data
        
        total_sessions = 0
        
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
        
        for i in os.listdir(path):
            if os.path.isdir(os.path.join(path, i)):
                for j in os.listdir(os.path.join(path, i)):
                    if 'behavior' in j:
                        behavior_old = scio.loadmat(os.path.join(path, i, j))
                        behavior = behavior_old.copy()
                        self.sessions += [i]
                        self.i_good_trials[total_sessions] = cat(behavior['i_good_trials']) - 1 # zero indexing in python

                        self.L_correct[total_sessions] = cat(behavior['L_hit_tmp'])
                        self.R_correct[total_sessions] = cat(behavior['R_hit_tmp'])
                        
                        self.early_lick[total_sessions] = cat(behavior['LickEarly_tmp'])
                        
                        self.L_wrong[total_sessions] = cat(behavior['L_miss_tmp'])
                        self.R_wrong[total_sessions] = cat(behavior['R_miss_tmp'])
                        
                        self.L_ignore[total_sessions] = cat(behavior['L_ignore_tmp'])
                        self.R_ignore[total_sessions] = cat(behavior['R_ignore_tmp'])
                        
                        self.stim_ON[total_sessions] = np.where(cat(behavior['StimDur_tmp']) == 1)
                        total_sessions += 1

                        
        self.total_sessions = total_sessions

    def plot_performance_over_sessions(self):
        
        reg = []
        opto_p = []
        
        for i in range(self.total_sessions):
            
            igood = self.i_good_trials[i]
            opto = self.stim_ON[i][0]
            igood_opto = np.setdiff1d(self.i_good_trials[i], self.stim_ON[i])
            
            # Filter out early lick
            opto = [o for o in opto if not self.early_lick[i][o]]
            igood_opto = [j for j in igood_opto if not self.early_lick[i][j]]

            reg += [np.sum([(self.L_correct[i][t] + self.R_correct[i][t]) for t in igood_opto]) / len(igood_opto)]
                
            opto_p += [np.sum([(self.L_correct[i][t] + self.R_correct[i][t]) for t in opto]) / len(opto)]

            
        plt.plot(reg, 'g-')
        plt.plot(opto_p, 'r-')
        plt.title('Performance over time')
        plt.xticks(range(self.total_sessions), self.sessions, rotation = 45)
        plt.axhline(0.5)
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
            
            


                        
        

        