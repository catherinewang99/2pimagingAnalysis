# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 17:36:38 2023

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
from sklearn.linear_model import LogisticRegressionCV

class Sample(Session):
    
    def __init__(self, path, layer_num='all', guang=False, passive=False):
        
        # Inherit all parameters and functions of session.py
        super().__init__(path, layer_num, guang, passive)
        
        self.n = self.get_selective_neurons()

        
    def get_selective_neurons(self):
        
        neurons = []
        epochs = [range(self.sample, self.sample+6), 
                  range(self.delay, self.response), 
                  range(self.response, self.time_cutoff)]
        
        for i in epochs:
            neurons += self.get_epoch_selective(i)
        
        return list(set(neurons))
        
    def do_sample_neurons(self, numneurons=200):
        
        # Use all neurons to sample
        
        self.sample_neurons = np.random.choice(self.n, size = len(numneurons), replace=True)
        
        return self.sample_neurons
    
    def sample_trials(self, correct=50, error=10):
        
        LL = [t for t in np.where(self.L_correct)[0] if t  in self.i_good_trials]
        RR = [t for t in np.where(self.R_correct)[0] if t  in self.i_good_trials]
        RL = [t for t in np.where(self.R_wrong)[0] if t  in self.i_good_trials]
        LR = [t for t in np.where(self.L_wrong)[0] if t  in self.i_good_trials]
        
        if len(LL)<correct or len(RR)<correct:
            correct = 30
            error=5
            
        self.LL = np.random.choice(LL, size = correct, replace=False)
        self.RR = np.random.choice(RR, size = correct, replace=False)
        self.RL = np.random.choice(RL, size = error, replace=False)
        self.LR = np.random.choice(LR, size = error, replace=False)
        
        return (correct,error)
        
    def get_choice_matrix(self, timestep, lens):
        correct, error=lens
        R_choice, L_choice = dict(), dict()
        for i in range(5):
            R_choice[i] = []
            L_choice[i] = []
            for n in self.sample_neurons:
                R_choice[i] += [self.dff[0, t][n, timestep] for t in self.RR[int(i*correct/5):int((i+1)*correct/5)]]
                R_choice[i] += [self.dff[0, t][n, timestep] for t in self.LR[int(i*error/5):int((i+1)*error/5)]]
                L_choice[i] += [self.dff[0, t][n, timestep] for t in self.LL[int(i*correct/5):int((i+1)*correct/5)]]
                L_choice[i] += [self.dff[0, t][n, timestep] for t in self.RL[int(i*error/5):int((i+1)*error/5)]]
            # R_choice[i] = np.array(R_choice[i])
            # L_choice[i] = np.array(L_choice[i])
        return R_choice, L_choice
    
    def do_log_reg(self, timestep, lens):
        
        scores = []
        R_choice, L_choice = self.get_choice_matrix(timestep, lens)
        for i in range(5):
            # i is the held out fold
            train = [j for j in range(5) if j != i]
            
            X = []
            y = []
            for t in train:
                X += R_choice[t]
                y += [np.ones(len(R_choice[t]))] # Encode R as 1s
                X += L_choice[t]
                y += [np.zeros(len(L_choice[t]))] 
            y = cat(y)
            
            # This does the 4 cross-val automatically
            if type(timestep) == int:
                log_cv = LogisticRegressionCV(cv=4, random_state=0).fit(np.array(X).reshape(-1, 1), y.reshape(-1, 1))
                
                testX = np.array(R_choice[i] + L_choice[i])
                testy = cat((np.ones(len(R_choice[i])), np.zeros(len(L_choice[i]))))
                scores += [log_cv.score(testX.reshape(-1, 1), testy.reshape(-1, 1))]
            else:
                log_cv = LogisticRegressionCV(cv=4, random_state=0).fit(np.array(X), y)
                
                testX = np.array(R_choice[i] + L_choice[i])
                testy = cat((np.ones(len(R_choice[i])), np.zeros(len(L_choice[i]))))
                scores += [log_cv.score(testX, testy)]                
                
        return scores
    
    def run_iter_logreg(self, timestep, num_neurons, iterations=100):
        
        mean_accuracy = []
        
        
        for i in range(iterations):
            
            # print("##### ITERATION {} #######".format(i))
            
            self.do_sample_neurons(num_neurons)
            lens = self.sample_trials()
            
            acc = self.do_log_reg(timestep, lens)
            mean_accuracy += [np.mean(acc)]
            
        return mean_accuracy
    
    def run_iter_log_timesteps(self):
        
        acc = []
        for time in range(self.time_cutoff):
            print("##### TIMESTEP {} #######".format(time))

            score = np.mean(self.run_iter_logreg(time, len(self.n))) # Use all neurons to train
            acc += [score]
            
        return acc
        
    
    

            
        
            