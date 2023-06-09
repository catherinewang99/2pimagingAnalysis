# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 16:34:11 2023

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
from sys import path
from oasis.functions import gen_data, gen_sinusoidal_data, deconvolve, estimate_parameters
from oasis.plotting import simpleaxis
from oasis.oasis_methods import oasisAR1, oasisAR2
from session import Session
from scipy.signal import convolve



class Deconvolved(Session):
    
    def __init__(self, path, layer_num='all', guang=False, passive=False):
        # Inherit all parameters and functions of session.py
        super().__init__(path, layer_num, guang, passive)
            

        for n in range(self.num_neurons):
            for t in range(self.num_trials):
                
                y = self.dff[0, t][n, :self.time_cutoff]
                
                # Do the deconvolution
                c, s, b, g, lam = deconvolve(y)
    
                self.dff[0, t][n] = np.zeros(len(self.dff[0, t][n]))
                self.dff[0, t][n, :self.time_cutoff] = s
    

    

    
    def plot_PSTH_sidebyside(self, neuron_num):
        
        R, L = self.get_trace_matrix(neuron_num)
        title = "Neuron {}: Control".format(neuron_num)


        # f, axarr = plt.subplots(2,2, sharex='col', sharey = 'row')
        f, axarr = plt.subplots(1,2, sharex='col')
        
    


        
        R_av, L_av = np.mean(R, axis = 0), np.mean(L, axis = 0)
        
        left_err = np.std(L, axis=0) / np.sqrt(len(L)) 
        right_err = np.std(R, axis=0) / np.sqrt(len(R))
                    

        
        axarr[0].plot(L_av, 'r-')
        axarr[0].plot(R_av, 'b-')
        axarr[0].axvline(7, linestyle = '--')
        axarr[0].axvline(13, linestyle = '--')
        axarr[0].axvline(28, linestyle = '--')
        
        x = range(self.time_cutoff)

        axarr[0].fill_between(x, L_av - left_err, 
                 L_av + left_err,
                 color=['#ffaeb1'])
        axarr[0].fill_between(x, R_av - right_err, 
                 R_av + right_err,
                 color=['#b4b2dc'])
        
        axarr[0].set_title(title)
        
    
        R, L = self.get_opto_trace_matrix(neuron_num)
        r, l = self.get_opto_trace_matrix(neuron_num)
        title = "Neuron {}: Opto".format(neuron_num)

                


        R_av, L_av = np.mean(R, axis = 0), np.mean(L, axis = 0)
        vmax = max(cat([R_av, L_av]))

        left_err = np.std(L, axis=0) / np.sqrt(len(L)) 
        right_err = np.std(R, axis=0) / np.sqrt(len(R))
                    

        
        axarr[1].plot(L_av, 'r-')
        axarr[1].plot(R_av, 'b-')
        axarr[1].axvline(7, linestyle = '--')
        axarr[1].axvline(13, linestyle = '--')
        axarr[1].axvline(28, linestyle = '--')
        axarr[1].hlines(y=vmax, xmin=13, xmax=18, linewidth=10, color='lightblue')
        
        x = range(self.time_cutoff)

        axarr[1].fill_between(x, L_av - left_err, 
                 L_av + left_err,
                 color=['#ffaeb1'])
        axarr[1].fill_between(x, R_av - right_err, 
                 R_av + right_err,
                 color=['#b4b2dc'])
        
        axarr[1].set_title(title)
        
        plt.show()
