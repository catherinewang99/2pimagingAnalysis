# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:10:18 2025

Investigate if there is an early lick cd that predicts the final cd over learning

@author: catherinewang
"""

import sys
sys.path.append("C:\scripts\Imaging analysis")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from alm_2p import session
from matplotlib.pyplot import figure
# import decon
from scipy.stats import chisquare
import pandas as pd
from activityMode import Mode
from scipy import stats
cat=np.concatenate

#%% paths

all_paths = [[    
            # r'F:\data\BAYLORCW032\python\2023_10_05',
            # r'F:\data\BAYLORCW034\python\2023_10_12',
            # r'F:\data\BAYLORCW036\python\2023_10_09',
            # r'F:\data\BAYLORCW035\python\2023_10_26',
            # r'F:\data\BAYLORCW037\python\2023_11_21',
            
            r'H:\data\BAYLORCW044\python\2024_05_22',
            r'H:\data\BAYLORCW044\python\2024_05_23',
            
            r'H:\data\BAYLORCW046\python\2024_05_29',
            r'H:\data\BAYLORCW046\python\2024_05_30',
            r'H:\data\BAYLORCW046\python\2024_05_31',
            ],

              [
             # r'F:\data\BAYLORCW032\python\2023_10_19',
            # r'F:\data\BAYLORCW034\python\2023_10_22',
            # r'F:\data\BAYLORCW036\python\2023_10_19',
            # r'F:\data\BAYLORCW035\python\2023_12_07',
            # r'F:\data\BAYLORCW037\python\2023_12_08',
            
            r'H:\data\BAYLORCW044\python\2024_06_06',
            r'H:\data\BAYLORCW044\python\2024_06_04',

            # r'H:\data\BAYLORCW046\python\2024_06_07', #sub out for below
            r'H:\data\BAYLORCW046\python\2024_06_24',
            r'H:\data\BAYLORCW046\python\2024_06_10',
            r'H:\data\BAYLORCW046\python\2024_06_11',
            ],


              [
             # r'F:\data\BAYLORCW032\python\2023_10_24',
            # r'F:\data\BAYLORCW034\python\2023_10_27',
            # r'F:\data\BAYLORCW036\python\2023_10_30',
            # r'F:\data\BAYLORCW035\python\2023_12_15',
            # r'F:\data\BAYLORCW037\python\2023_12_15',
            
            r'H:\data\BAYLORCW044\python\2024_06_19',
            r'H:\data\BAYLORCW044\python\2024_06_18',
            
            r'H:\data\BAYLORCW046\python\2024_06_28',
            r'H:\data\BAYLORCW046\python\2024_06_27',
            r'H:\data\BAYLORCW046\python\2024_06_26',
            
            ]]

#%% Work with early lick info


