# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 15:19:57 2023

@author: Catherine Wang
"""

import sys
sys.path.append("C:\scripts\Imaging analysis")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import session
from matplotlib.pyplot import figure
import decon
from scipy.stats import chisquare
import pandas as pd
from activityMode import Mode
#%%
paths = [[r'F:\data\BAYLORCW032\python\2023_10_08',
          r'F:\data\BAYLORCW032\python\2023_10_16',
          r'F:\data\BAYLORCW032\python\2023_10_25',
          r'F:\data\BAYLORCW032\python\cellreg\layer{}\1008_1016_1025pairs_proc.npy'],
         
         [ r'F:\data\BAYLORCW034\python\2023_10_12',
            r'F:\data\BAYLORCW034\python\2023_10_22',
            r'F:\data\BAYLORCW034\python\2023_10_27',
            r'F:\data\BAYLORCW034\python\cellreg\layer{}\1012_1022_1027pairs_proc.npy'],
         
         [r'F:\data\BAYLORCW036\python\2023_10_09',
            r'F:\data\BAYLORCW036\python\2023_10_19',
            r'F:\data\BAYLORCW036\python\2023_10_30',
            r'F:\data\BAYLORCW036\python\cellreg\layer{}\1009_1019_1030pairs_proc.npy'],
        ]
#%% Matched
#%% CW32 matched

# CONTRA
paths = [r'F:\data\BAYLORCW032\python\2023_10_05',
          r'F:\data\BAYLORCW032\python\2023_10_19',
          r'F:\data\BAYLORCW032\python\2023_10_24',]
for path in paths:
    
    l1 = Mode(path, use_reg=True, triple=True)
    
    l1.selectivity_optogenetics(p=0.01)
    l1.plot_CD_opto()

#%% CW34 matched
paths = [ r'F:\data\BAYLORCW034\python\2023_10_12',
           r'F:\data\BAYLORCW034\python\2023_10_22',
           r'F:\data\BAYLORCW034\python\2023_10_27',]
for path in paths:
    
    l1 = Mode(path, use_reg=True, triple=True)
    
    l1.selectivity_optogenetics()
    l1.plot_CD_opto()

#%% CW36 matched
paths = [r'F:\data\BAYLORCW036\python\2023_10_09',
           r'F:\data\BAYLORCW036\python\2023_10_19',
           r'F:\data\BAYLORCW036\python\2023_10_30',]
for path in paths:
    
    l1 = Mode(path, use_reg=True, triple=True)
    
    l1.selectivity_optogenetics(p=0.01)
    l1.plot_CD_opto()

#%% Unmatched

path = r'F:\data\BAYLORCW032\python\2023_10_24'
paths = [r'F:\data\BAYLORCW032\python\2023_10_05',
          r'F:\data\BAYLORCW032\python\2023_10_19',
          r'F:\data\BAYLORCW032\python\2023_10_24',]
for path in paths:
    l1 = session.Session(path)
    
    l1.selectivity_optogenetics()
#%% CW34 unmatched

paths = [ r'F:\data\BAYLORCW034\python\2023_10_12',
           r'F:\data\BAYLORCW034\python\2023_10_22',
           r'F:\data\BAYLORCW034\python\2023_10_27',]
for path in paths:
    
    l1 = session.Session(path)
    
    l1.selectivity_optogenetics()
#%% CW36 unmatched
paths = [r'F:\data\BAYLORCW036\python\2023_10_09',
           r'F:\data\BAYLORCW036\python\2023_10_19',
           r'F:\data\BAYLORCW036\python\2023_10_30',]
for path in paths:
    
    l1 = session.Session(path)
    
    l1.selectivity_optogenetics(p=0.01)