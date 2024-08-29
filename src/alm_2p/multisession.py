# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 11:24:51 2023

@author: Catherine Wang
"""

import numpy as np
from numpy import concatenate as cat
import matplotlib.pyplot as plt
from scipy import stats
import copy
import scipy.io as scio
from sklearn.preprocessing import normalize
from session import Session
import sympy
import time
import os




class Multisession():
    
    def __init__(self, paths, paired, layer_num='all'):
        
        """
        Parameters
        ----------
        path : list
            list of paths to the folders containing layers.mat and behavior.mat 
            files. Length of list is number of sessions to include.
        pairs : list or str
            List of paths of length n-1 where n is the total number of sessions.
            Alternatively, single path. Each .npy files contains the matched
            cells across the sessions.
        layer_num : str or int, optional
            Layer number to analyze (default is all the layers)
        """ 
        
        self.number_of_sessions = len(paths)
        sess = dict()
        
        counter = 0
        for path in paths:
            sess[counter] = Session(path, layer_num=layer_num)
            counter += 1
        
        paired = np.load(paired)