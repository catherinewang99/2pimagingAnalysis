# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 11:42:45 2023

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

class Mode(Session):
    
    def __init__(self, path, layer_num):
        super().__init__(self, path, layer_num) # inherit all parameters of session.py
        
    