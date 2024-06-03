# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 17:20:38 2024

@author: catherinewang

Testing sparse component analysis at discovering underlying latent factors 
Maybe it can discover behavior state related features?

"""
from sca.models import SCA
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
from sklearn.preprocessing import normalize
import random


path = r'F:\data\BAYLORCW036\python\2023_10_19'

l1 = session.Session(path, use_reg=True, triple=True)

F = None


# concatenate all trials together
for t in l1.i_good_trials:
    if F is None:
        F = l1.dff[0,t][:, 9:l1.time_cutoff] # take out first 1.5 seconds of baseline
    else:
        
        F = np.hstack((F, l1.dff[0,t][:, 9:l1.time_cutoff]))


F = F[l1.good_neurons]

X=F
K=8
sca = SCA(n_components=K)
latent = sca.fit_transform(X)