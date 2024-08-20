# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 16:42:16 2024

@author: catherinewang
"""

import sys
sys.path.append("C:\scripts\Imaging analysis")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from session import Session
from matplotlib.pyplot import figure
import decon
from scipy.stats import chisquare
import pandas as pd
from activityMode import Mode
from scipy import stats

from sklearn.manifold import TSNE


X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

X_embedded = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=3).fit_transform(X)
X_embedded




paths = all_matched_paths[6]

intialpath, finalpath = paths[1], paths[2]

# sample CD
if '43' in paths[1] or '38' in paths[1]:
    l1 = Mode(intialpath, use_reg=True, triple=False)
    l2 = Mode(finalpath, use_reg = True, triple=False)
else:
    l1 = Mode(intialpath, use_reg=True, triple=True)
    l2 = Mode(finalpath, use_reg = True, triple=True)

orthonormal_basis_initial, mean = l1.plot_CD(mode_input = 'stimulus')
orthonormal_basis_initial_choice, mean = l1.plot_CD(mode_input = 'choice')

orthonormal_basis, mean = l2.plot_CD(mode_input = 'stimulus')
orthonormal_basis_choice, mean = l2.plot_CD(mode_input = 'choice')
    
X = np.vstack((orthonormal_basis, orthonormal_basis_choice)).T
X_embedded = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=10).fit_transform(X)

f=plt.figure(figsize=(5,5))
plt.scatter(X_embedded[:,0], X_embedded[:,1])


X = np.vstack((orthonormal_basis_initial, orthonormal_basis_initial_choice)).T
X_embedded = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=10).fit_transform(X)

f=plt.figure(figsize=(5,5))
plt.scatter(X_embedded[:,0], X_embedded[:,1])
