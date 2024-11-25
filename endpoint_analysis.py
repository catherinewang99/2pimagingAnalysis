# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 12:43:34 2024

@author: catherinewang
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
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot
from numpy.linalg import norm

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def cos_sim(a,b):
    return np.dot(a, b)/(norm(a)*norm(b))
#%% Paths

# naivepath, learningpath, expertpath =[r'F:\data\BAYLORCW036\python\2023_10_09',
#             r'F:\data\BAYLORCW036\python\2023_10_19',
#             r'F:\data\BAYLORCW036\python\2023_10_30',]

# naivepath, learningpath, expertpath = [r'H:\data\BAYLORCW044\python\2024_05_23',
#                    r'H:\data\BAYLORCW044\python\2024_06_04',
#                   r'H:\data\BAYLORCW044\python\2024_06_18',]

naivepath, learningpath, expertpath = [r'H:\data\BAYLORCW044\python\2024_05_22',
                    r'H:\data\BAYLORCW044\python\2024_06_06',
                  r'H:\data\BAYLORCW044\python\2024_06_19',]

# naivepath, learningpath, expertpath = [r'H:\data\BAYLORCW046\python\2024_05_31',
#                     r'H:\data\BAYLORCW046\python\2024_06_11',
#                   r'H:\data\BAYLORCW046\python\2024_06_26']
#%%

path = expertpath
s2 = Mode(path, use_reg = True, triple=True)
proj_allDimR, proj_allDimL = s2.plot_CD(ctl=True, plot=False, auto_corr_return=True)

plt.scatter(np.ones(proj_allDimR.shape[0]), proj_allDimR[:, s2.response-1], color='b')
plt.scatter(np.ones(proj_allDimL.shape[0]), proj_allDimL[:, s2.response-1], color='r')




