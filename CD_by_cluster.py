# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 14:31:45 2024

@author: catherinewang

Use clusters to build various CD's in session instead of orthogonalizing to look 
for robust dimension
"""


import sys
sys.path.append("C:\scripts\Imaging analysis")
sys.path.append("Users/catherinewang/Desktop/Imaging analysis/2pimagingAnalysis/src")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from alm_2p import session
from activityMode import Mode
from matplotlib.pyplot import figure
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import stats
from numpy.linalg import norm
# from scipy.stats import norm
from sklearn import preprocessing

cat = np.concatenate
plt.rcParams['pdf.fonttype'] = '42' 


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
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

#%% PATHS

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

naivepath =r'F:\data\BAYLORCW032\python\2023_10_05'
learningpath =  r'F:\data\BAYLORCW032\python\2023_10_19'
expertpath =r'F:\data\BAYLORCW032\python\2023_10_24'

naivepath, learningpath, expertpath = [ r'F:\data\BAYLORCW034\python\2023_10_12',
    r'F:\data\BAYLORCW034\python\2023_10_22',
    r'F:\data\BAYLORCW034\python\2023_10_27',]

naivepath, learningpath, expertpath =[r'F:\data\BAYLORCW036\python\2023_10_09',
            r'F:\data\BAYLORCW036\python\2023_10_19',
            r'F:\data\BAYLORCW036\python\2023_10_30',]

naivepath, learningpath, expertpath = [r'H:\data\BAYLORCW044\python\2024_05_23',
                   r'H:\data\BAYLORCW044\python\2024_06_04',
                  r'H:\data\BAYLORCW044\python\2024_06_18',]

naivepath, learningpath, expertpath = [r'H:\data\BAYLORCW046\python\2024_05_31',
                    r'H:\data\BAYLORCW046\python\2024_06_11',
                  r'H:\data\BAYLORCW046\python\2024_06_26',]


# naivepath, learningpath, expertpath = [r'H:\data\BAYLORCW046\python\2024_05_31',
#                     r'/Users/catherinewang/Desktop/Imaging analysis/CW46/2024_06_11',
#                   r'H:\data\BAYLORCW046\python\2024_06_26',]


# naivepath, learningpath, expertpath =[r'F:\data\BAYLORCW035\python\2023_10_12',
#             r'F:\data\BAYLORCW035\python\2023_10_26',
#             r'F:\data\BAYLORCW035\python\2023_12_12',]

# naivepath, learningpath, expertpath =[r'F:\data\BAYLORCW035\python\2023_10_26',
#             r'F:\data\BAYLORCW035\python\2023_12_07',
#             r'F:\data\BAYLORCW035\python\2023_12_15',]

    
# naivepath, learningpath, expertpath =[r'F:\data\BAYLORCW037\python\2023_11_21',
#             r'F:\data\BAYLORCW037\python\2023_12_08',
#             r'F:\data\BAYLORCW037\python\2023_12_15',]

allpaths = [[    r'F:\data\BAYLORCW032\python\2023_10_05',
            # r'F:\data\BAYLORCW034\python\2023_10_12',
            r'F:\data\BAYLORCW036\python\2023_10_09',
            r'F:\data\BAYLORCW035\python\2023_10_26',
            r'F:\data\BAYLORCW037\python\2023_11_21',
            
            r'H:\data\BAYLORCW044\python\2024_05_22',
            r'H:\data\BAYLORCW044\python\2024_05_23',
            
            r'H:\data\BAYLORCW046\python\2024_05_29',
            r'H:\data\BAYLORCW046\python\2024_05_30',
            r'H:\data\BAYLORCW046\python\2024_05_31',
            ],

             [r'F:\data\BAYLORCW032\python\2023_10_19',
            # r'F:\data\BAYLORCW034\python\2023_10_22',
            r'F:\data\BAYLORCW036\python\2023_10_19',
            r'F:\data\BAYLORCW035\python\2023_12_07',
            r'F:\data\BAYLORCW037\python\2023_12_08',
            
            r'H:\data\BAYLORCW044\python\2024_06_06',
            r'H:\data\BAYLORCW044\python\2024_06_04',

            r'H:\data\BAYLORCW046\python\2024_06_07',
            r'H:\data\BAYLORCW046\python\2024_06_10',
            r'H:\data\BAYLORCW046\python\2024_06_11',
            ],


             [r'F:\data\BAYLORCW032\python\2023_10_24',
            # r'F:\data\BAYLORCW034\python\2023_10_27',
            r'F:\data\BAYLORCW036\python\2023_10_30',
            r'F:\data\BAYLORCW035\python\2023_12_15',
            r'F:\data\BAYLORCW037\python\2023_12_15',
            
            r'H:\data\BAYLORCW044\python\2024_06_19',
            r'H:\data\BAYLORCW044\python\2024_06_18',
            
            r'H:\data\BAYLORCW046\python\2024_06_24',
            r'H:\data\BAYLORCW046\python\2024_06_27',
            r'H:\data\BAYLORCW046\python\2024_06_26',
            
            ]]

#%% Run analysis