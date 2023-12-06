# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 13:25:04 2023

@author: Catherine Wang
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import zscore
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


path= r'F:\data\BAYLORCW032\python\2023_10_24'

rmap = np.load(path + '\\F_embedding.npy', allow_pickle=True)
rmap = rmap.item()

