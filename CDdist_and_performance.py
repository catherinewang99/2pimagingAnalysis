# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 11:19:58 2023

@author: Catherine Wang

Replicating Li et al 2016 Fig 4 with plots distance from CD as a function of performance
Uses multiple sessions per mouse/across mouse

Normalize each CD to [-1,1] per session
"""

import sys
sys.path.append("C:\scripts\Imaging analysis")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import session
from activityMode import Mode
from matplotlib.pyplot import figure
import numpy as np
from sklearn.decomposition import PCA



