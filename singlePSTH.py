# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 17:02:58 2023

@author: Catherine Wang
"""
import sys
sys.path.append("C:\scripts\Imaging analysis")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import session
from neuralFuncs import plot_average_PSTH


layer_1 = scio.loadmat(r'E:\data\BAYLORCW022\python\2022_12_15\layer_1.mat')
behavior = scio.loadmat(r'E:\data\BAYLORCW022\python\2022_12_15\behavior.mat')


l1 = session.Session(layer_1, 1, behavior)

num_neurons = l1.num_neurons

av_dff = []
for i in range(l1.num_trials):
    av_dff += [[l1.dff[0, i][3, :50]]]
plt.plot(np.mean(av_dff, axis = 0)[0])
