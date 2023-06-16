# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 18:31:54 2023

@author: Catherine Wang
"""
import sys
sys.path.append("C:\scripts\Imaging analysis")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import session
from matplotlib.pyplot import figure
from numpy import concatenate as cat
from sklearn.preprocessing import normalize
import quality
import bootstrap

##### Run in notebook so it goes faster #####

path = r'F:\data\BAYLORCW021\python\2023_02_13'
d1 = bootstrap.Sample(path)

scores1 = d1.run_iter_log_timesteps()

plt.plot(scores1)
plt.show()
path = r'F:\data\BAYLORCW021\python\2023_04_27'
d = bootstrap.Sample(path)

scores = d.run_iter_log_timesteps()

plt.plot(scores)

plt.show()
numneurons = np.arange(500,2800, 300)
acc = []
for num in numneurons:
    acc += [np.mean(d.run_iter_logreg(range(d.time_cutoff), num))]
    
plt.plot(numneurons, acc)
plt.show()