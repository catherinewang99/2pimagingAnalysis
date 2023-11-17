# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 15:22:11 2023

@author: Catherine Wang
"""


import sys
sys.path.append("C:\scripts\Imaging analysis")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import session
from matplotlib.pyplot import figure
# import decon
from scipy.stats import chisquare
import pandas as pd
plt.rcParams['pdf.fonttype'] = '42' 

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
# paths = [[
#             r'F:\data\BAYLORCW032\python\2023_10_08',
#               r'F:\data\BAYLORCW036\python\2023_10_09',
#           ],
    
# path = [       [r'F:\data\BAYLORCW036\python\2023_10_19',
#           r'F:\data\BAYLORCW032\python\2023_10_16'
#          ],
         
paths =[
        
        [
            r'F:\data\BAYLORCW032\python\2023_10_08',
              r'F:\data\BAYLORCW036\python\2023_10_09',
          ],

         [
            r'F:\data\BAYLORCW032\python\2023_10_25',
            r'F:\data\BAYLORCW036\python\2023_10_30',
            ],
        ]


# paths = [ r'F:\data\BAYLORCW034\python\2023_10_12',
#             r'F:\data\BAYLORCW034\python\2023_10_22',
#             r'F:\data\BAYLORCW034\python\2023_10_27']

# paths = [r'F:\data\BAYLORCW036\python\2023_10_09',
#             r'F:\data\BAYLORCW036\python\2023_10_19',
#             r'F:\data\BAYLORCW036\python\2023_10_30']
counter = 1
for path_ in paths:
    counter += 1
    lick, stim, reward, num_neurons = [],[],[],0
    for path in path_:
    
        l1 = session.Session(path, use_reg=True, triple=True)
        
        tstim, tlick, treward, tmixed = l1.single_neuron_sel('Chen 2017')
        
        lick += tlick
        stim += tstim
        reward += treward
        num_neurons += len(l1.good_neurons)

    f, axarr = plt.subplots(1,3, sharey='row', figsize=(20,5))
    x = np.arange(-6.97,4,l1.fs)[:l1.time_cutoff]
    
    axarr[0].plot(x, np.sum([lick[:61], lick[61:]], axis=0)/num_neurons, color='magenta')
    axarr[0].set_title('Lick direction cell')
    axarr[1].plot(x, np.sum([stim[:61], stim[61:]], axis=0)/num_neurons, color='lime')
    axarr[1].set_title('Object location cell')
    axarr[2].plot(x, np.sum([reward[:61], reward[61:]], axis=0)/num_neurons, color='cyan')
    axarr[2].set_title('Outcome cell')

    for i in range(3):
        
        axarr[i].axvline(0, color = 'grey', alpha=0.5, ls = '--')
        axarr[i].axvline(-4.3, color = 'grey', alpha=0.5, ls = '--')
        axarr[i].axvline(-3, color = 'grey', alpha=0.5, ls = '--')
    plt.savefig(r'F:\data\SFN 2023\newsingle_neuron_sel{}.pdf'.format(counter))
        
    plt.show()
