# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 09:59:53 2023

@author: Catherine Wang

Make cross scatter plots of CD_action_naive, CD_delay_naive and CD_action_expert
CD_delay_expert all combinations (2x2 plot)

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

paths = [r'F:\data\BAYLORCW032\python\2023_10_05',
          # r'F:\data\BAYLORCW032\python\2023_10_19',
          r'F:\data\BAYLORCW032\python\2023_10_24',]

CD_action_naive, CD_delay_naive, CD_action_expert, CD_delay_expert = [],[],[],[]

f, axarr = plt.subplots(2, 2, sharex='col', figsize=(21,10))



for i in range(len(paths)):
    l1 = Mode(paths[i], use_reg = True) # only match across sessions

    # orthonormal_basis, mean = l1.plot_behaviorally_relevant_modes(plot=False) # one method
    epoch = range(l1.response - 9, l1.response)
    orthonormal_basis_delay, _ = l1.func_compute_epoch_decoder([l1.PSTH_r_train_correct, 
                                                                    l1.PSTH_l_train_correct], epoch)
    epoch = range(l1.response +6, l1.response+12)
    orthonormal_basis_action, _ = l1.func_compute_epoch_decoder([l1.PSTH_r_train_correct, 
                                                                    l1.PSTH_l_train_correct], epoch)    
    
    idx = 0
    if not i:
        # CD_action_naive = orthonormal_basis[:, 5]
        # CD_delay_naive = orthonormal_basis[:, 1]
        CD_action_naive = orthonormal_basis_action[idx]
        CD_delay_naive = orthonormal_basis_delay[idx]
    else:
        # CD_action_expert = orthonormal_basis[:, 5]
        # CD_delay_expert = orthonormal_basis[:, 1]
        CD_action_expert = orthonormal_basis_action[idx]
        CD_delay_expert = orthonormal_basis_delay[idx]
        
    axarr[i,i].scatter(orthonormal_basis_action[idx], orthonormal_basis_delay[idx])
    axarr[i,i].axhline(0, color='grey')
    axarr[i,i].axvline(0, color='grey')

axarr[0,1].scatter(CD_action_naive, CD_delay_expert)
axarr[0,1].axhline(0, color='grey')
axarr[0,1].axvline(0, color='grey')

axarr[1,0].scatter(CD_action_expert, CD_delay_naive)
axarr[1,0].axhline(0, color='grey')
axarr[1,0].axvline(0, color='grey')

plt.show()

