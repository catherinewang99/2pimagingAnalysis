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
# paths = [    r'F:\data\BAYLORCW034\python\2023_10_12',
#             # r'F:\data\BAYLORCW034\python\2023_10_22',
#             r'F:\data\BAYLORCW034\python\2023_10_27',]

paths = [r'F:\data\BAYLORCW036\python\2023_10_09',
            # r'F:\data\BAYLORCW036\python\2023_10_19',
            r'F:\data\BAYLORCW036\python\2023_10_30',]

# naivepath, learningpath, expertpath =[r'F:\data\BAYLORCW037\python\2023_11_21',
#             r'F:\data\BAYLORCW037\python\2023_12_08',
#             r'F:\data\BAYLORCW037\python\2023_12_15',]
paths = [r'F:\data\BAYLORCW037\python\2023_11_21',
         r'F:\data\BAYLORCW037\python\2023_12_15']
paths = [r'F:\data\BAYLORCW035\python\2023_10_26',
         r'F:\data\BAYLORCW035\python\2023_12_15']

CD_action_naive, CD_delay_naive, CD_action_expert, CD_delay_expert = [],[],[],[]

f, axarr = plt.subplots(2, 2, sharex='col', figsize=(21,10))



for i in range(len(paths)):
    l1 = Mode(paths[i], use_reg = True,triple=True) # only match across sessions

    orthonormal_basis, mean = l1.plot_behaviorally_relevant_modes(plot=False) # one method
    
    # epoch = range(l1.response - 9, l1.response)
    # orthonormal_basis_delay, _ = l1.func_compute_epoch_decoder([l1.PSTH_r_train_correct, 
    #                                                                 l1.PSTH_l_train_correct], epoch)
    # epoch = range(l1.response +6, l1.response+12)
    # orthonormal_basis_action, _ = l1.func_compute_epoch_decoder([l1.PSTH_r_train_correct, 
                                                                    # l1.PSTH_l_train_correct], epoch)    
    
    idx = 0
    if not i:
        CD_action_naive = orthonormal_basis[:, 5]
        CD_delay_naive = orthonormal_basis[:, 1]
        print(orthonormal_basis[:, 5].shape)
        # CD_action_naive = orthonormal_basis_action[idx]
        # CD_delay_naive = orthonormal_basis_delay[idx]
    else:
        CD_action_expert = orthonormal_basis[:, 5]
        CD_delay_expert = orthonormal_basis[:, 1]
        # CD_action_expert = orthonormal_basis_action[idx]
        # CD_delay_expert = orthonormal_basis_delay[idx]
        
    axarr[i,i].scatter(orthonormal_basis[:, 5], orthonormal_basis[:, 1])
    # axarr[i,i].scatter(orthonormal_basis_action[idx], orthonormal_basis_delay[idx])
    axarr[i,i].axhline(0, color='grey')
    axarr[i,i].axvline(0, color='grey')

axarr[0,1].scatter(CD_action_naive, CD_delay_expert)
axarr[0,1].axhline(0, color='grey')
axarr[0,1].axvline(0, color='grey')

axarr[1,0].scatter(CD_action_expert, CD_delay_naive)
axarr[1,0].axhline(0, color='grey')
axarr[1,0].axvline(0, color='grey')

axarr[0,0].set_ylabel('CD_delay_naive')
axarr[0,0].set_xlabel('CD_action_naive')
axarr[0,1].set_ylabel('CD_delay_expert')
axarr[1,0].set_xlabel('CD_action_expert')


plt.show()

#%% Aggregate all mice:
allpaths = [[r'F:\data\BAYLORCW032\python\2023_10_05',
          # r'F:\data\BAYLORCW032\python\2023_10_19',
          r'F:\data\BAYLORCW032\python\2023_10_24',],
# paths = [    r'F:\data\BAYLORCW034\python\2023_10_12',
#             # r'F:\data\BAYLORCW034\python\2023_10_22',
#             r'F:\data\BAYLORCW034\python\2023_10_27',]

         [r'F:\data\BAYLORCW036\python\2023_10_09',
            # r'F:\data\BAYLORCW036\python\2023_10_19',
            r'F:\data\BAYLORCW036\python\2023_10_30',],

# naivepath, learningpath, expertpath =[r'F:\data\BAYLORCW037\python\2023_11_21',
#             r'F:\data\BAYLORCW037\python\2023_12_08',
#             r'F:\data\BAYLORCW037\python\2023_12_15',]
        [r'F:\data\BAYLORCW037\python\2023_11_21',
         r'F:\data\BAYLORCW037\python\2023_12_15'],
        [r'F:\data\BAYLORCW035\python\2023_10_26',
         r'F:\data\BAYLORCW035\python\2023_12_15']]
    
CD_action_naive, CD_delay_naive, CD_action_expert, CD_delay_expert = [],[],[],[]

f, axarr = plt.subplots(2, 2, sharex='col', figsize=(21,10))


for paths in allpaths:    
    for i in range(len(paths)):
        l1 = Mode(paths[i], use_reg = True,triple=True) # only match across sessions
    
        orthonormal_basis, mean = l1.plot_behaviorally_relevant_modes(plot=False) # one method
        
        # epoch = range(l1.response - 9, l1.response)
        # orthonormal_basis_delay, _ = l1.func_compute_epoch_decoder([l1.PSTH_r_train_correct, 
        #                                                                 l1.PSTH_l_train_correct], epoch)
        # epoch = range(l1.response +6, l1.response+12)
        # orthonormal_basis_action, _ = l1.func_compute_epoch_decoder([l1.PSTH_r_train_correct, 
                                                                        # l1.PSTH_l_train_correct], epoch)    
        
        idx = 0
        if not i:
            CD_action_naive =np.append(CD_action_naive, orthonormal_basis[:, 5])
            CD_delay_naive = np.append(CD_delay_naive, orthonormal_basis[:, 1])
            print(orthonormal_basis[:, 5].shape)
            # CD_action_naive = orthonormal_basis_action[idx]
            # CD_delay_naive = orthonormal_basis_delay[idx]
        else:
            CD_action_expert = np.append(CD_action_expert, orthonormal_basis[:, 5])
            CD_delay_expert = np.append(CD_delay_expert, orthonormal_basis[:, 1])
            # CD_action_expert = orthonormal_basis_action[idx]
            # CD_delay_expert = orthonormal_basis_delay[idx]
            
        axarr[i,i].scatter(orthonormal_basis[:, 5], orthonormal_basis[:, 1], color='b')
        # axarr[i,i].scatter(orthonormal_basis_action[idx], orthonormal_basis_delay[idx])
        axarr[i,i].axhline(0, color='grey')
        axarr[i,i].axvline(0, color='grey')

axarr[0,1].scatter(CD_action_naive, CD_delay_expert)
axarr[0,1].axhline(0, color='grey')
axarr[0,1].axvline(0, color='grey')

axarr[1,0].scatter(CD_action_expert, CD_delay_naive)
axarr[1,0].axhline(0, color='grey')
axarr[1,0].axvline(0, color='grey')

axarr[0,0].set_ylabel('CD_delay_naive')
axarr[0,0].set_xlabel('CD_action_naive')
axarr[0,1].set_ylabel('CD_delay_expert')
axarr[1,0].set_xlabel('CD_action_expert')

# axarr[0].set_ylabel('CD_delay_naive')
# axarr[0].set_xlabel('CD_action_naive')
# axarr[1].set_ylabel('CD_delay_expert')
# axarr[1].set_xlabel('CD_action_expert')
f.suptitle('CD weights scattered (n={} neurons)'.format(CD_action_expert.shape[0]))

plt.show()

#%% Abandon CD, use t-stat instead


paths = [r'F:\data\BAYLORCW032\python\2023_10_05',
          # r'F:\data\BAYLORCW032\python\2023_10_19',
          r'F:\data\BAYLORCW032\python\2023_10_24',]


f, axarr = plt.subplots(2, 2, sharex='col', figsize=(21,10))



for i in range(len(paths)):
    l1 = session.Session(paths[i], use_reg = True) # only match across sessions


    if not i:
        # CD_action_naive = orthonormal_basis[:, 5]
        # CD_delay_naive = orthonormal_basis[:, 1]
        epoch = range(l1.response +6, l1.response+12)
        CD_action_naive, _, _ = l1.get_epoch_tstat(epoch, l1.good_neurons)
        epoch = range(l1.response - 9, l1.response)
        CD_delay_naive, _, _ = l1.get_epoch_tstat(epoch, l1.good_neurons)
    else:
        # CD_action_expert = orthonormal_basis[:, 5]
        # CD_delay_expert = orthonormal_basis[:, 1]
        epoch = range(l1.response +6, l1.response+12)
        CD_action_expert, _, _ = l1.get_epoch_tstat(epoch, l1.good_neurons)
        epoch = range(l1.response - 9, l1.response)
        CD_delay_expert, _, _ = l1.get_epoch_tstat(epoch, l1.good_neurons)
        
axarr[0,0].scatter(CD_action_naive, CD_delay_naive)
axarr[0,0].axhline(0, color='grey')
axarr[0,0].axvline(0, color='grey')
axarr[0,0].set_xlabel('CD_action_naive')
axarr[0,0].set_ylabel('CD_delay_naive')

axarr[0,1].scatter(CD_action_naive, CD_delay_expert)
axarr[0,1].axhline(0, color='grey')
axarr[0,1].axvline(0, color='grey')
axarr[0,1].set_xlabel('CD_action_naive')
axarr[0,1].set_ylabel('CD_delay_expert')

axarr[1,0].scatter(CD_action_expert, CD_delay_naive)
axarr[1,0].axhline(0, color='grey')
axarr[1,0].axvline(0, color='grey')
axarr[1,0].set_xlabel('CD_action_expert')
axarr[1,0].set_ylabel('CD_delay_naive')

axarr[1,1].scatter(CD_action_expert, CD_delay_expert)
axarr[1,1].axhline(0, color='grey')
axarr[1,1].axvline(0, color='grey')
axarr[1,1].set_xlabel('CD_action_expert')
axarr[1,1].set_ylabel('CD_delay_expert')

plt.show()