# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 12:09:11 2023

@author: Catherine Wang

Calculate CD for trained and apply to naive and learning sessions for
choice, stim, and action
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

naivepath =r'F:\data\BAYLORCW032\python\2023_10_05'
learningpath =  r'F:\data\BAYLORCW032\python\2023_10_19'
expertpath =r'F:\data\BAYLORCW032\python\2023_10_24'

# naivepath, learningpath, expertpath = [ r'F:\data\BAYLORCW034\python\2023_10_12',
#     r'F:\data\BAYLORCW034\python\2023_10_22',
#     r'F:\data\BAYLORCW034\python\2023_10_27',]

# naivepath, learningpath, expertpath =[r'F:\data\BAYLORCW036\python\2023_10_09',
#             r'F:\data\BAYLORCW036\python\2023_10_19',
#             r'F:\data\BAYLORCW036\python\2023_10_30',]

# naivepath, learningpath, expertpath =[r'F:\data\BAYLORCW035\python\2023_10_12',
#             r'F:\data\BAYLORCW035\python\2023_10_26',
#             r'F:\data\BAYLORCW035\python\2023_12_12',]

# naivepath, learningpath, expertpath =[r'F:\data\BAYLORCW035\python\2023_10_26',
#             r'F:\data\BAYLORCW035\python\2023_12_07',
#             r'F:\data\BAYLORCW035\python\2023_12_15',]

    
# naivepath, learningpath, expertpath =[r'F:\data\BAYLORCW037\python\2023_11_21',
#             r'F:\data\BAYLORCW037\python\2023_12_08',
#             r'F:\data\BAYLORCW037\python\2023_12_15',]
#%% Choice dimension unmatched

path = expertpath
l1 = Mode(path)
orthonormal_basis, mean = l1.plot_CD(mode_input='action')

path =learningpath
l1 = Mode(path)
orthonormal_basis, mean = l1.plot_CD(mode_input='action')

path = naivepath
l1 = Mode(path)
orthonormal_basis, mean = l1.plot_CD(mode_input='action')

#%% Choice dimension define on learning sess

path = learningpath
l1 = Mode(path, use_reg = True, triple=True)
orthonormal_basis, mean = l1.plot_CD()

path =expertpath
l1 = Mode(path, use_reg = True, triple=True)
l1.plot_appliedCD(orthonormal_basis, mean)

path = naivepath
l1 = Mode(path, use_reg = True, triple=True)
l1.plot_appliedCD(orthonormal_basis, mean) 

#%% Choice dimension

path = expertpath
l1 = Mode(path, use_reg = True, triple=True)
orthonormal_basis, mean = l1.plot_CD()#save = r'F:\data\Fig 2\CDchoice_expert_CW37.pdf')

path =learningpath
l1 = Mode(path, use_reg = True, triple=True)
l1.plot_appliedCD(orthonormal_basis, mean)#, save = r'F:\data\Fig 2\CDchoice_learning_CW37.pdf')

path = naivepath
l1 = Mode(path, use_reg = True, triple=True)
l1.plot_appliedCD(orthonormal_basis, mean)#, save = r'F:\data\Fig 2\CDchoice_naive_CW37.pdf')

#%% Stim dimension

path = expertpath
l1 = Mode(path, use_reg = True, triple=True)
orthonormal_basis, mean = l1.plot_CD(mode_input='stimulus')#, save = r'F:\data\Fig 2\CDstim_expert_CW37.pdf')

path = learningpath
l1 = Mode(path, use_reg = True, triple=True)
l1.plot_appliedCD(orthonormal_basis, mean)#, save = r'F:\data\Fig 2\CDstim_learning_CW37.pdf')

path = naivepath
l1 = Mode(path, use_reg = True, triple=True)
l1.plot_appliedCD(orthonormal_basis, mean)#,  save = r'F:\data\Fig 2\CDstim_naive_CW37.pdf')


#%% Action dimension

path = expertpath
l1 = Mode(path, use_reg = True, triple=True)
orthonormal_basis, mean = l1.plot_CD(mode_input='action',save = r'F:\data\Fig 2\CDchoice_expert_CW36.pdf')

path = learningpath
l1 = Mode(path, use_reg = True, triple=True)
l1.plot_appliedCD(orthonormal_basis, mean, save = r'F:\data\Fig 2\CDchoice_learning_CW36.pdf')

path = naivepath
l1 = Mode(path, use_reg = True, triple=True)
l1.plot_appliedCD(orthonormal_basis, mean, save = r'F:\data\Fig 2\CDchoice_naive_CW36.pdf')

#%% Use Full method

naivepath, learningpath, expertpath = [r'F:\data\BAYLORCW032\python\2023_10_05',
          r'F:\data\BAYLORCW032\python\2023_10_19',
          r'F:\data\BAYLORCW032\python\2023_10_24',]
# naivepath, learningpath, expertpath =[r'F:\data\BAYLORCW036\python\2023_10_09',
#             r'F:\data\BAYLORCW036\python\2023_10_19',
#             r'F:\data\BAYLORCW036\python\2023_10_30',]
path = expertpath
l1 = Mode(path, use_reg = True, triple=True)
orthonormal_basis, mean = l1.plot_behaviorally_relevant_modes()

path = learningpath
l1 = Mode(path, use_reg = True, triple=True)
l1.plot_behaviorally_relevant_modes_appliedCD(orthonormal_basis, mean)

path = naivepath
l1 = Mode(path, use_reg = True, triple=True)
l1.plot_behaviorally_relevant_modes_appliedCD(orthonormal_basis, mean)

