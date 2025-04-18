
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 15:14:50 2023

@author: Catherine Wang
"""

import sys
sys.path.append("C:\scripts\Imaging analysis")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from alm_2p import session



# path = r'F:\data\BAYLORCW030\python\2023_06_09'
# path = r'F:\data\BAYLORCW021\python\2023_04_06'
# path = r'F:\data\BAYLORCW021\python\2023_02_13'

# path = r'F:\data\BAYLORCW021\python\2023_04_25'
# path = r'F:\data\BAYLORCW021\python\2023_04_27'

path = r'F:\data\BAYLORCW030\python\2023_06_03'

path = r'F:\data\BAYLORCW030\python\2023_07_10'

path = r'F:\data\BAYLORCW032\python\2023_10_24'
path = r'F:\data\BAYLORCW037\python\2023_11_21'
path = r'F:\data\BAYLORCW034\python\2023_10_27'
path = r'F:\data\BAYLORCW037\python\2023_12_15'

l1 = session.Session(path)#, use_reg=True, triple=True)

# l1 = decon.Deconvolved(path)


# path = r'F:\data\GC225\python\2022_02_14'

# gc = session.Session(path, 4, guang = True)
# l1.crop_trials(160)

# l2 = session.Session(layer_2, 2, behavior)
# l1.crop_trials(250)

# View neuron single trial psth

# for j in range(l1.num_trials):
#     l1.plot_single_trial_PSTH(j, 8)

# View the neurons
# for i in range(l1.num_neurons):
#     l1.plot_raster_and_PSTH(i)

# for i in range(l2.num_neurons):
#     l2.plot_PSTH(i)
    
# Get contra and ipsi neurons

# contra, ipsi, _, _ = l1.contra_ipsi_pop()

# # Plot contra neurons
# for i in contra:
#     l1.plot_PSTH(i)

# # Plot ipsi neurons
# for i in ipsi:
#     l1.plot_PSTH(i)
    
# Get population average plots
# l1.plot_contra_ipsi_pop()

for i in l1.good_neurons[:10]:
    l1.plot_raster_and_PSTH(i)
    # l1.plot_rasterPSTH_sidebyside(i)
    
    # l1.plot_raster_and_PSTH(i, bias= True)


# Plot rasters for delay selective neurons:

# for n in gc.get_delay_selective(p = 0.01/gc.num_neurons):
# #     # if l1.filter_by_deltas(n):



# #         # plt.show()
# #     # l1.plot_selectivity(n)
#     gc.plot_raster_and_PSTH(n)
#         # l1.plot_rasterPSTH_sidebyside(n)

    
# for n in l1.get_epoch_selective(range(l1.delay, l1.response)):
# # # #     # if l1.filter_by_deltas(n):

# # # #         # plt.show()
# # # #     # l1.plot_selectivity(n)
#     l1.plot_rasterPSTH_sidebyside(n)
# #     l1.plot_raster_and_PSTH(n, bias= True)
#%% Look for good example neurons that become robust after being silenced
#  [r'H:\data\BAYLORCW044\python\2024_05_22',
#   r'H:\data\BAYLORCW044\python\2024_06_06',
# r'H:\data\BAYLORCW044\python\2024_06_19'],

 
#     [r'H:\data\BAYLORCW044\python\2024_05_23',
#      r'H:\data\BAYLORCW044\python\2024_06_04',
# r'H:\data\BAYLORCW044\python\2024_06_18'],

#     [r'H:\data\BAYLORCW046\python\2024_05_29',
#      r'H:\data\BAYLORCW046\python\2024_06_24',
#      r'H:\data\BAYLORCW046\python\2024_06_28'], # Modified dates (use 6/7, 6/24?)


#     [r'H:\data\BAYLORCW046\python\2024_05_30',
#      r'H:\data\BAYLORCW046\python\2024_06_10',
#      r'H:\data\BAYLORCW046\python\2024_06_27'],

#     [r'H:\data\BAYLORCW046\python\2024_05_31',
#      r'H:\data\BAYLORCW046\python\2024_06_11',
#      r'H:\data\BAYLORCW046\python\2024_06_26'
#      ]

naivepath, learningpath, expertpath =[r'F:\data\BAYLORCW036\python\2023_10_09',
 r'F:\data\BAYLORCW036\python\2023_10_19',
 r'F:\data\BAYLORCW036\python\2023_10_30',
]
s1 = session.Session(learningpath, use_reg = True, triple=True, baseline_normalization="median_zscore")
for i in s1.get_epoch_selective(range(s1.delay, s1.response)):
    s1.plot_rasterPSTH_sidebyside(i)
    
l2 =  session.Session(expertpath, use_reg = True, triple=True, baseline_normalization="median_zscore")
n = l2.good_neurons[np.where(s1.good_neurons == 1315)[0][0]]
l2.plot_rasterPSTH_sidebyside(n)

l0 =  session.Session(naivepath, use_reg = True, triple=True, baseline_normalization="median_zscore")
n = l0.good_neurons[np.where(s1.good_neurons == 1315)[0][0]]
l0.plot_rasterPSTH_sidebyside(n)

#%% Look for neurons that are not delay selective in naive but selective in later 
all_matched_paths = [
    
             [r'F:\data\BAYLORCW032\python\2023_10_05',
               r'F:\data\BAYLORCW032\python\2023_10_19',
               r'F:\data\BAYLORCW032\python\2023_10_24',
           ],
         
           # [ r'F:\data\BAYLORCW034\python\2023_10_12',
           #    r'F:\data\BAYLORCW034\python\2023_10_22',
           #    r'F:\data\BAYLORCW034\python\2023_10_27',
           #    r'F:\data\BAYLORCW034\python\cellreg\layer{}\1012_1022_1027pairs_proc.npy'],
         
             [r'F:\data\BAYLORCW036\python\2023_10_09',
             r'F:\data\BAYLORCW036\python\2023_10_19',
             r'F:\data\BAYLORCW036\python\2023_10_30',
            ],
         
          [r'F:\data\BAYLORCW037\python\2023_11_21',
             r'F:\data\BAYLORCW037\python\2023_12_08',
             r'F:\data\BAYLORCW037\python\2023_12_15',],
         
          [r'F:\data\BAYLORCW035\python\2023_10_26',
             r'F:\data\BAYLORCW035\python\2023_12_07',
             r'F:\data\BAYLORCW035\python\2023_12_15',],
         
         [r'H:\data\BAYLORCW044\python\2024_05_22',
          r'H:\data\BAYLORCW044\python\2024_06_06',
        r'H:\data\BAYLORCW044\python\2024_06_19'],

         
            [r'H:\data\BAYLORCW044\python\2024_05_23',
             r'H:\data\BAYLORCW044\python\2024_06_04',
        r'H:\data\BAYLORCW044\python\2024_06_18'],

            # [r'H:\data\BAYLORCW046\python\2024_05_29',
            #  r'H:\data\BAYLORCW046\python\2024_06_07',
            #  r'H:\data\BAYLORCW046\python\2024_06_24'],

            [r'H:\data\BAYLORCW046\python\2024_05_29',
             r'H:\data\BAYLORCW046\python\2024_06_24',
             r'H:\data\BAYLORCW046\python\2024_06_28'],

            [r'H:\data\BAYLORCW046\python\2024_05_30',
             r'H:\data\BAYLORCW046\python\2024_06_10',
             r'H:\data\BAYLORCW046\python\2024_06_27'],

            [r'H:\data\BAYLORCW046\python\2024_05_31',
             r'H:\data\BAYLORCW046\python\2024_06_11',
             r'H:\data\BAYLORCW046\python\2024_06_26'
             ]
         
        ]
for paths in all_matched_paths:
    
    naivepath, learningpath, expertpath =  [r'F:\data\BAYLORCW035\python\2023_10_26',
       r'F:\data\BAYLORCW035\python\2023_12_07',
       r'F:\data\BAYLORCW035\python\2023_12_15',]
    s1 = session.Session(learningpath, use_reg = True, triple=True, baseline_normalization="median_zscore")
    l0 =  session.Session(naivepath, use_reg = True, triple=True, baseline_normalization="median_zscore")
    l2 =  session.Session(expertpath, use_reg = True, triple=True, baseline_normalization="median_zscore")
    
    naivedelaysel = l0.get_epoch_selective(range(l0.delay, l0.response))
    for i in s1.get_epoch_selective(range(s1.delay, s1.response)):
        n = l0.good_neurons[np.where(s1.good_neurons == i)[0][0]]
        nexp = l2.good_neurons[np.where(s1.good_neurons == i)[0][0]]
        if n not in naivedelaysel:
            l0.plot_rasterPSTH_sidebyside(n)
            s1.plot_rasterPSTH_sidebyside(i)
            l2.plot_rasterPSTH_sidebyside(nexp)

savepath = r'H:\COSYNE 2025\proportion_susceptible.pdf'
#%% LEARNING
naivepath, learningpath, expertpath = [r'H:\data\BAYLORCW046\python\2024_05_29',
 r'H:\data\BAYLORCW046\python\2024_06_07',
 r'H:\data\BAYLORCW046\python\2024_06_24']

# naivepath, learningpath, expertpath = [r'H:\data\BAYLORCW044\python\2024_05_23',
#                    r'H:\data\BAYLORCW044\python\2024_06_04',
#                   r'H:\data\BAYLORCW044\python\2024_06_18',]

naivepath, learningpath, expertpath = [r'H:\data\BAYLORCW046\python\2024_05_31',
             r'H:\data\BAYLORCW046\python\2024_06_11',
             r'H:\data\BAYLORCW046\python\2024_06_26'
             ]

s1 = session.Session(naivepath, use_reg=True, triple=True)
s2 = session.Session(learningpath, use_reg=True, triple=True)
s3 = session.Session(expertpath, use_reg=True, triple=True)
#%%
sample_sel = s1.get_epoch_selective(range(s1.response, s1.time_cutoff), p=0.000001)
sample_sel_idx = [np.where(s1.good_neurons == i)[0][0] for i in sample_sel]

#%%
stages = ['naive', 'learning', 'expert']
for idx, sess in enumerate([s1,s2,s3]):
    sess.plot_rasterPSTH_sidebyside(sess.good_neurons[sample_sel_idx[5]], save=r'F:\data\Fig 1\example neurons\action_{}_{}.pdf'.format(stages[idx], sample_sel_idx[5]))
    # sess.plot_raster_and_PSTH(sess.good_neurons[sample_sel_idx[0]])
    
#%% Perturbation
path = r'H:\data\BAYLORCW046\python\2024_06_26'
s1 = session.Session(path, use_reg=True, triple=True)
#%%
neurons = [434, 404, 443, 19, 38,59]
idx = [np.where(s1.good_neurons == n)[0][0] for n in neurons]
# sel_n = s1.get_epoch_selective(range(s1.delay, s1.response), p=0.0001)
for n in idx:
    s1.plot_rasterPSTH_sidebyside(s1.good_neurons[n])
#%%
path = r'H:\data\BAYLORCW046\python\2024_06_11'
s1 = session.Session(path, use_reg=True, triple=True)
for n in idx:
    s1.plot_rasterPSTH_sidebyside(s1.good_neurons[n])
#%%
agg_mice_paths = [
    
            [r'F:\data\BAYLORCW032\python\2023_10_05',
              r'F:\data\BAYLORCW032\python\2023_10_19',
              r'F:\data\BAYLORCW032\python\2023_10_24',
          ],
         
           # [ r'F:\data\BAYLORCW034\python\2023_10_12',
           #    r'F:\data\BAYLORCW034\python\2023_10_22',
           #    r'F:\data\BAYLORCW034\python\2023_10_27',
           #    r'F:\data\BAYLORCW034\python\cellreg\layer{}\1012_1022_1027pairs_proc.npy'],
         
            [r'F:\data\BAYLORCW036\python\2023_10_09',
            r'F:\data\BAYLORCW036\python\2023_10_19',
            r'F:\data\BAYLORCW036\python\2023_10_30',
           ],
         
         [r'F:\data\BAYLORCW037\python\2023_11_21',
            r'F:\data\BAYLORCW037\python\2023_12_08',
            r'F:\data\BAYLORCW037\python\2023_12_15',],
         
         [r'F:\data\BAYLORCW035\python\2023_10_26',
            r'F:\data\BAYLORCW035\python\2023_12_07',
            r'F:\data\BAYLORCW035\python\2023_12_15',],
         
         [r'H:\data\BAYLORCW044\python\2024_05_22',
          r'H:\data\BAYLORCW044\python\2024_06_06',
        r'H:\data\BAYLORCW044\python\2024_06_19'],

         
            [r'H:\data\BAYLORCW044\python\2024_05_23',
             r'H:\data\BAYLORCW044\python\2024_06_04',
        r'H:\data\BAYLORCW044\python\2024_06_18'],

            [r'H:\data\BAYLORCW046\python\2024_05_29',
             r'H:\data\BAYLORCW046\python\2024_06_07',
             r'H:\data\BAYLORCW046\python\2024_06_24'],


            [r'H:\data\BAYLORCW046\python\2024_05_30',
             r'H:\data\BAYLORCW046\python\2024_06_10',
             r'H:\data\BAYLORCW046\python\2024_06_27'],

            [r'H:\data\BAYLORCW046\python\2024_05_31',
             r'H:\data\BAYLORCW046\python\2024_06_11',
             r'H:\data\BAYLORCW046\python\2024_06_26'
             ]
         
        ]
    