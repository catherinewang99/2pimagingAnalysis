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
import session
from matplotlib.pyplot import figure



# path = r'F:\data\BAYLORCW030\python\2023_06_09'
# path = r'F:\data\BAYLORCW021\python\2023_04_06'
# path = r'F:\data\BAYLORCW021\python\2023_02_13'

# path = r'F:\data\BAYLORCW021\python\2023_04_25'
# path = r'F:\data\BAYLORCW021\python\2023_04_27'

path = r'F:\data\BAYLORCW030\python\2023_06_03'

path = r'F:\data\BAYLORCW030\python\2023_07_07'

l1 = session.Session(path)

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
    l1.plot_rasterPSTH_sidebyside(i)
    
    # l1.plot_raster_and_PSTH(i, bias= True)


# Plot rasters for delay selective neurons:

# for n in gc.get_delay_selective(p = 0.01/gc.num_neurons):
# #     # if l1.filter_by_deltas(n):



# #         # plt.show()
# #     # l1.plot_selectivity(n)
#     gc.plot_raster_and_PSTH(n)
#         # l1.plot_rasterPSTH_sidebyside(n)

    
for n in l1.get_epoch_selective(range(l1.delay, l1.response)):
# # #     # if l1.filter_by_deltas(n):

# # #         # plt.show()
# # #     # l1.plot_selectivity(n)
    l1.plot_rasterPSTH_sidebyside(n)
#     l1.plot_raster_and_PSTH(n, bias= True)

        
    
    
    
    