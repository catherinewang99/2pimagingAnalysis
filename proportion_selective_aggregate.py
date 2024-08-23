# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 15:35:30 2023

@author: Catherine Wang
"""

import sys
sys.path.append("C:\scripts\Imaging analysis")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from session import Session
from matplotlib.pyplot import figure
# import decon
from scipy.stats import chisquare
import pandas as pd
plt.rcParams['pdf.fonttype'] = '42' 
cat=np.concatenate

#%% Paths
all_paths = [[    
            r'F:\data\BAYLORCW032\python\2023_10_05',
            # r'F:\data\BAYLORCW034\python\2023_10_12',
            r'F:\data\BAYLORCW036\python\2023_10_09',
            r'F:\data\BAYLORCW035\python\2023_10_26',
            r'F:\data\BAYLORCW037\python\2023_11_21',
            
            r'F:\data\BAYLORCW036\python\2023_10_16',
            r'F:\data\BAYLORCW035\python\2023_10_12',
        r'F:\data\BAYLORCW035\python\2023_11_02',

        r'H:\data\BAYLORCW044\python\2024_05_22',
        r'H:\data\BAYLORCW044\python\2024_05_23',
        r'H:\data\BAYLORCW044\python\2024_05_24',
        
        r'H:\data\BAYLORCW046\python\2024_05_29',
        r'H:\data\BAYLORCW046\python\2024_05_30',
        r'H:\data\BAYLORCW046\python\2024_05_31',
            ],
             [r'F:\data\BAYLORCW032\python\2023_10_19',
            # r'F:\data\BAYLORCW034\python\2023_10_22',
            r'F:\data\BAYLORCW036\python\2023_10_19',
            r'F:\data\BAYLORCW035\python\2023_12_07',
            r'F:\data\BAYLORCW037\python\2023_12_08',

        r'F:\data\BAYLORCW032\python\2023_10_18',
        r'F:\data\BAYLORCW035\python\2023_10_25',
            r'F:\data\BAYLORCW035\python\2023_11_27',
            r'F:\data\BAYLORCW035\python\2023_11_29',
            r'F:\data\BAYLORCW037\python\2023_11_28',
            
        r'H:\data\BAYLORCW044\python\2024_06_06',
        r'H:\data\BAYLORCW044\python\2024_06_04',
        r'H:\data\BAYLORCW044\python\2024_06_03',
        r'H:\data\BAYLORCW044\python\2024_06_12',

        r'H:\data\BAYLORCW046\python\2024_06_07',
        r'H:\data\BAYLORCW046\python\2024_06_10',
        r'H:\data\BAYLORCW046\python\2024_06_11',
        r'H:\data\BAYLORCW046\python\2024_06_19',

        ],
        [r'F:\data\BAYLORCW032\python\2023_10_24',
            # r'F:\data\BAYLORCW034\python\2023_10_27',
            r'F:\data\BAYLORCW036\python\2023_10_30',
            r'F:\data\BAYLORCW035\python\2023_12_15',
            r'F:\data\BAYLORCW037\python\2023_12_15',
            
            r'F:\data\BAYLORCW036\python\2023_10_28',
        r'F:\data\BAYLORCW035\python\2023_12_12',
            r'F:\data\BAYLORCW035\python\2023_12_14',
            r'F:\data\BAYLORCW035\python\2023_12_16',
            r'F:\data\BAYLORCW037\python\2023_12_13',
            
            r'H:\data\BAYLORCW044\python\2024_06_19',
            r'H:\data\BAYLORCW044\python\2024_06_18',
            r'H:\data\BAYLORCW044\python\2024_06_17',
            
            r'H:\data\BAYLORCW046\python\2024_06_24',
            r'H:\data\BAYLORCW046\python\2024_06_27',
            r'H:\data\BAYLORCW046\python\2024_06_26',
            r'H:\data\BAYLORCW046\python\2024_06_25',

        ]]

agg_matched_paths = [[    
            r'F:\data\BAYLORCW032\python\2023_10_05',
            # r'F:\data\BAYLORCW034\python\2023_10_12',
            r'F:\data\BAYLORCW036\python\2023_10_09',
            r'F:\data\BAYLORCW035\python\2023_10_26',
            r'F:\data\BAYLORCW037\python\2023_11_21',
            

        r'H:\data\BAYLORCW044\python\2024_05_22',
        r'H:\data\BAYLORCW044\python\2024_05_23',
        
        r'H:\data\BAYLORCW046\python\2024_05_29',
        r'H:\data\BAYLORCW046\python\2024_05_30',
        r'H:\data\BAYLORCW046\python\2024_05_31',
            ],
             [r'F:\data\BAYLORCW032\python\2023_10_19',
            # r'F:\data\BAYLORCW034\python\2023_10_22',
            r'F:\data\BAYLORCW036\python\2023_10_19',
            r'F:\data\BAYLORCW035\python\2023_12_07',
            r'F:\data\BAYLORCW037\python\2023_12_08',

            
        r'H:\data\BAYLORCW044\python\2024_06_06',
        r'H:\data\BAYLORCW044\python\2024_06_04',

        r'H:\data\BAYLORCW046\python\2024_06_07',
        r'H:\data\BAYLORCW046\python\2024_06_10',
        r'H:\data\BAYLORCW046\python\2024_06_11',

        ],
        [r'F:\data\BAYLORCW032\python\2023_10_24',
            # r'F:\data\BAYLORCW034\python\2023_10_27',
            r'F:\data\BAYLORCW036\python\2023_10_30',
            r'F:\data\BAYLORCW035\python\2023_12_15',
            r'F:\data\BAYLORCW037\python\2023_12_15',
            
            
            r'H:\data\BAYLORCW044\python\2024_06_19',
            r'H:\data\BAYLORCW044\python\2024_06_18',
            
            r'H:\data\BAYLORCW046\python\2024_06_24',
            r'H:\data\BAYLORCW046\python\2024_06_27',
            r'H:\data\BAYLORCW046\python\2024_06_26',

        ]]

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






#%% Proportion contra ipsi per epoch
all_contra = []
all_ipsi = []
f, ax =plt.subplots(3, 3, sharey=True, figsize=(10,10))

out_counter = 0
for paths in agg_matched_paths: # For each stage of naive learning expert
    sampleprop, delayprop, responseprop = [],[],[]
    for path in paths:
        
        l1 = Session(path, use_reg=True, triple=True)
        periods = [range(l1.sample, l1.delay), range(l1.delay, l1.response), range(l1.response, l1.time_cutoff)]
        contras, ipsis = l1.plot_number_of_sig_neurons(return_nums=True)

        period = periods[0]
        contra, ipsi = np.sum(contras[period]), np.sum(ipsis[period])
        sampleprop += [(contra/(contra+ipsi), ipsi/(contra+ipsi))]
        
        period = periods[1]
        contra, ipsi = np.sum(contras[period]), np.sum(ipsis[period])
        delayprop += [(contra/(contra+ipsi), ipsi/(contra+ipsi))]
        
        period = periods[2]
        contra, ipsi = np.sum(contras[period]), np.sum(ipsis[period])
        responseprop += [(contra/(contra+ipsi), ipsi/(contra+ipsi))]
        
        
    ax[out_counter, 0].bar([0], [np.mean(sampleprop, axis=0)[0]], yerr = [np.std(sampleprop, axis=0)[0]], color='blue')
    ax[out_counter, 0].bar([1], [np.mean(sampleprop, axis=0)[1]], yerr = [np.std(sampleprop, axis=0)[1]], color='red')
    ax[out_counter, 0].set_xticks([])

    ax[out_counter, 1].bar([0], [np.mean(delayprop, axis=0)[0]], yerr = [np.std(delayprop, axis=0)[0]], color='blue')
    ax[out_counter, 1].bar([1], [np.mean(delayprop, axis=0)[1]], yerr = [np.std(delayprop, axis=0)[1]], color='red')
    ax[out_counter, 1].set_xticks([])

    ax[out_counter, 2].bar([0], [np.mean(responseprop, axis=0)[0]], yerr = [np.std(responseprop, axis=0)[0]], color='blue')
    ax[out_counter, 2].bar([1], [np.mean(responseprop, axis=0)[1]], yerr = [np.std(responseprop, axis=0)[1]], color='red')
    ax[out_counter, 2].set_xticks([])
    
    for i in range(3):
        ax[out_counter, i].axhline(0.5, ls = '--', color='black')

    out_counter += 1
    
ax[0, 0].set_xticks([0, 1], ['Contra-pref.', 'Ipsi-pref.'], rotation=40)

ax[0, 0].set_title('Sample')
ax[0, 1].set_title('Delay')
ax[0, 2].set_title('Response')

ax[1, 0].set_title('Learning')
ax[2, 0].set_title('Expert')

#%% Previous work

# path = r'F:\data\BAYLORCW032\python\2023_10_05'
# l1 = session.Session(path, use_reg=True)
# l1.plot_number_of_sig_neurons()
# # l1.plot_number_of_sig_neurons(save=True, y_axis=[-65,65])

# path = r'F:\data\BAYLORCW032\python\2023_10_24'
# l1 = session.Session(path, use_reg=True)
# l1.plot_number_of_sig_neurons()
# # l1.plot_number_of_sig_neurons(save=True,y_axis=[-65,65])



# path = r'F:\data\BAYLORCW034\python\2023_10_12'
# l1 = session.Session(path, use_reg=True)
# # l1.plot_number_of_sig_neurons()
# stim_neurons, choice_neurons, outcome_neurons, stim_sel, choice_sel, outcome_sel = l1.stim_choice_outcome_selectivity()
# # l1.plot_number_of_sig_neurons(save=True, y_axis = [-330, 300])

# path = r'F:\data\BAYLORCW034\python\2023_10_27'
# l1 = session.Session(path, use_reg=True)
# # l1.plot_number_of_sig_neurons()
# stim_neurons, choice_neurons, outcome_neurons, stim_sel, choice_sel, outcome_sel = l1.stim_choice_outcome_selectivity()
# # l1.plot_number_of_sig_neurons(save=True)


# path = r'F:\data\BAYLORCW035\python\2023_10_11'
# l1 = session.Session(path)
# l1.plot_number_of_sig_neurons(save=True)

# path = r'F:\data\BAYLORCW035\python\2023_10_??'
# l1 = session.Session(path)
# l1.plot_number_of_sig_neurons(save=True)



# path = r'F:\data\BAYLORCW036\python\2023_10_09'
# l1 = session.Session(path)
# l1.plot_number_of_sig_neurons(save=True)

# path = r'F:\data\BAYLORCW036\python\2023_10_28'
# l1 = session.Session(path)
# l1.plot_number_of_sig_neurons(save=True)


### Proportion of selective for stim out  ###

### Naive sessions ###

# path = r'F:\data\BAYLORCW030\python\2023_06_21'
# path = r'F:\data\BAYLORCW036\python\2023_10_17'
# path = r'F:\data\BAYLORCW034\python\2023_10_22'

# for path in paths:
#     l1 = session.Session(path)
    
#     tstim, tlick, treward, tmixed = l1.single_neuron_sel('Chen 2017')
# path = r'F:\data\BAYLORCW032\python\2023_10_05'
# l1 = session.Session(path)
# tstim, tlick, treward, tmixed = l1.single_neuron_sel('Chen 2017', save=True)

# path = r'F:\data\BAYLORCW032\python\2023_10_24'
# l1 = session.Session(path)
# tstim, tlick, treward, tmixed = l1.single_neuron_sel('Chen 2017', save=True)



# path = r'F:\data\BAYLORCW034\python\2023_10_10'
# l1 = session.Session(path)
# tstim, tlick, treward, tmixed = l1.single_neuron_sel('Chen 2017', save=True)

# path = r'F:\data\BAYLORCW036\python\2023_10_09'
# l1 = session.Session(path)
# tstim, tlick, treward, tmixed = l1.single_neuron_sel('Chen 2017', save=True)


### Trained sessions ###

### Proportion of stim choice outcome neurons ###




### Selectivity trace for stim choice outcome ###





path = r'F:\data\BAYLORCW036\python\2023_10_09'
l1 = session.Session(path)
stim_neurons, choice_neurons, outcome_neurons, stim_sel, choice_sel, outcome_sel = l1.stim_choice_outcome_selectivity()

path = r'F:\data\BAYLORCW036\python\2023_10_28'
l1 = session.Session(path)
stim_neurons, choice_neurons, outcome_neurons, stim_sel, choice_sel, outcome_sel = l1.stim_choice_outcome_selectivity()


# path = r'F:\data\BAYLORCW035\python\2023_10_11'
# l1 = session.Session(path)
# stim_neurons, choice_neurons, outcome_neurons, stim_sel, choice_sel, outcome_sel = l1.stim_choice_outcome_selectivity()

# path = r'F:\data\BAYLORCW035\python\2023_10_??'
# l1 = session.Session(path)
# stim_neurons, choice_neurons, outcome_neurons, stim_sel, choice_sel, outcome_sel = l1.stim_choice_outcome_selectivity()



# path = r'F:\data\BAYLORCW034\python\2023_10_10'
# l1 = session.Session(path)
# stim_neurons, choice_neurons, outcome_neurons, stim_sel, choice_sel, outcome_sel = l1.stim_choice_outcome_selectivity()

# path = r'F:\data\BAYLORCW034\python\2023_10_24'
# l1 = session.Session(path)
# stim_neurons, choice_neurons, outcome_neurons, stim_sel, choice_sel, outcome_sel = l1.stim_choice_outcome_selectivity()



# path = r'F:\data\BAYLORCW032\python\2023_10_05'
# l1 = session.Session(path)
# stim_neurons, choice_neurons, outcome_neurons, stim_sel, choice_sel, outcome_sel = l1.stim_choice_outcome_selectivity()

# path = r'F:\data\BAYLORCW032\python\2023_10_24'
# l1 = session.Session(path)
# stim_neurons, choice_neurons, outcome_neurons, stim_sel, choice_sel, outcome_sel = l1.stim_choice_outcome_selectivity()
