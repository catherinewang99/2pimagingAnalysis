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
import session
from matplotlib.pyplot import figure
# import decon
from scipy.stats import chisquare
import pandas as pd
plt.rcParams['pdf.fonttype'] = '42' 



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
