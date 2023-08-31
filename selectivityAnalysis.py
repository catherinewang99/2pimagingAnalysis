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
from matplotlib.pyplot import figure
import decon
from scipy.stats import chisquare
import pandas as pd

# from neuralFuncs import plot_average_PSTH
# path = r'F:\data\BAYLORCW021\python\2023_01_25'

# l1 = session.Session(path, 5)

# path = r'F:\data\BAYLORCW027\python\2023_05_05'

# path = r'F:\data\BAYLORCW021\python\2023_05_03'
# path = r'F:\data\BAYLORCW021\python\2023_02_13'
# # path = r'F:\data\BAYLORCW021\python\2023_04_06'

path = r'F:\data\BAYLORCW021\python\2023_04_27'

# path = r'F:\data\BAYLORCW021\python\2023_04_06'
# # path = r'F:\data\BAYLORCW021\python\2023_05_03'
# path = r'F:\data\BAYLORCW021\python\2023_02_08'
# path = r'F:\data\BAYLORCW030\python\2023_07_03'

path = r'F:\data\BAYLORCW030\python\2023_07_12'

l1 = session.Session(path, layer_num=2)
# l1 = decon.Deconvolved(path)
## Single neuron selectivity

# l1.single_neuron_sel('Susu method')

# stim, lick, reward, mixed = l1.single_neuron_sel('Chen 2017')


## Population analysis

# l1.plot_three_selectivity()

# l1.population_sel_timecourse()

# l1.plot_number_of_sig_neurons()

l1.selectivity_table_by_epoch()

l1.selectivity_optogenetics()

## For selective neurons
# l1 = session.Session(path)

# stim_neurons, choice_neurons, outcome_neurons, stim_sel, choice_sel, outcome_sel = l1.stim_choice_outcome_selectivity()
# n_numneurons = l1.num_neurons

# tstim, tlick, treward, tmixed = l1.single_neuron_sel('Chen 2017')

# path = r'F:\data\BAYLORCW021\python\2023_04_27'
# l2 = session.Session(path)
# tstim_neurons, tchoice_neurons, toutcome_neurons, tstim_sel, tchoice_sel, toutcome_sel = l2.stim_choice_outcome_selectivity()


# x = np.array([1,2,3])
# plt.bar(x-0.2, [len(tstim_neurons)/len(l2.good_neurons), 
#                 len(tchoice_neurons)/len(l2.good_neurons), 
#                 len(toutcome_neurons)/len(l2.good_neurons)], width = 0.2, label = 'Trained')
# plt.bar(x+0.2, [len(stim_neurons)/len(l1.good_neurons), 
#                 len(choice_neurons)/len(l1.good_neurons), 
#                 len(outcome_neurons)/len(l1.good_neurons)], width = 0.2, label = 'Naive')
# plt.legend()

# tstim, tlick, treward, tmixed = l2.single_neuron_sel('Chen 2017')

# t_numneurons = l2.num_neurons

# npop = np.zeros(n_numneurons)
# npop[outcome_neurons] = 1

# tpop = np.zeros(t_numneurons)
# tpop[toutcome_neurons] = 1

# obs = np.array([np.sum(np.random.choice(npop, size = 100, replace=True)), np.sum(np.random.choice(tpop, size = 100, replace=True))])
# chisquare(obs)


# import scipy.stats as stats
# from scipy.stats import chi2_contingency
# res = chi2_contingency([[len(outcome_neurons), n_numneurons - len(outcome_neurons)], 
#                            [len(toutcome_neurons), t_numneurons - len(toutcome_neurons)]])



# from scipy.stats import ttest_ind, ttest_ind_from_stats
# ttest_ind(stim_sel, tstim_sel, equal_var=False)
# ttest_ind(outcome_sel, toutcome_sel, equal_var=False)

# ttest_ind(lick[7:13]/, tlick[7:13])
