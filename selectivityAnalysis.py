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

# from neuralFuncs import plot_average_PSTH
# path = r'F:\data\BAYLORCW021\python\2023_01_25'

# l1 = session.Session(path, 5)

# path = r'F:\data\BAYLORCW027\python\2023_05_05'

# path = r'F:\data\BAYLORCW021\python\2023_05_03'
path = r'F:\data\BAYLORCW021\python\2023_02_13'
# path = r'F:\data\BAYLORCW021\python\2023_04_06'

path = r'F:\data\BAYLORCW021\python\2023_04_27'

# path = r'F:\data\BAYLORCW021\python\2023_05_03'
# path = r'F:\data\BAYLORCW021\python\2023_02_08'


l1 = session.Session(path)

## Single neuron selectivity

# l1.single_neuron_sel('Susu method')

stim, lick, reward, mixed = l1.single_neuron_sel('Chen 2017')


## Population analysis

# l1.plot_three_selectivity()

# l1.population_sel_timecourse()

# l1.plot_number_of_sig_neurons()

# l1.selectivity_table_by_epoch()

# l1.selectivity_optogenetics()

## For selective neurons

# l1.stim_choice_outcome_selectivity()
    