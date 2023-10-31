# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:11:12 2023

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


#NAIVE

paths = [r'F:\data\BAYLORCW032\python\2023_10_05',
        # r'F:\data\BAYLORCW034\python\2023_10_12',
        r'F:\data\BAYLORCW036\python\2023_10_09']

total_n = 0
s,c,o = 0,0,0
for path in paths:
    l1 = session.Session(path)
    stim_neurons, choice_neurons, _, outcome_neurons = l1.single_neuron_sel('Susu method')
    s += len(stim_neurons)
    c += len(choice_neurons)
    o += len(outcome_neurons)
    
    total_n += len(l1.good_neurons)

naivestim, naivechoice, naiveoutcome = s/total_n, c/total_n, o/total_n
#LEARNING
paths = [r'F:\data\BAYLORCW032\python\2023_10_16',
        # r'F:\data\BAYLORCW034\python\2023_10_22',
        r'F:\data\BAYLORCW036\python\2023_10_19']

total_n = 0
s,c,o = 0,0,0
for path in paths:
    l1 = session.Session(path)
    stim_neurons, choice_neurons, _, outcome_neurons = l1.single_neuron_sel('Susu method')
    s += len(stim_neurons)
    c += len(choice_neurons)
    o += len(outcome_neurons)
    
    total_n += len(l1.good_neurons)

learningstim, learningchoice, learningoutcome = s/total_n, c/total_n, o/total_n

#EXPERT

paths = [r'F:\data\BAYLORCW032\python\2023_10_24',
        # r'F:\data\BAYLORCW034\python\2023_10_27',
        r'F:\data\BAYLORCW036\python\2023_10_28']

total_n = 0
s,c,o = 0,0,0
for path in paths:
    l1 = session.Session(path)
    stim_neurons, choice_neurons, _, outcome_neurons = l1.single_neuron_sel('Susu method')
    s += len(stim_neurons)
    c += len(choice_neurons)
    o += len(outcome_neurons)
    
    total_n += len(l1.good_neurons)
expertstim, expertchoice, expertoutcome = s/total_n, c/total_n, o/total_n


plt.plot([1,4,7], [naivestim, learningstim, expertstim], color='green',label='Stimulus')
plt.plot([1,4,7], [naivechoice, learningchoice, expertchoice], color='purple', label='Choice')
plt.plot([1,4,7], [naiveoutcome, learningoutcome, expertoutcome], color='dodgerblue', label='Outcome')

plt.scatter([1,4,7], [naivestim, learningstim, expertstim], color='lightgreen', marker = 'o', s=150, alpha = 0.5)
plt.scatter([1,4,7], [naivechoice, learningchoice, expertchoice], color='violet', marker = 'o', s=150, alpha = 0.5)
plt.scatter([1,4,7], [naiveoutcome, learningoutcome, expertoutcome], color='lightskyblue',  marker = 'o', s=150, alpha = 0.5)

plt.xticks([1,4,7], ['Naive', 'Learning', 'Expert'])
plt.xlabel('Training stage')
plt.ylabel('Proportion of selective neurons')
plt.legend()

plt.savefig(r'F:\data\SFN 2023\SCO_proportions_over_learning.pdf')
plt.show()




