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


#%% SUSU METHOD

p = 0.05
#NAIVE

paths = [
        r'F:\data\BAYLORCW032\python\2023_10_08',
        # r'F:\data\BAYLORCW034\python\2023_10_12',
        r'F:\dat a\BAYLORCW036\python\2023_10_09'
        ]

total_n = 0
s,c,o = 0,0,0
naivestim, naivechoice, naiveoutcome = [],[],[]
for path in paths:
    l1 = session.Session(path, use_reg=True, triple=True)
    stim_neurons, choice_neurons, _, outcome_neurons = l1.single_neuron_sel('Susu method', p=p)
    s = len(stim_neurons)
    c = len(choice_neurons)
    o = len(outcome_neurons)
    
    total_n = len(l1.good_neurons)

    naivestim += [s/total_n]
    naivechoice += [c/total_n]
    naiveoutcome += [o/total_n]
#LEARNING
paths = [
        r'F:\data\BAYLORCW032\python\2023_10_16',
        # r'F:\data\BAYLORCW034\python\2023_10_22',
        r'F:\data\BAYLORCW036\python\2023_10_19'
        ]

total_n = 0
s,c,o = 0,0,0
learningstim, learningchoice, learningoutcome = [],[],[]

for path in paths:
    l1 = session.Session(path, use_reg=True, triple=True)
    stim_neurons, choice_neurons, _, outcome_neurons = l1.single_neuron_sel('Susu method', p=p)
    s = len(stim_neurons)
    c = len(choice_neurons)
    o = len(outcome_neurons)
    
    total_n = len(l1.good_neurons)
    learningstim += [s/total_n]
    learningchoice += [c/total_n]
    learningoutcome += [o/total_n]
    

#EXPERT

paths = [
        r'F:\data\BAYLORCW032\python\2023_10_25',
        # r'F:\data\BAYLORCW034\python\2023_10_27',
        r'F:\data\BAYLORCW036\python\2023_10_30'
        ]

total_n = 0
s,c,o = 0,0,0
expertstim, expertchoice, expertoutcome = [],[],[]

for path in paths:
    l1 = session.Session(path, use_reg=True, triple=True)
    stim_neurons, choice_neurons, _, outcome_neurons = l1.single_neuron_sel('Susu method', p=p)
    s = len(stim_neurons)
    c = len(choice_neurons)
    o = len(outcome_neurons)
    
    total_n = len(l1.good_neurons)
    expertstim += [s/total_n]
    expertchoice += [c/total_n]
    expertoutcome += [o/total_n]
    



#Error bars
naivestimerr, naivechoiceerr, naiveoutcomeerr = np.std(naivestim)/np.sqrt(3), np.std(naivechoice)/np.sqrt(3), np.std(naiveoutcome)/np.sqrt(3)
learningstimerr, learningchoiceerr, learningoutcomeerr = np.std(learningstim)/np.sqrt(3), np.std(learningchoice)/np.sqrt(3), np.std(learningoutcome)/np.sqrt(3)
expertstimerr, expertchoiceerr, expertoutcomeerr = np.std(expertstim)/np.sqrt(3), np.std(expertchoice)/np.sqrt(3), np.std(expertoutcome)/np.sqrt(3)

#Mean

naivestim, naivechoice, naiveoutcome = np.mean(naivestim), np.mean(naivechoice), np.mean(naiveoutcome)
learningstim, learningchoice, learningoutcome = np.mean(learningstim), np.mean(learningchoice), np.mean(learningoutcome)
expertstim, expertchoice, expertoutcome = np.mean(expertstim), np.mean(expertchoice), np.mean(expertoutcome)

plt.plot([1,4,7], [naivestim, learningstim, expertstim], color='green',label='Stimulus')
plt.plot([1,4,7], [naivechoice, learningchoice, expertchoice], color='purple', label='Choice')
plt.plot([1,4,7], [naiveoutcome, learningoutcome, expertoutcome], color='dodgerblue', label='Outcome')

plt.errorbar([1,4,7], [naivestim, learningstim, expertstim], yerr = [naivestimerr, naivechoiceerr, naiveoutcomeerr], color='green')
plt.errorbar([1,4,7], [naivechoice, learningchoice, expertchoice], yerr = [learningstimerr, learningchoiceerr, learningoutcomeerr], color='purple')
plt.errorbar([1,4,7], [naiveoutcome, learningoutcome, expertoutcome], yerr = [expertstimerr, expertchoiceerr, expertoutcomeerr], color='dodgerblue')

plt.scatter([1,4,7], [naivestim, learningstim, expertstim], color='lightgreen', marker = 'o', s=150, alpha = 0.5)
plt.scatter([1,4,7], [naivechoice, learningchoice, expertchoice], color='violet', marker = 'o', s=150, alpha = 0.5)
plt.scatter([1,4,7], [naiveoutcome, learningoutcome, expertoutcome], color='lightskyblue',  marker = 'o', s=150, alpha = 0.5)

plt.xticks([1,4,7], ['Naive', 'Learning', 'Expert'])
plt.xlabel('Training stage')
plt.ylabel('Proportion of selective neurons')
plt.legend()

# plt.savefig(r'F:\data\SFN 2023\SCO_proportions_over_learning.pdf')
plt.show()


#%% Chen et al method

p = 0.05
#NAIVE

paths = [
        r'F:\data\BAYLORCW032\python\2023_10_08',
        # r'F:\data\BAYLORCW034\python\2023_10_12',
        r'F:\data\BAYLORCW036\python\2023_10_09'
        ]

total_n = 0
s,c,o = 0,0,0
naivestim, naivechoice, naiveoutcome = [],[],[]
for path in paths:
    l1 = session.Session(path, use_reg=True, triple=True)
    s, c, _, o = l1.single_neuron_sel('Chen proportions')
    # s = len(stim_neurons)
    # c = len(choice_neurons)
    # o = len(outcome_neurons)
    
    total_n = len(l1.good_neurons)

    naivestim += [s/total_n]
    naivechoice += [c/total_n]
    naiveoutcome += [o/total_n]
#LEARNING
paths = [
        r'F:\data\BAYLORCW032\python\2023_10_16',
        # r'F:\data\BAYLORCW034\python\2023_10_22',
        r'F:\data\BAYLORCW036\python\2023_10_19'
        ]

total_n = 0
s,c,o = 0,0,0
learningstim, learningchoice, learningoutcome = [],[],[]

for path in paths:
    l1 = session.Session(path, use_reg=True, triple=True)
    s, c, _, o = l1.single_neuron_sel('Chen proportions')
    # s = len(stim_neurons)
    # c = len(choice_neurons)
    # o = len(outcome_neurons)
    
    total_n = len(l1.good_neurons)
    learningstim += [s/total_n]
    learningchoice += [c/total_n]
    learningoutcome += [o/total_n]
    

#EXPERT

paths = [
        r'F:\data\BAYLORCW032\python\2023_10_25',
        # r'F:\data\BAYLORCW034\python\2023_10_27',
        r'F:\data\BAYLORCW036\python\2023_10_30'
        ]

total_n = 0
s,c,o = 0,0,0
expertstim, expertchoice, expertoutcome = [],[],[]

for path in paths:
    l1 = session.Session(path, use_reg=True, triple=True)
    s, c, _, o = l1.single_neuron_sel('Chen proportions')
    # s += len(stim_neurons)
    # c += len(choice_neurons)
    # o += len(outcome_neurons)
    
    total_n = len(l1.good_neurons)
    expertstim += [s/total_n]
    expertchoice += [c/total_n]
    expertoutcome += [o/total_n]
    



#Error bars
naivestimerr, naivechoiceerr, naiveoutcomeerr = np.std(naivestim)/np.sqrt(3), np.std(naivechoice)/np.sqrt(3), np.std(naiveoutcome)/np.sqrt(3)
learningstimerr, learningchoiceerr, learningoutcomeerr = np.std(learningstim)/np.sqrt(3), np.std(learningchoice)/np.sqrt(3), np.std(learningoutcome)/np.sqrt(3)
expertstimerr, expertchoiceerr, expertoutcomeerr = np.std(expertstim)/np.sqrt(3), np.std(expertchoice)/np.sqrt(3), np.std(expertoutcome)/np.sqrt(3)

#Mean

naivestim, naivechoice, naiveoutcome = np.mean(naivestim), np.mean(naivechoice), np.mean(naiveoutcome)
learningstim, learningchoice, learningoutcome = np.mean(learningstim), np.mean(learningchoice), np.mean(learningoutcome)
expertstim, expertchoice, expertoutcome = np.mean(expertstim), np.mean(expertchoice), np.mean(expertoutcome)

plt.plot([1,4,7], [naivestim, learningstim, expertstim], color='green',label='Stimulus')
plt.plot([1,4,7], [naivechoice, learningchoice, expertchoice], color='purple', label='Choice')
plt.plot([1,4,7], [naiveoutcome, learningoutcome, expertoutcome], color='dodgerblue', label='Outcome')

plt.errorbar([1,4,7], [naivestim, learningstim, expertstim], yerr = [naivestimerr, naivechoiceerr, naiveoutcomeerr], color='green')
plt.errorbar([1,4,7], [naivechoice, learningchoice, expertchoice], yerr = [learningstimerr, learningchoiceerr, learningoutcomeerr], color='purple')
plt.errorbar([1,4,7], [naiveoutcome, learningoutcome, expertoutcome], yerr = [expertstimerr, expertchoiceerr, expertoutcomeerr], color='dodgerblue')

plt.scatter([1,4,7], [naivestim, learningstim, expertstim], color='lightgreen', marker = 'o', s=150, alpha = 0.5)
plt.scatter([1,4,7], [naivechoice, learningchoice, expertchoice], color='violet', marker = 'o', s=150, alpha = 0.5)
plt.scatter([1,4,7], [naiveoutcome, learningoutcome, expertoutcome], color='lightskyblue',  marker = 'o', s=150, alpha = 0.5)

plt.xticks([1,4,7], ['Naive', 'Learning', 'Expert'])
plt.xlabel('Training stage')
plt.ylabel('Proportion of selective neurons')
plt.legend()

plt.savefig(r'F:\data\SFN 2023\SCO_proportions_over_learning_chenmethod.pdf')
plt.show()


#%% Different t-test METHOD

p = 0.01
#NAIVE

paths = [
        r'F:\data\BAYLORCW032\python\2023_10_08',
        r'F:\data\BAYLORCW034\python\2023_10_12',
        r'F:\data\BAYLORCW036\python\2023_10_09']

total_n = 0
s,c,o = 0,0,0
naivestim, naivechoice, naiveoutcome = [],[],[]
for path in paths:
    l1 = session.Session(path, use_reg=True, triple=True)
    stim_neurons, _, _, outcome_neurons = l1.single_neuron_sel('Susu method', p=p)
    choice_neurons = l1.get_epoch_selective(range(l1.delay +6, l1.response), p=p, lickdir=True)
    s = len(stim_neurons)
    c = len(choice_neurons)
    o = len(outcome_neurons)
    
    total_n = len(l1.good_neurons)

    naivestim += [s/total_n]
    naivechoice += [c/total_n]
    naiveoutcome += [o/total_n]
#LEARNING
paths = [
        r'F:\data\BAYLORCW032\python\2023_10_16',
        r'F:\data\BAYLORCW034\python\2023_10_22',
        r'F:\data\BAYLORCW036\python\2023_10_19']

total_n = 0
s,c,o = 0,0,0
learningstim, learningchoice, learningoutcome = [],[],[]

for path in paths:
    l1 = session.Session(path, use_reg=True, triple=True)
    stim_neurons, _, _, outcome_neurons = l1.single_neuron_sel('Susu method', p=p)
    choice_neurons = l1.get_epoch_selective(range(l1.delay +6, l1.response), p=p,lickdir=True)

    s = len(stim_neurons)
    c = len(choice_neurons)
    o = len(outcome_neurons)
    
    total_n = len(l1.good_neurons)
    learningstim += [s/total_n]
    learningchoice += [c/total_n]
    learningoutcome += [o/total_n]
    

#EXPERT

paths = [
        r'F:\data\BAYLORCW032\python\2023_10_25',
        r'F:\data\BAYLORCW034\python\2023_10_27',
        r'F:\data\BAYLORCW036\python\2023_10_30']

total_n = 0
s,c,o = 0,0,0
expertstim, expertchoice, expertoutcome = [],[],[]

for path in paths:
    l1 = session.Session(path, use_reg=True, triple=True)
    stim_neurons, _, _, outcome_neurons = l1.single_neuron_sel('Susu method', p=p)
    choice_neurons = l1.get_epoch_selective(range(l1.delay +6, l1.response), p=p,lickdir=True)

    s += len(stim_neurons)
    c += len(choice_neurons)
    o += len(outcome_neurons)
    
    total_n += len(l1.good_neurons)
    expertstim += [s/total_n]
    expertchoice += [c/total_n]
    expertoutcome += [o/total_n]
    



#Error bars
naivestimerr, naivechoiceerr, naiveoutcomeerr = np.std(naivestim)/np.sqrt(3), np.std(naivechoice)/np.sqrt(3), np.std(naiveoutcome)/np.sqrt(3)
learningstimerr, learningchoiceerr, learningoutcomeerr = np.std(learningstim)/np.sqrt(3), np.std(learningchoice)/np.sqrt(3), np.std(learningoutcome)/np.sqrt(3)
expertstimerr, expertchoiceerr, expertoutcomeerr = np.std(expertstim)/np.sqrt(3), np.std(expertchoice)/np.sqrt(3), np.std(expertoutcome)/np.sqrt(3)

#Mean

naivestim, naivechoice, naiveoutcome = np.mean(naivestim), np.mean(naivechoice), np.mean(naiveoutcome)
learningstim, learningchoice, learningoutcome = np.mean(learningstim), np.mean(learningchoice), np.mean(learningoutcome)
expertstim, expertchoice, expertoutcome = np.mean(expertstim), np.mean(expertchoice), np.mean(expertoutcome)

plt.plot([1,4,7], [naivestim, learningstim, expertstim], color='green',label='Stimulus')
plt.plot([1,4,7], [naivechoice, learningchoice, expertchoice], color='purple', label='Choice')
plt.plot([1,4,7], [naiveoutcome, learningoutcome, expertoutcome], color='dodgerblue', label='Outcome')

plt.errorbar([1,4,7], [naivestim, learningstim, expertstim], yerr = [naivestimerr, naivechoiceerr, naiveoutcomeerr], color='green')
plt.errorbar([1,4,7], [naivechoice, learningchoice, expertchoice], yerr = [learningstimerr, learningchoiceerr, learningoutcomeerr], color='purple')
plt.errorbar([1,4,7], [naiveoutcome, learningoutcome, expertoutcome], yerr = [expertstimerr, expertchoiceerr, expertoutcomeerr], color='dodgerblue')

plt.scatter([1,4,7], [naivestim, learningstim, expertstim], color='lightgreen', marker = 'o', s=150, alpha = 0.5)
plt.scatter([1,4,7], [naivechoice, learningchoice, expertchoice], color='violet', marker = 'o', s=150, alpha = 0.5)
plt.scatter([1,4,7], [naiveoutcome, learningoutcome, expertoutcome], color='lightskyblue',  marker = 'o', s=150, alpha = 0.5)

plt.xticks([1,4,7], ['Naive', 'Learning', 'Expert'])
plt.xlabel('Training stage')
plt.ylabel('Proportion of selective neurons')
plt.legend()

# plt.savefig(r'F:\data\SFN 2023\SCO_proportions_over_learning.pdf')
plt.show()
