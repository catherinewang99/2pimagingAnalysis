# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 14:48:59 2023

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

paths = [
        r'F:\data\BAYLORCW032\python\2023_10_08',
        # r'F:\data\BAYLORCW034\python\2023_10_12',
        r'F:\data\BAYLORCW036\python\2023_10_09',
        r'F:\data\BAYLORCW035\python\2023_10_26',
        # r'F:\data\BAYLORCW037\python\2023_11_21',
        ]

stimnonpref, stimpref = [], []
choicenonpref, choicepref = [], []
outcomenonpref, outcomepref = [], []
for path in paths:
    l1 = session.Session(path, use_reg=True, triple=True)
    _, _, numoutcome, stim_sel, choice_sel, outcome_sel = l1.stim_choice_outcome_selectivity()

    snp, sp = stim_sel
    stimnonpref += snp
    stimpref += sp
    
    snp, sp = choice_sel
    choicenonpref += snp
    choicepref += sp
    
    snp, sp = outcome_sel
    outcomenonpref += snp
    outcomepref += sp
     
    print(len(numoutcome))


f, axarr = plt.subplots(3,1, figsize=(5,15))
plt.setp(axarr, ylim=(-0.1,1.0))

x = np.arange(-6.97,4,l1.fs)[:l1.time_cutoff]
titles = ['Stimulus selective', 'Choice selective', 'Outcome selective']

err = np.std(stimpref, axis=0) / np.sqrt(len(stimpref)) 
err += np.std(stimnonpref, axis=0) / np.sqrt(len(stimnonpref))
sel = np.mean(stimpref, axis=0) - np.mean(stimnonpref, axis=0) 
axarr[0].plot(x, sel, color='green')
        
axarr[0].fill_between(x, sel - err, 
          sel + err,
          color='lightgreen')

axarr[0].set_title(titles[0])


err = np.std(choicepref, axis=0) / np.sqrt(len(choicepref)) 
err += np.std(choicenonpref, axis=0) / np.sqrt(len(choicenonpref))
sel = np.mean(choicepref, axis=0) - np.mean(choicenonpref, axis=0) 
axarr[1].plot(x, sel, color='purple')
        
axarr[1].fill_between(x, sel - err, 
          sel + err,
          color='violet')
axarr[1].set_title(titles[1])
        


err = np.std(outcomepref, axis=0) / np.sqrt(len(outcomepref)) 
err += np.std(outcomenonpref, axis=0) / np.sqrt(len(outcomenonpref))
sel = np.mean(outcomepref, axis=0) - np.mean(outcomenonpref, axis=0) 
axarr[2].plot(x, sel, color='dodgerblue')
        
axarr[2].fill_between(x, sel - err, 
          sel + err,
          color='lightskyblue')

axarr[2].set_title(titles[2])

###########################################

axarr[0].set_ylabel('Selectivity')
axarr[1].set_xlabel('Time from Go cue (s)')

for i in range(3):
    
    axarr[i].axvline(0, color = 'grey', alpha=0.5, ls = '--')
    axarr[i].axvline(-4.3, color = 'grey', alpha=0.5, ls = '--')
    axarr[i].axvline(-3, color = 'grey', alpha=0.5, ls = '--')        
    axarr[i].axhline(0, color = 'grey', alpha=0.5, ls = '--')

# plt.savefig(r'F:\data\Fig 2\naive_stim_choice_outcome_selectivitynocw32.pdf')
plt.show()

#%%
# LEARNING

paths = [
        r'F:\data\BAYLORCW032\python\2023_10_16',
        # r'F:\data\BAYLORCW034\python\2023_10_22',
        r'F:\data\BAYLORCW036\python\2023_10_19',
        r'F:\data\BAYLORCW035\python\2023_12_07',
        r'F:\data\BAYLORCW037\python\2023_12_08',
        ]

stimnonpref, stimpref = [], []
choicenonpref, choicepref = [], []
outcomenonpref, outcomepref = [], []
for path in paths:
    l1 = session.Session(path, use_reg=True, triple=True)
    _, _, _, stim_sel, choice_sel, outcome_sel = l1.stim_choice_outcome_selectivity()

    snp, sp = stim_sel
    stimnonpref += snp
    stimpref += sp
    
    snp, sp = choice_sel
    choicenonpref += snp
    choicepref += sp
    
    snp, sp = outcome_sel
    outcomenonpref += snp
    outcomepref += sp
     


f, axarr = plt.subplots(3,1, figsize=(5,15))
plt.setp(axarr, ylim=(-0.1,1.0))

x = np.arange(-6.97,4,l1.fs)[:l1.time_cutoff]
titles = ['Stimulus selective', 'Choice selective', 'Outcome selective']

err = np.std(stimpref, axis=0) / np.sqrt(len(stimpref)) 
err += np.std(stimnonpref, axis=0) / np.sqrt(len(stimnonpref))
sel = np.mean(stimpref, axis=0) - np.mean(stimnonpref, axis=0) 
axarr[0].plot(x, sel, color='green')
        
axarr[0].fill_between(x, sel - err, 
          sel + err,
          color='lightgreen')

axarr[0].set_title(titles[0])


err = np.std(choicepref, axis=0) / np.sqrt(len(choicepref)) 
err += np.std(choicenonpref, axis=0) / np.sqrt(len(choicenonpref))
sel = np.mean(choicepref, axis=0) - np.mean(choicenonpref, axis=0) 
axarr[1].plot(x, sel, color='purple')
        
axarr[1].fill_between(x, sel - err, 
          sel + err,
          color='violet')
axarr[1].set_title(titles[1])
        


err = np.std(outcomepref, axis=0) / np.sqrt(len(outcomepref)) 
err += np.std(outcomenonpref, axis=0) / np.sqrt(len(outcomenonpref))
sel = np.mean(outcomepref, axis=0) - np.mean(outcomenonpref, axis=0) 
axarr[2].plot(x, sel, color='dodgerblue')
        
axarr[2].fill_between(x, sel - err, 
          sel + err,
          color='lightskyblue')

axarr[2].set_title(titles[2])

###########################################

axarr[0].set_ylabel('Selectivity')
axarr[1].set_xlabel('Time from Go cue (s)')

for i in range(3):
    
    axarr[i].axvline(0, color = 'grey', alpha=0.5, ls = '--')
    axarr[i].axvline(-4.3, color = 'grey', alpha=0.5, ls = '--')
    axarr[i].axvline(-3, color = 'grey', alpha=0.5, ls = '--')        
    axarr[i].axhline(0, color = 'grey', alpha=0.5, ls = '--')

plt.savefig(r'F:\data\Fig 2\learning_stim_choice_outcome_selectivitynocw32.pdf')
plt.show()

#EXPERT
paths = [
        # r'F:\data\BAYLORCW032\python\2023_10_25',
        # r'F:\data\BAYLORCW034\python\2023_10_27',
        r'F:\data\BAYLORCW036\python\2023_10_30',
        r'F:\data\BAYLORCW035\python\2023_12_15',
        r'F:\data\BAYLORCW037\python\2023_12_15',
        ]

stimnonpref, stimpref = [], []
choicenonpref, choicepref = [], []
outcomenonpref, outcomepref = [], []
for path in paths:
    l1 = session.Session(path, use_reg=True, triple=True)
    _, _, _, stim_sel, choice_sel, outcome_sel = l1.stim_choice_outcome_selectivity()

    snp, sp = stim_sel
    stimnonpref += snp
    stimpref += sp
    
    snp, sp = choice_sel
    choicenonpref += snp
    choicepref += sp
    
    snp, sp = outcome_sel
    outcomenonpref += snp
    outcomepref += sp
     
f, axarr = plt.subplots(3,1, figsize=(5,15))
plt.setp(axarr, ylim=(-0.1,1.0))

x = np.arange(-6.97,4,l1.fs)[:l1.time_cutoff]
titles = ['Stimulus selective', 'Choice selective', 'Outcome selective']

err = np.std(stimpref, axis=0) / np.sqrt(len(stimpref)) 
err += np.std(stimnonpref, axis=0) / np.sqrt(len(stimnonpref))
sel = np.mean(stimpref, axis=0) - np.mean(stimnonpref, axis=0) 
axarr[0].plot(x, sel, color='green')
        
axarr[0].fill_between(x, sel - err, 
          sel + err,
          color='lightgreen')

axarr[0].set_title(titles[0])


err = np.std(choicepref, axis=0) / np.sqrt(len(choicepref)) 
err += np.std(choicenonpref, axis=0) / np.sqrt(len(choicenonpref))
sel = np.mean(choicepref, axis=0) - np.mean(choicenonpref, axis=0) 
axarr[1].plot(x, sel, color='purple')
        
axarr[1].fill_between(x, sel - err, 
          sel + err,
          color='violet')
axarr[1].set_title(titles[1])
        


err = np.std(outcomepref, axis=0) / np.sqrt(len(outcomepref)) 
err += np.std(outcomenonpref, axis=0) / np.sqrt(len(outcomenonpref))
sel = np.mean(outcomepref, axis=0) - np.mean(outcomenonpref, axis=0) 
axarr[2].plot(x, sel, color='dodgerblue')
        
axarr[2].fill_between(x, sel - err, 
          sel + err,
          color='lightskyblue')

axarr[2].set_title(titles[2])

###########################################

axarr[0].set_ylabel('Selectivity')
axarr[1].set_xlabel('Time from Go cue (s)')

for i in range(3):
    
    axarr[i].axvline(0, color = 'grey', alpha=0.5, ls = '--')
    axarr[i].axvline(-4.3, color = 'grey', alpha=0.5, ls = '--')
    axarr[i].axvline(-3, color = 'grey', alpha=0.5, ls = '--')        
    axarr[i].axhline(0, color = 'grey', alpha=0.5, ls = '--')

plt.savefig(r'F:\data\Fig 2\trained_stim_choice_outcome_selectivitynocw32.pdf')
plt.show()

#%% Action mode selectivity trace:
    
allpaths = [ [
        # r'F:\data\BAYLORCW032\python\2023_10_08',
        r'F:\data\BAYLORCW034\python\2023_10_12',
        r'F:\data\BAYLORCW036\python\2023_10_09',
        r'F:\data\BAYLORCW035\python\2023_10_26',
        # r'F:\data\BAYLORCW037\python\2023_11_21',
        ],
    
    
    [
        # r'F:\data\BAYLORCW032\python\2023_10_16',
        r'F:\data\BAYLORCW034\python\2023_10_22',
        r'F:\data\BAYLORCW036\python\2023_10_19',
        r'F:\data\BAYLORCW035\python\2023_12_07',
        # r'F:\data\BAYLORCW037\python\2023_12_08',
        ],
    
    
    [
        # r'F:\data\BAYLORCW032\python\2023_10_25',
        r'F:\data\BAYLORCW034\python\2023_10_27',
        r'F:\data\BAYLORCW036\python\2023_10_30',
        r'F:\data\BAYLORCW035\python\2023_12_15',
        # r'F:\data\BAYLORCW037\python\2023_12_15',
        ]
    ]
f, axarr = plt.subplots(1,3, figsize=(15,5))
plt.setp(axarr, ylim=(-0.1,1.0))

x = np.arange(-6.97,4,l1.fs)[:l1.time_cutoff]
titles = ['Stimulus selective', 'Choice selective', 'Outcome selective']
for i in range(3):
    
    stimnonpref, stimpref = [], []
    
    paths = allpaths[i]
    for path in paths:
        l1 = session.Session(path, use_reg=True, triple=True)
        action_sel = l1.stim_choice_outcome_selectivity(action=True)
    
        snp, sp = action_sel
        stimnonpref += snp
        stimpref += sp
        
    
    err = np.std(stimpref, axis=0) / np.sqrt(len(stimpref)) 
    err += np.std(stimnonpref, axis=0) / np.sqrt(len(stimnonpref))
    sel = np.mean(stimpref, axis=0) - np.mean(stimnonpref, axis=0) 
    axarr[i].plot(x, sel, color='goldenrod')
            
    axarr[i].fill_between(x, sel - err, 
              sel + err,
              color='wheat')
    
    axarr[i].set_title(titles[0])

    
    axarr[i].axvline(0, color = 'grey', alpha=0.5, ls = '--')
    axarr[i].axvline(-4.3, color = 'grey', alpha=0.5, ls = '--')
    axarr[i].axvline(-3, color = 'grey', alpha=0.5, ls = '--')        
    axarr[i].axhline(0, color = 'grey', alpha=0.5, ls = '--')
    
plt.savefig(r'F:\data\Fig 2\action_NLE_selectivity_minuscw3237.pdf')
plt.show()