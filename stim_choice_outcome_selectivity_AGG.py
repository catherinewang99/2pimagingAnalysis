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

#%% Plot all at once av over neurons
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


f, axarr = plt.subplots(3,3, sharey='row', sharex = True, figsize=(18,18))
# plt.setp(axarr, ylim=(-0.2,1.2))

for j in range(3):
    
    stimnonpref, stimpref = np.zeros(61), np.zeros(61)
    choicenonpref, choicepref = np.zeros(61), np.zeros(61)
    actionnonpref, actionpref = np.zeros(61), np.zeros(61)
    
    for path in agg_matched_paths[j]:
        l1 = session.Session(path, use_reg=True, triple=True)
        # adjusted_p = 0.05 / np.sqrt(len(l1.good_neurons))
        adjusted_p = 0.01
        _, _, _, numoutcome, stim_sel, choice_sel, outcome_sel, action_sel = l1.stim_choice_outcome_selectivity(plot=False, p=adjusted_p,
                                                                                                                downsample='04' in path)
        snp, sp = stim_sel
        if len(snp) != 0 and np.sum(snp) != 0:
            stimnonpref = np.vstack((stimnonpref, snp))
            stimpref = np.vstack((stimpref, sp))
        
        snp, sp = choice_sel
        if len(snp) != 0 and np.sum(snp) != 0:

            choicenonpref = np.vstack((choicenonpref, snp))
            choicepref = np.vstack((choicepref, sp))
         
        snp, sp = action_sel
        if len(snp) != 0 and np.sum(snp) != 0:
    
            actionnonpref = np.vstack((actionnonpref, snp))
            actionpref = np.vstack((actionpref, sp))
            
    print(len(stimpref), len(choicepref), len(actionpref))

    x = np.arange(-6.97,4,1/6)[:61]
    titles = ['Stimulus selective', 'Choice selective', 'Outcome selective', 'Action selective']
    
    stimnonpref, stimpref = stimnonpref[1:,:], stimpref[1:,:]
    choicenonpref, choicepref = choicenonpref[1:,:], choicepref[1:,:]
    actionnonpref, actionpref = actionnonpref[1:,:], actionpref[1:,:]
    
    err = np.std((stimpref - stimnonpref), axis=0) / np.sqrt(len(stimpref)) 
    sel = np.mean((stimpref - stimnonpref), axis=0)
    axarr[0,j].plot(x, sel, color='green')
            
    axarr[0,j].fill_between(x, sel - err, 
              sel + err,
              color='lightgreen')
    
    axarr[0,j].set_title(titles[0])
    
    err = np.std((choicepref - choicenonpref), axis=0) / np.sqrt(len(choicepref)) 
    sel = np.mean((choicepref - choicenonpref), axis=0)
    # err = np.std(choicepref, axis=0) / np.sqrt(len(choicepref)) 
    # err += np.std(choicenonpref, axis=0) / np.sqrt(len(choicenonpref))
    # sel = np.mean(choicepref, axis=0) - np.mean(choicenonpref, axis=0) 
    axarr[1,j].plot(x, sel, color='purple')
            
    axarr[1,j].fill_between(x, sel - err, 
              sel + err,
              color='violet')
    axarr[1,j].set_title(titles[1])
    
    err = np.std((actionpref - actionnonpref), axis=0) / np.sqrt(len(actionpref)) 
    sel = np.mean((actionpref - actionnonpref), axis=0)
    # err = np.std(actionpref, axis=0) / np.sqrt(len(actionpref)) 
    # err += np.std(actionnonpref, axis=0) / np.sqrt(len(actionnonpref))
    # sel = np.mean(actionpref, axis=0) - np.mean(actionnonpref, axis=0) 
    axarr[2,j].plot(x, sel, color='goldenrod')
            
    axarr[2,j].fill_between(x, sel - err, 
              sel + err,
              color='wheat')
    
    axarr[2,j].set_title(titles[3])
    
    
for i in range(3):
    for j in range(3):
        axarr[j,i].axvline(0, color = 'grey', alpha=0.5, ls = '--')
        axarr[j,i].axvline(-4.3, color = 'grey', alpha=0.5, ls = '--')
        axarr[j,i].axvline(-3, color = 'grey', alpha=0.5, ls = '--')        
        axarr[j,i].axhline(0, color = 'grey', alpha=0.5, ls = '--')
        
        
axarr[0,0].set_ylabel('Selectivity')
axarr[2,1].set_xlabel('Time from Go cue (s)')



plt.savefig(r'F:\data\Fig 2\NLE_all_selectivityagg_new_p01.pdf')
plt.show()


#%% Plot all at once av over FOV
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


f, axarr = plt.subplots(4,3, sharey='row', sharex = True, figsize=(24,18))
plt.setp(axarr, ylim=(-0.2,1.2))

for j in range(3):
    
    stimnonpref, stimpref = [], []
    choicenonpref, choicepref = [], []
    outcomenonpref, outcomepref = [], []
    actionnonpref, actionpref = [], []
    
    for path in agg_matched_paths[j]:
        l1 = session.Session(path, use_reg=True, triple=True)
        _, _, _, numoutcome, stim_sel, choice_sel, outcome_sel, action_sel = l1.stim_choice_outcome_selectivity(plot=False, 
                                                                                                                downsample='04' in path)
    
        snp, sp = stim_sel
        stimnonpref += snp
        stimpref += sp
        
        snp, sp = choice_sel
        choicenonpref += snp
        choicepref += sp
        
        snp, sp = outcome_sel
        outcomenonpref += snp
        outcomepref += sp
         
        snp, sp = action_sel
        actionnonpref += snp
        actionpref += sp
        
        print(len(numoutcome))
    
    
    x = np.arange(-6.97,4,1/6)[:61]
    titles = ['Stimulus selective', 'Choice selective', 'Outcome selective', 'Action selective']
    
    err = np.std(stimpref, axis=0) / np.sqrt(len(stimpref)) 
    err += np.std(stimnonpref, axis=0) / np.sqrt(len(stimnonpref))
    sel = np.mean(stimpref, axis=0) - np.mean(stimnonpref, axis=0) 
    axarr[0,j].plot(x, sel, color='green')
            
    axarr[0,j].fill_between(x, sel - err, 
              sel + err,
              color='lightgreen')
    
    axarr[0,j].set_title(titles[0])

    err = np.std(choicepref, axis=0) / np.sqrt(len(choicepref)) 
    err += np.std(choicenonpref, axis=0) / np.sqrt(len(choicenonpref))
    sel = np.mean(choicepref, axis=0) - np.mean(choicenonpref, axis=0) 
    axarr[1,j].plot(x, sel, color='purple')
            
    axarr[1,j].fill_between(x, sel - err, 
              sel + err,
              color='violet')
    axarr[1,j].set_title(titles[1])
            
    
    
    err = np.std(outcomepref, axis=0) / np.sqrt(len(outcomepref)) 
    err += np.std(outcomenonpref, axis=0) / np.sqrt(len(outcomenonpref))
    sel = np.mean(outcomepref, axis=0) - np.mean(outcomenonpref, axis=0) 
    axarr[2,j].plot(x, sel, color='dodgerblue')
            
    axarr[2,j].fill_between(x, sel - err, 
              sel + err,
              color='lightskyblue')
    
    axarr[2,j].set_title(titles[2])
    
    
    err = np.std(actionpref, axis=0) / np.sqrt(len(actionpref)) 
    err += np.std(actionnonpref, axis=0) / np.sqrt(len(actionnonpref))
    sel = np.mean(actionpref, axis=0) - np.mean(actionnonpref, axis=0) 
    axarr[3,j].plot(x, sel, color='goldenrod')
            
    axarr[3,j].fill_between(x, sel - err, 
              sel + err,
              color='wheat')
    
    axarr[3,j].set_title(titles[3])
    
    
for i in range(3):
    for j in range(4):
        axarr[j,i].axvline(0, color = 'grey', alpha=0.5, ls = '--')
        axarr[j,i].axvline(-4.3, color = 'grey', alpha=0.5, ls = '--')
        axarr[j,i].axvline(-3, color = 'grey', alpha=0.5, ls = '--')        
        axarr[j,i].axhline(0, color = 'grey', alpha=0.5, ls = '--')
        
        
axarr[0,0].set_ylabel('Selectivity')
axarr[2,1].set_xlabel('Time from Go cue (s)')



plt.savefig(r'F:\data\Fig 2\NLE_all_selectivityagg.pdf')
plt.show()

#%% Do it by selectivity type
allpaths =  [[
        # r'F:\data\BAYLORCW032\python\2023_10_05',
        # r'F:\data\BAYLORCW034\python\2023_10_12',
        r'F:\data\BAYLORCW036\python\2023_10_09',
        r'F:\data\BAYLORCW035\python\2023_10_26',
        r'F:\data\BAYLORCW037\python\2023_11_21',
        ],
        [
        # r'F:\data\BAYLORCW032\python\2023_10_19',
        # r'F:\data\BAYLORCW034\python\2023_10_22',
        r'F:\data\BAYLORCW036\python\2023_10_19',
        r'F:\data\BAYLORCW035\python\2023_12_07',
        r'F:\data\BAYLORCW037\python\2023_12_08',
        ],
        [
        # r'F:\data\BAYLORCW032\python\2023_10_24',
        # r'F:\data\BAYLORCW034\python\2023_10_27',
        r'F:\data\BAYLORCW036\python\2023_10_30',
        r'F:\data\BAYLORCW035\python\2023_12_15',
        r'F:\data\BAYLORCW037\python\2023_12_15',
        ]]



f, axarr = plt.subplots(1,3, figsize=(25,5))
plt.setp(axarr, ylim=(-0.2,1.5))

for j in range(3):
    
    stimnonpref, stimpref = [], []
    choicenonpref, choicepref = [], []
    outcomenonpref, outcomepref = [], []
    actionnonpref, actionpref = [], []
    
    for path in allpaths[j]:
        l1 = session.Session(path, use_reg=True, triple=True)
        _, _, _, numoutcome, stim_sel, choice_sel, outcome_sel, action_sel = l1.stim_choice_outcome_selectivity(plot=False)
    
        snp, sp = stim_sel
        stimnonpref += snp
        stimpref += sp
        
        snp, sp = choice_sel
        choicenonpref += snp
        choicepref += sp
        
        snp, sp = outcome_sel
        outcomenonpref += snp
        outcomepref += sp
         
        snp, sp = action_sel
        actionnonpref += snp
        actionpref += sp
        
        print(len(numoutcome))
    
    
    x = np.arange(-6.97,4,l1.fs)[:l1.time_cutoff]
    titles = ['Stimulus selective', 'Choice selective', 'Outcome selective', 'Action selective']
    
    # err = np.std(stimpref, axis=0) / np.sqrt(len(stimpref)) 
    # err += np.std(stimnonpref, axis=0) / np.sqrt(len(stimnonpref))stimnonpref
    err = np.std(np.vstack((stimpref, stimnonpref)),axis=0) / np.sqrt(len(stimpref) + len(stimnonpref))
    sel = np.mean(stimpref, axis=0) - np.mean(stimnonpref, axis=0) 
    axarr[j].plot(x, sel, color='green')
            
    axarr[j].fill_between(x, sel - err, 
              sel + err,
              color='lightgreen')
    
    axarr[j].set_title(titles[0])

    # # err = np.std(choicepref, axis=0) / np.sqrt(len(choicepref)) 
    # # err += np.std(choicenonpref, axis=0) / np.sqrt(len(choicenonpref))
    # err = np.std(np.vstack((choicepref, choicenonpref)),axis=0) / np.sqrt(len(choicepref) + len(choicenonpref))
    # sel = np.mean(choicepref, axis=0) - np.mean(choicenonpref, axis=0) 
    # axarr[j].plot(x, sel, color='purple')
            
    # axarr[j].fill_between(x, sel - err, 
    #           sel + err,
    #           color='violet')
    # axarr[j].set_title(titles[1])
            
    
    
    # err = np.std(outcomepref, axis=0) / np.sqrt(len(outcomepref)) 
    # err += np.std(outcomenonpref, axis=0) / np.sqrt(len(outcomenonpref))
    # sel = np.mean(outcomepref, axis=0) - np.mean(outcomenonpref, axis=0) 
    # axarr[j].plot(x, sel, color='dodgerblue')
            
    # axarr[j].fill_between(x, sel - err, 
    #           sel + err,
    #           color='lightskyblue')
    
    # axarr[j].set_title(titles[2])
    
    
    # # err = np.std(actionpref, axis=0) / np.sqrt(len(actionpref)) 
    # # err += np.std(actionnonpref, axis=0) / np.sqrt(len(actionnonpref))
    # err = np.std(np.vstack((actionpref, actionnonpref)),axis=0) / np.sqrt(len(actionpref) + len(actionnonpref))
    # sel = np.mean(actionpref, axis=0) - np.mean(actionnonpref, axis=0) 
    # axarr[j].plot(x, sel, color='goldenrod')
            
    # axarr[j].fill_between(x, sel - err, 
    #           sel + err,
    #           color='wheat')
    
    # axarr[j].set_title(titles[3])



axarr[0].set_ylabel('Selectivity')
axarr[0].set_xlabel('Time from Go cue (s)')

for i in range(3):
    
    axarr[i].axvline(0, color = 'grey', alpha=0.5, ls = '--')
    axarr[i].axvline(-4.3, color = 'grey', alpha=0.5, ls = '--')
    axarr[i].axvline(-3, color = 'grey', alpha=0.5, ls = '--')        
    axarr[i].axhline(0, color = 'grey', alpha=0.5, ls = '--')

plt.savefig(r'F:\data\Fig 2\NLE_stim_selectivitynocw32.pdf')
plt.show()



#%% By learning stage:
#NAIVE

paths = [
        # r'F:\data\BAYLORCW032\python\2023_10_08',
        # r'F:\data\BAYLORCW034\python\2023_10_12',
        r'F:\data\BAYLORCW036\python\2023_10_09',
        r'F:\data\BAYLORCW035\python\2023_10_26',
        r'F:\data\BAYLORCW037\python\2023_11_21',
        ]

stimnonpref, stimpref = [], []
choicenonpref, choicepref = [], []
outcomenonpref, outcomepref = [], []
actionnonpref, actionpref = [], []

for path in paths:
    l1 = session.Session(path, use_reg=True, triple=True)
    _, _, _, numoutcome, stim_sel, choice_sel, outcome_sel, action_sel = l1.stim_choice_outcome_selectivity()

    snp, sp = stim_sel
    stimnonpref += snp
    stimpref += sp
    
    snp, sp = choice_sel
    choicenonpref += snp
    choicepref += sp
    
    snp, sp = outcome_sel
    outcomenonpref += snp
    outcomepref += sp
     
    snp, sp = action_sel
    actionnonpref += snp
    actionpref += sp
    
    print(len(numoutcome))


f, axarr = plt.subplots(4,1, figsize=(5,15))
plt.setp(axarr, ylim=(-0.1,1.0))

x = np.arange(-6.97,4,l1.fs)[:l1.time_cutoff]
titles = ['Stimulus selective', 'Choice selective', 'Outcome selective', 'Action selective']

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


err = np.std(actionpref, axis=0) / np.sqrt(len(actionpref)) 
err += np.std(actionnonpref, axis=0) / np.sqrt(len(actionnonpref))
sel = np.mean(actionpref, axis=0) - np.mean(actionnonpref, axis=0) 
axarr[3].plot(x, sel, color='goldenrod')
        
axarr[3].fill_between(x, sel - err, 
          sel + err,
          color='wheat')

axarr[3].set_title(titles[3])


###########################################

axarr[0].set_ylabel('Selectivity')
axarr[1].set_xlabel('Time from Go cue (s)')

for i in range(4):
    
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