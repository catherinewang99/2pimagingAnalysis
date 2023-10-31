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

paths = [r'F:\data\BAYLORCW032\python\2023_10_05',
        # r'F:\data\BAYLORCW034\python\2023_10_12',
        r'F:\data\BAYLORCW036\python\2023_10_09']

stimnonpref, stimpref = [], []
choicenonpref, choicepref = [], []
outcomenonpref, outcomepref = [], []
for path in paths:
    l1 = session.Session(path)
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
     


f, axarr = plt.subplots(1,3, sharey='row', figsize=(15,5))
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

plt.savefig(r'F:\data\SFN 2023\naive_stim_choice_outcome_selectivity.pdf')
plt.show()

#TRAINED

paths = [r'F:\data\BAYLORCW032\python\2023_10_24',
        # r'F:\data\BAYLORCW034\python\2023_10_27',
        r'F:\data\BAYLORCW036\python\2023_10_28']

stimnonpref, stimpref = [], []
choicenonpref, choicepref = [], []
outcomenonpref, outcomepref = [], []
for path in paths:
    l1 = session.Session(path)
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
     


f, axarr = plt.subplots(1,3, sharey='row', figsize=(15,5))
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

plt.savefig(r'F:\data\SFN 2023\trained_stim_choice_outcome_selectivity.pdf')
plt.show()
