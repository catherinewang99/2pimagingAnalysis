# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:15:58 2024

@author: catherinewang
"""

import sys
sys.path.append("C:\scripts\Imaging analysis")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from session import Session
import behavior
cat = np.concatenate
plt.rcParams['pdf.fonttype'] = '42' 

from matplotlib.pyplot import figure
from sklearn.preprocessing import normalize
import quality
from scipy import stats
from scipy.stats import zscore
from activityMode import Mode
import behavior
from numpy.linalg import norm

#%% Summary of behavior Phase 2 across 5 opto stages

perf_init = []
perf_firstopto = []
perf_mid = []
perf_secondopto = []
perf_final = []

b = behavior.Behavior(r'H:\data\Behavior data\BAYLORCW038\python_behavior', behavior_only=True)
perf = b.plot_performance_over_sessions(all=True, color_background=[2,3,4,5,6,7,8,9, 11, 12, 14,15,16,17,18, 19, 20, 21,22,23], return_vals=True)
perf_init += [perf[0:2]]
perf_firstopto += [perf[2:10]]
perf_mid += [[perf[x] for x in [10, 13]]]
perf_secondopto += [perf[14:23]]
perf_final += [[perf[24]]]


b = behavior.Behavior(r'H:\data\Behavior data\BAYLORCW039\python_behavior', behavior_only=True)
perf = b.plot_performance_over_sessions(all=True, color_background=[2,3,4,5,8,9,10,11,12,13,14,15,16,20,21,22,23,24], return_vals=True)
perf_init += [perf[0:2]]
perf_firstopto += [[perf[x] for x in [2,3,4,5]]]
perf_mid += [perf[6:8]]
perf_secondopto += [perf[8:17]]
perf_final += [perf[17:20]]


b = behavior.Behavior(r'H:\data\Behavior data\BAYLORCW041\python_behavior', behavior_only=True)
perf = b.plot_performance_over_sessions(all=True, color_background=[3,4,5,6,7,8,9,13,14,15,16,17,18,19,20], return_vals=True)
perf_init += [perf[0:3]]
perf_firstopto += [perf[3:10]]
perf_mid += [perf[10:13]]
perf_secondopto += [perf[13:21]]
perf_final += [perf[21:]]

b = behavior.Behavior(r'H:\data\Behavior data\BAYLORCW043\python_behavior', behavior_only=True)
perf = b.plot_performance_over_sessions(all=True, color_background=[3,4,5,6,7,8,9,13,14,15,16,17,18], return_vals=True)
perf_init += [perf[0:3]]
perf_firstopto += [perf[3:10]]
perf_mid += [perf[10:13]]
perf_secondopto += [perf[13:19]]
perf_final += [perf[19:]]


b = behavior.Behavior(r'H:\data\Behavior data\BAYLORCW042\python_behavior', behavior_only=True)
perf = b.plot_performance_over_sessions(all=True, color_background=[2,3,4,5,6,7,8,10,11,12,14,15,16,17,18], return_vals=True)
perf_init += [perf[0:2]]
perf_firstopto += [perf[2:9]]
perf_mid += [[perf[x] for x in [9, 13]]]
perf_secondopto +=  [[perf[x] for x in [10,11,12,14,15,16,17,18]]]
perf_final += [[perf[19]]]

allperfs = [perf_init, perf_firstopto, perf_mid, perf_secondopto, perf_final]
for i in range(5):
    maxlen = max([len(arr) for arr in allperfs[i]])
    fig = plt.figure(figsize =(maxlen, 12)) 
    
    for idx in range(5):
        
        correctarr = allperfs[i][idx]
        plt.plot(correctarr, color='black', alpha =0.75)        
        plt.ylabel('% correct')
        plt.axhline(y=0.7, alpha = 0.5, color='orange')
        plt.axhline(y=0.5, alpha = 0.5, color='red', ls = '--')
        plt.ylim(0.18, 0.92)
        
        plt.scatter(len(correctarr)-1, correctarr[-1], marker='o', s=650, alpha = 0.5, color = 'g')
    # plt.savefig(r'H:\Fig 4\alllearningcurves_{}.pdf'.format(i),transparent=True)
    
    plt.show()
#%% Bar graph showing behavior diff after opto corruption


fig = plt.figure(figsize =(6, 10)) 

# Use all sessions:
# First round
plt.scatter(np.ones(len(cat(perf_firstopto)))*(-0.2), cat(perf_firstopto))
plt.scatter(np.ones(len(cat(perf_mid)))*0.2, cat(perf_mid))
plt.bar(-0.2, np.mean(cat(perf_firstopto)), 0.4, fill=False)
plt.bar(0.2, np.mean(cat(perf_mid)), 0.4, fill=False)

# Second round
plt.scatter(np.ones(len(cat(perf_secondopto)))*(1-0.2), cat(perf_secondopto))
plt.scatter(np.ones(len(cat(perf_final)))*1.2, cat(perf_final))
plt.bar(1-0.2, np.mean(cat(perf_secondopto)), 0.4, fill=False)
plt.bar(1.2, np.mean(cat(perf_final)), 0.4, fill=False)
plt.show()


# Only use last three sessions:
fig = plt.figure(figsize =(6, 8)) 
last_n = 3
# First round
plt.scatter(np.ones(5*last_n)*(-0.2), cat([p[-last_n:] for p in perf_firstopto]))
plt.scatter(np.ones(len(cat(perf_mid)))*0.2, cat(perf_mid))
plt.bar(-0.2, np.mean(cat([p[-last_n:] for p in perf_firstopto])), 0.4, fill=False)
plt.bar(0.2, np.mean(cat(perf_mid)), 0.4, fill=False)

# Second round
plt.scatter(np.ones(5*last_n)*(1-0.2), cat([p[-last_n:] for p in perf_secondopto]))
plt.scatter(np.ones(len(cat(perf_final)))*1.2, cat(perf_final))
plt.bar(1-0.2, np.mean(cat([p[-last_n:] for p in perf_secondopto])), 0.4, fill=False)
plt.bar(1.2, np.mean(cat(perf_final)), 0.4, fill=False) 
plt.ylabel("Performance")
plt.xticks([-0.2, 0.2, 0.8, 1.2], ["Opto round 1", "Middle perf.", "Opto round 2", "Final perf."])
plt.ylim(bottom=0.4)
plt.savefig(r'H:\Fig 4\perf_barscatter_rounds.pdf')
plt.show()
#%%
# Only look at the difference
# Only use last three sessions:
fig = plt.figure(figsize =(6, 8)) 

last_n = 1
# First round
av_perf = np.mean([p[-last_n:] for p in perf_firstopto],axis=1)
first_perf = [p[0] for p in perf_mid]
diff = first_perf - av_perf

plt.scatter(np.zeros(5), diff)
plt.bar([0], np.mean(diff), 0.7, fill=False)

# Second round
av_perf = np.mean([p[-last_n:] for p in perf_secondopto],axis=1)
first_perf = [p[0] for p in perf_mid]
diff = first_perf - av_perf

plt.scatter(np.ones(5), diff)
plt.bar([1], np.mean(diff), 0.7, fill=False)

plt.ylabel("Difference in performance")
plt.xticks([0, 1], ["Middle perf. - Opto round 1" , "Final perf. - Opto round 2"])
plt.savefig(r'H:\Fig 4\perfdiff_barscatter_rounds.pdf')
plt.show()
#%% Behavioral recovery no L/R info

all_paths = [[r'H:\data\BAYLORCW038\python\2024_02_05',
          r'H:\data\BAYLORCW038\python\2024_02_15',
          r'H:\data\BAYLORCW038\python\2024_03_15',]]

all_paths = [[r'H:\data\BAYLORCW038\python\2024_02_05',
          r'H:\data\BAYLORCW038\python\2024_02_15',
          r'H:\data\BAYLORCW038\python\2024_03_15',],
             
             ['H:\\data\\BAYLORCW039\\python\\2024_04_17', 
            'H:\\data\\BAYLORCW039\\python\\2024_04_24',
            'H:\\data\\BAYLORCW039\\python\\2024_05_06'],
              ['H:\\data\\BAYLORCW039\\python\\2024_04_18', 
            'H:\\data\\BAYLORCW039\\python\\2024_04_25',
            'H:\\data\\BAYLORCW039\\python\\2024_05_20'],
              
            ['H:\\data\\BAYLORCW041\\python\\2024_05_14', 
          'H:\\data\\BAYLORCW041\\python\\2024_05_23',
          'H:\\data\\BAYLORCW041\\python\\2024_06_07'],
            ['H:\\data\\BAYLORCW041\\python\\2024_05_13', 
          'H:\\data\\BAYLORCW041\\python\\2024_05_24',
          'H:\\data\\BAYLORCW041\\python\\2024_06_12'],
            ['H:\\data\\BAYLORCW041\\python\\2024_05_15', 
          'H:\\data\\BAYLORCW041\\python\\2024_05_28',
          'H:\\data\\BAYLORCW041\\python\\2024_06_11'],
            
            ['H:\\data\\BAYLORCW043\\python\\2024_05_20', 
          'H:\\data\\BAYLORCW043\\python\\2024_06_03',
          'H:\\data\\BAYLORCW043\\python\\2024_06_13'],
            ['H:\\data\\BAYLORCW043\\python\\2024_05_22', 
          'H:\\data\\BAYLORCW043\\python\\2024_06_04',
          'H:\\data\\BAYLORCW043\\python\\2024_06_14'],
            
            ['H:\\data\\BAYLORCW042\\python\\2024_06_05', 
          'H:\\data\\BAYLORCW042\\python\\2024_06_14',
          'H:\\data\\BAYLORCW042\\python\\2024_06_24']
            ]



performance_opto = []
performance_ctl = []
fig = plt.figure()
ticks = ["o", "X", "D"]
for paths in all_paths:
    counter = -1

    opt, ctl = [],[]
    for path in paths:
        counter += 1
        l1 = Session(path)
        stim_trials = np.where(l1.stim_ON)[0]
        control_trials = np.where(~l1.stim_ON)[0]
        
        perf_right, perf_left, perf_all = l1.performance_in_trials(stim_trials)
        opt += [perf_all]
        # plt.scatter(counter + 0.2, perf_right, c='b', marker='x')
        # plt.scatter(counter + 0.2, perf_left, c='r', marker='x')
       
        perf_rightctl, perf_left, perf_all_c = l1.performance_in_trials(control_trials)
        ctl += [perf_all_c]
        # plt.scatter(counter - 0.2, perf_rightctl, c='b', marker='o')
        # plt.scatter(counter - 0.2, perf_left, c='r', marker='o')
        plt.plot([counter - 0.2, counter + 0.2], [perf_all_c, perf_all], color='grey')
        
        
    performance_opto += [opt]
    performance_ctl += [ctl]


    plt.scatter(np.arange(3)+0.2, opt, color = 'red')
    plt.scatter(np.arange(3)-0.2, ctl, color = 'grey')
    
plt.bar(np.arange(3)+0.2, np.mean(performance_opto, axis=0), 0.4, color='red', alpha=0.5, label='Perturbation trials')

plt.bar(np.arange(3)-0.2, np.mean(performance_ctl, axis=0), 0.4, fill=False, label='Control trials')
plt.ylabel('Behavior performance')
plt.xticks(range(3), ["Before corruption", "Midpoint", "Final"])
plt.ylim([0.4,1])
plt.legend()
plt.savefig(r'H:\Fig 4\ctlopto_barscatter.pdf')
plt.show()