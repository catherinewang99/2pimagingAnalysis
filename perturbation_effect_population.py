# -*- coding: utf-8 -*-
"""
Created on Fri May 24 10:12:42 2024

@author: catherinewang

Quantify the population level effect of perturbation, creating similar figs as 
in Chen et al., 2021 (Fig S3) and Yang et al 2022 (Fig 6)

to show the contralateral and ipsilateral to imaging side effect of stimulation

"""

import sys
sys.path.append("C:\scripts\Imaging analysis")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
# from alm_2p import session
from matplotlib.pyplot import figure
from numpy import concatenate as cat
from sklearn.preprocessing import normalize
import quality
from scipy import stats
from scipy.stats import zscore
from activityMode import Mode
cat = np.concatenate

plt.rcParams['pdf.fonttype'] = '42' 
#%% CONTRA PATHS:
all_paths = [[    r'F:\data\BAYLORCW032\python\2023_10_05',
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

#%% Fraction change in dF/F0 by stimulation

ipsi_paths = [r'F:\data\BAYLORCW032\python\2023_10_23',
         r'F:\data\BAYLORCW036\python\2023_10_20',
         r'F:\data\BAYLORCW034\python\2023_10_24',
         r'F:\data\BAYLORCW035\python\2023_12_06',
         r'F:\data\BAYLORCW037\python\2023_11_22'
         ]

contra_paths = [
            r'F:\data\BAYLORCW032\python\2023_10_24',
            # r'F:\data\BAYLORCW034\python\2023_10_27',
            r'F:\data\BAYLORCW036\python\2023_10_30',
            r'F:\data\BAYLORCW035\python\2023_12_15',
            r'F:\data\BAYLORCW037\python\2023_12_15',
            
            r'H:\data\BAYLORCW044\python\2024_05_22',
            r'H:\data\BAYLORCW044\python\2024_05_23',
            
            r'H:\data\BAYLORCW046\python\2024_05_29',
            r'H:\data\BAYLORCW046\python\2024_05_30',
            r'H:\data\BAYLORCW046\python\2024_05_31',
            ]


all_ipsi_fracs = []


for path in ipsi_paths:

    l1 = quality.QC(path, use_background_sub=False)
    
    frac, sig_n = l1.stim_effect_per_neuron()
    if frac < 5 and frac > -5:
        all_ipsi_fracs += [frac]
    
all_contra_fracs = []
for path in contra_paths:

    l1 = quality.QC(path, use_background_sub=False)
    
    frac, sig_n = l1.stim_effect_per_neuron()
    
    if frac < 5 and frac > -5:
        all_contra_fracs += [frac]
       
    
f, axarr = plt.subplots(1,1, sharex='col', figsize=(5,5))
axarr.bar([0, 1], [np.mean(all_ipsi_fracs), np.mean(all_contra_fracs)])
axarr.scatter(np.zeros(len(all_ipsi_fracs)), all_ipsi_fracs)
axarr.scatter(np.ones(len(all_contra_fracs)), all_contra_fracs)
axarr.axhline(1, ls = '--', color='lightgrey')
axarr.set_ylabel('Fraction change in dF/F0')
axarr.set_xticks([0,1], ['Ipsilateral to imaging', 'Contralateral'])
plt.show()

#%% Fraction of neurons affected by stimulation

ipsi_frac_sup, ipsi_frac_exc = [], []
for path in ipsi_paths:

    l1 = quality.QC(path, use_background_sub=True)
    
    _, sig_n = l1.stim_effect_per_neuron()
        
    ipsi_frac_sup += [len(np.where(sig_n < 0)[0]) / len(sig_n)]
    ipsi_frac_exc += [len(np.where(sig_n > 0)[0]) / len(sig_n)]
 
contra_frac_sup, contra_frac_exc = [], []
for path in contra_paths:

    l1 = quality.QC(path, use_background_sub=True)
    
    _, sig_n = l1.stim_effect_per_neuron()
        
    contra_frac_sup += [len(np.where(sig_n < 0)[0]) / len(sig_n)]
    contra_frac_exc += [len(np.where(sig_n > 0)[0]) / len(sig_n)]
    
    
plt.barh([0, 1], [np.mean(ipsi_frac_exc), np.mean(contra_frac_exc)], color = 'r', edgecolor = 'black', label = 'Excited')
plt.barh([0, 1], [-np.mean(ipsi_frac_sup), -np.mean(contra_frac_sup)], color = 'b', edgecolor = 'black', label = 'Inhibited')
plt.scatter(cat((ipsi_frac_exc, -1 * np.array(ipsi_frac_sup))), np.zeros(len(cat((ipsi_frac_exc, ipsi_frac_sup)))), facecolors='none', edgecolors='grey')
plt.scatter(cat((contra_frac_exc, -1 * np.array(contra_frac_sup))), np.ones(len(cat((contra_frac_exc, contra_frac_sup)))), facecolors='none', edgecolors='grey')

plt.axvline(0)
plt.yticks([0,1], ['Ipsilateral to imaging', 'Contralateral'])
plt.ylabel('Condition')
plt.xlabel('Fraction of neurons with significant dF/F0 change')
plt.legend()
plt.show()
    
#%% Fraction of neurons affected by stim over learning

allsup, allexc = [], []
for i in range(3):
    contra_frac_sup, contra_frac_exc = [], []
    contra_paths = allpaths[i]
    for path in contra_paths:
    
        l1 = quality.QC(path=path, use_reg=True, triple=True, use_background_sub=True, baseline_normalization="median_zscore")
        
        _, sig_n = l1.stim_effect_per_neuron()
            
        contra_frac_sup += [len(np.where(sig_n < 0)[0]) / len(sig_n)]
        contra_frac_exc += [len(np.where(sig_n > 0)[0]) / len(sig_n)]
        
    allsup += [contra_frac_sup]
    allexc += [contra_frac_exc]

# total_n = 0
# for path in contra_paths:

#     l1 = quality.QC(path=path, use_reg=True, triple=True, use_background_sub=True, baseline_normalization="median_zscore")
#     total_n += len(l1.good_neurons)

f = plt.figure(figsize = (5,5))

for i in range(3):
    contra_frac_exc = allexc[i]
    contra_frac_sup = allsup[i]
    if i ==0:
        plt.barh([i], [np.mean(contra_frac_exc)], color = 'r', edgecolor = 'black', label = 'Excited')
        plt.barh([i], [-np.mean(contra_frac_sup)], color = 'b', edgecolor = 'black', label = 'Inhibited')
    else:
        plt.barh([i], [np.mean(contra_frac_exc)], color = 'r', edgecolor = 'black')
        plt.barh([i], [-np.mean(contra_frac_sup)], color = 'b', edgecolor = 'black')
    
    plt.scatter(cat((contra_frac_exc, -1 * np.array(contra_frac_sup))), np.ones(len(cat((contra_frac_exc, contra_frac_sup)))) * (i), facecolors='none', edgecolors='grey')

plt.axvline(0)
plt.yticks([0,1,2], ['Expert', 'Learning', 'Naive'])
plt.ylabel('Stage')
plt.xlabel('Fraction of neurons with significant dF/F0 change')
plt.title("Perturbation effect over learning (n = {} neurons)".format(total_n))
plt.legend()
plt.savefig(r'F:\data\Fig 3\opto_population_effect_overlearning.pdf')
plt.show()


## Try a violin plot

f = plt.figure(figsize = (5,5))
plt.violinplot(contra_frac_exc, vert=False)
plt.violinplot(-contra_frac_sup, vert=False)
plt.yticks([0,1,2], ['Expert', 'Learning', 'Naive'])


#%% Changes at over opto corruption

init_paths, mid_paths, final_paths = [[r'H:\\data\\BAYLORCW038\\python\\2024_02_05',
                                       r'H:\\data\\BAYLORCW039\\python\\2024_04_17',
                                       r'H:\\data\\BAYLORCW039\\python\\2024_04_18',
                                       r'H:\\data\\BAYLORCW041\\python\\2024_05_14',
                                       r'H:\\data\\BAYLORCW041\\python\\2024_05_13',
                                       r'H:\\data\\BAYLORCW041\\python\\2024_05_15',
                                       r'H:\\data\\BAYLORCW043\\python\\2024_05_20',
                                       r'H:\\data\\BAYLORCW043\\python\\2024_05_21',
                                       r'H:\\data\\BAYLORCW042\\python\\2024_06_05',],
                                      
                                       [r'H:\data\BAYLORCW038\python\2024_02_15',
                                        r'H:\\data\\BAYLORCW039\\python\\2024_04_24',
                                        r'H:\\data\\BAYLORCW039\\python\\2024_04_25',
                                        r'H:\\data\\BAYLORCW041\\python\\2024_05_23',
                                        r'H:\\data\\BAYLORCW041\\python\\2024_05_24',
                                        r'H:\\data\\BAYLORCW041\\python\\2024_05_28',
                                        r'H:\\data\\BAYLORCW043\\python\\2024_06_03',
                                        r'H:\\data\\BAYLORCW043\\python\\2024_06_04',
                                        r'H:\\data\\BAYLORCW042\\python\\2024_06_14',],
                                      
                                       [r'H:\\data\\BAYLORCW038\\python\\2024_03_15',
                                        r'H:\\data\\BAYLORCW039\\python\\2024_05_06',
                                        r'H:\\data\\BAYLORCW039\\python\\2024_05_08',
                                        r'H:\\data\\BAYLORCW041\\python\\2024_06_07',
                                        r'H:\\data\\BAYLORCW041\\python\\2024_06_12',
                                        r'H:\\data\\BAYLORCW041\\python\\2024_06_11',
                                        r'H:\\data\\BAYLORCW043\\python\\2024_06_13',
                                        r'H:\\data\\BAYLORCW043\\python\\2024_06_14',
                                        r'H:\\data\\BAYLORCW042\\python\\2024_06_24',]
                                    ]

# for path in paths:
#     l1 = quality.QC(path, use_reg=True, triple=True, use_background_sub=False)
    
#     frac, sig_n = l1.stim_effect_per_neuron()
    
p=0.001
period = np.arange(l1.response-6, l1.response)
all_init_fracs = []
for path in init_paths:

    l1 = quality.QC(path, use_background_sub=False)
    
    _, sig_n = l1.stim_effect_per_neuron(p=p, period=period)
        
    inh = len(np.where(sig_n < 0)[0]) / len(sig_n)
    exc = len(np.where(sig_n > 0)[0]) / len(sig_n)
    
    all_init_fracs += [[inh, exc]]
    
all_middle_fracs = []
for path in mid_paths:

    l1 = quality.QC(path, use_background_sub=False)
    
    _, sig_n = l1.stim_effect_per_neuron(p=p, period=period)
        
    inh = len(np.where(sig_n < 0)[0]) / len(sig_n)
    exc = len(np.where(sig_n > 0)[0]) / len(sig_n)
    
    all_middle_fracs += [[inh, exc]]
      
all_final_fracs = []
for path in final_paths:

    l1 = quality.QC(path, use_background_sub=False)
    
    _, sig_n = l1.stim_effect_per_neuron(p=p,period=period)
        
    inh = len(np.where(sig_n < 0)[0]) / len(sig_n)
    exc = len(np.where(sig_n > 0)[0]) / len(sig_n)
    
    all_final_fracs += [[inh, exc]]
    
plt.barh([2, 1, 0], [np.mean(all_init_fracs, axis=0)[1], np.mean(all_middle_fracs, axis=0)[1], np.mean(all_final_fracs, axis=0)[1]], color = 'r', edgecolor = 'black', label = 'Excited')
plt.barh([2, 1, 0], [-np.mean(all_init_fracs, axis=0)[0], -np.mean(all_middle_fracs, axis=0)[0], -np.mean(all_final_fracs, axis=0)[0]], color = 'b', edgecolor = 'black', label = 'Inhibited')

plt.scatter(cat((np.array(all_init_fracs)[:, 1], -1 * np.array(all_init_fracs)[:, 0])), np.ones(np.size(all_init_fracs)) * 2, facecolors='none', edgecolors='grey')
plt.scatter(cat((np.array(all_middle_fracs)[:, 1], -1 * np.array(all_middle_fracs)[:, 0])), np.ones(np.size(all_middle_fracs)), facecolors='none', edgecolors='grey')
plt.scatter(cat((np.array(all_final_fracs)[:, 1], -1 * np.array(all_final_fracs)[:, 0])), np.zeros(np.size(all_final_fracs)), facecolors='none', edgecolors='grey')

plt.axvline(0)
# plt.yticks([0,1], ['Ipsilateral to imaging', 'Contralateral'])
plt.yticks([0,1,2], ['Final', 'Middle', 'Initial'])
plt.ylabel('Session')
plt.xlabel('Fraction of neurons with significant dF/F0 change')
plt.legend()
plt.show()
    
#%% Compare proportion of neurons excited/inhibited by stim over stages
initial_paths = ['H:\\data\\BAYLORCW038\\python\\2024_02_05', 
              'H:\\data\\BAYLORCW039\\python\\2024_04_18',
              'H:\\data\\BAYLORCW039\\python\\2024_04_17', 
              ]
             
             # [ 'H:\\data\\BAYLORCW038\\python\\2024_02_15',
             #  'H:\\data\\BAYLORCW039\\python\\2024_04_25',
             #  'H:\\data\\BAYLORCW039\\python\\2024_04_24',
             #  ],
             
final_paths = ['H:\\data\\BAYLORCW038\\python\\2024_03_15',
              'H:\\data\\BAYLORCW039\\python\\2024_05_08',
              'H:\\data\\BAYLORCW039\\python\\2024_05_14']

ipsi_frac_sup, ipsi_frac_exc = [], []
for path in final_paths:

    l1 = quality.QC(path, use_background_sub=False)
    
    _, sig_n = l1.stim_effect_per_neuron()
        
    ipsi_frac_sup += [len(np.where(sig_n < 0)[0]) / len(sig_n)]
    ipsi_frac_exc += [len(np.where(sig_n > 0)[0]) / len(sig_n)]
 
contra_frac_sup, contra_frac_exc = [], []
for path in initial_paths:

    l1 = quality.QC(path, use_background_sub=False)
    
    _, sig_n = l1.stim_effect_per_neuron()
        
    contra_frac_sup += [len(np.where(sig_n < 0)[0]) / len(sig_n)]
    contra_frac_exc += [len(np.where(sig_n > 0)[0]) / len(sig_n)]
    
    
plt.barh([0, 1], [np.mean(ipsi_frac_exc), np.mean(contra_frac_exc)], color = 'r', edgecolor = 'black', label = 'Excited')
plt.barh([0, 1], [-np.mean(ipsi_frac_sup), -np.mean(contra_frac_sup)], color = 'b', edgecolor = 'black', label = 'Inhibited')
plt.scatter(cat((ipsi_frac_exc, -1 * np.array(ipsi_frac_sup))), np.zeros(len(cat((ipsi_frac_exc, ipsi_frac_sup)))), facecolors='none', edgecolors='grey')
plt.scatter(cat((contra_frac_exc, -1 * np.array(contra_frac_sup))), np.ones(len(cat((contra_frac_exc, contra_frac_sup)))), facecolors='none', edgecolors='grey')

plt.axvline(0)
plt.yticks([0,1], ['Final stage', 'Initial stage'])
plt.ylabel('Condition')
plt.xlabel('Fraction of neurons with significant dF/F0 change')
plt.legend()
plt.show()
    
    
    
    
    
    
    
    
    
    
    
    