# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 10:35:55 2023

@author: Catherine Wang

Script designed to go through matched neurons and filter out noisy neurons
"""


import sys
sys.path.append("C:\scripts\Imaging analysis")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from alm_2p.session import Session
from matplotlib.pyplot import figure
from scipy.stats import chisquare
import pandas as pd
from activityMode import Mode


paths = [r'F:\data\BAYLORCW032\python\2023_10_05',
           r'F:\data\BAYLORCW032\python\2023_10_19',
          r'F:\data\BAYLORCW032\python\2023_10_24',]


agg_mice_paths = [[r'F:\data\BAYLORCW032\python\2023_10_08',
          r'F:\data\BAYLORCW032\python\2023_10_16',
          r'F:\data\BAYLORCW032\python\2023_10_25',],

         
        [ r'F:\data\BAYLORCW034\python\2023_10_12',
              r'F:\data\BAYLORCW034\python\2023_10_22',
              r'F:\data\BAYLORCW034\python\2023_10_27'],
         
        [r'F:\data\BAYLORCW036\python\2023_10_09',
            r'F:\data\BAYLORCW036\python\2023_10_19',
            r'F:\data\BAYLORCW036\python\2023_10_30'],
    
        [r'F:\data\BAYLORCW035\python\2023_10_26',
            r'F:\data\BAYLORCW035\python\2023_12_07',
            r'F:\data\BAYLORCW035\python\2023_12_15',],
     
        [r'F:\data\BAYLORCW037\python\2023_11_21',
             r'F:\data\BAYLORCW037\python\2023_12_08',
             r'F:\data\BAYLORCW037\python\2023_12_15',]

        ]


agg_mice_paths = [
                # [r'H:\data\BAYLORCW041\python\2024_05_13',
                #     r'H:\data\BAYLORCW041\python\2024_05_24',
                #   r'H:\data\BAYLORCW041\python\2024_06_12',],
                  
                #    [r'H:\data\BAYLORCW041\python\2024_05_15',
                #    r'H:\data\BAYLORCW041\python\2024_05_28',
                #    r'H:\data\BAYLORCW041\python\2024_06_11',],
                 
                # [r'H:\data\BAYLORCW041\python\2024_05_14',
                # r'H:\data\BAYLORCW041\python\2024_05_23',
                # r'H:\data\BAYLORCW041\python\2024_06_07',],
                
                # [r'H:\data\BAYLORCW042\python\2024_06_05',
                # r'H:\data\BAYLORCW042\python\2024_06_14',
                # r'H:\data\BAYLORCW042\python\2024_06_24',],
                # [r'H:\data\BAYLORCW042\python\2024_06_06',
                # r'H:\data\BAYLORCW042\python\2024_06_18',
                # r'H:\data\BAYLORCW042\python\2024_06_26',],
                
              #   ['H:\\data\\BAYLORCW039\\python\\2024_04_17', 
              # 'H:\\data\\BAYLORCW039\\python\\2024_04_24',
              # 'H:\\data\\BAYLORCW039\\python\\2024_05_06'],
              #   ['H:\\data\\BAYLORCW039\\python\\2024_04_18', 
              # 'H:\\data\\BAYLORCW039\\python\\2024_04_25',
              # 'H:\\data\\BAYLORCW039\\python\\2024_05_08'],
                
                [r'H:\data\BAYLORCW038\python\2024_02_05',
                  r'H:\data\BAYLORCW038\python\2024_02_15',
                  r'H:\data\BAYLORCW038\python\2024_03_15',],
        ]

# agg_mice_paths = [
#                 # [r'H:\data\BAYLORCW044\python\2024_05_22',
#                 #     r'H:\data\BAYLORCW044\python\2024_06_06',
#                 #   r'H:\data\BAYLORCW044\python\2024_06_19',],
#                 # [r'H:\data\BAYLORCW044\python\2024_05_23',
#                 #     r'H:\data\BAYLORCW044\python\2024_06_04',
#                 #   r'H:\data\BAYLORCW044\python\2024_06_18',],
#                 # [r'H:\data\BAYLORCW044\python\2024_05_24',
#                 #     r'H:\data\BAYLORCW044\python\2024_06_05',
#                 #   r'H:\data\BAYLORCW044\python\2024_06_20',],
#                 [r'H:\data\BAYLORCW046\python\2024_05_29',
#                     r'H:\data\BAYLORCW046\python\2024_06_24',
#                   r'H:\data\BAYLORCW046\python\2024_06_28',],                
#                 # [r'H:\data\BAYLORCW046\python\2024_05_29',
#                 #     r'H:\data\BAYLORCW046\python\2024_06_07',
#                 #   r'H:\data\BAYLORCW046\python\2024_06_24',],
#                 # [r'H:\data\BAYLORCW046\python\2024_05_30',
#                 #     r'H:\data\BAYLORCW046\python\2024_06_10',
#                 #   r'H:\data\BAYLORCW046\python\2024_06_27',],
#                 # [r'H:\data\BAYLORCW046\python\2024_05_31',
#                 #   r'H:\data\BAYLORCW046\python\2024_06_11',
#                 # r'H:\data\BAYLORCW046\python\2024_06_26',],
                
#                 ]

num_layers = 5
for paths in agg_mice_paths:
    allkeep_ids = []
    for layer_num in range(1,num_layers+1):
        keep_ids = []
    
        for path in paths:
            total_neurons = 0   
            l1 = Session(path, layer_num = layer_num, use_reg=True, triple=True, filter_reg = False)
            reg = np.load(path + r'\layer{}_triple_registered_neurons.npy'.format(layer_num-1))
            neurons, _ = l1.get_pearsonscorr_neuron(cutoff=0.5, postreg = True)
            idx = [np.where(l1.good_neurons == n)[0][0] for n in neurons]
            keep_ids += [i for i in idx if i not in keep_ids] # Keep best neurons from each session using OR logic
            
            
            print("Proportion: ", len(neurons)/l1.num_neurons)
            print("Num neurons: ", len(neurons))
            
        allkeep_ids += [keep_ids]
        print("TOTAL NEURONS ", len(keep_ids))
        
    
    for layer_num in range(1,num_layers+1):
        for path in paths:
            l1 = Session(path, layer_num = layer_num, use_reg=True, triple=True, filter_reg = False)
            reg = np.load(path + r'\layer{}_triple_registered_neurons.npy'.format(layer_num-1))
    
            new_reg = reg[allkeep_ids[layer_num-1]]
            np.save(path + r'\layer{}_triple_registered_filtered_neurons.npy'.format(layer_num-1), new_reg)
            
            
#%% For pairwise matched sessions (instead of triple sessions)

agg_mice_paths = [
                # [r'H:\data\BAYLORCW038\python\2024_02_05', 
                #     r'H:\data\BAYLORCW038\python\2024_03_15'],
                  # [r'H:\data\BAYLORCW039\python\2024_04_24', 
                  #   r'H:\data\BAYLORCW039\python\2024_05_06'],
                  # [r'H:\data\BAYLORCW039\python\2024_04_18', 
                   #   r'H:\data\BAYLORCW039\python\2024_05_08'],
                   # ['H:\\data\\BAYLORCW043\\python\\2024_05_20', 
                   #  'H:\\data\\BAYLORCW043\\python\\2024_06_03'],
                   #   r'H:\data\BAYLORCW039\python\2024_05_08'],
                   # ['H:\\data\\BAYLORCW041\\python\\2024_05_14', 
                   #  'H:\\data\\BAYLORCW041\\python\\2024_05_'],

                   # ['H:\\data\\BAYLORCW041\\python\\2024_05_20', 
                   #  'H:\\data\\BAYLORCW041\\python\\2024_06_03'],

                   # ['H:\\data\\BAYLORCW041\\python\\2024_05_20', 
                   #  'H:\\data\\BAYLORCW041\\python\\2024_06_03'],
                   # [r'H:\\data\\BAYLORCW041\\python\\2024_05_13',
                   #  r'H:\\data\\BAYLORCW041\\python\\2024_06_12'],
                   # [r'H:\data\BAYLORCW041\python\2024_05_14',
                   # r'H:\data\BAYLORCW041\python\2024_06_07',],
                   [r'H:\data\BAYLORCW043\python\2024_05_20',
                   r'H:\data\BAYLORCW043\python\2024_06_03',],
                   [r'H:\data\BAYLORCW043\python\2024_06_04',
                   r'H:\data\BAYLORCW043\python\2024_06_14',],
                   [r'H:\data\BAYLORCW043\python\2024_06_06',
                   r'H:\data\BAYLORCW043\python\2024_06_13',]
                   ]


for paths in agg_mice_paths:
    allkeep_ids = []
    for layer_num in range(1,6):
        keep_ids = []
    
        for path in paths:
            total_neurons = 0   
            l1 = Session(path, layer_num = layer_num, use_reg=True, triple=False, filter_reg = False)
            reg = np.load(path + r'\layer{}_registered_neurons.npy'.format(layer_num-1))
            neurons, _ = l1.get_pearsonscorr_neuron(cutoff=0.5, postreg = True)
            idx = [np.where(l1.good_neurons == n)[0][0] for n in neurons]
            keep_ids += [i for i in idx if i not in keep_ids] # Keep best neurons from each session using OR logic
            
            
            print("Proportion: ", len(neurons)/l1.num_neurons)
            print("Num neurons: ", len(neurons))
            
        allkeep_ids += [keep_ids]
        print("TOTAL NEURONS ", len(keep_ids))
        
    
    for layer_num in range(1,6):
        for path in paths:
            l1 = Session(path, layer_num = layer_num, use_reg=True, triple=False, filter_reg = False)
            reg = np.load(path + r'\layer{}_registered_neurons.npy'.format(layer_num-1))
    
            new_reg = reg[allkeep_ids[layer_num-1]]
            np.save(path + r'\layer{}_registered_filtered_neurons.npy'.format(layer_num-1), new_reg)


#%% Make lists of selective neurons to be saved for triple matched sessions:
    
agg_mice_paths = [
    
            [r'F:\data\BAYLORCW032\python\2023_10_05',
              r'F:\data\BAYLORCW032\python\2023_10_19',
              r'F:\data\BAYLORCW032\python\2023_10_24',
          ],
         
           # [ r'F:\data\BAYLORCW034\python\2023_10_12',
           #    r'F:\data\BAYLORCW034\python\2023_10_22',
           #    r'F:\data\BAYLORCW034\python\2023_10_27',
           #    r'F:\data\BAYLORCW034\python\cellreg\layer{}\1012_1022_1027pairs_proc.npy'],
         
            [r'F:\data\BAYLORCW036\python\2023_10_09',
            r'F:\data\BAYLORCW036\python\2023_10_19',
            r'F:\data\BAYLORCW036\python\2023_10_30',
           ],
         
         [r'F:\data\BAYLORCW037\python\2023_11_21',
            r'F:\data\BAYLORCW037\python\2023_12_08',
            r'F:\data\BAYLORCW037\python\2023_12_15',],
         
         [r'F:\data\BAYLORCW035\python\2023_10_26',
            r'F:\data\BAYLORCW035\python\2023_12_07',
            r'F:\data\BAYLORCW035\python\2023_12_15',],
         
         [r'H:\data\BAYLORCW044\python\2024_05_22',
          r'H:\data\BAYLORCW044\python\2024_06_06',
        r'H:\data\BAYLORCW044\python\2024_06_19'],

         
            [r'H:\data\BAYLORCW044\python\2024_05_23',
             r'H:\data\BAYLORCW044\python\2024_06_04',
        r'H:\data\BAYLORCW044\python\2024_06_18'],

            [r'H:\data\BAYLORCW046\python\2024_05_29',
             r'H:\data\BAYLORCW046\python\2024_06_07',
             r'H:\data\BAYLORCW046\python\2024_06_24'],


            [r'H:\data\BAYLORCW046\python\2024_05_30',
             r'H:\data\BAYLORCW046\python\2024_06_10',
             r'H:\data\BAYLORCW046\python\2024_06_27'],

            [r'H:\data\BAYLORCW046\python\2024_05_31',
             r'H:\data\BAYLORCW046\python\2024_06_11',
             r'H:\data\BAYLORCW046\python\2024_06_26'
             ]
         
        ]

for paths in agg_mice_paths:
    
    naivepath, learningpath, expertpath = paths[0], paths[1], paths[2]
    
    # Compare to only the selective neurons included:
    all_sel_neurons = []
    l1 = Session(naivepath, use_reg = True, triple=True)
    sel = l1.get_epoch_selective(range(l1.time_cutoff), p=0.05)
    all_sel_neurons += [np.where(l1.good_neurons == s)[0] for s in sel]
    l1 = Session(learningpath, use_reg = True, triple=True)
    sel = l1.get_epoch_selective(range(l1.time_cutoff), p=0.05)
    all_sel_neurons += [np.where(l1.good_neurons == s)[0] for s in sel]
    l1 = Session(expertpath, use_reg = True, triple=True)
    sel = l1.get_epoch_selective(range(l1.time_cutoff), p=0.05)
    all_sel_neurons += [np.where(l1.good_neurons == s)[0] for s in sel]
    
    all_sel_neurons = list(set(cat(all_sel_neurons)))
    
    np.save(naivepath + '\selective_neurons.npy', all_sel_neurons)
    np.save(learningpath + '\selective_neurons.npy', all_sel_neurons)
    np.save(expertpath + '\selective_neurons.npy', all_sel_neurons)
