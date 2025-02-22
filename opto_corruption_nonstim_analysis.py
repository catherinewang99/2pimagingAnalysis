# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 16:21:35 2024

@author: Catherine Wang
"""

import sys
sys.path.append("C:\scripts\Imaging analysis")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from session import Session
from matplotlib.pyplot import figure
from numpy import concatenate as cat
from sklearn.preprocessing import normalize
import quality
from scipy import stats
from scipy.stats import zscore
from activityMode import Mode
import behavior
from numpy.linalg import norm


plt.rcParams['pdf.fonttype'] = '42' 

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def cos_sim(a,b):
    return np.dot(a, b)/(norm(a)*norm(b))

#%% Get number to make SANKEY diagram contra ipsi

agg_mice_paths = [[ r'H:\data\BAYLORCW041\python\2024_05_13',
   r'H:\data\BAYLORCW041\python\2024_06_12']]

agg_mice_paths = [
            # [r'H:\data\BAYLORCW038\python\2024_02_05',
            #   r'H:\data\BAYLORCW038\python\2024_02_15',
            #   r'H:\data\BAYLORCW038\python\2024_03_15',],
             
              ['H:\\data\\BAYLORCW039\\python\\2024_04_17', 
            'H:\\data\\BAYLORCW039\\python\\2024_04_24',
            'H:\\data\\BAYLORCW039\\python\\2024_05_06'],
              ['H:\\data\\BAYLORCW039\\python\\2024_04_18', 
            'H:\\data\\BAYLORCW039\\python\\2024_04_25',
            'H:\\data\\BAYLORCW039\\python\\2024_05_08'],
              
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

p=0.001
stage1, stage2 = 0, 1 
triple=True

og_SDR = []
c1, i1, ns1 = np.zeros(3),np.zeros(3),np.zeros(3)
for paths in agg_mice_paths: # For each mouse

    s1 = Session(paths[stage1], use_reg=True, triple=triple) # Naive
    # epoch = range(s1.response, s1.time_cutoff) # response selective
    epoch = range(s1.delay + 9, s1.response) # delay selective
    # epoch = range(s1.sample, s1.delay) # sample selective
    
    contra_neurons, ipsi_neurons, _, _ = s1.contra_ipsi_pop(epoch, p=p)
    naive_nonsel = [n for n in s1.good_neurons if n not in ipsi_neurons and n not in contra_neurons]

    og_SDR += [[len(contra_neurons), len(ipsi_neurons), len(naive_nonsel)]]

    # s2 = session.Session(paths[0][1], use_reg=True, triple=True) # Learning
    s2 = Session(paths[stage2], use_reg=True, triple=triple) # Expert

    # learning = sum([s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], epoch) for n in naive_sel])
    # expert = sum([s3.is_selective(s3.good_neurons[np.where(s1.good_neurons ==n)[0][0]], epoch) for n in naive_sel])
    
    for n in contra_neurons:
        if s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], epoch, p=p):
            ipsi, _, _ = s2.screen_preference(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], epoch)
            if not ipsi:
                c1[0] += 1
            else:
                c1[1] += 1
        else:
            c1[2] += 1
    
    for n in ipsi_neurons:
        if s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], epoch, p=p):
            ipsi, _, _ = s2.screen_preference(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], epoch)
            if not ipsi:
                i1[0] += 1
            else:
                i1[1] += 1
        else:
            i1[2] += 1
    

    for n in naive_nonsel:
        if s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], epoch, p=p):
            ipsi, _, _ = s2.screen_preference(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], epoch)
            if not ipsi:
                ns1[0] += 1
            else:
                ns1[1] += 1
        else:
            ns1[2] += 1
    
og_SDR = np.sum(og_SDR, axis=0)


#%% Plot bar graph showing proportion of recruitment into contra ipsi



#%% Changes at single cell level - sankey SDR
agg_mice_paths = [[['H:\\data\\BAYLORCW038\\python\\2024_02_05', 
                    'H:\\data\\BAYLORCW038\\python\\2024_03_15'],
                   ]]

agg_mice_paths = [[['H:\\data\\BAYLORCW039\\python\\2024_04_24', 
                    'H:\\data\\BAYLORCW039\\python\\2024_05_06'],
                   ]]
agg_mice_paths = [['H:\\data\\BAYLORCW038\\python\\2024_02_05', 
                    'H:\\data\\BAYLORCW038\\python\\2024_03_15'],
                   
                   ['H:\\data\\BAYLORCW039\\python\\2024_04_17', 
                    'H:\\data\\BAYLORCW039\\python\\2024_05_06'],
                   
                    [r'H:\\data\\BAYLORCW039\\python\\2024_04_18',
                     r'H:\\data\\BAYLORCW039\\python\\2024_05_08'],
                    
                    [r'H:\\data\\BAYLORCW041\\python\\2024_05_14',
                      r'H:\\data\\BAYLORCW041\\python\\2024_06_07'],
                    
                    [r'H:\\data\\BAYLORCW041\\python\\2024_05_13',
                      r'H:\\data\\BAYLORCW041\\python\\2024_06_12'],

                    [r'H:\\data\\BAYLORCW041\\python\\2024_05_15',
                     r'H:\\data\\BAYLORCW041\\python\\2024_06_11'],
                    ]
                  #   [r'H:\\data\\BAYLORCW043\\python\\2024_05_20',
                  #   r'H:\\data\\BAYLORCW043\\python\\2024_06_13'],

                  #   [r'H:\\data\\BAYLORCW043\\python\\2024_05_21',
                  #    r'H:\\data\\BAYLORCW043\\python\\2024_06_14']
                  # ]]
agg_mice_paths = [[ r'H:\data\BAYLORCW041\python\2024_05_24',
   r'H:\data\BAYLORCW041\python\2024_06_12']]


agg_mice_paths = [
            # [r'H:\data\BAYLORCW038\python\2024_02_05',
            #   r'H:\data\BAYLORCW038\python\2024_02_15',
            #   r'H:\data\BAYLORCW038\python\2024_03_15',],
             
              ['H:\\data\\BAYLORCW039\\python\\2024_04_17', 
            'H:\\data\\BAYLORCW039\\python\\2024_04_24',
            'H:\\data\\BAYLORCW039\\python\\2024_05_06'],
              ['H:\\data\\BAYLORCW039\\python\\2024_04_18', 
            'H:\\data\\BAYLORCW039\\python\\2024_04_25',
            'H:\\data\\BAYLORCW039\\python\\2024_05_08'],
              
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
            '-'], # ONLY IF INIT --> MID
            
              # ['-', 
            # 'H:\\data\\BAYLORCW043\\python\\2024_06_06',
            # 'H:\\data\\BAYLORCW043\\python\\2024_06_13'], #  ONLY IF MID --> FINAL
            
            #   ['-', 
            # 'H:\\data\\BAYLORCW043\\python\\2024_06_04',
            # 'H:\\data\\BAYLORCW043\\python\\2024_06_14'], # ONLY IF MID --> FINAL
            
             ['H:\\data\\BAYLORCW042\\python\\2024_06_05', 
           'H:\\data\\BAYLORCW042\\python\\2024_06_14',
           'H:\\data\\BAYLORCW042\\python\\2024_06_24']
            ]

p=0.001
stage1, stage2 = 0,1
triple=True


og_SDR = []
allstos, allnstos, alldtod, allnstod = [],[],[],[]
alldtons, allstons = [],[]
all_s1list, all_d1, all_r1, all_ns1 = [],[],[],[]
for paths in agg_mice_paths: # For each mouse/FOV
    stos = []
    nstos = []
    dtod = []
    nstod = []
    
    dtons, stons = [], []
    
    s1list, d1, r1, ns1 = np.zeros(4),np.zeros(4),np.zeros(4),np.zeros(4)

    # s1 = Session(paths[stage1], use_reg=True, triple=triple) # Learning
    if '43' in paths[1]:
        s1 = Session(paths[stage1], use_reg=True, triple=False, use_background_sub=False) # Naive
    else:
        s1 = Session(paths[stage1], use_reg=True, triple=triple, use_background_sub=False) # Naive

    sample_epoch = range(s1.sample, s1.delay+2)
    delay_epoch = range(s1.delay+9, s1.response)
    response_epoch = range(s1.response, s1.response + 12)
    
    naive_sample_sel = s1.get_epoch_selective(sample_epoch, p=p)
    
    naive_delay_sel = s1.get_epoch_selective(delay_epoch, p=p)
    naive_delay_sel = [n for n in naive_delay_sel if n not in naive_sample_sel]
    
    naive_response_sel = s1.get_epoch_selective(response_epoch, p=p)
    naive_response_sel = [n for n in naive_response_sel if n not in naive_sample_sel and n not in naive_delay_sel]

    naive_nonsel = [n for n in s1.good_neurons if n not in naive_sample_sel and n not in naive_delay_sel and n not in naive_response_sel]

    og_SDR += [[len(naive_sample_sel), len(naive_delay_sel), len(naive_response_sel), len(naive_nonsel)]]

    # s2 = Session(paths[stage2], use_reg=True, triple=triple) # Learning
    if '43' in paths[1]:
        s2 = Session(paths[stage2], use_reg=True, triple=False, use_background_sub=False) # Naive
    else:
        s2 = Session(paths[stage2], use_reg=True, triple=triple, use_background_sub=False) # Naive
    
    for n in naive_sample_sel:
        if s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], sample_epoch, p=p):
            s1list[0] += 1
            stos += [(n, s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]])]  #save sample to sample cells

        elif s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], delay_epoch, p=p):
            s1list[1] += 1
        elif s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], response_epoch, p=p):
            s1list[2] += 1
        else:
            s1list[3] += 1
            stons += [(n, s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]])]  #save sample to ns cells

    
    for n in naive_delay_sel:
        if s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], sample_epoch, p=p):
            d1[0] += 1
        elif s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], delay_epoch, p=p):
            d1[1] += 1
            dtod += [(n, s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]])]  #save delay to delay cells

        elif s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], response_epoch, p=p):
            d1[2] += 1
        else:
            d1[3] += 1
            dtons += [(n, s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]])]  #save delay to ns cells

    
    for n in naive_response_sel:
        if s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], sample_epoch, p=p):
            r1[0] += 1

        elif s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], delay_epoch, p=p):
            r1[1] += 1

        elif s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], response_epoch, p=p):
            r1[2] += 1
        else:
            r1[3] += 1
    
    
    for n in naive_nonsel:
        if s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], sample_epoch, p=p):
            ns1[0] += 1
            nstos += [(n, s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]])]  #save ns to sample cells

        elif s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], delay_epoch, p=p):
            ns1[1] += 1
            nstod += [(n, s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]])]  #save ns to delay cells

        elif s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], response_epoch, p=p):
            ns1[2] += 1
        else:
            ns1[3] += 1
            
    allstos += [stos]
    allnstos += [nstos]
    alldtod += [dtod]
    allnstod += [nstod]
    
    allstons += [stons]
    alldtons += [dtons]
    
    all_s1list += [s1list]
    all_d1 += [d1]
    all_r1 += [r1]
    all_ns1 += [ns1]

og_SDR = np.sum(og_SDR, axis=0)

#%% Plot bar graph showing proportion of recruitment into SDR neurons

# all_SDR_initmid = [all_s1list, all_d1, all_r1, all_ns1]
all_SDR_midfinal = [all_s1list, all_d1, all_r1, all_ns1]
titles = ['Sample', 'Delay', 'Response', 'Non-sel.']
for sel_type in range(4): # Go thru each type of SDR sel
    all_neurons = np.sum(all_SDR_initmid[sel_type], axis=1) # Total neurons to divide by
    
    f = plt.figure(figsize=(6,6))
    
    all_props = []
    for i in range(len(all_neurons)): #For each FOV
        props = all_SDR_initmid[sel_type][i] / all_neurons[i]
        all_props += [props]
        plt.scatter(np.arange(4)-0.2, props, marker='o')
    
    plt.bar(np.arange(4)-0.2, np.mean(all_props, axis=0), 0.4, fill=False, label="First round")
    plt.xticks(np.arange(4), ['Sample', 'Delay', 'Response', 'Non-sel.'])
    plt.axhline(0.5, ls='--', alpha=0.5)

    # Go thru the mid to final stages
    all_neurons = np.sum(all_SDR_midfinal[sel_type], axis=1) # Total neurons to divide by
        
    all_props = []
    for i in range(len(all_neurons)): #For each FOV
        props = all_SDR_midfinal[sel_type][i] / all_neurons[i]
        all_props += [props]
        plt.scatter(np.arange(4)+0.2, props, marker='o')
    
    plt.bar(np.arange(4)+0.2, np.mean(all_props, axis=0), 0.4, alpha = 0.2, fill='grey', label="Second round")

    plt.legend()
    plt.ylabel('Proportion of neurons')
    plt.xlabel('Pool that {} neurons recruited from'.format(titles[sel_type]))
    plt.title('{} selective neurons'.format(titles[sel_type]))
    plt.show()
# Bar plot showing just the within type proportions
f = plt.figure(figsize=(6,6))

for sel_type in range(4): # Go thru each type of SDR sel
    all_neurons = np.sum(all_SDR_initmid[sel_type], axis=1) # Total neurons to divide by
    
    all_props = []
    for i in range(len(all_neurons)): #For each FOV
        props = all_SDR_initmid[sel_type][i] / all_neurons[i]
        all_props += [props]
        plt.scatter((sel_type) - 0.2, props[sel_type], marker='o')
    if sel_type == 0:
        plt.bar((sel_type) - 0.2, np.mean(np.array(all_props)[:,sel_type]), 0.4, fill=False, label="First round of opto")
    else:
        plt.bar((sel_type) - 0.2, np.mean(np.array(all_props)[:,sel_type]), 0.4, fill=False)

    plt.xticks(np.arange(4), ['Sample', 'Delay', 'Response', 'Non-sel.'])
    plt.axhline(0.5, ls='--', alpha=0.5)

    # Go thru the mid to final stages
    all_neurons = np.sum(all_SDR_midfinal[sel_type], axis=1) # Total neurons to divide by
        
    all_props = []
    for i in range(len(all_neurons)): #For each FOV
        props = all_SDR_midfinal[sel_type][i] / all_neurons[i]
        all_props += [props]
        plt.scatter((sel_type)+0.2, props[sel_type], marker='o')
    
    if sel_type == 0:
        plt.bar((sel_type)+0.2, np.mean(np.array(all_props)[:,sel_type]), 0.4, alpha = 0.2, color='grey', label="Second round of opto")
    else:
        plt.bar((sel_type)+0.2, np.mean(np.array(all_props)[:,sel_type]), 0.4, alpha = 0.2, color='grey')


plt.legend()
plt.ylabel('Proportion of neurons retained')
plt.xlabel('Type of selectivity')
plt.title('Retention of selective neurons by epoch')
plt.savefig(r'H:\Fig 4\neural\retention_of_SDR_neurons.pdf')


#%% Plot the selectivity of recruited vs retained vs dropout neurons
# look at the accompanying recovery to stim


# Alternative method:

# intialpath, middlepath, finalpath = ['H:\\data\\BAYLORCW038\\python\\2024_02_05', 
#                          r'H:\data\BAYLORCW038\python\2024_02_15',
#                          'H:\\data\\BAYLORCW038\\python\\2024_03_15']
# intialpath, finalpath = ['H:\\data\\BAYLORCW039\\python\\2024_04_24', 
#                          'H:\\data\\BAYLORCW039\\python\\2024_05_06']
allcats = [allstos, allnstos, alldtod, allnstod, allstons, alldtons]
allcats = [alldtod, allnstod, alldtons] # delay only
by_FOV = True

for cats in allcats:
    num_neurons, counter = 0, 0
    for paths in agg_mice_paths:
        
        intialpath, finalpath =  paths[stage1], paths[stage2]
        if '43' in paths[1] or '38' in paths[1]:
            s1 = Session(intialpath, use_reg=True) # pre
            s2 = Session(finalpath, use_reg=True) # post
        else:
            s1 = Session(intialpath, use_reg=True, triple=True) # pre
            s2 = Session(finalpath, use_reg=True, triple=True) # post
        stos = np.array(cats[counter])
        counter += 1

        for i in range(len(stos)):
            
            if by_FOV and i == 0 :
                print("first addition")
                pre_sel = s1.plot_selectivity(stos[i,0], plot=False)
                post_sel = s2.plot_selectivity(stos[i,1], plot=False)
        
                pre_sel_opto = s1.plot_selectivity(stos[i,0], plot=False, opto=True)
                post_sel_opto = s2.plot_selectivity(stos[i,1], plot=False, opto=True)
                
            elif num_neurons + i == 0:
                print("first addition non FOV")
                pre_sel = s1.plot_selectivity(stos[i,0], plot=False)
                post_sel = s2.plot_selectivity(stos[i,1], plot=False)
        
                pre_sel_opto = s1.plot_selectivity(stos[i,0], plot=False, opto=True)
                post_sel_opto = s2.plot_selectivity(stos[i,1], plot=False, opto=True)
            else:
                pre_sel = np.vstack((pre_sel, s1.plot_selectivity(stos[i,0], plot=False)))
                post_sel = np.vstack((post_sel, s2.plot_selectivity(stos[i,1], plot=False)))
                
                pre_sel_opto = np.vstack((pre_sel_opto, s1.plot_selectivity(stos[i,0], plot=False, opto=True)))
                post_sel_opto = np.vstack((post_sel_opto, s2.plot_selectivity(stos[i,1], plot=False, opto=True)))
                
        if num_neurons == 0 and by_FOV:
            
            if stos.shape[0] == 1:
                agg_pre_sel = pre_sel
                agg_pre_sel_opto = pre_sel_opto
                
                agg_post_sel = post_sel
                agg_post_sel_opto = post_sel_opto
            else:
                agg_pre_sel = np.mean(pre_sel, axis=0)
                agg_pre_sel_opto = np.mean(pre_sel_opto, axis=0)
                
                agg_post_sel = np.mean(post_sel, axis=0)
                agg_post_sel_opto = np.mean(post_sel_opto, axis=0)
            
        elif by_FOV:
            if stos.shape[0] == 1:
                
                agg_pre_sel = np.vstack((agg_pre_sel, pre_sel))
                agg_pre_sel_opto = np.vstack((agg_pre_sel_opto, pre_sel_opto))
                
                agg_post_sel = np.vstack((agg_post_sel, post_sel))
                agg_post_sel_opto = np.vstack((agg_post_sel_opto, post_sel_opto))
                
            else:

                agg_pre_sel = np.vstack((agg_pre_sel, np.mean(pre_sel, axis=0)))
                agg_pre_sel_opto = np.vstack((agg_pre_sel_opto, np.mean(pre_sel_opto, axis=0)))
                
                agg_post_sel = np.vstack((agg_post_sel, np.mean(post_sel, axis=0)))
                agg_post_sel_opto = np.vstack((agg_post_sel_opto, np.mean(post_sel_opto, axis=0)))
                
        num_neurons += len(stos)
        
    f, ax = plt.subplots(1,2, sharey='row', figsize=(10,5))

    if by_FOV:
        sel = np.mean(agg_pre_sel, axis=0)
        err = np.std(agg_pre_sel, axis=0) / np.sqrt(len(agg_pre_sel)) 
        selo = np.mean(agg_pre_sel_opto, axis=0)
        erro = np.std(agg_pre_sel_opto, axis=0) / np.sqrt(len(agg_pre_sel_opto)) 
    else:
        sel = np.mean(pre_sel, axis=0)
        err = np.std(pre_sel, axis=0) / np.sqrt(len(pre_sel)) 
        selo = np.mean(pre_sel_opto, axis=0)
        erro = np.std(pre_sel_opto, axis=0) / np.sqrt(len(pre_sel_opto))
        
        
    x = np.arange(-6.97,5,s1.fs)[:s1.time_cutoff]
    ax[0].plot(x, sel, 'black')
            
    ax[0].fill_between(x, sel - err, 
              sel + err,
              color=['darkgray'])
    
    ax[0].plot(x, selo, 'r-')
            
    ax[0].fill_between(x, selo - erro, 
              selo + erro,
              color=['#ffaeb1']) 
    
    if by_FOV:
        sel = np.mean(agg_post_sel, axis=0)
        err = np.std(agg_post_sel, axis=0) / np.sqrt(len(agg_post_sel)) 
        selo = np.mean(agg_post_sel_opto, axis=0)
        erro = np.std(agg_post_sel_opto, axis=0) / np.sqrt(len(agg_post_sel_opto)) 
    else:
        sel = np.mean(post_sel, axis=0)
        err = np.std(post_sel, axis=0) / np.sqrt(len(post_sel)) 
        selo = np.mean(post_sel_opto, axis=0)
        erro = np.std(post_sel_opto, axis=0) / np.sqrt(len(post_sel_opto)) 
        
    x = np.arange(-6.97,5,s1.fs)[:s1.time_cutoff]
    ax[1].plot(x, sel, 'black')
            
    ax[1].fill_between(x, sel - err, 
              sel + err,
              color=['darkgray'])
    
    ax[1].plot(x, selo, 'r-')
            
    ax[1].fill_between(x, selo - erro, 
              selo + erro,
              color=['#ffaeb1']) 
    
    for i in range(2):
        ax[i].axvline(-4.3, color = 'grey', alpha=0.5, ls = '--')
        ax[i].axvline(-3, color = 'grey', alpha=0.5, ls = '--')
        ax[i].axvline(0, color = 'grey', alpha=0.5, ls = '--')
        ax[i].hlines(y=max(cat((selo, sel))), xmin=-3, xmax=-2, linewidth=10, color='red')
        
    ax[0].set_xlabel('Time from Go cue (s)')
    ax[0].set_ylabel('Selectivity')

    ax[0].set_title('Optogenetic effect on selectivity (n = {} neurons)'.format(num_neurons))                  
    # ax[1].set_title('Optogenetic effect on selectivity (n = {} neurons)'.format(num_neurons))         
    plt.show()         


#%% Quantify the robustness / modularity of recruited vs retained vs pruned neurons

allcats = [alldtod, allnstod, alldtons] # delay only
by_FOV = True
period = np.arange(s1.delay, s1.delay+6)
period = np.arange(s1.response-6, s1.response)
allpre_areas = []
allpost_areas = []
for cats in allcats:
    pre_area = []
    post_area = []
    
    num_neurons, counter = 0, 0
    for paths in agg_mice_paths: # For every FOV
        
        intialpath, finalpath =  paths[stage1], paths[stage2]
        if '43' in paths[1] or '38' in paths[1]:
            s1 = Session(intialpath, use_reg=True) # pre
            s2 = Session(finalpath, use_reg=True) # post
        else:
            s1 = Session(intialpath, use_reg=True, triple=True) # pre
            s2 = Session(finalpath, use_reg=True, triple=True) # post
        stos = np.array(cats[counter])
        counter += 1

        for i in range(len(stos)): # For every neuron
            
            if by_FOV and i == 0 :
                print("first addition")
                pre_sel = s1.plot_selectivity(stos[i,0], plot=False)
                post_sel = s2.plot_selectivity(stos[i,1], plot=False)
        
                pre_sel_opto = s1.plot_selectivity(stos[i,0], plot=False, opto=True)
                post_sel_opto = s2.plot_selectivity(stos[i,1], plot=False, opto=True)
                
            elif num_neurons + i == 0:
                print("first addition non FOV")
                pre_sel = s1.plot_selectivity(stos[i,0], plot=False)
                post_sel = s2.plot_selectivity(stos[i,1], plot=False)
        
                pre_sel_opto = s1.plot_selectivity(stos[i,0], plot=False, opto=True)
                post_sel_opto = s2.plot_selectivity(stos[i,1], plot=False, opto=True)
            else:
                pre_sel = np.vstack((pre_sel, s1.plot_selectivity(stos[i,0], plot=False)))
                post_sel = np.vstack((post_sel, s2.plot_selectivity(stos[i,1], plot=False)))
                
                pre_sel_opto = np.vstack((pre_sel_opto, s1.plot_selectivity(stos[i,0], plot=False, opto=True)))
                post_sel_opto = np.vstack((post_sel_opto, s2.plot_selectivity(stos[i,1], plot=False, opto=True)))
         
        # To aggregate over FOVs (unused)
        if num_neurons == 0 and by_FOV:
            
            if stos.shape[0] == 1:
                agg_pre_sel = pre_sel
                agg_pre_sel_opto = pre_sel_opto
                
                agg_post_sel = post_sel
                agg_post_sel_opto = post_sel_opto
            else:
                agg_pre_sel = np.mean(pre_sel, axis=0)
                agg_pre_sel_opto = np.mean(pre_sel_opto, axis=0)
                
                agg_post_sel = np.mean(post_sel, axis=0)
                agg_post_sel_opto = np.mean(post_sel_opto, axis=0)
            
        elif by_FOV:
            if stos.shape[0] == 1:
                
                agg_pre_sel = np.vstack((agg_pre_sel, pre_sel))
                agg_pre_sel_opto = np.vstack((agg_pre_sel_opto, pre_sel_opto))
                
                agg_post_sel = np.vstack((agg_post_sel, post_sel))
                agg_post_sel_opto = np.vstack((agg_post_sel_opto, post_sel_opto))
                
            else:

                agg_pre_sel = np.vstack((agg_pre_sel, np.mean(pre_sel, axis=0)))
                agg_pre_sel_opto = np.vstack((agg_pre_sel_opto, np.mean(pre_sel_opto, axis=0)))
                
                agg_post_sel = np.vstack((agg_post_sel, np.mean(post_sel, axis=0)))
                agg_post_sel_opto = np.vstack((agg_post_sel_opto, np.mean(post_sel_opto, axis=0)))
                
        num_neurons += len(stos)
    
        # For every FOV, get the area between curves
        if stos.shape[0] == 1: # Only one neuron
    
            pre_area += [np.trapz(y = pre_sel[period] - pre_sel_opto[period], dx = 1/s1.fs)]  
            post_area += [np.trapz(y = post_sel[period] - post_sel_opto[period], dx = 1/s1.fs)]  
            
        else: 
            
            pre_area += [np.trapz(y = np.mean(pre_sel, axis=0)[period] - np.mean(pre_sel_opto, axis=0)[period],
                                  dx = 1/s1.fs)]  
            post_area += [np.trapz(y = np.mean(post_sel, axis=0)[period] - np.mean(post_sel_opto, axis=0)[period],
                                   dx = 1/s1.fs)]  
    
    allpre_areas += [pre_area]
    allpost_areas += [post_area]
        
    
titles = ["Retained", "Recruited", "Pruned"]
f = plt.figure(figsize=(10,5))
plt.bar(np.arange(3)-0.2, np.mean(np.abs(allpre_areas), axis=1), 0.4, label="Before")
plt.bar(np.arange(3)+0.2, np.mean(np.abs(allpost_areas), axis=1), 0.4, label="After")
for i in range(3):
    plt.scatter(np.zeros(np.array(allpre_areas).shape[1])+i-0.2, np.array(np.abs(allpre_areas))[i,:])
    plt.scatter(np.zeros(np.array(allpost_areas).shape[1])+i+0.2, np.array(np.abs(allpost_areas))[i,:])
plt.xticks(np.arange(3), ['Retained', 'Recruited', 'Pruned'])
plt.xlabel('Type of neuron')
# plt.ylabel('Coupling strength (area between curves)')
plt.ylabel('Lack of robustness (area between curves)')
plt.legend()



#%% Retention of general selectivity
# FIXME
agg_mice_paths = [
            # [r'H:\data\BAYLORCW038\python\2024_02_05',
            #   r'H:\data\BAYLORCW038\python\2024_02_15',
            #   r'H:\data\BAYLORCW038\python\2024_03_15',],
             
              ['H:\\data\\BAYLORCW039\\python\\2024_04_17', 
            'H:\\data\\BAYLORCW039\\python\\2024_04_24',
            'H:\\data\\BAYLORCW039\\python\\2024_05_06'],
              ['H:\\data\\BAYLORCW039\\python\\2024_04_18', 
            'H:\\data\\BAYLORCW039\\python\\2024_04_25',
            'H:\\data\\BAYLORCW039\\python\\2024_05_08'],
              
            ['H:\\data\\BAYLORCW041\\python\\2024_05_14', 
          'H:\\data\\BAYLORCW041\\python\\2024_05_23',
          'H:\\data\\BAYLORCW041\\python\\2024_06_07'],
            ['H:\\data\\BAYLORCW041\\python\\2024_05_13', 
          'H:\\data\\BAYLORCW041\\python\\2024_05_24',
          'H:\\data\\BAYLORCW041\\python\\2024_06_12'],
            ['H:\\data\\BAYLORCW041\\python\\2024_05_15', 
          'H:\\data\\BAYLORCW041\\python\\2024_05_28',
          'H:\\data\\BAYLORCW041\\python\\2024_06_11'],
            
          #   ['H:\\data\\BAYLORCW043\\python\\2024_05_20', 
          # 'H:\\data\\BAYLORCW043\\python\\2024_06_03',
          # 'H:\\data\\BAYLORCW043\\python\\2024_06_13'], # NEEDS TO BE SWITCHED IF MID --> FINAL
            
             ['H:\\data\\BAYLORCW043\\python\\2024_05_20', 
           'H:\\data\\BAYLORCW043\\python\\2024_06_06',
           'H:\\data\\BAYLORCW043\\python\\2024_06_13'], # NEEDS TO BE SWITCHED IF INIT --> MID
            
             ['H:\\data\\BAYLORCW043\\python\\2024_05_22', 
           'H:\\data\\BAYLORCW043\\python\\2024_06_04',
           'H:\\data\\BAYLORCW043\\python\\2024_06_14'], # ONLY IF MID --> FINAL
            
             ['H:\\data\\BAYLORCW042\\python\\2024_06_05', 
           'H:\\data\\BAYLORCW042\\python\\2024_06_14',
           'H:\\data\\BAYLORCW042\\python\\2024_06_24']
            ]

p=0.001
stage1, stage2 = 1,2
triple=True


og_SDR = []
allstos = []
allnstos = []
all_s1list, all_d1, all_r1, all_ns1 = [],[],[],[]
for paths in agg_mice_paths: # For each mouse/FOV
    stos = []
    nstos = []
    s1list, d1, r1, ns1 = np.zeros(4),np.zeros(4),np.zeros(4),np.zeros(4)

    # s1 = Session(paths[stage1], use_reg=True, triple=triple) # Learning
    if '43' in paths[0]:
        s1 = Session(paths[stage1], use_reg=True, triple=False, use_background_sub=False) # Naive
    else:
        s1 = Session(paths[stage1], use_reg=True, triple=triple, use_background_sub=False) # Naive

    whole_epoch = range(s1.time_cutoff)

    naive_sel = s1.get_epoch_selective(whole_epoch, p=p)
    naive_nonsel = [n for n in s1.good_neurons if n not in naive_sel]

    # s2 = Session(paths[stage2], use_reg=True, triple=triple) # Learning
    if '43' in paths[0]:
        s2 = Session(paths[stage2], use_reg=True, triple=False, use_background_sub=False) # Naive
    else:
        s2 = Session(paths[stage2], use_reg=True, triple=triple, use_background_sub=False) # Naive
    
    for n in naive_sel:
        if s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], whole_epoch, p=p):
            s1list[0] += 1

        else:
            ns1[3] += 1
            # stos += [(n, s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]])]  #save sample to ns cells

    # for n in naive_nonsel:
    #     if s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], sample_epoch, p=p):
    #         ns1[0] += 1
    #         nstos += [(n, s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]])]  #save ns to sample cells

    #     elif s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], delay_epoch, p=p):
    #         ns1[1] += 1
    #         # nstos += [(n, s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]])]  #save ns to delay cells

    #     elif s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], response_epoch, p=p):
    #         ns1[2] += 1
    #     else:
    #         ns1[3] += 1
    allstos += [[stos]]
    allnstos += [[nstos]]
    
    all_s1list += [s1list]
    all_d1 += [d1]
    all_r1 += [r1]
    all_ns1 += [ns1]

og_SDR = np.sum(og_SDR, axis=0)


#%% Contributions of neurons to CD before and after as a scatter plot


# intialpath, middlepath, finalpath = ['H:\\data\\BAYLORCW038\\python\\2024_02_05', 
#                          r'H:\data\BAYLORCW038\python\2024_02_15',
#                          'H:\\data\\BAYLORCW038\\python\\2024_03_15']
# intialpath, finalpath = ['H:\\data\\BAYLORCW039\\python\\2024_04_24', 
#                          'H:\\data\\BAYLORCW039\\python\\2024_05_06']

# intialpath, finalpath = ['H:\\data\\BAYLORCW041\\python\\2024_05_14', 
#                          'H:\\data\\BAYLORCW041\\python\\2024_05_23']
# intialpath, middlepath, finalpath = [r'H:\data\BAYLORCW041\python\2024_05_13',
#    r'H:\data\BAYLORCW041\python\2024_05_24',
#   r'H:\data\BAYLORCW041\python\2024_06_12']

# intialpath, middlepath, finalpath = [r'H:\data\BAYLORCW042\python\2024_06_05',
# r'H:\data\BAYLORCW042\python\2024_06_14',
# r'H:\data\BAYLORCW042\python\2024_06_24',]

# intialpath, middlepath, finalpath = [r'H:\data\BAYLORCW042\python\2024_06_06',
# r'H:\data\BAYLORCW042\python\2024_06_18',
# r'H:\data\BAYLORCW042\python\2024_06_26',]
agg_mice_paths = [
            # [r'H:\data\BAYLORCW038\python\2024_02_05',
            #   r'H:\data\BAYLORCW038\python\2024_02_15',
            #   r'H:\data\BAYLORCW038\python\2024_03_15',],
             
              ['H:\\data\\BAYLORCW039\\python\\2024_04_17', 
            'H:\\data\\BAYLORCW039\\python\\2024_04_24',
            'H:\\data\\BAYLORCW039\\python\\2024_05_06'],
              ['H:\\data\\BAYLORCW039\\python\\2024_04_18', 
            'H:\\data\\BAYLORCW039\\python\\2024_04_25',
            'H:\\data\\BAYLORCW039\\python\\2024_05_08'],
              
            ['H:\\data\\BAYLORCW041\\python\\2024_05_14', 
          'H:\\data\\BAYLORCW041\\python\\2024_05_23',
          'H:\\data\\BAYLORCW041\\python\\2024_06_07'],
            ['H:\\data\\BAYLORCW041\\python\\2024_05_13', 
          'H:\\data\\BAYLORCW041\\python\\2024_05_24',
          'H:\\data\\BAYLORCW041\\python\\2024_06_12'],
            ['H:\\data\\BAYLORCW041\\python\\2024_05_15', 
          'H:\\data\\BAYLORCW041\\python\\2024_05_28',
          'H:\\data\\BAYLORCW041\\python\\2024_06_11'],
            
            #   ['H:\\data\\BAYLORCW043\\python\\2024_05_20', 
            # 'H:\\data\\BAYLORCW043\\python\\2024_06_03',
            # '-'], # NEEDS TO BE SWITCHED IF MID --> FINAL
            
               ['-', 
            'H:\\data\\BAYLORCW043\\python\\2024_06_06',
            'H:\\data\\BAYLORCW043\\python\\2024_06_13'], # NEEDS TO BE SWITCHED IF INIT --> MID
            
              ['-', 
            'H:\\data\\BAYLORCW043\\python\\2024_06_04',
            'H:\\data\\BAYLORCW043\\python\\2024_06_14'], # ONLY IF MID --> FINAL
            
             ['H:\\data\\BAYLORCW042\\python\\2024_06_05', 
           'H:\\data\\BAYLORCW042\\python\\2024_06_14',
           'H:\\data\\BAYLORCW042\\python\\2024_06_24']
            ]

r_stim, r_delay = [], []

for paths in agg_mice_paths:
    
    intialpath, finalpath = paths[1], paths[2]
    
    # sample CD
    if '43' in paths[1] or '38' in paths[1]:
        l1 = Mode(intialpath, use_reg=True, triple=False)
        l2 = Mode(finalpath, use_reg = True, triple=False)
    else:
        l1 = Mode(intialpath, use_reg=True, triple=True)
        l2 = Mode(finalpath, use_reg = True, triple=True)

    orthonormal_basis_initial, mean = l1.plot_CD(mode_input = 'stimulus')
    orthonormal_basis_initial_choice, mean = l1.plot_CD(mode_input = 'choice')
    
    orthonormal_basis, mean = l2.plot_CD(mode_input = 'stimulus')
    orthonormal_basis_choice, mean = l2.plot_CD(mode_input = 'choice')
    
    plt.scatter(orthonormal_basis_initial, orthonormal_basis)
    plt.title('Pearsons correlation: {}, p-val: {}'.format(stats.pearsonr(orthonormal_basis_initial, orthonormal_basis)[0], 
                                                           stats.pearsonr(orthonormal_basis_initial, orthonormal_basis)[1]))
    plt.xlabel('Initial sample CD values')
    plt.ylabel('Final sample CD values')
    plt.show()
    r_stim += [stats.pearsonr(orthonormal_basis_initial, orthonormal_basis)[0]]
    
    # delay CD
    
    
    plt.scatter(orthonormal_basis_initial_choice, orthonormal_basis_choice)
    plt.title('Pearsons correlation: {}, p-val: {}'.format(stats.pearsonr(orthonormal_basis_initial_choice, orthonormal_basis_choice)[0], 
                                                           stats.pearsonr(orthonormal_basis_initial_choice, orthonormal_basis_choice)[1]))
    plt.xlabel('Initial delay CD values')
    plt.ylabel('Final delay CD values')
    plt.show()
    r_delay += [stats.pearsonr(orthonormal_basis_initial_choice, orthonormal_basis_choice)[0]]
#%% Plot the R squared values of each FOV
# r_stimr1, r_delayr1 = r_stim, r_delay

f = plt.figure(figsize = (5,5))
plt.scatter(np.abs(r_stimr1), np.abs(r_delayr1), label="Round 1")
plt.scatter(np.abs(r_stim), np.abs(r_delay), label="Round 2")
plt.xlabel('R2 values for sample mode')
plt.ylabel('R2 values for delay mode')
plt.axhline(0, ls='--')
plt.axvline(0, ls='--')
plt.axhline(0.5, ls='--', alpha = 0.5)
plt.axvline(0.5, ls='--', alpha = 0.5)
plt.legend()
plt.savefig(r'H:\Fig 4\neural\R2_vals_sampledelaymodes.pdf')

#%% Stability of CD 


intialpath, middlepath, finalpath = ['H:\\data\\BAYLORCW038\\python\\2024_02_05', 
                         r'H:\data\BAYLORCW038\python\2024_02_15',
                         'H:\\data\\BAYLORCW038\\python\\2024_03_15']
intialpath, finalpath = ['H:\\data\\BAYLORCW039\\python\\2024_04_24', 
                         'H:\\data\\BAYLORCW039\\python\\2024_05_06']
intialpath, finalpath = ['H:\\data\\BAYLORCW043\\python\\2024_05_20', 
                         'H:\\data\\BAYLORCW043\\python\\2024_06_03']
intialpath, finalpath = ['H:\\data\\BAYLORCW041\\python\\2024_05_14', 
                         'H:\\data\\BAYLORCW041\\python\\2024_05_23']
intialpath, middlepath, finalpath = [r'H:\data\BAYLORCW041\python\2024_05_13',
   r'H:\data\BAYLORCW041\python\2024_05_24',
  r'H:\data\BAYLORCW041\python\2024_06_12']
l1 = Mode(intialpath, use_reg=True, triple=True)
orthonormal_basis, mean = l1.plot_CD(mode_input='stimulus')#, save = r'F:\data\Fig 2\CDstim_expert_CW37.pdf')

# l1 = Mode(middlepath)
l1 = Mode(middlepath, use_reg = True, triple=True)
l1.plot_appliedCD(orthonormal_basis, mean)

l1 = Mode(finalpath, use_reg = True, triple=True)
l1.plot_appliedCD(orthonormal_basis, mean)



#%% CD rotation
# TAKES A LONG TIME TO RUN!

intialpath, finalpath = ['H:\\data\\BAYLORCW038\\python\\2024_02_05', 
        'H:\\data\\BAYLORCW038\\python\\2024_03_15']

intialpath, finalpath = ['H:\\data\\BAYLORCW039\\python\\2024_04_17', 
                         'H:\\data\\BAYLORCW039\\python\\2024_04_24']
# Bootstrap
angles = []
angles_stim = []
for _ in range(50):

    l1 = Mode(intialpath, use_reg=True)
    orthonormal_basis_initial, mean = l1.plot_CD(plot=False)
    orthonormal_basis_initial_sample, mean = l1.plot_CD(mode_input = 'stimulus', plot=False)
    
    l1 = Mode(finalpath, use_reg = True)
    orthonormal_basis_final, mean = l1.plot_CD(plot=False)
    orthonormal_basis_final_sample, mean = l1.plot_CD(mode_input = 'stimulus', plot=False)

    angles += [cos_sim(orthonormal_basis_initial, orthonormal_basis_final)]
    angles_stim += [cos_sim(orthonormal_basis_initial_sample, orthonormal_basis_final_sample)]

## Benchmark -- look at rotation of stim mode
plt.bar([0,1], [np.mean(angles), np.mean(angles_stim)], 0.4, fill=False)

plt.scatter(np.zeros(50), angles)
plt.scatter(np.ones(50), angles_stim)
plt.axhline(0, ls = '--')

# plt.ylim(-0.3, 1)
plt.xticks(range(2), ["Delay", "Stimulus"])
plt.xlabel('Choice decoder mode')
plt.ylabel('Rotation over corruption')

plt.show()

plt.scatter(angles, angles_stim)
plt.axhline(0, ls = '--')
plt.axvline(0, ls = '--')
# plt.ylim(0, 1)
# plt.xlim(-1, 1)
plt.xlabel('Choice decoder angles')
plt.ylabel('Stimulus decoder angles')








