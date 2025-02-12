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
from alm_2p import session
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

p=0.001

og_SDR = []
c1, i1, ns1 = np.zeros(3),np.zeros(3),np.zeros(3)
for paths in agg_mice_paths: # For each mouse

    s1 = session.Session(paths[0], use_reg=True, triple=True) # Naive
    # epoch = range(s1.response, s1.time_cutoff) # response selective
    epoch = range(s1.delay + 9, s1.response) # delay selective
    # epoch = range(s1.sample, s1.delay) # sample selective
    
    contra_neurons, ipsi_neurons, _, _ = s1.contra_ipsi_pop(epoch, p=p)
    naive_nonsel = [n for n in s1.good_neurons if n not in ipsi_neurons and n not in contra_neurons]

    og_SDR += [[len(contra_neurons), len(ipsi_neurons), len(naive_nonsel)]]

    # s2 = session.Session(paths[0][1], use_reg=True, triple=True) # Learning
    s2 = session.Session(paths[1], use_reg=True, triple=True) # Expert

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

# p=0.0005
p=0.001

og_SDR = []
allstos = []
allnstos = []
s1list, d1, r1, ns1 = np.zeros(4),np.zeros(4),np.zeros(4),np.zeros(4)
for paths in agg_mice_paths: # For each mouse
    stos = []
    nstos = []
    if '41' in paths[0]:
        s1 = Session(paths[0], use_reg=True, triple=True, use_background_sub=False) # Naive
    else:
        s1 = Session(paths[0], use_reg=True, use_background_sub=False) # Naive
    # sample_epoch = range(s1.sample+2, s1.delay+2)
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

    # s2 = session.Session(paths[0][1], use_reg=True, triple=True) # Learning
    if '41' in paths[0]:
        s2 = Session(paths[1], use_reg=True, triple=True, use_background_sub=False) # Naive
    else:
        s2 = Session(paths[1], use_reg=True) # Expert
    
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
            # stos += [(n, s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]])]  #save sample to ns cells

    
    for n in naive_delay_sel:
        if s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], sample_epoch, p=p):
            d1[0] += 1
        elif s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], delay_epoch, p=p):
            d1[1] += 1
            # stos += [(n, s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]])]  #save delay to delay cells

        elif s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], response_epoch, p=p):
            d1[2] += 1
        else:
            d1[3] += 1
            # nstos += [(n, s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]])]  #save delay to ns cells

    
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
            # nstos += [(n, s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]])]  #save ns to delay cells

        elif s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], response_epoch, p=p):
            ns1[2] += 1
        else:
            ns1[3] += 1
    allstos += [[stos]]
    allnstos += [[nstos]]

og_SDR = np.sum(og_SDR, axis=0)

#%% Plot the selectivity of recruited sample neurons vs stable sample neurons
# look at the accompanying recovery to stim

stos = np.array(stos)
stos_old = stos
# stos = np.array(nstos)

intialpath, middlepath, finalpath = ['H:\\data\\BAYLORCW038\\python\\2024_02_05', 
                         r'H:\data\BAYLORCW038\python\2024_02_15',
                         'H:\\data\\BAYLORCW038\\python\\2024_03_15']
intialpath, finalpath = ['H:\\data\\BAYLORCW039\\python\\2024_04_24', 
                         'H:\\data\\BAYLORCW039\\python\\2024_05_06']
s1 = Session(intialpath, use_reg=True, use_background_sub=False) # pre
s2 = Session(finalpath, use_reg=True) # post


for i in range(stos.shape[0]):
    
    if i == 0:
        pre_sel = s1.plot_selectivity(stos[i,0], plot=False)
        post_sel = s2.plot_selectivity(stos[i,1], plot=False)

        pre_sel_opto = s1.plot_selectivity(stos[i,0], plot=False, opto=True)
        post_sel_opto = s2.plot_selectivity(stos[i,1], plot=False, opto=True)
    else:
        pre_sel = np.vstack((pre_sel, s1.plot_selectivity(stos[i,0], plot=False)))
        post_sel = np.vstack((post_sel, s2.plot_selectivity(stos[i,1], plot=False)))
        
        pre_sel_opto = np.vstack((pre_sel_opto, s1.plot_selectivity(stos[i,0], plot=False, opto=True)))
        post_sel_opto = np.vstack((post_sel_opto, s2.plot_selectivity(stos[i,1], plot=False, opto=True)))

f, ax = plt.subplots(1,2, sharey='row', figsize=(10,5))
sel = np.mean(pre_sel, axis=0)
err = np.std(pre_sel, axis=0) / np.sqrt(len(pre_sel)) 
selo = np.mean(pre_sel_opto, axis=0)
erro = np.std(pre_sel_opto, axis=0) / np.sqrt(len(pre_sel_opto)) 

x = np.arange(-5.97,5,s1.fs)[:s1.time_cutoff]
ax[0].plot(x, sel, 'black')
        
ax[0].fill_between(x, sel - err, 
          sel + err,
          color=['darkgray'])

ax[0].plot(x, selo, 'r-')
        
ax[0].fill_between(x, selo - erro, 
          selo + erro,
          color=['#ffaeb1']) 

sel = np.mean(post_sel, axis=0)
err = np.std(post_sel, axis=0) / np.sqrt(len(post_sel)) 
selo = np.mean(post_sel_opto, axis=0)
erro = np.std(post_sel_opto, axis=0) / np.sqrt(len(post_sel_opto)) 

x = np.arange(-5.97,5,s1.fs)[:s1.time_cutoff]
ax[1].plot(x, sel, 'black')
        
ax[1].fill_between(x, sel - err, 
          sel + err,
          color=['darkgray'])

ax[1].plot(x, selo, 'r-')
        
ax[1].fill_between(x, selo - erro, 
          selo + erro,
          color=['#ffaeb1']) 



ax[0].set_xlabel('Time from Go cue (s)')



#%% Selectivity recovery ALL SESSIONS NON MATCHED

paths = ['H:\\data\\BAYLORCW038\\python\\2024_02_05', 
        'H:\\data\\BAYLORCW038\\python\\2024_03_15']
paths = ['H:\\data\\BAYLORCW039\\python\\2024_04_17', 
         'H:\\data\\BAYLORCW039\\python\\2024_04_24']
agg_mice_paths = [
                    [r'H:\data\BAYLORCW038\python\2024_02_05',
                     'H:\\data\\BAYLORCW039\\python\\2024_04_17',
                     'H:\\data\\BAYLORCW039\\python\\2024_04_18',
                     'H:\\data\\BAYLORCW041\\python\\2024_05_14', 
                     'H:\\data\\BAYLORCW041\\python\\2024_05_13', 
                     'H:\\data\\BAYLORCW041\\python\\2024_05_15', 
                     'H:\\data\\BAYLORCW043\\python\\2024_05_20', 
                     'H:\\data\\BAYLORCW043\\python\\2024_05_21', 
                     'H:\\data\\BAYLORCW042\\python\\2024_06_05', 
                     'H:\\data\\BAYLORCW042\\python\\2024_06_06', 
                     ],
                    
                    [r'H:\data\BAYLORCW038\python\2024_02_15',
                     'H:\\data\\BAYLORCW039\\python\\2024_04_24',
                     'H:\\data\\BAYLORCW039\\python\\2024_04_25',
                     'H:\\data\\BAYLORCW041\\python\\2024_05_23',
                     'H:\\data\\BAYLORCW041\\python\\2024_05_24',
                     'H:\\data\\BAYLORCW041\\python\\2024_05_28',
                     'H:\\data\\BAYLORCW043\\python\\2024_06_03',
                     'H:\\data\\BAYLORCW043\\python\\2024_06_06',
                     'H:\\data\\BAYLORCW043\\python\\2024_06_04',
                     'H:\\data\\BAYLORCW042\\python\\2024_06_14',
                     'H:\\data\\BAYLORCW042\\python\\2024_06_18',
                     ],
                    
                    [r'H:\data\BAYLORCW038\python\2024_03_15',
                     'H:\\data\\BAYLORCW039\\python\\2024_05_06',
                     'H:\\data\\BAYLORCW039\\python\\2024_05_08',
                     'H:\\data\\BAYLORCW041\\python\\2024_06_07',
                     'H:\\data\\BAYLORCW041\\python\\2024_06_12',
                     'H:\\data\\BAYLORCW041\\python\\2024_06_11',
                     'H:\\data\\BAYLORCW043\\python\\2024_06_13',
                     'H:\\data\\BAYLORCW043\\python\\2024_06_14',
                     'H:\\data\\BAYLORCW042\\python\\2024_06_24'
                     ]
    
                    ]

by_FOV = True

f, axarr = plt.subplots(1,3, sharex='col', figsize=(15,5))  

for i in range(3):
    
    pref, nonpref, optop, optonp = np.zeros(61), np.zeros(61), np.zeros(61), np.zeros(61)
    all_control_sel, all_opto_sel = np.zeros(61), np.zeros(61)

    num_neurons = 0
    for path in agg_mice_paths[i]:
    
        
        l1 = session.Session(path) #, baseline_normalization="median_zscore")
        
        control_sel, opto_sel = l1.selectivity_optogenetics(p=0.01, lickdir=False, 
                                                            return_traces=True)
        
        # pref_, nonpref_, optop_, optonp_ = l1.selectivity_optogenetics(p=0.01, lickdir=True, return_traces=True)
        
        # pref = np.vstack((pref, pref_))
        # nonpref = np.vstack((nonpref, nonpref_))
        # optop = np.vstack((optop, optop_))
        # optonp = np.vstack((optonp, optonp_))
        
        if by_FOV:
            all_control_sel = np.vstack((all_control_sel, np.mean(control_sel, axis=0)))
            all_opto_sel = np.vstack((all_opto_sel, np.mean(opto_sel, axis=0)))
        else:
            all_control_sel = np.vstack((all_control_sel, control_sel))
            all_opto_sel = np.vstack((all_opto_sel, opto_sel))
        
        num_neurons += len(l1.selective_neurons)
        
    # pref, nonpref, optop, optonp = pref[1:], nonpref[1:], optop[1:], optonp[1:]
    
    # sel = np.mean(pref, axis = 0) - np.mean(nonpref, axis = 0)
    # err = np.std(pref, axis=0) / np.sqrt(len(pref)) 
    # err += np.std(nonpref, axis=0) / np.sqrt(len(nonpref))
    
    # selo = np.mean(optop, axis = 0) - np.mean(optonp, axis = 0)
    # erro = np.std(optop, axis=0) / np.sqrt(len(optop)) 
    # erro += np.std(optonp, axis=0) / np.sqrt(len(optonp))  
    
    all_control_sel, all_opto_sel = all_control_sel[1:], all_opto_sel[1:]

    sel = np.mean(all_control_sel, axis=0)
    err = np.std(all_control_sel, axis=0) / np.sqrt(len(all_control_sel))
    selo = np.mean(all_opto_sel, axis=0)
    erro = np.std(all_opto_sel, axis=0) / np.sqrt(len(all_opto_sel))
    
    x = np.arange(-6.97,4,l1.fs)[:l1.time_cutoff]
    axarr[i].plot(x, sel, 'black')
            
    axarr[i].fill_between(x, sel - err, 
              sel + err,
              color=['darkgray'])
    
    axarr[i].plot(x, selo, 'r-')
            
    axarr[i].fill_between(x, selo - erro, 
              selo + erro,
              color=['#ffaeb1'])       
    
    axarr[i].axvline(-4.3, color = 'grey', alpha=0.5, ls = '--')
    axarr[i].axvline(-3, color = 'grey', alpha=0.5, ls = '--')
    axarr[i].axvline(0, color = 'grey', alpha=0.5, ls = '--')
    axarr[i].hlines(y=max(cat((selo, sel))), xmin=-3, xmax=-2, linewidth=10, color='red')
    
    axarr[i].set_title('Optogenetic effect on selectivity (n = {} neurons)'.format(num_neurons))                  
    axarr[i].set_xlabel('Time from Go cue (s)')
    axarr[i].set_ylabel('Selectivity')
    axarr[i].set_ylim((-0.2, 1.2))

# plt.savefig(r'F:\data\Fig 3\lea_sel_recovery_ALL.pdf')
plt.show()

#%% Selectivity recovery MATCHED

agg_mice_paths = [
                    [
                        r'H:\data\BAYLORCW038\python\2024_02_05',
                     'H:\\data\\BAYLORCW039\\python\\2024_04_17',
                     'H:\\data\\BAYLORCW039\\python\\2024_04_18',
                     'H:\\data\\BAYLORCW041\\python\\2024_05_14', 
                     'H:\\data\\BAYLORCW041\\python\\2024_05_13', 
                     'H:\\data\\BAYLORCW041\\python\\2024_05_15', 
                     'H:\\data\\BAYLORCW042\\python\\2024_06_05', 
                     ],
                    
                    [
                        # r'H:\data\BAYLORCW038\python\2024_02_15',
                     'H:\\data\\BAYLORCW039\\python\\2024_04_24',
                     'H:\\data\\BAYLORCW039\\python\\2024_04_25',
                     'H:\\data\\BAYLORCW041\\python\\2024_05_23',
                     'H:\\data\\BAYLORCW041\\python\\2024_05_24',
                     'H:\\data\\BAYLORCW041\\python\\2024_05_28',
                     'H:\\data\\BAYLORCW042\\python\\2024_06_14',
                     ],
                    
                    [
                        r'H:\data\BAYLORCW038\python\2024_03_15',
                     'H:\\data\\BAYLORCW039\\python\\2024_05_06',
                     'H:\\data\\BAYLORCW039\\python\\2024_05_08',
                     'H:\\data\\BAYLORCW041\\python\\2024_06_07',
                     'H:\\data\\BAYLORCW041\\python\\2024_06_12',
                     'H:\\data\\BAYLORCW041\\python\\2024_06_11',
                     'H:\\data\\BAYLORCW042\\python\\2024_06_24'
                     ]
    
                    ]


by_FOV=True

f, axarr = plt.subplots(1,3, sharex='col', figsize=(15,5))  

for i in range(3):
    
    pref, nonpref, optop, optonp = np.zeros(61), np.zeros(61), np.zeros(61), np.zeros(61)
    all_control_sel, all_opto_sel = np.zeros(61), np.zeros(61)
    num_neurons = 0

    for path in agg_mice_paths[i]:
    
        if '43' in path or '38' in path:
            l1 = session.Session(path, use_reg=True, triple=False)
        else:
            l1 = session.Session(path, use_reg=True, triple=True)

        # pref_, nonpref_, optop_, optonp_ = l1.selectivity_optogenetics(p=0.001, lickdir=True, return_traces=True)
        control_sel, opto_sel = l1.selectivity_optogenetics(p=0.001, lickdir=False, 
                                                            return_traces=True)
        
        # pref_, nonpref_, optop_, optonp_ = l1.selectivity_optogenetics(p=0.01, lickdir=True, return_traces=True)
        
        # pref = np.vstack((pref, pref_))
        # nonpref = np.vstack((nonpref, nonpref_))
        # optop = np.vstack((optop, optop_))
        # optonp = np.vstack((optonp, optonp_))
        
        if by_FOV:
            all_control_sel = np.vstack((all_control_sel, np.mean(control_sel, axis=0)))
            all_opto_sel = np.vstack((all_opto_sel, np.mean(opto_sel, axis=0)))
        else:
            all_control_sel = np.vstack((all_control_sel, control_sel))
            all_opto_sel = np.vstack((all_opto_sel, opto_sel))
        
        # pref = np.vstack((pref, pref_))
        # nonpref = np.vstack((nonpref, nonpref_))
        # optop = np.vstack((optop, optop_))
        # optonp = np.vstack((optonp, optonp_))
        
        num_neurons += len(l1.selective_neurons)
        
    all_control_sel, all_opto_sel = all_control_sel[1:], all_opto_sel[1:]

    sel = np.mean(all_control_sel, axis=0)
    err = np.std(all_control_sel, axis=0) / np.sqrt(len(all_control_sel))
    selo = np.mean(all_opto_sel, axis=0)
    erro = np.std(all_opto_sel, axis=0) / np.sqrt(len(all_opto_sel))
    # pref, nonpref, optop, optonp = pref[1:], nonpref[1:], optop[1:], optonp[1:]
    
    # sel = np.mean(pref, axis = 0) - np.mean(nonpref, axis = 0)
    # err = np.std(pref, axis=0) / np.sqrt(len(pref)) 
    # err += np.std(nonpref, axis=0) / np.sqrt(len(nonpref))
    
    # selo = np.mean(optop, axis = 0) - np.mean(optonp, axis = 0)
    # erro = np.std(optop, axis=0) / np.sqrt(len(optop)) 
    # erro += np.std(optonp, axis=0) / np.sqrt(len(optonp))  
    
    x = np.arange(-6.97,4,l1.fs)[:l1.time_cutoff]
    axarr[i].plot(x, sel, 'black')
            
    axarr[i].fill_between(x, sel - err, 
              sel + err,
              color=['darkgray'])
    
    axarr[i].plot(x, selo, 'r-')
            
    axarr[i].fill_between(x, selo - erro, 
              selo + erro,
              color=['#ffaeb1'])       
    
    axarr[i].axvline(-4.3, color = 'grey', alpha=0.5, ls = '--')
    axarr[i].axvline(-3, color = 'grey', alpha=0.5, ls = '--')
    axarr[i].axvline(0, color = 'grey', alpha=0.5, ls = '--')
    axarr[i].hlines(y=max(cat((selo, sel))), xmin=-3, xmax=-2, linewidth=10, color='red')
    
    axarr[i].set_title('Optogenetic effect on selectivity (n = {} neurons)'.format(num_neurons))                  
    axarr[i].set_xlabel('Time from Go cue (s)')
    axarr[i].set_ylabel('Selectivity')
    axarr[i].set_ylim((-0.2, 1.0))

# plt.savefig(r'F:\data\Fig 3\lea_sel_recovery_ALL.pdf')
plt.show()

#%% Selectivity recovery but use OR operation on neurons

for path in agg_mice_paths:
    
    expert = Session(paths[1], use_reg=True, triple=False, filter_reg=False)
    expert.selectivity_optogenetics()
    selective_neurons = expert.selective_neurons
    
    naive = Session(paths[0], use_reg=True, triple=False, filter_reg=False)
    sel_n = [naive.good_neurons[np.where(expert.good_neurons ==n)[0][0]] for n in selective_neurons]
    naive.selectivity_optogenetics(selective_neurons = sel_n)

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

#%% CD recovery to stim


intialpath, middlepath, finalpath = ['H:\\data\\BAYLORCW038\\python\\2024_02_05', 
                         r'H:\data\BAYLORCW038\python\2024_02_15',
                         'H:\\data\\BAYLORCW038\\python\\2024_03_15']
intialpath, finalpath = ['H:\\data\\BAYLORCW039\\python\\2024_04_24', 
                         'H:\\data\\BAYLORCW039\\python\\2024_05_06']
intialpath, finalpath = ['H:\\data\\BAYLORCW039\\python\\2024_04_18', 
                         'H:\\data\\BAYLORCW039\\python\\2024_05_08']

intialpath, finalpath = ['H:\\data\\BAYLORCW043\\python\\2024_05_20', 
                         'H:\\data\\BAYLORCW043\\python\\2024_06_03']

intialpath, middlepath, finalpath = [r'H:\data\BAYLORCW041\python\2024_05_13',
   r'H:\data\BAYLORCW041\python\2024_05_24',
  r'H:\data\BAYLORCW041\python\2024_06_12']

intialpath, middlepath, finalpath = [r'H:\data\BAYLORCW042\python\2024_06_05',
r'H:\data\BAYLORCW042\python\2024_06_14',
r'H:\data\BAYLORCW042\python\2024_06_24',]
intialpath, middlepath, finalpath = [r'H:\data\BAYLORCW042\python\2024_06_06',
r'H:\data\BAYLORCW042\python\2024_06_18',
r'H:\data\BAYLORCW042\python\2024_06_26',]
  
# intialpath, _, finalpath =  [r'H:\data\BAYLORCW041\python\2024_05_15',
#           r'H:\data\BAYLORCW041\python\2024_05_28',
#           r'H:\data\BAYLORCW041\python\2024_06_11',]
 
# intialpath, _, finalpath = [r'H:\data\BAYLORCW041\python\2024_05_14',
# r'H:\data\BAYLORCW041\python\2024_05_23',
# r'H:\data\BAYLORCW041\python\2024_06_07',]

# intialpath, finalpath = ['H:\\data\\BAYLORCW043\\python\\2024_06_06', 
#                          'H:\\data\\BAYLORCW043\\python\\2024_06_13']


l1 = Mode(finalpath, use_reg=True, triple=True, filter_reg=True)
# orthonormal_basis, mean = l1.plot_CD(ctl=True)
l1.plot_CD_opto(ctl=True)
control_traces, opto_traces, error_bars, orthonormal_basis, mean, meantrain, meanstd = l1.plot_CD_opto(return_traces=True, return_applied=True,ctl=True)

l1 = Mode(middlepath, use_reg=True, triple=True, filter_reg=True)
# l1.plot_CD_opto()
l1.plot_CD_opto_applied(orthonormal_basis, mean, meantrain, meanstd)

l1 = Mode(intialpath, use_reg = True, triple=True, filter_reg=True)
# l1.plot_appliedCD(orthonormal_basis, mean)
# l1.plot_CD_opto()
l1.plot_CD_opto_applied(orthonormal_basis, mean, meantrain, meanstd)

#%% Applying post CD to pre CD:
    
intialpath, middlepath, finalpath = ['H:\\data\\BAYLORCW038\\python\\2024_02_05', 
                         r'H:\data\BAYLORCW038\python\2024_02_15',
                         'H:\\data\\BAYLORCW038\\python\\2024_03_15']
intialpath, middlepath, finalpath = ['H:\\data\\BAYLORCW039\\python\\2024_04_24', 
                         r'H:\data\BAYLORCW039\python\2024_02_15',
                         'H:\\data\\BAYLORCW039\\python\\2024_05_06']
l1 = Mode(finalpath, use_reg = True)
# orthonormal_basis, mean = l1.plot_CD(ctl=True)
l1.plot_CD_opto(ctl=False)
control_traces, opto_traces, error_bars, orthonormal_basis, mean, meantrain, meanstd = l1.plot_CD_opto(return_traces=True, return_applied=True,ctl=False)

l1 = Mode(intialpath, use_reg=True)
# l1.plot_appliedCD(orthonormal_basis, mean)
l1.plot_CD_opto_applied(orthonormal_basis, mean, meantrain, meanstd)

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






#%% Plot selectivity recovery as a bar graph only matched

all_paths = [['H:\\data\\BAYLORCW038\\python\\2024_02_05', 
              'H:\\data\\BAYLORCW039\\python\\2024_04_18',
              'H:\\data\\BAYLORCW039\\python\\2024_04_17', 
              ],
             
             [ 'H:\\data\\BAYLORCW038\\python\\2024_02_15',
              'H:\\data\\BAYLORCW039\\python\\2024_04_25',
              'H:\\data\\BAYLORCW039\\python\\2024_04_24',
              ],
             
             ['H:\\data\\BAYLORCW038\\python\\2024_03_15',
              'H:\\data\\BAYLORCW039\\python\\2024_05_08',
              'H:\\data\\BAYLORCW039\\python\\2024_05_14']]

ticks = ["o", "X", "D"]
naive_sel_recovery,learning_sel_recovery,expert_sel_recovery = [],[],[]
all_recovery = []
for paths in all_paths: # For each stage of training
    recovery = []
    for path in paths: # For each mouse
        
        l1 = Mode(path)
        # l1 = Mode(path, use_reg=True)

        temp = l1.modularity_proportion_by_CD(period = range(l1.delay+2, l1.delay+8))
        recovery += [temp]
        
        # temp, _ = l1.modularity_proportion(period = range(l1.delay, l1.delay+6))
        # recovery += [temp]

        # if temp > 0 and temp < 1: # Exclude values based on Chen et al method guideliens
        #     recovery += [temp]
    
    all_recovery += [recovery]
        

plt.bar(range(3), [np.mean(a) for a in all_recovery])
# plt.scatter(np.zeros(len(all_recovery[0])), all_recovery[0])
# plt.scatter(np.ones(len(all_recovery[1])), all_recovery[1])
# plt.scatter(np.ones(len(all_recovery[2]))+1, all_recovery[2])

for i in range(len(all_paths)):
    
    plt.scatter([0,1,2], np.array(all_recovery)[:, i], marker = ticks[i], label = "FOV {}".format(i + 1))

plt.xticks(range(3), ['Before corruption', 'Midpoint', 'Final'])
plt.ylabel('Modularity')
plt.legend()

plt.show()

# Add t-test:

tstat, p_val = stats.ttest_ind(all_recovery[1], all_recovery[2], equal_var=False, permutations = np.inf, alternative='less')
print("mod diff p-value: ", p_val)

#%% Plot derivative of recovery as a bar graph

all_paths = [[r'H:\\data\\BAYLORCW038\\python\\2024_02_05',
               r'H:\\data\\BAYLORCW039\\python\\2024_04_17',
               r'H:\\data\\BAYLORCW039\\python\\2024_04_18',
               r'H:\\data\\BAYLORCW041\\python\\2024_05_14',
               r'H:\\data\\BAYLORCW041\\python\\2024_05_13',
               r'H:\\data\\BAYLORCW041\\python\\2024_05_15',
               r'H:\\data\\BAYLORCW043\\python\\2024_05_20',
               r'H:\\data\\BAYLORCW043\\python\\2024_05_21'],
              
               [r'H:\data\BAYLORCW038\python\2024_02_15',
                r'H:\\data\\BAYLORCW039\\python\\2024_04_24',
                r'H:\\data\\BAYLORCW039\\python\\2024_04_25',
                r'H:\\data\\BAYLORCW041\\python\\2024_05_23',
                r'H:\\data\\BAYLORCW041\\python\\2024_05_24',
                r'H:\\data\\BAYLORCW041\\python\\2024_05_28',
                r'H:\\data\\BAYLORCW043\\python\\2024_06_03',
                r'H:\\data\\BAYLORCW043\\python\\2024_06_04'],
              
               [r'H:\\data\\BAYLORCW038\\python\\2024_03_15',
                r'H:\\data\\BAYLORCW039\\python\\2024_05_06',
                r'H:\\data\\BAYLORCW039\\python\\2024_05_08',
                r'H:\\data\\BAYLORCW041\\python\\2024_06_07',
                r'H:\\data\\BAYLORCW041\\python\\2024_06_12',
                r'H:\\data\\BAYLORCW041\\python\\2024_06_11',
                r'H:\\data\\BAYLORCW043\\python\\2024_06_13',
                r'H:\\data\\BAYLORCW043\\python\\2024_06_14']
            ]

agg_mice_paths = [
                    [r'H:\data\BAYLORCW038\python\2024_02_05',
                     'H:\\data\\BAYLORCW039\\python\\2024_04_17',
                     'H:\\data\\BAYLORCW039\\python\\2024_04_18',
                     'H:\\data\\BAYLORCW041\\python\\2024_05_14', 
                     'H:\\data\\BAYLORCW041\\python\\2024_05_13', 
                     'H:\\data\\BAYLORCW041\\python\\2024_05_15', 
                     'H:\\data\\BAYLORCW043\\python\\2024_05_20', 
                     'H:\\data\\BAYLORCW043\\python\\2024_05_21', 
                     'H:\\data\\BAYLORCW042\\python\\2024_06_05', 
                     'H:\\data\\BAYLORCW042\\python\\2024_06_06', 
                     ],
                    
                    [r'H:\data\BAYLORCW038\python\2024_02_15',
                     'H:\\data\\BAYLORCW039\\python\\2024_04_24',
                     'H:\\data\\BAYLORCW039\\python\\2024_04_25',
                     'H:\\data\\BAYLORCW041\\python\\2024_05_23',
                     'H:\\data\\BAYLORCW041\\python\\2024_05_24',
                     'H:\\data\\BAYLORCW041\\python\\2024_05_28',
                     'H:\\data\\BAYLORCW043\\python\\2024_06_03',
                     'H:\\data\\BAYLORCW043\\python\\2024_06_06',
                     'H:\\data\\BAYLORCW043\\python\\2024_06_04',
                     'H:\\data\\BAYLORCW042\\python\\2024_06_14',
                     'H:\\data\\BAYLORCW042\\python\\2024_06_18',
                     ],
                    
                    [r'H:\data\BAYLORCW038\python\2024_03_15',
                     'H:\\data\\BAYLORCW039\\python\\2024_05_06',
                     'H:\\data\\BAYLORCW039\\python\\2024_05_08',
                     'H:\\data\\BAYLORCW041\\python\\2024_06_07',
                     'H:\\data\\BAYLORCW041\\python\\2024_06_12',
                     'H:\\data\\BAYLORCW041\\python\\2024_06_11',
                     'H:\\data\\BAYLORCW043\\python\\2024_06_13',
                     'H:\\data\\BAYLORCW043\\python\\2024_06_14',
                     'H:\\data\\BAYLORCW042\\python\\2024_06_24'
                     ]
    
                    ]

ticks = ["o", "X", "D"]
naive_sel_recovery,learning_sel_recovery,expert_sel_recovery = [],[],[]
all_recovery, all_recovery_ctl, all_diff = [], [], []
for paths in agg_mice_paths: # For each stage of training
    recovery = []
    r_ctl = []
    d_diff = []
    for path in paths: # For each mouse
        
        l1 = Mode(path)
        # l1 = Mode(path, use_reg=True)

        der,ctl,diff = l1.selectivity_derivative(period = range(l1.delay, l1.delay+5))
        recovery += [np.mean(der)]
        r_ctl += [np.mean(ctl)]
        d_diff += [np.mean(diff)]
        # temp, _ = l1.modularity_proportion(period = range(l1.delay, l1.delay+6))
        # recovery += [temp]

        # if temp > 0 and temp < 1: # Exclude values based on Chen et al method guideliens
        #     recovery += [temp]
    
    all_recovery += [recovery]
    all_recovery_ctl += [r_ctl]
    all_diff += [d_diff]

f = plt.figure(figsize=(7,5))
# plt.bar(range(3), [np.mean(a) for a in all_recovery])

plt.bar(np.arange(3)+0.2, [np.mean(a) for a in all_recovery], 0.4, label='Perturbation')
plt.scatter(np.zeros(len(all_recovery[0]))+0.2, all_recovery[0])
plt.scatter(np.ones(len(all_recovery[1]))+0.2, all_recovery[1])
plt.scatter(np.ones(len(all_recovery[2]))+1+0.2, all_recovery[2])


plt.bar(np.arange(3)-0.2, [np.mean(a) for a in all_recovery_ctl], 0.4, label = 'Control')
plt.scatter(np.zeros(len(all_recovery_ctl[0]))-0.2, all_recovery_ctl[0])
plt.scatter(np.ones(len(all_recovery_ctl[1]))-0.2, all_recovery_ctl[1])
plt.scatter(np.ones(len(all_recovery_ctl[2]))+1-0.2, all_recovery_ctl[2])



for i in range(len(all_paths[0])):
    
    plt.plot([-0.2, 0.2], [all_recovery_ctl[0][i], all_recovery[0][i]], color='grey', alpha = 0.5)
    plt.plot([0.8, 1.2], [all_recovery_ctl[1][i], all_recovery[1][i]], color='grey', alpha = 0.5)
    plt.plot([1.8, 2.2], [all_recovery_ctl[2][i], all_recovery[2][i]], color='grey', alpha = 0.5)


plt.xticks(range(3), ['Before corruption', 'Midpoint', 'Final'])
plt.ylabel('Derivative of selectivity')
plt.legend()

plt.show()


plt.bar(np.arange(3), [np.mean(a) for a in all_diff])
plt.scatter(np.zeros(len(all_diff[0])), all_diff[0])
plt.scatter(np.ones(len(all_diff[1])), all_diff[1])
plt.scatter(np.ones(len(all_diff[2]))+1, all_diff[2])



plt.xticks(range(3), ['Before corruption', 'Midpoint', 'Final'])
plt.ylabel('Diff in derivative of selectivity (pert-ctl)')

plt.show()



# Add t-test:

tstat, p_val = stats.ttest_ind(all_recovery_ctl[0], all_recovery[0], equal_var=False, permutations = np.inf, alternative='less')
print("mod diff p-value: ", p_val)
tstat, p_val = stats.ttest_ind(all_recovery_ctl[1], all_recovery[1], equal_var=False, permutations = np.inf, alternative='less')
print("mod diff p-value: ", p_val)
tstat, p_val = stats.ttest_ind(all_recovery_ctl[2], all_recovery[2], equal_var=False, permutations = np.inf, alternative='less')
print("mod diff p-value: ", p_val)




#%% Look at the contribution of individual neurons to coupling effect
agg_mice_paths = [
                    [
                        r'H:\data\BAYLORCW038\python\2024_02_05',
                     'H:\\data\\BAYLORCW039\\python\\2024_04_17',
                     'H:\\data\\BAYLORCW039\\python\\2024_04_18',
                     'H:\\data\\BAYLORCW041\\python\\2024_05_14', 
                     'H:\\data\\BAYLORCW041\\python\\2024_05_13', 
                     'H:\\data\\BAYLORCW041\\python\\2024_05_15', 
                     'H:\\data\\BAYLORCW042\\python\\2024_06_05', 
                     ],
                    
                    [
                        r'H:\data\BAYLORCW038\python\2024_03_15',

                        # r'H:\data\BAYLORCW038\python\2024_02_15',
                     'H:\\data\\BAYLORCW039\\python\\2024_04_24',
                     'H:\\data\\BAYLORCW039\\python\\2024_04_25',
                     'H:\\data\\BAYLORCW041\\python\\2024_05_23',
                     'H:\\data\\BAYLORCW041\\python\\2024_05_24',
                     'H:\\data\\BAYLORCW041\\python\\2024_05_28',
                     'H:\\data\\BAYLORCW042\\python\\2024_06_14',
                     ],
                    
                    [
                        r'H:\data\BAYLORCW038\python\2024_03_15',
                     'H:\\data\\BAYLORCW039\\python\\2024_05_06',
                     'H:\\data\\BAYLORCW039\\python\\2024_05_08',
                     'H:\\data\\BAYLORCW041\\python\\2024_06_07',
                     'H:\\data\\BAYLORCW041\\python\\2024_06_12',
                     'H:\\data\\BAYLORCW041\\python\\2024_06_11',
                     'H:\\data\\BAYLORCW042\\python\\2024_06_24'
                     ]
    
                    ]


all_stages_susc = []
all_pvals = []
for i in range(2):

    susc = []
    pvals = []
    
    for path in agg_mice_paths[i]:
    
        if '43' in path or '38' in path:
            l1 = session.Session(path, use_reg=True, triple=False)
        else:
            l1 = session.Session(path, use_reg=True, triple=True)
            
            
        all_susc, p = l1.susceptibility(period = range(l1.delay, l1.delay+int(1/l1.fs)))
    
        susc += [all_susc]
        pvals += [p]
        
    all_stages_susc += [susc]
    all_pvals += [pvals]
    
#%% Plot the change in susc over stages
fov = 4
f=plt.figure(figsize=(10,10))
plt.scatter(np.zeros(len(all_stages_susc[0][fov])), all_stages_susc[0][fov])
plt.scatter(np.ones(len(all_stages_susc[1][fov])), all_stages_susc[1][fov])
for i in range(len(all_stages_susc[0][fov])):
    plt.plot([0,1], [all_stages_susc[0][fov][i], all_stages_susc[1][fov][i]], color='grey')
plt.show()
# plt.scatter(all_stages_susc[0][fov], all_stages_susc[1][fov])
plt.show()
#%% Consider significance

# number of significantly susceptible cells over learning per FOV
learning_sig, expert_sig = [], []
for fov in range(len(agg_mice_paths[1])):
    learning_sig += [np.sum(all_pvals[0][fov])]
    expert_sig += [np.sum(all_pvals[1][fov])]

plt.scatter(np.zeros(len(learning_sig)), learning_sig)
plt.scatter(np.ones(len(expert_sig)), expert_sig)


#%% Replot with only significantly modulated neurons 

# for fov in range(len(agg_mice_paths[1])):
for fov in [4]:
    f=plt.figure(figsize=(10,10))

    keep_idx = np.where(all_pvals[0][fov])[0]
    keep_idx = np.append(keep_idx, np.where(all_pvals[1][fov])[0])

    plt.scatter(np.zeros(len(np.array(all_stages_susc[0][fov])[keep_idx])), np.array(all_stages_susc[0][fov])[keep_idx])

    plt.scatter(np.ones(len(np.array(all_stages_susc[1][fov])[keep_idx])), np.array(all_stages_susc[1][fov])[keep_idx])
    for i in keep_idx:
        plt.plot([0,1], [np.array(all_stages_susc[0][fov][i]), np.array(all_stages_susc[1][fov][i])], color='grey')
 
#%% Compare this to their contribution to the choice CD
path = agg_mice_paths[0][fov]
l1 = Mode(path, use_reg=True, triple=True)
mode_init, _ = l1.plot_CD()

path = agg_mice_paths[1][fov]
l1 = Mode(path, use_reg=True, triple=True)
mode_mid, _ = l1.plot_CD()
#%% 
# plt.scatter(mode_init, mode_mid)
incr = []
# for i in range(len(all_stages_susc[0][fov])):
for i in keep_idx:

    if all_stages_susc[0][fov][i] < all_stages_susc[1][fov][i]: # if becomes more suscp
        incr += [i]
        plt.scatter(all_stages_susc[0][fov][i], all_stages_susc[1][fov][i], color='red')
    else:
        plt.scatter(all_stages_susc[0][fov][i], all_stages_susc[1][fov][i], color='blue')

# plt.plot(range(20), range(20))

#%%
# plt.scatter(np.array(all_stages_susc[1][fov])[incr], mode_init[incr])
plt.scatter(mode_init[incr], mode_mid[incr])

#%% Correlate modularity with behaavioral recovery (do with more data points)



