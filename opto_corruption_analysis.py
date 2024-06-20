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

#%% Contributions of neurons to CD before and after
intialpath, middlepath, finalpath = ['H:\\data\\BAYLORCW038\\python\\2024_02_05', 
                         r'H:\data\BAYLORCW038\python\2024_02_15',
                         'H:\\data\\BAYLORCW038\\python\\2024_03_15']
intialpath, finalpath = ['H:\\data\\BAYLORCW039\\python\\2024_04_24', 
                         'H:\\data\\BAYLORCW039\\python\\2024_05_06']
# sample CD

l1 = Mode(intialpath, use_reg=True)
orthonormal_basis_initial, mean = l1.plot_CD(mode_input = 'stimulus')
orthonormal_basis_initial_choice, mean = l1.plot_CD(mode_input = 'choice')

l1 = Mode(finalpath, use_reg = True)
orthonormal_basis, mean = l1.plot_CD(mode_input = 'stimulus')
orthonormal_basis_choice, mean = l1.plot_CD(mode_input = 'choice')

plt.scatter(orthonormal_basis_initial, orthonormal_basis)
plt.title('Pearsons correlation: {}, p-val: {}'.format(stats.pearsonr(orthonormal_basis_initial, orthonormal_basis)[0], 
                                                       stats.pearsonr(orthonormal_basis_initial, orthonormal_basis)[1]))
plt.xlabel('Initial sample CD values')
plt.ylabel('Final sample CD values')
plt.show()

# delay CD


plt.scatter(orthonormal_basis_initial_choice, orthonormal_basis_choice)
plt.title('Pearsons correlation: {}, p-val: {}'.format(stats.pearsonr(orthonormal_basis_initial_choice, orthonormal_basis_choice)[0], 
                                                       stats.pearsonr(orthonormal_basis_initial_choice, orthonormal_basis_choice)[1]))
plt.xlabel('Initial delay CD values')
plt.ylabel('Final delay CD values')
plt.show()

#%% Selectivity recovery

paths = ['H:\\data\\BAYLORCW038\\python\\2024_02_05', 
        'H:\\data\\BAYLORCW038\\python\\2024_03_15']
paths = ['H:\\data\\BAYLORCW039\\python\\2024_04_17', 
         'H:\\data\\BAYLORCW039\\python\\2024_04_24']
for path in paths:
    
    l1 = Session(path, use_reg=True)
    l1.selectivity_optogenetics()
    
#%% Selectivity recovery but use OR operation on neurons
paths = ['H:\\data\\BAYLORCW039\\python\\2024_04_24', 
         'H:\\data\\BAYLORCW039\\python\\2024_05_06']

paths = ['H:\\data\\BAYLORCW038\\python\\2024_02_05', 
        'H:\\data\\BAYLORCW038\\python\\2024_03_15']

paths = ['H:\\data\\BAYLORCW039\\python\\2024_04_18', 
         'H:\\data\\BAYLORCW039\\python\\2024_05_08']

paths = [r'H:\data\BAYLORCW041\python\2024_05_13',
   r'H:\data\BAYLORCW041\python\2024_05_24',
  r'H:\data\BAYLORCW041\python\2024_06_12']
paths = ['H:\\data\\BAYLORCW043\\python\\2024_06_06', 
        'H:\\data\\BAYLORCW043\\python\\2024_06_13']

# paths =  [r'H:\data\BAYLORCW041\python\2024_05_15',
#          r'H:\data\BAYLORCW041\python\2024_05_28',
#          r'H:\data\BAYLORCW041\python\2024_06_11',]
 
# paths = [r'H:\data\BAYLORCW041\python\2024_05_14',
# r'H:\data\BAYLORCW041\python\2024_05_23',
# r'H:\data\BAYLORCW041\python\2024_06_07',]
# for path in paths:
    
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
    
l1 = Mode(intialpath, use_reg=True)
orthonormal_basis, mean = l1.plot_CD(mode_input='choice')#, save = r'F:\data\Fig 2\CDstim_expert_CW37.pdf')

# l1 = Mode(middlepath)

l1 = Mode(finalpath, use_reg = True)
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
  
intialpath, _, finalpath =  [r'H:\data\BAYLORCW041\python\2024_05_15',
          r'H:\data\BAYLORCW041\python\2024_05_28',
          r'H:\data\BAYLORCW041\python\2024_06_11',]
 
intialpath, _, finalpath = [r'H:\data\BAYLORCW041\python\2024_05_14',
r'H:\data\BAYLORCW041\python\2024_05_23',
r'H:\data\BAYLORCW041\python\2024_06_07',]

intialpath, finalpath = ['H:\\data\\BAYLORCW043\\python\\2024_06_06', 
                         'H:\\data\\BAYLORCW043\\python\\2024_06_13']


l1 = Mode(finalpath, use_reg=True, triple=False, filter_reg=False)
# orthonormal_basis, mean = l1.plot_CD(ctl=True)
l1.plot_CD_opto(ctl=True)
control_traces, opto_traces, error_bars, orthonormal_basis, mean, meantrain, meanstd = l1.plot_CD_opto(return_traces=True, return_applied=True,ctl=True)

# l1 = Mode(middlepath, use_reg=True, triple=True, filter_reg=False)
# # l1.plot_CD_opto()
# l1.plot_CD_opto_applied(orthonormal_basis, mean, meantrain, meanstd)

l1 = Mode(intialpath, use_reg = True, triple=False, filter_reg=False)
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




#%% Behavioral recovery no L/R info

all_paths = [[r'H:\data\BAYLORCW038\python\2024_02_05',
          r'H:\data\BAYLORCW038\python\2024_02_15',
          r'H:\data\BAYLORCW038\python\2024_03_15',]]

all_paths = [['H:\\data\\BAYLORCW039\\python\\2024_04_17', 
            'H:\\data\\BAYLORCW039\\python\\2024_04_24',
            'H:\\data\\BAYLORCW039\\python\\2024_05_06'],
              ['H:\\data\\BAYLORCW039\\python\\2024_04_18', 
            'H:\\data\\BAYLORCW039\\python\\2024_04_25',
            'H:\\data\\BAYLORCW039\\python\\2024_05_20']]

performance_opto = []
performance_ctl = []
fig = plt.figure()
ticks = ["o", "X", "D"]
t = 0
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


    plt.scatter(np.arange(3)+0.2, opt, color = 'red', marker = ticks[t], label = "FOV {}".format(t + 1))
    plt.scatter(np.arange(3)-0.2, ctl, color = 'grey', marker = ticks[t])
    t += 1
    
plt.bar(np.arange(3)+0.2, np.mean(performance_opto, axis=0), 0.4, fill=False)

plt.bar(np.arange(3)-0.2, np.mean(performance_ctl, axis=0), 0.4, fill=False)
plt.ylabel('Behavior performance')
plt.xticks(range(3), ["Before corruption", "Midpoint", "Final"])
plt.ylim([0.4,1])
plt.legend()
plt.show()

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

ticks = ["o", "X", "D"]
naive_sel_recovery,learning_sel_recovery,expert_sel_recovery = [],[],[]
all_recovery, all_recovery_ctl = [], []
for paths in all_paths: # For each stage of training
    recovery = []
    r_ctl = []
    for path in paths: # For each mouse
        
        l1 = Mode(path)
        # l1 = Mode(path, use_reg=True)

        der,ctl,_ = l1.selectivity_derivative(period = range(l1.delay+2, l1.delay+8))
        recovery += [np.mean(der)]
        r_ctl += [np.mean(ctl)]
        # temp, _ = l1.modularity_proportion(period = range(l1.delay, l1.delay+6))
        # recovery += [temp]

        # if temp > 0 and temp < 1: # Exclude values based on Chen et al method guideliens
        #     recovery += [temp]
    
    all_recovery += [recovery]
    all_recovery_ctl += [r_ctl]

# plt.bar(range(3), [np.mean(a) for a in all_recovery])

plt.bar(np.arange(3)+0.2, [np.mean(a) for a in all_recovery], 0.4, label='Control')
plt.scatter(np.zeros(len(all_recovery[0]))+0.2, all_recovery[0])
plt.scatter(np.ones(len(all_recovery[1]))+0.2, all_recovery[1])
plt.scatter(np.ones(len(all_recovery[2]))+1+0.2, all_recovery[2])


plt.bar(np.arange(3)-0.2, [np.mean(a) for a in all_recovery_ctl], 0.4, label = 'Perturbation')
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

# Add t-test:

tstat, p_val = stats.ttest_ind(all_recovery_ctl[0], all_recovery[0], equal_var=False, permutations = np.inf, alternative='less')
print("mod diff p-value: ", p_val)
tstat, p_val = stats.ttest_ind(all_recovery_ctl[1], all_recovery[1], equal_var=False, permutations = np.inf, alternative='less')
print("mod diff p-value: ", p_val)
tstat, p_val = stats.ttest_ind(all_recovery_ctl[2], all_recovery[2], equal_var=False, permutations = np.inf, alternative='less')
print("mod diff p-value: ", p_val)






#%% Correlate modularity with behaavioral recovery (do with more data points)



