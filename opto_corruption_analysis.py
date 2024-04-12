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
# p=0.0005
p=0.001

og_SDR = []
allstos = []
allnstos = []
s1list, d1, r1, ns1 = np.zeros(4),np.zeros(4),np.zeros(4),np.zeros(4)
for paths in agg_mice_paths: # For each mouse
    stos = []
    nstos = []
    s1 = Session(paths[0][0], use_reg=True, use_background_sub=False) # Naive
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
    s2 = Session(paths[0][1], use_reg=True) # Expert
    
    for n in naive_sample_sel:
        if s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], sample_epoch, p=p):
            s1list[0] += 1
            # stos += [(n, s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]])]  #save sample to sample cells

        elif s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], delay_epoch, p=p):
            s1list[1] += 1
        elif s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], response_epoch, p=p):
            s1list[2] += 1
        else:
            s1list[3] += 1
    
    for n in naive_delay_sel:
        if s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], sample_epoch, p=p):
            d1[0] += 1
        elif s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], delay_epoch, p=p):
            d1[1] += 1
            stos += [(n, s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]])]  #save sample to sample cells

        elif s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], response_epoch, p=p):
            d1[2] += 1
        else:
            d1[3] += 1
    
    
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
            # nstos += [(n, s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]])]  #save ns to sample cells

        elif s2.is_selective(s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]], delay_epoch, p=p):
            ns1[1] += 1
            nstos += [(n, s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]])]  #save ns to sample cells

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
stos = np.array(nstos)

intialpath, middlepath, finalpath = ['H:\\data\\BAYLORCW038\\python\\2024_02_05', 
                         r'H:\data\BAYLORCW038\python\2024_02_15',
                         'H:\\data\\BAYLORCW038\\python\\2024_03_15']

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

# sample CD

l1 = Mode(intialpath, use_reg=True)
orthonormal_basis_initial, mean = l1.plot_CD(mode_input = 'stimulus',ctl=True)
orthonormal_basis_initial_choice, mean = l1.plot_CD(mode_input = 'choice',ctl=True)

l1 = Mode(finalpath, use_reg = True)
orthonormal_basis, mean = l1.plot_CD(mode_input = 'stimulus',ctl=True)
orthonormal_basis_choice, mean = l1.plot_CD(mode_input = 'choice',ctl=True)

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

for path in paths:
    
    l1 = Session(path, use_reg=True)
    l1.selectivity_optogenetics()
    
#%% CD recovery


intialpath, middlepath, finalpath = ['H:\\data\\BAYLORCW038\\python\\2024_02_05', 
                         r'H:\data\BAYLORCW038\python\2024_02_15',
                         'H:\\data\\BAYLORCW038\\python\\2024_03_15']

    
l1 = Mode(intialpath, use_reg=True)
# orthonormal_basis, mean = l1.plot_CD(ctl=True)
l1.plot_CD_opto(ctl=True)
control_traces, opto_traces, error_bars, orthonormal_basis, mean, meantrain, meanstd = l1.plot_CD_opto(return_traces=True, return_applied=True,ctl=True)

# l1 = Mode(middlepath)
# l1.plot_CD_opto()

l1 = Mode(finalpath, use_reg = True)
# l1.plot_appliedCD(orthonormal_basis, mean)
# l1.plot_CD_opto()
l1.plot_CD_opto_applied(orthonormal_basis, mean, meantrain, meanstd)
                
#%% CD rotation
# TAKES A LONG TIME TO RUN!

intialpath, finalpath = ['H:\\data\\BAYLORCW038\\python\\2024_02_05', 
        'H:\\data\\BAYLORCW038\\python\\2024_03_15']

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

plt.ylim(-0.3, 1)
plt.xticks(range(2), ["Delay", "Stimulus"])
plt.xlabel('Choice decoder mode')
plt.ylabel('Rotation over corruption')

plt.show()

plt.scatter(angles, angles_stim)
plt.axhline(0, ls = '--')
plt.axvline(0, ls = '--')
plt.ylim(0, 1)
plt.xlim(-1, 1)
plt.xlabel('Choice decoder angles')
plt.ylabel('Stimulus decoder angles')

#%% Behavioral progress

b = behavior.Behavior(r'H:\data\Behavior data\BAYLORCW038\python_behavior', behavior_only=True)
b.learning_progression(window = 200)



#%% Behavioral recovery no L/R info

all_paths = [[r'H:\data\BAYLORCW038\python\2024_02_05',
          r'H:\data\BAYLORCW038\python\2024_02_15',
          r'H:\data\BAYLORCW038\python\2024_03_15',]]

performance_opto = []
performance_ctl = []
fig = plt.figure()
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


    plt.scatter(np.arange(3)+0.2, opt)
    plt.scatter(np.arange(3)-0.2, ctl)
    
    
plt.bar(np.arange(3)+0.2, np.mean(performance_opto, axis=0), 0.4, fill=False)

plt.bar(np.arange(3)-0.2, np.mean(performance_ctl, axis=0), 0.4, fill=False)

plt.xticks(range(3), ["Initial", "Day 7", "Day 30"])
# plt.ylim([0.4,1])
# plt.legend()
# plt.savefig(r'F:\data\Fig 1\beh_opto.pdf')
plt.show()

#%% Correlate modularity with behaavioral recovery (do with more data points)



