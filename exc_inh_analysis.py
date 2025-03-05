# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 10:38:51 2025

Use this to further flush out the difference between excited vs inhibited cells
Mixes susc and input vector analysis


@author: catherinewang
"""
import sys
sys.path.append("C:\scripts\Imaging analysis")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from alm_2p import session
from matplotlib.pyplot import figure
from scipy.stats import chisquare
import pandas as pd
from activityMode import Mode
from scipy import stats
cat=np.concatenate
from scipy.stats import pearsonr
from numpy.linalg import norm

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

all_paths = [[    
            r'F:\data\BAYLORCW032\python\2023_10_05',
            # r'F:\data\BAYLORCW034\python\2023_10_12',
            r'F:\data\BAYLORCW036\python\2023_10_09',
            r'F:\data\BAYLORCW035\python\2023_10_26',
            r'F:\data\BAYLORCW037\python\2023_11_21',
            
            r'F:\data\BAYLORCW036\python\2023_10_16',
            r'F:\data\BAYLORCW035\python\2023_10_12',
        r'F:\data\BAYLORCW035\python\2023_11_02',

        r'H:\data\BAYLORCW044\python\2024_05_22',
        r'H:\data\BAYLORCW044\python\2024_05_23',
        r'H:\data\BAYLORCW044\python\2024_05_24',
        
        r'H:\data\BAYLORCW046\python\2024_05_29',
        r'H:\data\BAYLORCW046\python\2024_05_30',
        r'H:\data\BAYLORCW046\python\2024_05_31',
            ],
             [r'F:\data\BAYLORCW032\python\2023_10_19',
            # r'F:\data\BAYLORCW034\python\2023_10_22',
            r'F:\data\BAYLORCW036\python\2023_10_19',
            r'F:\data\BAYLORCW035\python\2023_12_07',
            r'F:\data\BAYLORCW037\python\2023_12_08',

        r'F:\data\BAYLORCW032\python\2023_10_18',
        r'F:\data\BAYLORCW035\python\2023_10_25',
            r'F:\data\BAYLORCW035\python\2023_11_27',
            r'F:\data\BAYLORCW035\python\2023_11_29',
            r'F:\data\BAYLORCW037\python\2023_11_28',
            
        r'H:\data\BAYLORCW044\python\2024_06_06',
        r'H:\data\BAYLORCW044\python\2024_06_04',
        r'H:\data\BAYLORCW044\python\2024_06_03',
        r'H:\data\BAYLORCW044\python\2024_06_12',

        r'H:\data\BAYLORCW046\python\2024_06_07',
        r'H:\data\BAYLORCW046\python\2024_06_10',
        r'H:\data\BAYLORCW046\python\2024_06_11',
        r'H:\data\BAYLORCW046\python\2024_06_19',

        ],
        [r'F:\data\BAYLORCW032\python\2023_10_24',
            # r'F:\data\BAYLORCW034\python\2023_10_27',
            r'F:\data\BAYLORCW036\python\2023_10_30',
            r'F:\data\BAYLORCW035\python\2023_12_15',
            r'F:\data\BAYLORCW037\python\2023_12_15',
            
            r'F:\data\BAYLORCW036\python\2023_10_28',
        r'F:\data\BAYLORCW035\python\2023_12_12',
            r'F:\data\BAYLORCW035\python\2023_12_14',
            r'F:\data\BAYLORCW035\python\2023_12_16',
            r'F:\data\BAYLORCW037\python\2023_12_13',
            
            r'H:\data\BAYLORCW044\python\2024_06_19',
            r'H:\data\BAYLORCW044\python\2024_06_18',
            r'H:\data\BAYLORCW044\python\2024_06_17',
            
            r'H:\data\BAYLORCW046\python\2024_06_24',
            r'H:\data\BAYLORCW046\python\2024_06_27',
            r'H:\data\BAYLORCW046\python\2024_06_26',
            r'H:\data\BAYLORCW046\python\2024_06_25',

        ]]

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

all_matched_paths = [
    
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

            # [r'H:\data\BAYLORCW046\python\2024_05_29',
            #  r'H:\data\BAYLORCW046\python\2024_06_07',
            #  r'H:\data\BAYLORCW046\python\2024_06_24'],

            [r'H:\data\BAYLORCW046\python\2024_05_29',
             r'H:\data\BAYLORCW046\python\2024_06_24',
             r'H:\data\BAYLORCW046\python\2024_06_28'],

            [r'H:\data\BAYLORCW046\python\2024_05_30',
             r'H:\data\BAYLORCW046\python\2024_06_10',
             r'H:\data\BAYLORCW046\python\2024_06_27'],

            [r'H:\data\BAYLORCW046\python\2024_05_31',
             r'H:\data\BAYLORCW046\python\2024_06_11',
             r'H:\data\BAYLORCW046\python\2024_06_26'
             ]
         
        ]
#%% Distribution of excited vs supressed neurons

all_naive_susc, all_lea_susc, all_exp_susc = [],[],[]
p_naive, p_learning, p_expert = [],[],[]
p_s=0.01
p=0.01
by_trial_type = False
for paths in all_matched_paths: # For each mouse/FOV

    s1 = session.Session(paths[0], use_reg=True, triple=True) # Naive
    s2 = session.Session(paths[1], use_reg=True, triple=True) # Learning
    s3 = session.Session(paths[2], use_reg=True, triple=True) # Expert
    
    # p_s = p_s / len(s1.good_neurons)
    stim_period = range(s1.delay+int(0.5/s1.fs), s1.delay+int(1.3/s1.fs))
    
    sample_epoch = range(s1.sample, s1.delay)
    delay_epoch = range(s1.delay+int(1.5 * 1/s1.fs), s1.response)
    response_epoch = range(s1.response, s1.response + int(2*1/s1.fs))
    
    naive_susc_tval, naive_susc = s1.susceptibility(period = stim_period, p=p_s, by_trial_type=by_trial_type, exc_supr=True, flexible=True)
    lea_susc_tval, lea_susc = s2.susceptibility(period = stim_period, p=p_s, by_trial_type=by_trial_type, exc_supr=True, flexible=True)
    exp_susc_tval, exp_susc = s3.susceptibility(period = stim_period, p=p_s, by_trial_type=by_trial_type, exc_supr=True, flexible=True)

    # bucket = delay_epoch
    
    # naive_susc = [n for n in naive_susc if n in s1.get_epoch_selective(bucket, p=p)]
    # lea_susc = [n for n in lea_susc if n in s2.get_epoch_selective(bucket, p=p)]
    # exp_susc = [n for n in exp_susc if n in s3.get_epoch_selective(bucket, p=p)]
    
    # naive_delay = [n for n in s1.get_epoch_selective(delay_epoch, p=p) if n not in s1.get_epoch_selective(sample_epoch, p=p)]
    # learning_delay = [n for n in s2.get_epoch_selective(delay_epoch, p=p) if n not in s2.get_epoch_selective(sample_epoch, p=p)]
    # expert_delay = [n for n in s3.get_epoch_selective(delay_epoch, p=p) if n not in s3.get_epoch_selective(sample_epoch, p=p)]
    
    # naive_susc_tval = [naive_susc_tval[n] for n in range(len(naive_susc)) if naive_susc[n] in naive_delay]
    # lea_susc_tval = [lea_susc_tval[n] for n in range(len(lea_susc)) if lea_susc[n] in learning_delay]
    # exp_susc_tval = [exp_susc_tval[n] for n in range(len(exp_susc)) if exp_susc[n] in expert_delay]


    # Get the proportion that are excited
    if len(naive_susc_tval) == 0:
        all_naive_susc += [0]
    else:
        all_naive_susc += [sum(np.array(naive_susc_tval) < 0) / len(naive_susc_tval)]
        
    if len(lea_susc_tval) == 0:
        all_lea_susc += [0]
    else:
        all_lea_susc += [sum(np.array(lea_susc_tval) < 0)  / len(lea_susc_tval)]
        
    if len(exp_susc_tval) == 0:
        all_exp_susc += [0]
    else:       
        all_exp_susc += [sum(np.array(exp_susc_tval) < 0) / len(exp_susc_tval)]
    
    # all_naive_susc += [len(naive_susc) / len(s1.selective_neurons)]
    # all_lea_susc += [len(lea_susc)  / len(s2.selective_neurons)]
    # all_exp_susc += [len(exp_susc) / len(s3.selective_neurons)]
    
    # p_naive += [p_n]
    # p_learning += [p_l]
    # p_expert += [p_e]


p_naive = [np.array(p) for p in p_naive] 
p_learning = [np.array(p) for p in p_learning] 
p_expert = [np.array(p) for p in p_expert] 

# Plot as a bar graph the proportion

f=plt.figure()

plt.bar(range(3), [np.mean(all_naive_susc), np.mean(all_lea_susc), np.mean(all_exp_susc)])
plt.scatter(np.zeros(len(all_naive_susc)), all_naive_susc)
plt.scatter(np.ones(len(all_lea_susc)), all_lea_susc)
plt.scatter(np.ones(len(all_exp_susc))*2, all_exp_susc)

for i in range(len(all_exp_susc)):
    plt.plot([0,1], [all_naive_susc[i], all_lea_susc[i]], color='grey')
    plt.plot([1,2], [all_lea_susc[i], all_exp_susc[i]], color='grey')
    
plt.ylabel('Proportion of excited neurons')
plt.xticks([0,1,2], ['Naive', 'Learning', 'Expert'])
plt.ylim(bottom=0.5)
plt.title('Proportion of excited susceptible neurons over learning')
plt.show()


#%% Projections of input vector of exc inh cells and alignment w CD calculated only with susc neurons

CD_angle, rotation_learning = [], []
all_deltas = []
decoding_acc = []
cd_delta = []
input_cd = []
p_s = 0.05

for paths in all_matched_paths:

    l1 = session.Session(paths[1], use_reg=True, triple=True) # Learning
    stim_period = range(l1.delay+int(0.5/l1.fs), l1.delay+int(1.2/l1.fs))
    lea_susc_n, lea_susc_tval = l1.susceptibility(period = stim_period, p=p_s, return_n = True, exc_supr=True)
    good_neurons = np.array(lea_susc_n)[np.where(np.array(lea_susc_tval) > 0)[0]]
    if len(good_neurons) < 5:
        print(paths[1] + " no inh neurons")
        continue
    l1 = Mode(paths[1], lickdir=False, use_reg = True, triple=True, 
              good_neurons = good_neurons,
              proportion_train=1, proportion_opto_train=1)
    
    input_vector, delta = l1.input_vector(by_trialtype=False, plot=True, return_delta = True)
    cd_choice, _ = l1.plot_CD(mode_input='choice', plot=False)
    _, cd_delta_lea = l1.input_vector(by_trialtype=False, plot=True, return_delta = True, plot_ctl_opto=False)
    
    
    
    l2 = session.Session(paths[2], use_reg=True, triple=True) # Learning
    exp_susc_n, exp_susc_tval = l2.susceptibility(period = stim_period, p=p_s, return_n = True, exc_supr=True)
    good_neurons = np.array(exp_susc_n)[np.where(np.array(exp_susc_tval) > 0)[0]]
    
    l2 = Mode(paths[2], lickdir=False, use_reg = True, triple=True, 
              good_neurons = good_neurons,
              proportion_train=1, proportion_opto_train=1)

    input_vector_exp, delta_exp = l2.input_vector(by_trialtype=False, plot=True, return_delta = True)
    cd_choice_exp, _ = l2.plot_CD(mode_input='choice', plot=False)
    _, cd_delta_exp = l2.input_vector(by_trialtype=False, plot=True, return_delta = True, plot_ctl_opto=False)

    # Angle between trial type input vector and CD
    CD_angle += [(cos_sim(input_vector, cd_choice), cos_sim(input_vector_exp, cd_choice_exp))]
    # rotation_learning += [cos_sim(input_vector, input_vector_exp)]
    all_deltas += [(delta, delta_exp)]
    cd_delta += [(cd_delta_lea, cd_delta_exp)]
    input_cd += [(input_vector, input_vector_exp)]
    
CD_angle, rotation_learning, all_deltas, cd_delta = np.array(CD_angle), np.array(rotation_learning), np.array(all_deltas), np.array(np.abs(cd_delta))
# input_cd = np.array(input_cd)

#Plot
# Plot angle between choice CD and input vector

plt.bar([0,1],np.mean(CD_angle, axis=0))
plt.scatter(np.zeros(len(CD_angle)), np.array(CD_angle)[:, 0])
plt.scatter(np.ones(len(CD_angle)), np.array(CD_angle)[:, 1])
for i in range(len(CD_angle)):
    plt.plot([0,1],[CD_angle[i,0], CD_angle[i,1]], color='grey')
plt.xticks([0,1],['Learning','Expert'])
plt.ylabel('Dot product')
plt.title('Input vector alignment to choice CD')
plt.show()

# Plot the deltas over learning
plt.bar([0,1],np.mean(all_deltas, axis=0))
plt.scatter(np.zeros(len(all_deltas)), np.array(all_deltas)[:, 0])
plt.scatter(np.ones(len(all_deltas)), np.array(all_deltas)[:, 1])
# plt.scatter(np.ones(len(all_deltas))*2, np.array(all_deltas)[:, 2])
for i in range(len(all_deltas)):
    plt.plot([0,1],[all_deltas[i,0], all_deltas[i,1]], color='grey')
    # plt.plot([1,2],[all_deltas[i,1], all_deltas[i,2]], color='grey')
plt.xticks([0,1],['Learning','Expert'])
plt.ylabel('Delta (ctl-stim)')
plt.title('Delta of input vector btw control and stim condition')
plt.show()
stats.ttest_rel(np.array(all_deltas)[:, 0], np.array(all_deltas)[:, 1])


#%% Projections of input vector of exc inh cells and alignment w CD calculated  with all neurons; test on susc


CD_angle, CD_angle_filtered, rotation_learning = [], [], []
all_deltas = []
decoding_acc = []
cd_delta = []
input_cd = []
susc_n = []
p_s = 0.01

for paths in all_matched_paths:

    l1 = Mode(paths[1], lickdir=False, use_reg = True, triple=True, 
              proportion_train=1, proportion_opto_train=1)
    
    stim_period = range(l1.delay+int(0.5/l1.fs), l1.delay+int(1.5/l1.fs))
    lea_susc_n, lea_susc_tval = l1.susceptibility(period = stim_period, p=p_s, return_n = True, exc_supr=True, flexible=True)
    
    input_vector, delta = l1.input_vector(by_trialtype=False, plot=True, return_delta = True)
    cd_choice, _ = l1.plot_CD(mode_input='choice', plot=False)
    
    exc_n = np.array(lea_susc_n)[np.where(np.array(lea_susc_tval) < 0)[0]] # Excited only
    exc_n_idx = [np.where(l1.good_neurons == n)[0][0] for n in exc_n]
    
    if len(exc_n_idx) < 2:
        exc_n_idx = []

    
    l2 = Mode(paths[2], lickdir=False, use_reg = True, triple=True, 
              proportion_train=1, proportion_opto_train=1)
    exp_susc_n, exp_susc_tval = l2.susceptibility(period = stim_period, p=p_s, return_n = True, exc_supr=True,flexible=True)
    
    input_vector_exp, delta_exp = l2.input_vector(by_trialtype=False, plot=True, return_delta = True)
    cd_choice_exp, _ = l2.plot_CD(mode_input='choice', plot=False)

    exc_n_expert = np.array(exp_susc_n)[np.where(np.array(exp_susc_tval) < 0)[0]] # Excited only
    exc_n_idx_expert = [np.where(l2.good_neurons == n)[0][0] for n in exc_n_expert]
        
    if len(exc_n_idx_expert) < 2:
        exc_n_idx_expert = []
    
    # Angle between trial type input vector and CD
    CD_angle += [(cos_sim(input_vector, cd_choice), cos_sim(input_vector_exp, cd_choice_exp))]
    CD_angle_filtered += [(cos_sim(input_vector[exc_n_idx], cd_choice[exc_n_idx]), 
                           cos_sim(input_vector_exp[exc_n_idx_expert], cd_choice_exp[exc_n_idx_expert]))]
    # rotation_learning += [cos_sim(input_vector, input_vector_exp)]
    susc_n += [(exc_n_idx, exc_n_idx_expert)]
    all_deltas += [(delta, delta_exp)]
    # cd_delta += [(cd_delta_lea, cd_delta_exp)]
    input_cd += [(input_vector, input_vector_exp)]
    
CD_angle, rotation_learning, all_deltas, cd_delta = np.array(CD_angle), np.array(rotation_learning), np.array(all_deltas), np.array(np.abs(cd_delta))
CD_angle_filtered = np.array(CD_angle_filtered)

#Plot
# Plot angle between choice CD and input vector

plt.bar([0,1],np.mean(CD_angle, axis=0))
plt.scatter(np.zeros(len(CD_angle)), np.array(CD_angle)[:, 0])
plt.scatter(np.ones(len(CD_angle)), np.array(CD_angle)[:, 1])
for i in range(len(CD_angle)):
    plt.plot([0,1],[CD_angle[i,0], CD_angle[i,1]], color='grey')
plt.xticks([0,1],['Learning','Expert'])
plt.ylabel('Dot product')
plt.title('Input vector alignment to choice CD')
plt.show()

# Plot the deltas over learning
plt.bar([0,1],np.mean(all_deltas, axis=0))
plt.scatter(np.zeros(len(all_deltas)), np.array(all_deltas)[:, 0])
plt.scatter(np.ones(len(all_deltas)), np.array(all_deltas)[:, 1])
# plt.scatter(np.ones(len(all_deltas))*2, np.array(all_deltas)[:, 2])
for i in range(len(all_deltas)):
    plt.plot([0,1],[all_deltas[i,0], all_deltas[i,1]], color='grey')
    # plt.plot([1,2],[all_deltas[i,1], all_deltas[i,2]], color='grey')
plt.xticks([0,1],['Learning','Expert'])
plt.ylabel('Delta (ctl-stim)')
plt.title('Delta of input vector btw control and stim condition')
plt.show()
stats.ttest_rel(np.array(all_deltas)[:, 0], np.array(all_deltas)[:, 1])


# Plot angle between choice CD and input vector

plt.bar([0,1],np.nanmean(CD_angle_filtered, axis=0))
plt.scatter(np.zeros(len(CD_angle_filtered)), np.array(CD_angle_filtered)[:, 0])
plt.scatter(np.ones(len(CD_angle_filtered)), np.array(CD_angle_filtered)[:, 1])
for i in range(len(CD_angle_filtered)):
    plt.plot([0,1],[CD_angle_filtered[i,0], CD_angle_filtered[i,1]], color='grey')
plt.xticks([0,1],['Learning','Expert'])
plt.ylabel('Dot product')
plt.title('Input vector alignment to choice CD')
plt.show()






