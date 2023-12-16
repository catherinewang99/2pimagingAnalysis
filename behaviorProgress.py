# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 15:29:46 2023

@author: Catherine Wang
"""

import sys
sys.path.append("C:\scripts\Imaging analysis")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import session
import behavior
plt.rcParams['pdf.fonttype'] = '42' 



#%% Plot behavior in opto trials over learning
import session

paths = [r'F:\data\BAYLORCW032\python\2023_10_08',
          r'F:\data\BAYLORCW032\python\2023_10_16',
          r'F:\data\BAYLORCW032\python\2023_10_25',]

paths = [r'F:\data\BAYLORCW032\python\2023_10_05',
          r'F:\data\BAYLORCW032\python\2023_10_19',
          r'F:\data\BAYLORCW032\python\2023_10_24',]


paths = [    r'F:\data\BAYLORCW034\python\2023_10_12',
            r'F:\data\BAYLORCW034\python\2023_10_22',
            r'F:\data\BAYLORCW034\python\2023_10_27',
            r'F:\data\BAYLORCW034\python\2023_11_22']

paths = [r'F:\data\BAYLORCW036\python\2023_10_09',
            r'F:\data\BAYLORCW036\python\2023_10_19',
            r'F:\data\BAYLORCW036\python\2023_10_30',]

paths = [r'F:\data\BAYLORCW035\python\2023_10_12',
            r'F:\data\BAYLORCW035\python\2023_10_26',
            r'F:\data\BAYLORCW035\python\2023_12_14',]

paths = [r'F:\data\BAYLORCW037\python\2023_11_21',
            r'F:\data\BAYLORCW037\python\2023_11_28',
            r'F:\data\BAYLORCW037\python\2023_12_15',]

performance_opto = []
performance_ctl = []
counter = -1
fig = plt.figure()

for path in paths:
    counter += 1
    l1 = session.Session(path)
    stim_trials = np.where(l1.stim_ON)[0]
    control_trials = np.where(~l1.stim_ON)[0]
    
    perf_right, perf_left, perf_all = l1.performance_in_trials(stim_trials)
    performance_opto += [perf_all]
    plt.scatter(counter + 0.2, perf_right, c='b', marker='x')
    plt.scatter(counter + 0.2, perf_left, c='r', marker='x')
   
    perf_rightctl, perf_left, perf_all = l1.performance_in_trials(control_trials)
    performance_ctl += [perf_all]
    plt.scatter(counter - 0.2, perf_rightctl, c='b', marker='o')
    plt.scatter(counter - 0.2, perf_left, c='r', marker='o')
   

plt.scatter(counter + 0.2, perf_right, c='b', marker='x', label="Perturbation trials")
plt.scatter(counter - 0.2, perf_rightctl, c='b', marker='o', label="Control trials")

plt.bar(np.arange(len(paths))+0.2, performance_opto, 0.4, fill=False)

plt.bar(np.arange(len(paths))-0.2, performance_ctl, 0.4, fill=False)

# plt.xticks(range(len(paths)), ["Naive", "Learning", "Expert"])
# plt.ylim([0.4,1])
plt.legend()
plt.show()


    
#%% Plot learning progression

# b = behavior.Behavior('F:\data\Behavior data\BAYLORCW028\python_behavior', behavior_only=True)
# b.learning_progression(imaging=True)

b = behavior.Behavior('F:\data\Behavior data\BAYLORCW035\python_behavior', behavior_only=True)
b.learning_progression(imaging=True)
b.learning_progression(window=150)

b = behavior.Behavior('F:\data\Behavior data\BAYLORCW037\python_behavior', behavior_only=True)
b.learning_progression(imaging=True)
b.learning_progression()

# b = behavior.Behavior('F:\data\Behavior data\BAYLORCW021\python_behavior', behavior_only=True)
# b.learning_progression_no_EL(imaging=True)
# b.learning_progression_no_EL(window = 200,save=True)

# b = behavior.Behavior('F:\data\Behavior data\BAYLORCW021\python_behavior', behavior_only=True)
# b.learning_progression(window = 100)
# b.learning_progression(window = 100, imaging=True)

# b = behavior.Behavior('F:\data\Behavior data\BAYLORCW027\python_behavior', behavior_only=True)
# b.learning_progression(window = 50)

#%% Compare learning curves ####

# window = 200
# b = behavior.Behavior('F:\data\Behavior data\BAYLORCW032\python_behavior', behavior_only=True)
# delays, acc_arr_32, numtrials_32 = b.learning_progression(return_results = True, window=window)


# b = behavior.Behavior('F:\data\Behavior data\BAYLORCW034\python_behavior', behavior_only=True)
# delays, acc_arr, numtrials = b.learning_progression(return_results = True, window=window)

# b = behavior.Behavior('F:\data\Behavior data\BAYLORCW036\python_behavior', behavior_only=True)
# delays, acc_arr_36, numtrials_36 = b.learning_progression(return_results = True, window=window)


# plt.scatter(numtrials_32[4], (acc_arr_32[numtrials_32[0]:numtrials_32[4]] - acc_arr_32[numtrials_32[0]])[-1], marker='o', s=150, alpha = 0.5, color = 'r')
# plt.scatter(numtrials_32[10] - numtrials_32[6], (acc_arr_32[numtrials_32[6]:numtrials_32[10]] - acc_arr_32[numtrials_32[6]])[-1], marker='o', s=150, alpha = 0.5, color = 'r')
# plt.scatter(numtrials_32[6] - numtrials_32[4], (acc_arr_32[numtrials_32[4]:numtrials_32[6]] - acc_arr_32[numtrials_32[4]])[-1], marker='o', s=150, alpha = 0.5, color = 'g')
# plt.scatter(numtrials_32[12] - numtrials_32[10], (acc_arr_32[numtrials_32[10]:numtrials_32[12]] - acc_arr_32[numtrials_32[10]])[-1], marker='o', s=150, alpha = 0.5, color = 'g')


# plt.scatter(numtrials_36[2], (acc_arr_36[numtrials_36[0]:numtrials_36[2]] - acc_arr_36[numtrials_36[0]])[-1], marker='o', s=150, alpha = 0.5, color = 'r')
# plt.scatter(numtrials_36[5] - numtrials_36[2], (acc_arr_36[numtrials_36[2]:numtrials_36[5]] - acc_arr_36[numtrials_36[2]])[-1], marker='o', s=150, alpha = 0.5, color = 'g')

# plt.scatter(numtrials[2], (acc_arr[numtrials[0]:numtrials[2]] - acc_arr[numtrials[0]])[-1], marker='o', s=150, alpha = 0.5, color = 'r')
# plt.scatter(numtrials[6] - numtrials[4], (acc_arr[numtrials[4]:numtrials[6]] - acc_arr[numtrials[4]])[-1], marker='o', s=150, alpha = 0.5, color = 'r')
# plt.scatter(numtrials[4] - numtrials[2], (acc_arr[numtrials[2]:numtrials[4]] - acc_arr[numtrials[2]])[-1], marker='o', s=150, alpha = 0.5, color = 'g')
# plt.scatter(numtrials[8] - numtrials[6], (acc_arr[numtrials[6]:numtrials[8]] - acc_arr[numtrials[6]])[-1], marker='o', s=150, alpha = 0.5, color = 'g')


# plt.plot(acc_arr_32[numtrials_32[0]:numtrials_32[4]] - acc_arr_32[numtrials_32[0]], color='r')
# plt.plot(acc_arr_32[numtrials_32[6]:numtrials_32[10]] - acc_arr_32[numtrials_32[6]], color='r')
# plt.plot(acc_arr_32[numtrials_32[4]:numtrials_32[6]] - acc_arr_32[numtrials_32[4]], color='g')
# plt.plot(acc_arr_32[numtrials_32[10]:numtrials_32[12]] - acc_arr_32[numtrials_32[10]], color='g')

# plt.plot(acc_arr_36[numtrials_36[0]:numtrials_36[2]] - acc_arr_36[numtrials_36[0]], color='r')
# plt.plot(acc_arr_36[numtrials_36[2]:numtrials_36[5]] - acc_arr_36[numtrials_36[2]], color='g')

# plt.plot(acc_arr[numtrials[0]:numtrials[2]] - acc_arr[numtrials[0]], color='r')
# plt.plot(acc_arr[numtrials[4]:numtrials[6]] - acc_arr[numtrials[4]], color='r', label='Full delay')
# plt.plot(acc_arr[numtrials[2]:numtrials[4]] - acc_arr[numtrials[2]], color='g', label='Varied delay')
# plt.plot(acc_arr[numtrials[6]:numtrials[8]] - acc_arr[numtrials[6]], color='g')


# plt.ylabel('Change in performance accuracy')
# plt.xlabel('Number of trials')

# plt.legend()
# plt.savefig(r'F:\data\SFN 2023\comparedelayleearning.pdf',transparent=True)
# plt.show()


#%% Compare all learning curves together

# f, axarr = plt.subplots(2, 1, sharex='col', figsize=(10,6))

# all_acc = []
# all_EL = []
# for idx in [34, 32, 36]:
#     b = behavior.Behavior('F:\data\Behavior data\BAYLORCW0{}\python_behavior'.format(idx), behavior_only=True)
#     earlylicksarr, correctarr, _ = b.get_acc_EL(window=100)
#     axarr[0].plot(correctarr, 'g', alpha =0.75)        
#     axarr[0].set_ylabel('% correct')
#     axarr[0].axhline(y=0.7, alpha = 0.5, color='orange')
#     axarr[0].axhline(y=0.5, alpha = 0.5, color='red', ls = '--')
#     axarr[0].set_ylim(0, 1)
    
#     # Early licking
    
#     axarr[1].plot(earlylicksarr, 'b', alpha=0.75)        
#     axarr[1].set_ylabel('% Early licks')
#     axarr[1].set_xlabel('Trials')
# plt.savefig(r'F:\data\SFN 2023\alllearningcurves.pdf',transparent=True)
# plt.show()

#%% Plot session to match GLM HMM
# sessions = ['20230215', '20230322', '20230323',  '20230403', '20230406', '20230409', '20230411',
#             '20230413', '20230420', '20230421', '20230423', '20230424', '20230427',
#             '20230426', '20230503', '20230508', '20230509', '20230510', '20230511', '20230512',
#             '20230516', '20230517']

# b = behavior.Behavior('F:\data\Behavior data\BAYLORCW021\python_behavior', behavior_only=True, glmhmm=sessions)
# b.learning_progression()



#%% Plot over all imaging sessions

# b = behavior.Behavior('F:\data\BAYLORCW022\python')
# b.plot_LR_performance_over_sessions()
# b.plot_early_lick()

# b = behavior.Behavior('F:\data\BAYLORCW021\python')
# b.plot_LR_performance_over_sessions()
# b.plot_early_lick()

# b = behavior.Behavior('F:\data\BAYLORCW030\python')
# b.plot_LR_performance_over_sessions()
# b.plot_early_lick()

#%% Plot single session performance - diagnostic session

# b = behavior.Behavior(r'F:\data\BAYLORCW022\python\2023_03_04', single=True)
# b.plot_single_session(save=True)

# b = behavior.Behavior(r'F:\data\BAYLORCW021\python\2023_05_17', single=True)
# b.plot_single_session_multidose(save=True)

# b = behavior.Behavior(r'F:\data\BAYLORCW027\python\2023_04_10', single=True)
# b.plot_single_session()

# b = behavior.Behavior(r'F:\data\BAYLORCW034\python\2023_10_11', single=True)
# b.plot_single_session()

# b = behavior.Behavior(r'F:\data\BAYLORCW021\python\2023_04_25', single=True)
# b.plot_single_session(save=True)

# b = behavior.Behavior(r'F:\data\BAYLORCW021\python\2023_05_23', single=True)
# b.plot_single_session_multidose(save=True)

# b = behavior.Behavior(r'F:\data\BAYLORCW021\python\2023_02_27', single=True)
# b.plot_single_session(save=True)

# b = behavior.Behavior(r'F:\data\BAYLORCW021\python\2023_03_03', single=True)
# b.plot_single_session(save=True)