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



#%% Plot behavior in opto trials over learning with left/right info
import session

paths = [r'F:\data\BAYLORCW032\python\2023_10_08',
          r'F:\data\BAYLORCW032\python\2023_10_16',
          r'F:\data\BAYLORCW032\python\2023_10_25',]

paths = [r'F:\data\BAYLORCW032\python\2023_10_05',
          r'F:\data\BAYLORCW032\python\2023_10_19',
          r'F:\data\BAYLORCW032\python\2023_10_24',]


# paths = [    r'F:\data\BAYLORCW034\python\2023_10_12',
#             r'F:\data\BAYLORCW034\python\2023_10_22',
#             r'F:\data\BAYLORCW034\python\2023_10_27',
#             r'F:\data\BAYLORCW034\python\2023_11_22']

paths = [r'F:\data\BAYLORCW036\python\2023_10_09',
            r'F:\data\BAYLORCW036\python\2023_10_19',
            r'F:\data\BAYLORCW036\python\2023_10_30',]

paths = [r'F:\data\BAYLORCW035\python\2023_10_12',
            r'F:\data\BAYLORCW035\python\2023_10_26',
            r'F:\data\BAYLORCW035\python\2023_12_16',]

paths = [r'F:\data\BAYLORCW037\python\2023_11_21',
            r'F:\data\BAYLORCW037\python\2023_11_28',
            r'F:\data\BAYLORCW037\python\2023_12_15',]

all_paths = [[r'F:\data\BAYLORCW032\python\2023_10_05',
          r'F:\data\BAYLORCW032\python\2023_10_19',
          r'F:\data\BAYLORCW032\python\2023_10_24',],


             [r'F:\data\BAYLORCW036\python\2023_10_09',
            r'F:\data\BAYLORCW036\python\2023_10_19',
            r'F:\data\BAYLORCW036\python\2023_10_30',],

        [r'F:\data\BAYLORCW035\python\2023_10_12',
            r'F:\data\BAYLORCW035\python\2023_10_26',
            r'F:\data\BAYLORCW035\python\2023_12_16',],

        [r'F:\data\BAYLORCW037\python\2023_11_21',
            r'F:\data\BAYLORCW037\python\2023_11_28',
            r'F:\data\BAYLORCW037\python\2023_12_15',]]

performance_opto = []
performance_ctl = []
fig = plt.figure()
for paths in all_paths:
    counter = -1

    opt, ctl = [],[]
    for path in paths:
        counter += 1
        l1 = session.Session(path)
        stim_trials = np.where(l1.stim_ON)[0]
        control_trials = np.where(~l1.stim_ON)[0]
        
        perf_right, perf_left, perf_all = l1.performance_in_trials(stim_trials)
        opt += [perf_all]
        plt.scatter(counter + 0.2, perf_right, c='b', marker='x')
        plt.scatter(counter + 0.2, perf_left, c='r', marker='x')
       
        perf_rightctl, perf_left, perf_all = l1.performance_in_trials(control_trials)
        ctl += [perf_all]
        plt.scatter(counter - 0.2, perf_rightctl, c='b', marker='o')
        plt.scatter(counter - 0.2, perf_left, c='r', marker='o')
        
    performance_opto += [opt]
    performance_ctl += [ctl]

plt.scatter(counter + 0.2, perf_right, c='b', marker='x', label="Perturbation trials")
plt.scatter(counter - 0.2, perf_rightctl, c='b', marker='o', label="Control trials")

plt.bar(np.arange(3)+0.2, np.mean(performance_opto, axis=0), 0.4, fill=False)

plt.bar(np.arange(3)-0.2, np.mean(performance_ctl, axis=0), 0.4, fill=False)

# plt.xticks(range(len(paths)), ["Naive", "Learning", "Expert"])
# plt.ylim([0.4,1])
plt.legend()
plt.savefig(r'F:\data\Fig 1\beh_opto.pdf')
plt.show()

#%% Plot behavior in opto trials over learning NO left/right info
all_paths = [[r'F:\data\BAYLORCW032\python\2023_10_05',
          r'F:\data\BAYLORCW032\python\2023_10_19',
          r'F:\data\BAYLORCW032\python\2023_10_24',],


             [r'F:\data\BAYLORCW036\python\2023_10_09',
            r'F:\data\BAYLORCW036\python\2023_10_19',
            r'F:\data\BAYLORCW036\python\2023_10_30',],
             
             [ r'F:\data\BAYLORCW034\python\2023_10_12',
            r'F:\data\BAYLORCW034\python\2023_10_22',
            r'F:\data\BAYLORCW034\python\2023_10_27'],

        [r'F:\data\BAYLORCW035\python\2023_10_12',
            r'F:\data\BAYLORCW035\python\2023_10_26',
            r'F:\data\BAYLORCW035\python\2023_12_16',],

        [r'F:\data\BAYLORCW037\python\2023_11_21',
            r'F:\data\BAYLORCW037\python\2023_11_28',
            r'F:\data\BAYLORCW037\python\2023_12_15',]]

performance_opto = []
performance_ctl = []
fig = plt.figure()
for paths in all_paths:
    counter = -1

    opt, ctl = [],[]
    for path in paths:
        counter += 1
        l1 = session.Session(path)
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

plt.xticks(range(3), ["Naive", "Learning", "Expert"])
# plt.ylim([0.4,1])
# plt.legend()
plt.savefig(r'F:\data\Fig 1\beh_opto.pdf')
plt.show()
    
#%% Plot learning progression

# b = behavior.Behavior('F:\data\Behavior data\BAYLORCW028\python_behavior', behavior_only=True)
# b.learning_progression(imaging=True)

# b = behavior.Behavior('F:\data\Behavior data\BAYLORCW032\python_behavior', behavior_only=True)
# # b.learning_progression(imaging=True)
# b.learning_progression(window=150, save =r'F:\data\Fig 1\CW32.pdf')

# b = behavior.Behavior('F:\data\Behavior data\BAYLORCW037\python_behavior', behavior_only=True)
# # b.learning_progression(imaging=True)
# b.learning_progression(window=200, save =r'F:\data\Fig 1\CW37.pdf')

# b = behavior.Behavior('F:\data\Behavior data\BAYLORCW021\python_behavior', behavior_only=True)
# b.learning_progression_no_EL(imaging=True)
# b.learning_progression_no_EL(window = 200,save=True)

# b = behavior.Behavior('F:\data\Behavior data\BAYLORCW021\python_behavior', behavior_only=True)
# b.learning_progression(window = 100)
# b.learning_progression(window = 100, imaging=True)

# b = behavior.Behavior(r'H:\data\Behavior data\BAYLORCW038\python_behavior', behavior_only=True)
# # b.learning_progression(window = 150, include_delay=False, color_background=[2,3,4,5,6,7,8,9, 11, 12, 14,15,16,17,18, 19, 20, 21,22,23])
# b.plot_performance_over_sessions(all=True, color_background=[2,3,4,5,6,7,8,9, 11, 12, 14,15,16,17,18, 19, 20, 21,22,23])

# b = behavior.Behavior(r'H:\data\Behavior data\BAYLORCW039\python_behavior', behavior_only=True)
# b.learning_progression(window = 50, include_delay=False, color_background=[2,3,4,5,8,9,10,11,12,13,14,15,16,20,21,22,23,24])
# b.plot_performance_over_sessions(all=True, color_background=[2,3,4,5,8,9,10,11,12,13,14,15,16,20,21,22,23,24])


b = behavior.Behavior(r'H:\data\Behavior data\BAYLORCW041\python_behavior', behavior_only=True)
b.learning_progression(window = 50, include_delay=False, color_background=[])
b.plot_performance_over_sessions(all=True, color_background=[])


# b = behavior.Behavior(r'H:\data\Behavior data\BAYLORCW040\python_behavior', behavior_only=True)
# b.learning_progression(window = 50)


#%% Compare learning curves ####

# window = 200
# b = behavior.Behavior('F:\data\Behavior data\BAYLORCW032\python_behavior', behavior_only=True)
# delays, acc_arr_32, numtrials_32 = b.learning_progression(return_results = True, window=window)


# b = behavior.Behavior('F:\data\Behavior data\BAYLORCW034\python_behavior', behavior_only=True)
# delays, acc_arr, numtrials = b.learning_progression(return_results = True, window=window)

# b = behavior.Behavior('F:\data\Behavior data\BAYLORCW036\python_behavior', behavior_only=True)
# delays, acc_arr_36, numtrials_36 = b.learning_progression(return_results = True, window=window)



# # New data 
# b = behavior.Behavior('F:\data\Behavior data\BAYLORCW035\python_behavior', behavior_only=True)
# delays, acc_arr_35, numtrials_35 = b.learning_progression(return_results = True, window=window)

# b = behavior.Behavior('F:\data\Behavior data\BAYLORCW037\python_behavior', behavior_only=True)
# delays, acc_arr_37, numtrials_37 = b.learning_progression(return_results = True, window=window)


# # Points
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

# # Lines
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


# #lines
# plt.plot(acc_arr_35[numtrials_35[0]:numtrials_35[4]] - acc_arr_35[numtrials_35[0]], color='r')
# plt.plot(acc_arr_35[numtrials_35[-3]:] - acc_arr_35[numtrials_35[-3]], color='r')
# plt.plot(acc_arr_35[numtrials_35[4]:numtrials_35[7]] - acc_arr_35[numtrials_35[4]], color='g')
# plt.plot(acc_arr_35[numtrials_35[23]:numtrials_35[27]] - acc_arr_35[numtrials_35[23]], color='g')

# plt.plot(acc_arr_37[numtrials_37[0]:numtrials_37[5]] - acc_arr_37[numtrials_37[0]], color='r')
# plt.plot(acc_arr_37[numtrials_37[10]:numtrials_37[13]] - acc_arr_37[numtrials_37[10]], color='r')
# plt.plot(acc_arr_37[numtrials_37[5]:numtrials_37[10]] - acc_arr_37[numtrials_37[5]], color='g')
# plt.plot(acc_arr_37[numtrials_37[13]:numtrials_37[15]] - acc_arr_37[numtrials_37[13]], color='g')

# #points

# plt.scatter(numtrials_35[4], (acc_arr_35[numtrials_35[0]:numtrials_35[4]] - acc_arr_35[numtrials_35[0]])[-1], marker='o', s=150, alpha = 0.5, color = 'r')
# plt.scatter(len(acc_arr_35) - numtrials_35[-3], (acc_arr_35[numtrials_35[-3]:] - acc_arr_35[numtrials_35[-3]])[-1], marker='o', s=150, alpha = 0.5, color = 'r')
# plt.scatter(numtrials_35[7] - numtrials_35[4], (acc_arr_35[numtrials_35[4]:numtrials_35[7]] - acc_arr_35[numtrials_35[4]])[-1], marker='o', s=150, alpha = 0.5, color = 'g')
# plt.scatter(numtrials_35[27] - numtrials_35[23], (acc_arr_35[numtrials_35[23]:numtrials_35[27]] - acc_arr_35[numtrials_35[23]])[-1], marker='o', s=150, alpha = 0.5, color = 'g')

# plt.scatter(numtrials_37[5], (acc_arr_37[numtrials_37[0]:numtrials_37[5]] - acc_arr_37[numtrials_37[0]])[-1], marker='o', s=150, alpha = 0.5, color = 'r')
# plt.scatter(numtrials_37[13] - numtrials_37[10], (acc_arr_37[numtrials_37[10]:numtrials_37[13]] - acc_arr_37[numtrials_37[10]])[-1], marker='o', s=150, alpha = 0.5, color = 'r')
# plt.scatter(numtrials_37[10] - numtrials_37[5], (acc_arr_37[numtrials_37[5]:numtrials_37[10]] - acc_arr_37[numtrials_37[4]])[-1], marker='o', s=150, alpha = 0.5, color = 'g')
# plt.scatter(numtrials_37[15] - numtrials_37[13], (acc_arr_37[numtrials_37[13]:numtrials_37[15]] - acc_arr_37[numtrials_37[13]])[-1], marker='o', s=150, alpha = 0.5, color = 'g')


# plt.ylabel('Change in performance accuracy')
# plt.xlabel('Number of trials')

# plt.legend()
# plt.savefig(r'F:\data\Fig 1\comparedelayleearning.pdf',transparent=True)
# plt.show()


#%% Compare all learning curves together

f, axarr = plt.subplots(2, 1, sharex='col', figsize=(10,6))

all_acc = []
all_EL = []
for idx in [34, 32, 36, 37]:
    b = behavior.Behavior('F:\data\Behavior data\BAYLORCW0{}\python_behavior'.format(idx), behavior_only=True)
    earlylicksarr, correctarr, _ = b.get_acc_EL(window=100)
    axarr[0].plot(correctarr, 'g', alpha =0.75)        
    axarr[0].set_ylabel('% correct')
    axarr[0].axhline(y=0.7, alpha = 0.5, color='orange')
    axarr[0].axhline(y=0.5, alpha = 0.5, color='red', ls = '--')
    axarr[0].set_ylim(0, 1)
    
    # Early licking
    
    axarr[1].plot(earlylicksarr, 'b', alpha=0.75)        
    axarr[1].set_ylabel('% Early licks')
    axarr[1].set_xlabel('Trials')
plt.savefig(r'F:\data\Fig 1\alllearningcurves.pdf',transparent=True)
plt.show()

#%% Separate learning curves by naive, learning, expert AGG all mice

f, axarr = plt.subplots(1, 3, sharex='col', figsize=(20,6))

all_acc = []
all_EL = []
mice_id = [32, 34, 36, 37, 35]

sess = [[(0,4),(0,2), (0,2),(0,5),(0,4)],
        [(4,11),(2,13),(2,15), (5,21), (4,40)],
        [(11,14),(13,15), (15,17),(21,24), (40,43)]
        ]

length = [4, 35, 4]
name = ['naive', 'learning', 'expert']
for i in range(3):
    fig = plt.figure(figsize =(length[i], 12)) 

    for idx in range(5):
        b = behavior.Behavior('F:\data\Behavior data\BAYLORCW0{}\python_behavior'.format(mice_id[idx]), 
                              behavior_only=True)
        _, correctarr, _ = b.get_acc_EL(window=150, sessions = sess[i][idx])
        
        plt.plot(correctarr, 'g', alpha =0.75)        
        plt.ylabel('% correct')
        plt.axhline(y=0.7, alpha = 0.5, color='orange')
        plt.axhline(y=0.5, alpha = 0.5, color='red', ls = '--')
        plt.ylim(0.18, 0.92)
        
        plt.scatter(len(correctarr), correctarr[-1], marker='o', s=650, alpha = 0.5, color = 'g')
    plt.savefig(r'F:\data\Fig 1\alllearningcurves_{}.pdf'.format(name[i]),transparent=True)

    plt.show()
        # axarr[i].plot(correctarr, 'g', alpha =0.75)        
        # axarr[i].set_ylabel('% correct')
        # axarr[i].axhline(y=0.7, alpha = 0.5, color='orange')
        # axarr[i].axhline(y=0.5, alpha = 0.5, color='red', ls = '--')
        # axarr[i].set_ylim(0.25, 0.92)
        
        # axarr[i].scatter(len(correctarr), correctarr[-1], marker='o', s=200, alpha = 0.5, color = 'g')
        
# plt.savefig(r'F:\data\Fig 1\alllearningcurves_sep.pdf',transparent=True)

plt.show()


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

b = behavior.Behavior('H:\\data\\BAYLORCW039\\python\\2024_05_15', single=True)
b.plot_single_session(save=True)


# b = behavior.Behavior('H:\\data\\BAYLORCW041\\python\\2024_05_14', single=True)
# b.plot_single_session(save=True)


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