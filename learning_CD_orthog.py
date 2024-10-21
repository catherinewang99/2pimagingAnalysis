# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 11:07:50 2024

@author: catherinewang

Look at the CD average in learning sessions and see if CD expert can be orthogonalized out
"""



import sys
sys.path.append("C:\scripts\Imaging analysis")
sys.path.append("Users/catherinewang/Desktop/Imaging analysis/2pimagingAnalysis/src")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from alm_2p import session
from activityMode import Mode
from matplotlib.pyplot import figure
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import stats
# from scipy.stats import norm
from sklearn import preprocessing

cat = np.concatenate
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
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def gs(X, row_vecs=True, norm = True):
    if not row_vecs:
        X = X.T
    Y = X[0:1,:].copy()
    for i in range(1, X.shape[0]):
        proj = np.diag((X[i,:].dot(Y.T)/np.linalg.norm(Y,axis=1)**2).flat).dot(Y)
        Y = np.vstack((Y, X[i,:] - proj.sum(0)))
    if norm:
        Y = np.diag(1/np.linalg.norm(Y,axis=1)).dot(Y)
    if row_vecs:
        return Y
    else:
        return Y.T
    
    
#%% PATHS
all_matched_paths = [
    
            [r'F:\data\BAYLORCW032\python\2023_10_05',
              r'F:\data\BAYLORCW032\python\2023_10_19',
              r'F:\data\BAYLORCW032\python\2023_10_24',
          ],
         
            [ r'F:\data\BAYLORCW034\python\2023_10_12',
                r'F:\data\BAYLORCW034\python\2023_10_22',
                r'F:\data\BAYLORCW034\python\2023_10_27',],
         
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
paths = [[r'F:\data\BAYLORCW032\python\2023_10_08',
          r'F:\data\BAYLORCW032\python\2023_10_16',
          r'F:\data\BAYLORCW032\python\2023_10_25',
          r'F:\data\BAYLORCW032\python\cellreg\layer{}\1008_1016_1025pairs_proc.npy'],
         
         [ r'F:\data\BAYLORCW034\python\2023_10_12',
            r'F:\data\BAYLORCW034\python\2023_10_22',
            r'F:\data\BAYLORCW034\python\2023_10_27',
            r'F:\data\BAYLORCW034\python\cellreg\layer{}\1012_1022_1027pairs_proc.npy'],
         
         [r'F:\data\BAYLORCW036\python\2023_10_09',
            r'F:\data\BAYLORCW036\python\2023_10_19',
            r'F:\data\BAYLORCW036\python\2023_10_30',
            r'F:\data\BAYLORCW036\python\cellreg\layer{}\1009_1019_1030pairs_proc.npy'],
        ]

naivepath =r'F:\data\BAYLORCW032\python\2023_10_05'
learningpath =  r'F:\data\BAYLORCW032\python\2023_10_19'
expertpath =r'F:\data\BAYLORCW032\python\2023_10_24'

naivepath, learningpath, expertpath = [ r'F:\data\BAYLORCW034\python\2023_10_12',
    r'F:\data\BAYLORCW034\python\2023_10_22',
    r'F:\data\BAYLORCW034\python\2023_10_27',]

# naivepath, learningpath, expertpath =[r'F:\data\BAYLORCW036\python\2023_10_09',
#             r'F:\data\BAYLORCW036\python\2023_10_19',
#             r'F:\data\BAYLORCW036\python\2023_10_30',]

# naivepath, learningpath, expertpath = [r'H:\data\BAYLORCW044\python\2024_05_23',
#                    r'H:\data\BAYLORCW044\python\2024_06_04',
#                   r'H:\data\BAYLORCW044\python\2024_06_18',]

naivepath, learningpath, expertpath = [r'H:\data\BAYLORCW044\python\2024_05_22',
                    r'H:\data\BAYLORCW044\python\2024_06_06',
                  r'H:\data\BAYLORCW044\python\2024_06_19',]

# naivepath, learningpath, expertpath = [r'H:\data\BAYLORCW046\python\2024_05_31',
#                     r'H:\data\BAYLORCW046\python\2024_06_11',
#                   r'H:\data\BAYLORCW046\python\2024_06_26']

#%%
path = expertpath
s1 = Mode(path, use_reg = True, triple=True)
orthonormal_basis, mean_exp = s1.plot_CD(ctl=True)

path = learningpath
s2 = Mode(path, use_reg = True, triple=True)
orthonormal_basis_learning, mean = s2.plot_CD(ctl=True)

# CD_final - CD_average; get: CD_final, CDaverage - CDfinal
exp_lea_stack = np.vstack((orthonormal_basis, orthonormal_basis_learning))
# orthog_exp_lea = s2.Gram_Schmidt_process(exp_lea_stack.T)
# Q, R = np.linalg.qr(exp_lea_stack.T, mode='complete')
# exp_CD = Q[0]
# lea_CD = Q[1]
exp_CD, lea_CD = gs(exp_lea_stack)

# Project onto data:
    
# CDfinal on expert
s1.plot_appliedCD(exp_CD, mean_exp)
# CDavg on expert
s1.plot_appliedCD(orthonormal_basis_learning, mean)
# CDleftover on expert
s1.plot_appliedCD(lea_CD, mean_exp)

# CDfinal on learning
s2.plot_appliedCD(exp_CD, mean_exp)
# CDavg on learning
s2.plot_appliedCD(orthonormal_basis_learning, mean)
# CDleftover on expert
s2.plot_appliedCD(lea_CD, mean)

# CD_average - CD_final; get: CD_average, CDfinal - CDaverage
lea_exp_stack = np.vstack((orthonormal_basis_learning, orthonormal_basis))
lea_CD, exp_CD = gs(lea_exp_stack)
# Q, R = np.linalg.qr(lea_exp_stack.T, mode='complete')
# lea_CD = Q[0]
# exp_CD = Q[1]
# Project onto data:
    
# CDfinal - CDaverage on expert
s1.plot_appliedCD(exp_CD, mean_exp)
# CDavg on expert
s1.plot_appliedCD(orthonormal_basis_learning, mean)
# CDavg on expert
s1.plot_appliedCD(lea_CD, mean_exp)

# CDfinal - CDaverage on learning
s2.plot_appliedCD(exp_CD, mean_exp)
# CDavg on learning
s2.plot_appliedCD(orthonormal_basis_learning, mean)
# CDavg on learning
s2.plot_appliedCD(lea_CD, mean)

#%% Look at decoding accuracy 
mode_input = 'choice'
pers=False
accs_expert = []
accs_learning = []

for paths in all_matched_paths:
    s1 = Mode(paths[2], use_reg = True, triple=True)
    orthonormal_basis, mean_exp = s1.plot_CD(ctl=True, plot=False)
    
    s2 = Mode(paths[1], use_reg = True, triple=True)
    orthonormal_basis_learning, mean = s2.plot_CD(ctl=True, plot=False)
    
    # CD_final - CD_average; get: CD_final, CDaverage - CDfinal
    # exp_lea_stack = np.vstack((orthonormal_basis_learning, orthonormal_basis))
    exp_lea_stack = np.vstack((orthonormal_basis, orthonormal_basis_learning))
    _, lea_CD = gs(exp_lea_stack)
    
    ### Get decoding accuracies ###
    
    orthonormal_basis, mean_exp, db, acc_final = s1.decision_boundary(mode_input=mode_input, persistence=pers, ctl=True)
    acc_average = s1.decision_boundary_appliedCD(mode_input, orthonormal_basis_learning, mean_exp, db, persistence=pers)
    acc_leftover = s1.decision_boundary_appliedCD(mode_input, lea_CD, mean_exp, db, persistence=pers)
    
    acc_final = np.mean(acc_final)
    acc_final = acc_final if acc_final > 0.5 else 1-acc_final
    acc_average = np.mean(acc_average)
    acc_average = acc_average if acc_average > 0.5 else 1-acc_average
    acc_leftover = np.mean(acc_leftover)
    acc_leftover = acc_leftover if acc_leftover > 0.5 else 1-acc_leftover
    accs_expert += [[acc_final, acc_average, acc_leftover]]
    
    
    _, mean, db, acc_average = s2.decision_boundary(mode_input=mode_input, persistence=pers, ctl=True)
    acc_final = s2.decision_boundary_appliedCD(mode_input, orthonormal_basis, mean_exp, db, persistence=pers)
    acc_leftover = s2.decision_boundary_appliedCD(mode_input, lea_CD, mean, db, persistence=pers)
    
    acc_final = np.mean(acc_final)
    acc_final = acc_final if acc_final > 0.5 else 1-acc_final
    acc_average = np.mean(acc_average)
    acc_average = acc_average if acc_average > 0.5 else 1-acc_average
    acc_leftover = np.mean(acc_leftover)
    acc_leftover = acc_leftover if acc_leftover > 0.5 else 1-acc_leftover
    accs_learning += [[acc_final, acc_average, acc_leftover]]

    
f = plt.figure()

plt.bar(np.arange(3)-0.2, np.mean(accs_learning, axis=0), 0.4, label='Learning')
plt.bar(np.arange(3)+0.2, np.mean(accs_expert, axis=0), 0.4, label='Expert')
for i in range(len(all_matched_paths)):
    plt.scatter(np.arange(3)-0.2, accs_learning[i])
    plt.scatter(np.arange(3)+0.2, accs_expert[i])
    for j in range(3):
        plt.plot([j-0.2, j+0.2], [accs_learning[i][j], accs_expert[i][j]], alpha=0.5, color='grey')

plt.xticks([0,1,2], ['CD_expert', 'CD_average', 'CD_average-CD_expert'])
plt.axhline(0.5, ls='--', color='grey')
plt.legend()
plt.ylabel('Decoding accuracy')

#%% Look at if the extra dimension in learning is less robust in opto trials
path = expertpath
s1 = Mode(path, use_reg = True, triple=True)
orthonormal_basis, mean_exp = s1.plot_CD(ctl=True)

path = learningpath
s2 = Mode(path, use_reg = True, triple=True)
orthonormal_basis_learning, mean = s2.plot_CD(ctl=True)

# CD_final - CD_average; get: CD_final, CDaverage - CDfinal
exp_lea_stack = np.vstack((orthonormal_basis, orthonormal_basis_learning))
exp_CD, lea_CD = gs(exp_lea_stack)

# Project onto data:
# Expert: 
# CDfinal on expert
orthonormal_basis, mean, meantrain, meanstd = s1.plot_CD_opto(return_traces = False, return_applied=True, ctl=True)
# CDavg on expert
s1.plot_CD_opto_applied(orthonormal_basis_learning, mean, meantrain, meanstd)
# CDleftover on expert
s1.plot_CD_opto_applied(lea_CD, mean, meantrain, meanstd)
    
# Learning: 
# CDavg on learning
orthonormal_basis_learning, mean, meantrain, meanstd = s2.plot_CD_opto(return_traces = False, return_applied=True, ctl=True)
# CDexp on learning
s2.plot_CD_opto_applied(exp_CD, mean, meantrain, meanstd)
# CDleftover on learning
s2.plot_CD_opto_applied(lea_CD, mean, meantrain, meanstd)
# What about CD_final-CDaverage?
lea_exp_stack = np.vstack((orthonormal_basis_learning, orthonormal_basis))
lea_CD, exp_CD = gs(lea_exp_stack)
s2.plot_CD_opto_applied(exp_CD, mean, meantrain, meanstd)

#%% Decoding accuracy: control vs opto, learning only

mode_input = 'choice'
pers=False
accs_control = []
accs_learning = []

for paths in all_matched_paths:
    
    s1 = Mode(paths[2], use_reg = True, triple=True, proportion_train=1)
    orthonormal_basis, mean_exp = s1.plot_CD(ctl=True, plot=False)
    
    s2 = Mode(paths[1], use_reg = True, triple=True, proportion_train=1)
    orthonormal_basis_learning, mean = s2.plot_CD(ctl=True, plot=False)
    
    # CD_final - CD_average; get: CD_final, CDaverage - CDfinal
    # exp_lea_stack = np.vstack((orthonormal_basis_learning, orthonormal_basis))
    exp_lea_stack = np.vstack((orthonormal_basis, orthonormal_basis_learning))
    _, lea_CD = gs(exp_lea_stack)
    
    ### Get decoding accuracies ###
    
    # Control
    _, mean, db, acc_average = s2.decision_boundary(mode_input=mode_input, persistence=pers, ctl=True)
    acc_final = s2.decision_boundary_appliedCD(mode_input, orthonormal_basis, mean_exp, db, persistence=pers)
    acc_leftover = s2.decision_boundary_appliedCD(mode_input, lea_CD, mean, db, persistence=pers)
    
    acc_final = np.mean(acc_final)
    acc_average = np.mean(acc_average)
    acc_leftover = np.mean(acc_leftover)
    accs_control += [[acc_final, acc_average, acc_leftover]]
    
    # Opto
    _, mean, db, acc_average = s2.decision_boundary(mode_input=mode_input, 
                                                    persistence=pers, ctl=True, opto=True)
    acc_final = s2.decision_boundary_appliedCD(mode_input, orthonormal_basis, mean_exp, db, persistence=pers, opto=True)
    acc_leftover = s2.decision_boundary_appliedCD(mode_input, lea_CD, mean, db, persistence=pers, opto=True)
    
    acc_final = np.mean(acc_final)
    acc_average = np.mean(acc_average)
    acc_leftover = np.mean(acc_leftover)
    accs_learning += [[acc_final, acc_average, acc_leftover]]

    
f = plt.figure()

plt.bar(np.arange(3)+0.2, np.mean(accs_learning, axis=0), 0.4, label='Opto')
plt.bar(np.arange(3)-0.2, np.mean(accs_control, axis=0), 0.4, label='Control')
for i in range(len(all_matched_paths)):
    plt.scatter(np.arange(3)+0.2, accs_learning[i])
    plt.scatter(np.arange(3)-0.2, accs_control[i])
    for j in range(3):
        plt.plot([j-0.2, j+0.2], [accs_control[i][j], accs_learning[i][j]], alpha=0.5, color='grey')

plt.xticks([0,1,2], ['CD_expert', 'CD_average', 'CD_leftover'])
plt.axhline(0.5, ls='--', color='grey')
plt.legend()
plt.ylabel('Decoding accuracy')

#%% Look at decoding accuracy on opto trials for different CDs

mode_input = 'choice'
pers=False
accs_expert = []
accs_learning = []

for paths in all_matched_paths:
    s1 = Mode(paths[2], use_reg = True, triple=True)
    orthonormal_basis, mean_exp = s1.plot_CD(ctl=True, plot=False)
    
    s2 = Mode(paths[1], use_reg = True, triple=True)
    orthonormal_basis_learning, mean = s2.plot_CD(ctl=True, plot=False)
    
    # CD_final - CD_average; get: CD_final, CDaverage - CDfinal
    # exp_lea_stack = np.vstack((orthonormal_basis_learning, orthonormal_basis))
    exp_lea_stack = np.vstack((orthonormal_basis, orthonormal_basis_learning))
    _, lea_CD = gs(exp_lea_stack)
    
    ### Get decoding accuracies ###
    
    orthonormal_basis, mean_exp, db, acc_final = s1.decision_boundary(mode_input=mode_input, persistence=pers, ctl=True, opto=True)
    acc_average = s1.decision_boundary_appliedCD(mode_input, orthonormal_basis_learning, mean_exp, db, persistence=pers, opto=True)
    acc_leftover = s1.decision_boundary_appliedCD(mode_input, lea_CD, mean_exp, db, persistence=pers, opto=True)
    
    acc_final = np.mean(acc_final)
    # acc_final = acc_final if acc_final > 0.5 else 1-acc_final
    acc_average = np.mean(acc_average)
    # acc_average = acc_average if acc_average > 0.5 else 1-acc_average
    acc_leftover = np.mean(acc_leftover)
    # acc_leftover = acc_leftover if acc_leftover > 0.5 else 1-acc_leftover
    accs_expert += [[acc_final, acc_average, acc_leftover]]
    
    
    _, mean, db, acc_average = s2.decision_boundary(mode_input=mode_input, persistence=pers, ctl=True, opto=True)
    acc_final = s2.decision_boundary_appliedCD(mode_input, orthonormal_basis, mean_exp, db, persistence=pers, opto=True)
    acc_leftover = s2.decision_boundary_appliedCD(mode_input, lea_CD, mean, db, persistence=pers, opto=True)
    
    acc_final = np.mean(acc_final)
    # acc_final = acc_final if acc_final > 0.5 else 1-acc_final
    acc_average = np.mean(acc_average)
    # acc_average = acc_average if acc_average > 0.5 else 1-acc_average
    acc_leftover = np.mean(acc_leftover)
    # acc_leftover = acc_leftover if acc_leftover > 0.5 else 1-acc_leftover
    accs_learning += [[acc_final, acc_average, acc_leftover]]

    
f = plt.figure()

plt.bar(np.arange(3)-0.2, np.mean(accs_learning, axis=0), 0.4, label='Learning')
plt.bar(np.arange(3)+0.2, np.mean(accs_expert, axis=0), 0.4, label='Expert')
for i in range(len(all_matched_paths)):
    plt.scatter(np.arange(3)-0.2, accs_learning[i])
    plt.scatter(np.arange(3)+0.2, accs_expert[i])
    for j in range(3):
        plt.plot([j-0.2, j+0.2], [accs_learning[i][j], accs_expert[i][j]], alpha=0.5, color='grey')

plt.xticks([0,1,2], ['CD_expert', 'CD_average', 'CD_average-CD_expert'])
plt.axhline(0.5, ls='--', color='grey')
plt.legend()
plt.ylabel('Decoding accuracy')



#%% Look at measures of robustness for different CDs
path = expertpath
s1 = Mode(path, use_reg = True, triple=True)
orthonormal_basis, mean_exp = s1.plot_CD(ctl=True)

path = learningpath
s2 = Mode(path, use_reg = True, triple=True)
orthonormal_basis_learning, mean = s2.plot_CD(ctl=True)

# CD_final - CD_average; get: CD_final, CDaverage - CDfinal
exp_lea_stack = np.vstack((orthonormal_basis, orthonormal_basis_learning))
exp_CD, lea_CD = gs(exp_lea_stack)

# Project onto data:
# Expert: 
# CDfinal on expert
orthonormal_basis, mean, meantrain, meanstd = s1.plot_CD_opto(return_traces = False, return_applied=True, ctl=True)
mod = s1.modularity_proportion_by_CD(period=range(s1.delay, s1.delay+int(1/s1.fs)))
rob = s1.modularity_proportion_by_CD(period=range(s1.delay-int(2*1/s1.fs), s1.response))
print('Mod/rob: ', mod, rob)

# CDavg on expert
s1.plot_CD_opto_applied(orthonormal_basis_learning, mean, meantrain, meanstd)
mod = s1.modularity_proportion_by_CD(period=range(s1.delay, s1.delay+int(1/s1.fs)),
                                     applied = (orthonormal_basis_learning, mean))
rob = s1.modularity_proportion_by_CD(period=range(s1.delay-int(2*1/s1.fs), s1.response),
                                     applied = (orthonormal_basis_learning, mean))
print('Mod/rob: ', mod, rob)
# CDleftover on expert
s1.plot_CD_opto_applied(lea_CD, mean, meantrain, meanstd)
mod = s1.modularity_proportion_by_CD(period=range(s1.delay, s1.delay+int(1/s1.fs)),
                                     applied = (lea_CD, mean))
rob = s1.modularity_proportion_by_CD(period=range(s1.delay-int(2*1/s1.fs), s1.response),
                                     applied = (lea_CD, mean))
print('Mod/rob: ', mod, rob) 
# Learning: 
# CDavg on learning
orthonormal_basis_learning, mean, meantrain, meanstd = s2.plot_CD_opto(return_traces=False, return_applied=True, ctl=True)
mod = s2.modularity_proportion_by_CD(period=range(s2.delay, s2.delay+int(1/s2.fs)))
rob = s2.modularity_proportion_by_CD(period=range(s2.delay-int(2*1/s2.fs), s2.response))
print('Mod/rob: ', mod, rob)
# CDexp on learning
s2.plot_CD_opto_applied(exp_CD, mean, meantrain, meanstd)
mod = s2.modularity_proportion_by_CD(period=range(s2.delay, s2.delay+int(1/s2.fs)),
                                     applied = (exp_CD, mean))
rob = s2.modularity_proportion_by_CD(period=range(s2.delay-int(2*1/s2.fs), s2.response),
                                     applied = (exp_CD, mean))
print('Mod/rob: ', mod, rob)
# CDleftover on learning
s2.plot_CD_opto_applied(lea_CD, mean, meantrain, meanstd)
mod = s2.modularity_proportion_by_CD(period=range(s2.delay, s2.delay+int(1/s2.fs)),
                                     applied = (lea_CD, mean))
rob = s2.modularity_proportion_by_CD(period=range(s2.delay-int(2*1/s2.fs), s2.response),
                                     applied = (lea_CD, mean))
print('Mod/rob: ', mod, rob)

#%% Measure robustness across different CDs in learning session

mode_input = 'choice'
pers=False
all_mod = []
all_rob = []

for paths in all_matched_paths:
    s1 = Mode(paths[2], use_reg = True, triple=True, proportion_train=1)
    orthonormal_basis, mean_exp = s1.plot_CD(ctl=True, plot=False)
    
    s2 = Mode(paths[1], use_reg = True, triple=True, proportion_train=1)
    orthonormal_basis_learning, mean = s2.plot_CD(ctl=True, plot=False)
    
    # CD_final - CD_average; get: CD_final, CDaverage - CDfinal
    # exp_lea_stack = np.vstack((orthonormal_basis_learning, orthonormal_basis))
    exp_lea_stack = np.vstack((orthonormal_basis, orthonormal_basis_learning))
    exp_CD, lea_CD = gs(exp_lea_stack)
    
    ### Get measures of robustness ###
    
    # CD expert    
    mod_exp = s2.modularity_proportion_by_CD(period=range(s2.delay, s2.delay+int(1/s2.fs)),
                                             applied = (exp_CD, mean_exp))
    rob_exp = s2.modularity_proportion_by_CD(period=range(s2.delay-int(2*1/s2.fs), s2.response),
                                         applied = (exp_CD, mean_exp))
    # CD average    
    mod_avg = s2.modularity_proportion_by_CD(period=range(s2.delay, s2.delay+int(1/s2.fs)))
    rob_avg = s2.modularity_proportion_by_CD(period=range(s2.delay-int(2*1/s2.fs), s2.response))

    # CD leftover    
    mod_left = s2.modularity_proportion_by_CD(period=range(s2.delay, s2.delay+int(1/s2.fs)),
                                              applied = (lea_CD, mean))
    rob_left = s2.modularity_proportion_by_CD(period=range(s2.delay-int(2*1/s2.fs), s2.response),
                                              applied = (lea_CD, mean))
    all_mod += [[mod_exp, mod_avg, mod_left]]
    all_rob += [[rob_exp, rob_avg, rob_left]]

f = plt.figure()
plt.bar(np.arange(3), np.mean(all_mod, axis=0))
for i in range(len(all_matched_paths)):
    plt.scatter(np.arange(3), all_mod[i])
    for j in range(3):
        plt.plot(np.arange(3), all_mod[i], alpha=0.2, color='grey')
plt.xticks([0,1,2], ['CD_expert', 'CD_average', 'CD_leftover'])
plt.ylim(bottom=-1, top=20)
plt.ylabel('Modularity')

f = plt.figure()
plt.bar(np.arange(3), np.mean(all_rob, axis=0))
for i in range(len(all_matched_paths)):
    plt.scatter(np.arange(3), all_rob[i])
    for j in range(3):
        plt.plot(np.arange(3), all_rob[i], alpha=0.2, color='grey')
plt.xticks([0,1,2], ['CD_expert', 'CD_average', 'CD_leftover'])
plt.ylabel('Robustness')