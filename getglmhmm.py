# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 15:27:28 2023

@author: Catherine Wang

Get biased trials from GLM-HMM
"""


import sys
sys.path.append(r'C:\scripts\Behavior state\code\glmhmm')
sys.path.append(r'C:\scripts\Behavior state\CW21')

import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.model_selection import KFold
from glmhmm import glm_hmm
from glmhmm.utils import permute_states, find_best_fit, compare_top_weights
from glmhmm.visualize import plot_model_params, plot_loglikelihoods, plot_weights
from glmhmm.utils import convert_ll_bits, reshape_obs, compObs
import scipy.io as sio
import pandas as pd
import pickle
import matplotlib as mpl
from scipy.io import savemat
from glmhmm.visualize import plot_example_sessions

cat = np.concatenate

matrix = sio.loadmat(r'C:\scripts\Behavior state\CW21\matrix.mat')['aggall']
sess = sio.loadmat(r'C:\scripts\Behavior state\CW21\sessind.mat')['startstop']

instr = (cat(matrix[:, 0]))
choice_history = cat(cat(matrix[:, 1]))
rewarded_choice = cat(cat(matrix[:, 2]))
choice_output = cat(cat(matrix[:, 3]))
sess = cat(cat(cat(sess)))
earlylick = cat(cat(matrix[:, 4]))
responsetime = cat(cat(cat(cat(matrix[:, 5]))))
stimulus = np.array([1 if i == 'r' else -1 for i in instr ])

df2 = pd.DataFrame()
df2['CS'] = stimulus[:]
for i in range(1,2):
    df2['stim' + str(i)] = cat((np.zeros(i), stimulus[:-i]))
for i in range(1,2):
    df2['choice' + str(i)] = cat((np.zeros(i), choice_history[:-i]))
for i in range(1,2):
    df2['reward' + str(i)] = cat((np.zeros(i), rewarded_choice[:-i]))
df2['EL'] = earlylick
df2['bias'] = 1


x_d1 = df2.to_numpy().astype(float)
y_d1 = choice_output
sess = [s-1 for s in sess]
labels = df2.columns.to_numpy()



###### TRAIN CROSS VALIDATED MODEL #######

folds = 5
train_inds = [0,321,829,1355,1856,2238] # Naive days
# train_inds = [2634,3152,3660,4132/4570,4987] # Trained days
alltrials = range(2238)

best_fit_GLMHMMs_d1 = np.zeros((folds),dtype=object)


for j in range(folds):
    start, stop = train_inds[j], train_inds[j+1]-1
    inds = [i for i in alltrials if i not in range(start,stop)]
    x_train_d1, y_train_d1 = x_d1[inds], y_d1[inds] # Held out 4 sessions
    
    N = x_train_d1.shape[0]
    K = 3 # number of latent states
    C = 2 # number of observation classes
    D = x_train_d1.shape[1] # number of GLM inputs (regressors)

    # Instantiate the model
    real_GLMHMM = glm_hmm.GLMHMM(N,D,C,K,observations="bernoulli",gaussianPrior=1)
    
    inits = 20 # set the number of initializations
    maxiter = 250 # maximum number of iterations of EM to allow for each fit
    tol = 1e-3
    
    # store values for each initialization
    lls_all_d1 = np.zeros((inits,250))
    A_all_d1 = np.zeros((inits,K,K))
    w_all_d1 = np.zeros((inits,K,D,C))
    sess = None # Initialize this to evaluate separately for each sesssion

    # fit the model for each initialization
    print('\n fits for direct pathway cohort')
    for i in range(inits):
        t0 = time.time()
        # initialize the weights
        # weights input: shape of input, low and high of initial random weight distribution, last term is bias
        A_init,w_init,pi_init = real_GLMHMM.generate_params(weights=['GLM',-0.2,1.2,x_train_d1,y_train_d1,0])

        # fit the model                     
        lls_all_d1[i,:],A_all_d1[i,:,:],w_all_d1[i,:,:],pi0 = real_GLMHMM.fit(y_train_d1,x_train_d1,A_init,w_init,
                                                                                 maxiter=maxiter,sess=sess,tol=tol) 
        minutes = (time.time() - t0)/60
        print('initialization %s complete in %.2f minutes' %(i+1, minutes))

    # store results from best fit
#     bestix = find_best_fit(lls_all_d1)
    best_fit_GLMHMMs_d1[j] = real_GLMHMM

    fig, ax = plt.subplots()
    topixs = plot_loglikelihoods(lls_all_d1,0.1,ax,startix=5) # set the x-axis startix > 0 to see better view of final lls
    print('Number of top matching lls within threshold: ', len(topixs))

        
        
        # first, permute the weights according to the value of a particular regressor (here we pick cues) so that the states
    # will be the same for each fit 
    w_permuted = np.zeros_like(w_all_d1[:,:,:,1])
    order = np.zeros((inits,K))
    for i in range(inits):
        w_permuted[i],order[i] = permute_states(w_all_d1[i,:,:,1],method='weight value',param='weights',ix=1)

    np.set_printoptions(precision=2,suppress=True)
    # now let's check if the weights for the top fits match up
    compare_top_weights(w_permuted,topixs,tol=0.5)
    
    bestix = find_best_fit(lls_all_d1) # find the initialization that led to the best fit
    A_permuted, _ = permute_states(A_all_d1[bestix], method='order', order=order[bestix].astype(int))

    variance = real_GLMHMM.computeVariance(x_train_d1,y_train_d1,A_permuted,w_permuted[bestix,:,:,np.newaxis],gaussPrior=1)

    # plot the inferred transition probabilities
    fig, ax = plt.subplots(1,1)
    plot_model_params(A_permuted,ax,precision='%.3f')

    # plot the inferred weights probabilities
    fig, ax = plt.subplots(1,1)
    colors = np.array([[39,110,167],[237,177,32],[233,0,111],[176,100,245]])/255
    colors = ['red', 'orange', 'gold', 'green', 'blue', 'violet']

    xlabels = list(np.load('labels.npy', allow_pickle = True))
    legend = ['state 1', 'state 2', 'state 3', 'state 4', 'state 5', 'state 6']
    plot_weights(w_permuted[bestix],ax,xlabels=xlabels,switch=False,style='.-',color=colors,error=None,label=legend)
    # ax.text(0.43,-0.25,'prev choice',transform=ax.transAxes)
    ax.legend()
    
    colors = np.array([[39,110,167],[237,177,32],[233,0.0001,111]])/255
    colors = ['red', 'orange', 'gold', 'green', 'blue', 'violet']
    sessions_d1 = [0,N]

    # permute the order of the state probabilities
    _,order_d1 = permute_states(real_GLMHMM.w[:,:,1],method='weight value',param='weights',ix=1)
    pstate_permuted_d1,_ = permute_states(real_GLMHMM.pStates,method='order',param='pstates',order=order_d1)

    fig = plt.figure(figsize=(10,6))
    # format axes (start left, start bottom, width, height)
    # axes = [plt.axes([0, 0.7, 0.95, 0.20]),plt.axes([1.1, 0.7, 1, 0.20]),\
    #         plt.axes([0, 0.35, 0.95, 0.20]),plt.axes([1.1, 0.35, 1, 0.20]),\
    #         plt.axes([0, 0, 0.95, 0.20]),plt.axes([1.1, 0, 1, 0.20])]

    axes = plt.axes([0, 0.7, 0.95, 0.20])

    plot_example_sessions(pstate_permuted_d1,sessions_d1,axes,colors=colors)

###### TEST CROSS VALIDATED MODEL #######

# For each fold, plot the predicted state predictions:
from glmhmm.visualize import plot_example_sessions

for j in range(folds):
    
    start, stop = train_inds[j], train_inds[j+1]-1
    x_test_d1 = x_d1[start:stop]
    y_test_d1 = y_d1[start:stop]
    
    
    real_GLMHMM = best_fit_GLMHMMs_d1[j]

    fit_glmhmm,x,y = real_GLMHMM, x_test_d1, y_test_d1
    phi = np.zeros((len(y),fit_glmhmm.k,fit_glmhmm.c))

    for i in range(fit_glmhmm.k):
        phi[:,i,:] = compObs(x,fit_glmhmm.w[i])
    ll,alpha,alpha_prior,cs = fit_glmhmm.forwardPass(y,fit_glmhmm.A,phi)
    pred_choice = np.zeros((len(y)))


    pBack,beta,zhatBack = fit_glmhmm.backwardPass(y, fit_glmhmm.A, phi, alpha, cs)
    
    
    colors = np.array([[39,110,167],[237,177,32],[233,0.0001,111]])/255
    colors = ['red', 'orange', 'gold', 'green', 'blue', 'violet']
    sessions_d1 = [0,pBack.shape[0]]

    # permute the order of the state probabilities

    # _,order_d1 = permute_states(real_GLMHMM.w[:,:,1],method='weight value',param='weights',ix=1)
    # pstate_permuted_d1,_ = permute_states(real_GLMHMM.pStates,method='order',param='pstates',order=order_d1)

    fig = plt.figure(figsize=(10,6))
    # format axes (start left, start bottom, width, height)
    # axes = [plt.axes([0, 0.7, 0.95, 0.20]),plt.axes([1.1, 0.7, 1, 0.20]),\
    #         plt.axes([0, 0.35, 0.95, 0.20]),plt.axes([1.1, 0.35, 1, 0.20]),\
    #         plt.axes([0, 0, 0.95, 0.20]),plt.axes([1.1, 0, 1, 0.20])]

    axes = plt.axes([0, 0.7, 0.95, 0.20])

    plot_example_sessions(pBack,sessions_d1,axes,colors=colors)