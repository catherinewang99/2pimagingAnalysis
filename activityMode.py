# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 11:42:45 2023

@author: Catherine Wang
"""
import numpy as np
from numpy import concatenate as cat
import matplotlib.pyplot as plt
from scipy import stats
import copy
import scipy.io as scio
from sklearn.preprocessing import normalize
from session import Session
import sympy
import time

class Mode(Session):
    
    def __init__(self, path, layer_num='all', time_epochs = [7, 13, 28]):
        # Inherit all parameters and functions of session.py
        super().__init__(path, layer_num=layer_num) 
        
        for n in range(self.num_neurons):
            r, l = self.get_trace_matrix(n)
            r_err, l_err = self.get_trace_matrix_error(n)
            r_opto, l_opto = self.get_opto_trace_matrix(n)
            r_opto_err, l_opto_err = self.get_opto_trace_matrix(n, error=True)
            
            r_train, l_train, r_test, l_test = self.train_test_split_data(r, l)
            r_err_train, l_err_train, r_err_test, l_err_test = self.train_test_split_data(r_err, l_err)
            r_opto_train, l_opto_train, r_opto_test, l_opto_test = self.train_test_split_data(r_opto, l_opto)
            r_opto_err_train, l_opto_err_train, r_opto_err_test, l_opto_err_test = self.train_test_split_data(r_opto_err, l_opto_err)
            
            if n == 0:
                self.PSTH_r_train_correct = np.reshape(r_train, (1,-1))
                self.PSTH_l_train_correct = np.reshape(l_train, (1,-1))
                self.PSTH_r_train_error = np.reshape(r_err_train, (1,-1))
                self.PSTH_l_train_error = np.reshape(l_err_train, (1,-1))
                self.PSTH_r_train_opto = np.reshape(r_opto_train, (1,-1))
                self.PSTH_l_train_opto = np.reshape(l_opto_train, (1,-1))
                self.PSTH_r_train_opto_err = np.reshape(r_opto_err_train, (1,-1))
                self.PSTH_l_train_opto_err = np.reshape(l_opto_err_train, (1,-1))

                self.PSTH_r_test_correct = np.reshape(r_test, (1,-1))
                self.PSTH_l_test_correct = np.reshape(l_test, (1,-1))
                self.PSTH_r_test_error = np.reshape(r_err_test, (1,-1))
                self.PSTH_l_test_error = np.reshape(l_err_test, (1,-1))
                self.PSTH_r_test_opto = np.reshape(r_opto_test, (1,-1))
                self.PSTH_l_test_opto = np.reshape(l_opto_test, (1,-1))
                self.PSTH_r_test_opto_err = np.reshape(r_opto_err_test, (1,-1))
                self.PSTH_l_test_opto_err = np.reshape(l_opto_err_test, (1,-1))
            else:
                self.PSTH_r_train_correct = np.concatenate((self.PSTH_r_train_correct, np.reshape(r_train, (1,-1))), axis = 0)
                self.PSTH_l_train_correct = np.concatenate((self.PSTH_l_train_correct, np.reshape(l_train, (1,-1))), axis = 0)
                self.PSTH_r_train_error = np.concatenate((self.PSTH_r_train_error, np.reshape(r_err_train, (1,-1))), axis = 0)
                self.PSTH_l_train_error = np.concatenate((self.PSTH_l_train_error, np.reshape(l_err_train, (1,-1))), axis = 0)
                self.PSTH_r_train_opto = np.concatenate((self.PSTH_r_train_opto, np.reshape(r_opto_train, (1,-1))), axis = 0)
                self.PSTH_l_train_opto = np.concatenate((self.PSTH_l_train_opto, np.reshape(l_opto_train, (1,-1))), axis = 0)
                self.PSTH_r_train_opto_err = np.concatenate((self.PSTH_r_train_opto_err, np.reshape(r_opto_err_train, (1,-1))), axis = 0)
                self.PSTH_l_train_opto_err = np.concatenate((self.PSTH_l_train_opto_err, np.reshape(l_opto_err_train, (1,-1))), axis = 0)
                
                self.PSTH_r_test_correct = np.concatenate((self.PSTH_r_test_correct, np.reshape(r_test, (1,-1))), axis = 0)
                self.PSTH_l_test_correct = np.concatenate((self.PSTH_l_test_correct, np.reshape(l_test, (1,-1))), axis = 0)
                self.PSTH_r_test_error = np.concatenate((self.PSTH_r_test_error, np.reshape(r_err_test, (1,-1))), axis = 0)
                self.PSTH_l_test_error = np.concatenate((self.PSTH_l_test_error, np.reshape(l_err_test, (1,-1))), axis = 0)
                self.PSTH_r_test_opto = np.concatenate((self.PSTH_r_test_opto, np.reshape(r_opto_test, (1,-1))), axis = 0)
                self.PSTH_l_test_opto = np.concatenate((self.PSTH_l_test_opto, np.reshape(l_opto_test, (1,-1))), axis = 0)
                self.PSTH_r_test_opto_err = np.concatenate((self.PSTH_r_test_opto_err, np.reshape(r_opto_err_test, (1,-1))), axis = 0)
                self.PSTH_l_test_opto_err = np.concatenate((self.PSTH_l_test_opto_err, np.reshape(l_opto_err_test, (1,-1))), axis = 0)
        
        self.T_cue_aligned_sel = np.arange(self.time_cutoff)
        self.time_epochs = time_epochs
        
        self.start_t = 3
    
    def lick_incorrect_direction(self, direction):
        ## Returns list of indices of lick left correct trials
        
        # TODO: Check what error trials mean in this case
        if direction == 'l':
            idx = np.where(self.L_wrong == 1)[0]
        elif direction == 'r':
            idx = np.where(self.R_wrong == 1)[0]
        else:
            raise Exception("Sorry, only 'r' or 'l' input accepted!")
            
        early_idx = np.where(self.early_lick == 1)[0]
        
        idx = [i for i in idx if i not in early_idx]
        
        idx = [i for i in idx if i in self.i_good_trials]
        
        return idx
    
    def get_trace_matrix_error(self, neuron_num):
        
        ## Returns matrix of all trial firing rates of a single neuron for lick left
        ## and lick right trials. Firing rates are normalized with individual trial
        ## baselines as well as overall firing rate z-score normalized.
        
        right_trials = self.lick_incorrect_direction('r')
        left_trials = self.lick_incorrect_direction('l')
        
        # Filter out opto trials
        right_trials = [r for r in right_trials if not self.stim_ON[r]]
        left_trials = [r for r in left_trials if not self.stim_ON[r]]
        
        R_av_dff = []
        for i in right_trials:
            # R_av_dff += [self.normalize_by_baseline(self.dff[0, i][neuron_num, :self.time_cutoff])]
            R_av_dff += [self.dff[0, i][neuron_num, :self.time_cutoff]]

        L_av_dff = []
        for i in left_trials:
            # L_av_dff += [self.normalize_by_baseline(self.dff[0, i][neuron_num, :self.time_cutoff])]
            L_av_dff += [self.dff[0, i][neuron_num, :self.time_cutoff]]
            
            
        return R_av_dff, L_av_dff
    
    def get_opto_trace_matrix_error(self, neuron_num):
        
        
        right_trials = self.lick_incorrect_direction('r')
        left_trials = self.lick_incorrect_direction('l')
        
        # Filter for opto trials
        right_trials = [r for r in right_trials if self.stim_ON[r]]
        left_trials = [r for r in left_trials if self.stim_ON[r]]

        
        R_av_dff = []
        for i in right_trials:
            
            R_av_dff += [self.dff[0, i][neuron_num, :self.time_cutoff]]
        
        L_av_dff = []
        for i in left_trials:

            L_av_dff += [self.dff[0, i][neuron_num, :self.time_cutoff]]
            
        
            
        return R_av_dff, L_av_dff
    
    def basis_col(self, A):
        # Bases
    
        # basis_col(A) produces a basis for the subspace of Eucldiean n-space 
        # spanned by the vectors {u1,u2,...}, where the matrix A is formed from 
        # these vectors as its columns. That is, the subspace is the column space 
        # of A. The columns of the matrix that is returned are the basis vectors 
        # for the subspace. These basis vectors will be a subset of the original 
        # vectors. An error is returned if a basis for the zero vector space is 
        # attempted to be produced.
    
        # For example, if the vector space V = span{u1,u2,...}, where u1,u2,... are
        # row vectors, then set A to be [u1' u2' ...].
    
        # For example, if the vector space V = Col(B), where B is an m x n matrix,
        # then set A to be equal to B.
    
        matrix_size = np.shape(A)
    
        m = matrix_size[0]
        n = matrix_size[1]
    
        if np.array_equal(A, np.zeros((m,n))):
            raise ValueError('There does not exist a basis for the zero vector space.')
        elif n == 1:
            basis = A
        else:
            flag = 0
    
            if n == 2:
                multiple = A[0,1]/A[0,0]
                count = 0
    
                for i in range(m):
                    if A[i,1]/A[i,0] == multiple:
                        count = count + 1
    
                if count == m:
                    basis = A[:,0].reshape(-1, 1)
                    flag = 1
    
            if flag == 0:
                ref_A, pivot_columns = sympy.Matrix(A).rref() # double check if works
    
                B = np.zeros((m, len(pivot_columns)))
    
                for i in range(len(pivot_columns)):
                    B[:,i] = A[:,pivot_columns[i]]
    
                basis = B
    
        return basis
    
    def is_orthogonal_set(self, A):
        """
        Orthogonal Sets
    
        Determines if a set of vectors in Euclidean n-space is orthogonal. The matrix A
        is formed from these vectors as its columns. That is, the subspace spanned by the
        set of vectors is the column space of A. The value 1 is returned if the set is
        orthogonal. The value 0 is returned if the set is not orthogonal.
    
        For example, if the set of row vectors (u1, u2, ...) is to be determined for
        orthogonality, set A to be equal to np.array([u1, u2, ...]).T.
        """
        matrix_size = A.shape
        n = matrix_size[1]
        tolerance = 1e-10
    
        if n == 1:
            return 1
        else:
            G = A.T @ A - np.eye(n)
            if np.max(np.abs(G)) <= tolerance:
                return 1
            else:
                return 0
    
    def is_orthonormal_set(self, A):
        """
        Orthonormal Sets
    
        Determines if a set of vectors in Euclidean n-space is orthonormal. The matrix A
        is formed from these vectors as its columns. That is, the subspace spanned by the
        set of vectors is the column space of A. The value 1 is returned if the set is
        orthonormal. The value 0 is returned if the set is not orthonormal. An error is
        returned if a set containing only zero vectors is attempted to be determined for
        orthonormality.
    
        For example, if the set of row vectors (u1, u2, ...) is to be determined for
        orthonormality, set A to be equal to np.array([u1, u2, ...]).T.
        """
        matrix_size = A.shape
        m = matrix_size[0]
        n = matrix_size[1]
        tolerance = 1e-10
    
        if np.allclose(A, np.zeros((m, n))):
            raise ValueError('The set that contains just zero vectors cannot be orthonormal.')
        elif n == 1:
            if np.abs(np.linalg.norm(A[:, 0]) - 1) <= tolerance:
                return 1
            else:
                return 0
        else:
            if self.is_orthogonal_set(A) == 1:
                length_counter = 0
                for i in range(n):
                    if np.abs(np.linalg.norm(A[:, i]) - 1) <= tolerance:
                        length_counter += 1
    
                if length_counter == n:
                    return 1
                else:
                    return 0
            else:
                return 0



        
    def Gram_Schmidt_process(self, A):
        """
        Gram-Schmidt Process
        
        Gram_Schmidt_process(A) produces an orthonormal basis for the subspace of
        Eucldiean n-space spanned by the vectors {u1,u2,...}, where the matrix A 
        is formed from these vectors as its columns. That is, the subspace is the
        column space of A. The columns of the matrix that is returned are the 
        orthonormal basis vectors for the subspace. An error is returned if an
        orthonormal basis for the zero vector space is attempted to be produced.
    
        For example, if the vector space V = span{u1,u2,...}, where u1,u2,... are
        row vectors, then set A to be [u1' u2' ...].
    
        For example, if the vector space V = Col(B), where B is an m x n matrix,
        then set A to be equal to B.
        """
        matrix_size = np.shape(A)
    
        m = matrix_size[0]
        n = matrix_size[1]
    
        if np.array_equal(A, np.zeros((m,n))):
            raise ValueError('There does not exist any type of basis for the zero vector space.')
        elif n == 1:
            orthonormal_basis = A[:, 0]/np.linalg.norm(A[:, 0])
        else:
            flag = 0
    
            if self.is_orthonormal_set(A) == 1:
                self.orthonormal_basis = A
                flag = 1
    
            if flag == 0:
                if np.linalg.matrix_rank(A) != n:
                    A = self.basis_col(A)
                
                matrix_size = np.shape(A)
                m = matrix_size[0]
                n = matrix_size[1]
    
                orthonormal_basis = A[:, 0]/np.linalg.norm(A[:, 0])
    
                for i in range(1, n):
                    u = A[:, i]
                    v = np.zeros((m, 1))
    
                    for j in range(i):
                        v -= np.dot(u, orthonormal_basis[:, j]) * orthonormal_basis[:, j]
    
                    v_ = u + v
                    orthonormal_basis[:, i] = v_/np.linalg.norm(v_)
    
        return orthonormal_basis
    
    def train_test_split_data(self, r, l):
        
        # Splits data into train and test sets (50/50 split)
        
        r_idx, l_idx = np.random.permutation(np.arange(len(r))), np.random.permutation(np.arange(len(l)))
        
        r_train_idx, l_train_idx = r_idx[:round(len(r) / 2)], l_idx[:round(len(l) / 2)]
        
        r_test_idx, l_test_idx = r_idx[round(len(r) / 2):], l_idx[round(len(l) / 2):]
        
        r_train, l_train = np.mean(np.array(r)[r_train_idx], axis = 0), np.mean(np.array(l)[l_train_idx], axis = 0)
        r_test, l_test = np.mean(np.array(r)[r_test_idx], axis = 0), np.mean(np.array(l)[l_test_idx], axis = 0)
        
        return r_train, l_train, r_test, l_test
    
    def func_compute_activity_modes_DRT(self, input_, ctl=True):
    
        # Inputs: Left Right Correct Error traces of ALL neurons that are selective
        #           time stamps for analysis?
        #           time epochs
        # Outputs: Orthonormal basis (nxn) where n = # of neurons
        #           activity variance of each dimension (nx1)
        
        # Actual method uses SVD decomposition
        
        T_cue_aligned_sel = self.T_cue_aligned_sel 
        time_epochs = self.time_epochs
    
        t_sample = time_epochs[0]
        t_delay = time_epochs[1]
        t_response = time_epochs[2]
        
        if ctl:
            PSTH_yes_correct, PSTH_no_correct = input_
        else:
            PSTH_yes_correct, PSTH_no_correct, PSTH_yes_error, PSTH_no_error = input_
    
        activityRL = np.concatenate((PSTH_yes_correct, PSTH_no_correct), axis=1)
        activityRL = activityRL - np.mean(activityRL, axis=1, keepdims=True) # remove?
        u, s, v = np.linalg.svd(activityRL.T)
        proj_allDim = activityRL.T @ v
    
        # Variance of each dimension normalized
        var_s = np.square(np.diag(s[0:proj_allDim.shape[1]]))
        var_allDim = var_s / np.sum(var_s)
    
        # Relevant choice dims
        CD_stim_mode = [] # Sample period
        CD_choice_mode = [] # Late delay period
        CD_outcome_mode = [] # Response period
        CD_sample_mode = [] # wt during the first 400 ms of the sample epoch
        CD_delay_mode = []
        CD_go_mode = []
        Ramping_mode = []
        GoDirection_mode = [] # To calculate the go direction (GD), we subtracted (rlick-right, t + rlick-left, t)/2 after the Go cue (Tgo < t < Tgo + 0.1 s) from that before the Go cue (Tgo - 0.1 s < t < Tgo), followed by normalization by its own norm. 
        
        if ctl:
            
            wt = (PSTH_yes_correct - PSTH_no_correct)/2
            i_t = np.where((T_cue_aligned_sel > t_sample) & (T_cue_aligned_sel < t_delay))[0]
            CD_stim_mode = np.mean(wt[:, i_t], axis=1)
        
            wt = (PSTH_yes_correct - PSTH_no_correct)/2
            i_t = np.where((T_cue_aligned_sel > t_delay) & (T_cue_aligned_sel < t_response))[0]
            CD_choice_mode = np.mean(wt[:, i_t], axis=1)
            
            wt = (PSTH_yes_correct + PSTH_no_correct)/2
            i_t = np.where((T_cue_aligned_sel > t_response) & (T_cue_aligned_sel < (t_response + 12)))[0]
            CD_outcome_mode = np.mean(wt[:, i_t], axis=1)
            
           
            wt = (PSTH_yes_correct - PSTH_no_correct)/2
            i_t = np.where((T_cue_aligned_sel > (t_sample + 1)) & (T_cue_aligned_sel < (t_sample + 3)))[0]
            CD_sample_mode = np.mean(wt[:, i_t], axis=1)
            
            i_t = np.where((T_cue_aligned_sel > (t_response - 3)) & (T_cue_aligned_sel < (t_response - 1)))[0]
            CD_delay_mode = np.mean(wt[:, i_t], axis=1)
            
            i_t = np.where((T_cue_aligned_sel > (t_response + 1)) & (T_cue_aligned_sel < (t_response + 3)))[0]
            CD_go_mode = np.mean(wt[:, i_t], axis=1)
            
            wt = (PSTH_yes_correct + PSTH_no_correct)/2
            i_t1 = np.where((T_cue_aligned_sel > (t_sample-3)) & (T_cue_aligned_sel < (t_sample-1)))[0]
            i_t2 = np.where((T_cue_aligned_sel > (t_response-3)) & (T_cue_aligned_sel < (t_response-1)))[0]
            Ramping_mode = np.mean(wt[:, i_t2], axis=1) - np.mean(wt[:, i_t1], axis=1)
            
            i_t1 = np.where((T_cue_aligned_sel > (t_response-2)) & (T_cue_aligned_sel < t_response))[0]
            i_t2 = np.where((T_cue_aligned_sel > t_response) & (T_cue_aligned_sel < (t_response+2)))[0]
            GoDirection_mode = np.mean(wt[:, i_t2], axis=1) - np.mean(wt[:, i_t1], axis=1)
            
        elif not ctl:
            
            wt = (PSTH_yes_correct + PSTH_yes_error) / 2 - (PSTH_no_correct + PSTH_no_error) / 2
            i_t = np.where((T_cue_aligned_sel > t_sample) & (T_cue_aligned_sel < t_delay))[0]
            CD_stim_mode = np.mean(wt[:, i_t], axis=1)
        
            wt = (PSTH_yes_correct + PSTH_no_error) / 2 - (PSTH_no_correct + PSTH_yes_error) / 2
            i_t = np.where((T_cue_aligned_sel > t_delay) & (T_cue_aligned_sel < t_response))[0]
            CD_choice_mode = np.mean(wt[:, i_t], axis=1)
            
            wt = (PSTH_yes_correct + PSTH_no_correct) / 2 - (PSTH_yes_error + PSTH_no_error) / 2
            i_t = np.where((T_cue_aligned_sel > t_response) & (T_cue_aligned_sel < (t_response + 12)))[0]
            CD_outcome_mode = np.mean(wt[:, i_t], axis=1)
            
           
            wt = PSTH_yes_correct - PSTH_no_correct
            i_t = np.where((T_cue_aligned_sel > (t_sample + 1)) & (T_cue_aligned_sel < (t_sample + 3)))[0]
            CD_sample_mode = np.mean(wt[:, i_t], axis=1)
            
            i_t = np.where((T_cue_aligned_sel > (t_response - 3)) & (T_cue_aligned_sel < (t_response - 1)))[0]
            CD_delay_mode = np.mean(wt[:, i_t], axis=1)
            
            i_t = np.where((T_cue_aligned_sel > (t_response + 1)) & (T_cue_aligned_sel < (t_response + 3)))[0]
            CD_go_mode = np.mean(wt[:, i_t], axis=1)
            
            wt = (PSTH_yes_correct + PSTH_no_correct)/2
            i_t1 = np.where((T_cue_aligned_sel > (t_sample-3)) & (T_cue_aligned_sel < (t_sample-1)))[0]
            i_t2 = np.where((T_cue_aligned_sel > (t_response-3)) & (T_cue_aligned_sel < (t_response-1)))[0]
            Ramping_mode = np.mean(wt[:, i_t2], axis=1) - np.mean(wt[:, i_t1], axis=1)
            
            i_t1 = np.where((T_cue_aligned_sel > (t_response-2)) & (T_cue_aligned_sel < t_response))[0]
            i_t2 = np.where((T_cue_aligned_sel > t_response) & (T_cue_aligned_sel < (t_response+2)))[0]
            GoDirection_mode = np.mean(wt[:, i_t2], axis=1) - np.mean(wt[:, i_t1], axis=1)

        
        CD_stim_mode = CD_stim_mode / np.linalg.norm(CD_stim_mode)
        CD_choice_mode = CD_choice_mode / np.linalg.norm(CD_choice_mode)
        CD_outcome_mode = CD_outcome_mode / np.linalg.norm(CD_outcome_mode)
        CD_sample_mode = CD_sample_mode / np.linalg.norm(CD_sample_mode)
        CD_delay_mode = CD_delay_mode / np.linalg.norm(CD_delay_mode)
        CD_go_mode = CD_go_mode / np.linalg.norm(CD_go_mode)
        Ramping_mode = Ramping_mode / np.linalg.norm(Ramping_mode)
        GoDirection_mode = GoDirection_mode / np.linalg.norm(GoDirection_mode)
        
        # Reshape all activity modes
        
        CD_stim_mode = np.reshape(CD_stim_mode, (-1, 1)) 
        CD_choice_mode = np.reshape(CD_choice_mode, (-1, 1)) 
        CD_outcome_mode = np.reshape(CD_outcome_mode, (-1, 1))
        CD_sample_mode = np.reshape(CD_sample_mode, (-1, 1)) 
        CD_delay_mode = np.reshape(CD_delay_mode, (-1, 1)) 
        CD_go_mode = np.reshape(CD_go_mode, (-1, 1)) 
        Ramping_mode = np.reshape(Ramping_mode, (-1, 1)) 
        GoDirection_mode = np.reshape(GoDirection_mode, (-1, 1)) 
        
        start_time = time.time()
        input_ = np.concatenate((CD_stim_mode, CD_choice_mode, CD_outcome_mode, CD_sample_mode, CD_delay_mode, CD_go_mode, Ramping_mode, GoDirection_mode, v), axis=1)
        # orthonormal_basis = self.Gram_Schmidt_process(input_)
        orthonormal_basis, _ = np.linalg.qr(input_, mode='complete')  # lmao
        
        proj_allDim = np.dot(activityRL.T, orthonormal_basis)
        var_allDim = np.sum(proj_allDim**2, axis=0)
        var_allDim = var_allDim[~np.isnan(var_allDim)]
        
        var_allDim = var_allDim / np.sum(var_allDim)
        
        print("Runtime: {} secs".format(time.time() - start_time))
        return orthonormal_basis, var_allDim

    
    def plot_activity_modes_err(self):
        # plot activity modes
        # all trials
        

        orthonormal_basis, var_allDim = self.func_compute_activity_modes_DRT([self.PSTH_r_train_correct, 
                                                                            self.PSTH_l_train_correct, 
                                                                            self.PSTH_r_train_error, 
                                                                            self.PSTH_l_train_error], ctl=False)
        
        activityRL_train= np.concatenate((self.PSTH_r_train_correct, 
                                        self.PSTH_l_train_correct, 
                                        self.PSTH_r_train_error, 
                                        self.PSTH_l_train_error), axis=1)
    
        activityRL_test= np.concatenate((self.PSTH_r_test_correct, 
                                        self.PSTH_l_test_correct), axis=1)
        
        activityRLerr_test = np.concatenate((self.PSTH_r_test_error, 
                                             self.PSTH_l_test_error), axis = 1)
        
        
        T_cue_aligned_sel = self.T_cue_aligned_sel
        

        
        
        # Correct trials
        activityRL_test = activityRL_test - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activityRL_test.shape[1]))  # remove mean
        proj_allDim = np.dot(activityRL_test.T, orthonormal_basis)
        
        # Error trials
        activityRLerr_test = activityRLerr_test - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activityRLerr_test.shape[1]))  # remove mean
        proj_allDim_err = np.dot(activityRLerr_test.T, orthonormal_basis)
        
        fig, axs = plt.subplots(4, 4, figsize=(12, 16))
        for i_pc in range(16):
            ax = axs.flatten()[i_pc]
            ax.plot(T_cue_aligned_sel, proj_allDim[:len(T_cue_aligned_sel), i_pc], 'b')
            ax.plot(T_cue_aligned_sel, proj_allDim[len(T_cue_aligned_sel):, i_pc], 'r')
            ax.plot(T_cue_aligned_sel, proj_allDim_err[:len(T_cue_aligned_sel), i_pc], color=[.7, .7, 1])
            ax.plot(T_cue_aligned_sel, proj_allDim_err[len(T_cue_aligned_sel):, i_pc], color=[1, .7, .7])
            ax.set_title("Mode {}".format(i_pc + 1))
            
        axs[0, 0].set_ylabel('Activity proj.')
        axs[3, 0].set_xlabel('Time')
        
        return proj_allDim[:len(T_cue_aligned_sel), i_pc], proj_allDim[len(T_cue_aligned_sel):, i_pc]
        # plt.show()

    def plot_activity_modes_ctl(self):
        
        T_cue_aligned_sel = self.T_cue_aligned_sel


            
        orthonormal_basis, var_allDim = self.func_compute_activity_modes_DRT([self.PSTH_r_train_correct, 
                                                                            self.PSTH_l_train_correct], ctl = True)
        
        activityRL_train= np.concatenate((self.PSTH_r_train_correct, 
                                        self.PSTH_l_train_correct), axis=1)

        activityRL_test= np.concatenate((self.PSTH_r_test_correct, 
                                        self.PSTH_l_test_correct), axis=1)
        
    
    
        # Correct trials
        activityRL_test = activityRL_test - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activityRL_test.shape[1]))  # remove mean
        proj_allDim = np.dot(activityRL_test.T, orthonormal_basis)
        

        fig, axs = plt.subplots(4, 4, figsize=(12, 16))
        for i_pc in range(16):
            ax = axs.flatten()[i_pc]
            ax.plot(T_cue_aligned_sel, proj_allDim[:len(T_cue_aligned_sel), i_pc], 'b')
            ax.plot(T_cue_aligned_sel, proj_allDim[len(T_cue_aligned_sel):, i_pc], 'r')
            ax.set_title("Mode {}".format(i_pc + 1))
            
        axs[0, 0].set_ylabel('Activity proj.')
        axs[3, 0].set_xlabel('Time')
        
        return proj_allDim[:len(T_cue_aligned_sel), i_pc], proj_allDim[len(T_cue_aligned_sel):, i_pc]


    def plot_activity_modes_opto(self, error = False):
        
        T_cue_aligned_sel = self.T_cue_aligned_sel

            
        # TODO: should I only train on correct trials?
        orthonormal_basis, var_allDim = self.func_compute_activity_modes_DRT([self.PSTH_r_train_correct, 
                                                                            self.PSTH_l_train_correct], ctl = True)
        
        activityRL_train= np.concatenate((self.PSTH_r_train_correct, 
                                        self.PSTH_l_train_correct), axis=1)

        activityRL_test= np.concatenate((self.PSTH_r_test_correct, 
                                        self.PSTH_l_test_correct), axis=1)
        
        activityRLerr_test = np.concatenate((self.PSTH_r_test_opto, 
                                             self.PSTH_l_test_opto), axis = 1)
        
        if error:
            
            activityRLerr_test = np.concatenate((self.PSTH_r_test_opto_err, 
                                                 self.PSTH_l_test_opto_err), axis = 1)
    
    
        # Correct trials
        activityRL_test = activityRL_test - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activityRL_test.shape[1]))  # remove mean
        proj_allDim = np.dot(activityRL_test.T, orthonormal_basis)
        
        # Opto trials
        activityRLerr_test = activityRLerr_test - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activityRLerr_test.shape[1]))  # remove mean
        proj_allDim_err = np.dot(activityRLerr_test.T, orthonormal_basis)
        
        fig, axs = plt.subplots(4, 4, figsize=(12, 16))
        for i_pc in range(16):
            ax = axs.flatten()[i_pc]
            ax.plot(T_cue_aligned_sel, proj_allDim[:len(T_cue_aligned_sel), i_pc], 'b')
            ax.plot(T_cue_aligned_sel, proj_allDim[len(T_cue_aligned_sel):, i_pc], 'r')
            ax.plot(T_cue_aligned_sel, proj_allDim_err[:len(T_cue_aligned_sel), i_pc], color=[.7, .7, 1])
            ax.plot(T_cue_aligned_sel, proj_allDim_err[len(T_cue_aligned_sel):, i_pc], color=[1, .7, .7])
            ax.set_title("Mode {}".format(i_pc + 1))
            
        axs[0, 0].set_ylabel('Activity proj. with opto trials')
        axs[3, 0].set_xlabel('Time')
        
        return proj_allDim[:len(T_cue_aligned_sel), i_pc], proj_allDim[len(T_cue_aligned_sel):, i_pc]

        

    
        
    def plot_behaviorally_relevant_modes(self):
        # plot behaviorally relevant activity modes only
        # separates trials into train vs test sets
        mode_ID = np.array([1, 2, 6, 3, 7, 8, 9])
        mode_name = ['stimulus', 'choice', 'action', 'outcome', 'ramping', 'go', 'response']
        
        orthonormal_basis, var_allDim = self.func_compute_activity_modes_DRT([self.PSTH_r_train_correct, 
                                                                            self.PSTH_l_train_correct, 
                                                                            self.PSTH_r_train_error, 
                                                                            self.PSTH_l_train_error], ctl=False)
        
        activityRL_train= np.concatenate((self.PSTH_r_train_correct, 
                                        self.PSTH_l_train_correct, 
                                        self.PSTH_r_train_error, 
                                        self.PSTH_l_train_error), axis=1)

        activityRL_test= np.concatenate((self.PSTH_r_test_correct, 
                                        self.PSTH_l_test_correct), axis=1)
        
        activityRLerr_test = np.concatenate((self.PSTH_r_test_error, 
                                             self.PSTH_l_test_error), axis = 1)
        
        
        T_cue_aligned_sel = self.T_cue_aligned_sel
        
        # Correct trials
        activityRL_test = activityRL_test - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activityRL_test.shape[1]))  # remove mean
        proj_allDim = np.dot(activityRL_test.T, orthonormal_basis)
        var_allDim = np.sum(proj_allDim ** 2, axis=0)
        var_allDim = var_allDim[~np.isnan(var_allDim)]

        var_allDim /= np.sum(var_allDim)
        
        
        # Error trials
        activityRLerr_test = activityRLerr_test - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activityRLerr_test.shape[1]))  # remove mean
        proj_allDim_err = np.dot(activityRLerr_test.T, orthonormal_basis)
        
        plt.figure()
        plt.bar(mode_ID, var_allDim[mode_ID-1])
        plt.xlabel('Activity modes')
        plt.ylabel('Frac var.')
        plt.title(f'Total Cross Validated Var Explained: {np.sum(var_allDim[mode_ID]):.4f}')
        
        n_plot = 0
        plt.figure()
        for i_mode in mode_ID:
            n_plot += 1
            print(f'plotting mode {n_plot}')
            
            proj_iPC_allBtstrp = np.zeros((20, activityRL_test.shape[1]))
            projErr_iPC_allBtstrp = np.zeros((20, activityRLerr_test.shape[1]))
            for i_btstrp in range(20):
                i_sample = np.random.choice(range(activityRL_test.shape[0]), activityRL_test.shape[0], replace=True)
                proj_iPC_allBtstrp[i_btstrp,:] = np.dot(activityRL_test[i_sample,:].T, orthonormal_basis[i_sample, i_mode])
                projErr_iPC_allBtstrp[i_btstrp,:] = np.dot(activityRLerr_test[i_sample,:].T, orthonormal_basis[i_sample, i_mode])
            
            plt.subplot(2, 4, n_plot)
            self.func_plot_mean_and_sem(T_cue_aligned_sel, projErr_iPC_allBtstrp[:,:len(T_cue_aligned_sel)], '#6666ff', '#ccccff', 2)
            self.func_plot_mean_and_sem(T_cue_aligned_sel, projErr_iPC_allBtstrp[:,len(T_cue_aligned_sel):], '#ff6666', '#ffcccc', 2)
            self.func_plot_mean_and_sem(T_cue_aligned_sel, proj_iPC_allBtstrp[:,:len(T_cue_aligned_sel)], 'b', '#9999ff', 2)
            self.func_plot_mean_and_sem(T_cue_aligned_sel, proj_iPC_allBtstrp[:,len(T_cue_aligned_sel):], 'r', '#ff9999', 2)
            
            # y_scale = np.mean(np.concatenate((proj_iPC_allBtstrp, projErr_iPC_allBtstrp)))
            # plt.plot([-2.6,-2.6],[min(y_scale), max(y_scale)]*1.2,'k:') 
            # plt.plot([-1.3,-1.3],[min(y_scale), max(y_scale)]*1.2,'k:')
            # plt.plot([0,0],[min(y_scale), max(y_scale)]*1.2,'k:')
            
            # plt.xlim([-3.2, 2.2])
            plt.title(f'mode {mode_name[n_plot-1]}')

        plt.subplot(2, 4, 1)
        plt.ylabel('Activity proj.')
        plt.xlabel('Time')
        
        return None
    
    def plot_behaviorally_relevant_modes_opto(self, error=False):
        # plot behaviorally relevant activity modes only
        # separates trials into train vs test sets
        mode_ID = np.array([1, 2, 6, 3, 7, 8, 9])
        mode_name = ['stimulus', 'choice', 'action', 'outcome', 'ramping', 'go', 'response']
        
        orthonormal_basis, var_allDim = self.func_compute_activity_modes_DRT([self.PSTH_r_train_correct, 
                                                                            self.PSTH_l_train_correct], ctl=True)
        
        activityRL_train= np.concatenate((self.PSTH_r_train_correct, 
                                        self.PSTH_l_train_correct), axis=1)

        activityRL_test= np.concatenate((self.PSTH_r_test_correct, 
                                        self.PSTH_l_test_correct), axis=1)
        
        activityRLerr_test = np.concatenate((self.PSTH_r_test_opto, 
                                             self.PSTH_l_test_opto), axis = 1)
        if error:
            
            activityRLerr_test = np.concatenate((self.PSTH_r_test_opto_err, 
                                                 self.PSTH_l_test_opto_err), axis = 1)
        
        T_cue_aligned_sel = self.T_cue_aligned_sel
        
        # Correct trials
        activityRL_test = activityRL_test - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activityRL_test.shape[1]))  # remove mean
        proj_allDim = np.dot(activityRL_test.T, orthonormal_basis)
        var_allDim = np.sum(proj_allDim ** 2, axis=0)
        var_allDim = var_allDim[~np.isnan(var_allDim)]

        var_allDim /= np.sum(var_allDim)
        
        
        # Error trials
        activityRLerr_test = activityRLerr_test - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activityRLerr_test.shape[1]))  # remove mean
        proj_allDim_err = np.dot(activityRLerr_test.T, orthonormal_basis)
        
        plt.figure()
        plt.bar(mode_ID, var_allDim[mode_ID-1])
        plt.xlabel('Activity modes')
        plt.ylabel('Frac var.')
        plt.title(f'Total Cross Validated Var Explained: {np.sum(var_allDim[mode_ID]):.4f}')
        
        n_plot = 0
        plt.figure()
        for i_mode in mode_ID:
            n_plot += 1

            print(f'plotting mode {n_plot}')
            
            proj_iPC_allBtstrp = np.zeros((20, activityRL_test.shape[1]))
            projErr_iPC_allBtstrp = np.zeros((20, activityRLerr_test.shape[1]))
            for i_btstrp in range(20):
                i_sample = np.random.choice(range(activityRL_test.shape[0]), activityRL_test.shape[0], replace=True)
                proj_iPC_allBtstrp[i_btstrp,:] = np.dot(activityRL_test[i_sample,:].T, orthonormal_basis[i_sample, i_mode])
                projErr_iPC_allBtstrp[i_btstrp,:] = np.dot(activityRLerr_test[i_sample,:].T, orthonormal_basis[i_sample, i_mode])
            
            plt.subplot(2, 4, n_plot)
            self.func_plot_mean_and_sem(T_cue_aligned_sel, projErr_iPC_allBtstrp[:,:len(T_cue_aligned_sel)], '#6666ff', '#ccccff', 2)
            self.func_plot_mean_and_sem(T_cue_aligned_sel, projErr_iPC_allBtstrp[:,len(T_cue_aligned_sel):], '#ff6666', '#ffcccc', 2)
            self.func_plot_mean_and_sem(T_cue_aligned_sel, proj_iPC_allBtstrp[:,:len(T_cue_aligned_sel)], 'b', '#9999ff', 2)
            self.func_plot_mean_and_sem(T_cue_aligned_sel, proj_iPC_allBtstrp[:,len(T_cue_aligned_sel):], 'r', '#ff9999', 2)
            
            # y_scale = np.mean(np.concatenate((proj_iPC_allBtstrp, projErr_iPC_allBtstrp)))
            # plt.plot([-2.6,-2.6],[min(y_scale), max(y_scale)]*1.2,'k:') 
            # plt.plot([-1.3,-1.3],[min(y_scale), max(y_scale)]*1.2,'k:')
            # plt.plot([0,0],[min(y_scale), max(y_scale)]*1.2,'k:')
            
            # plt.xlim([-3.2, 2.2])
            plt.title(f'mode {mode_name[n_plot-1]}')

        plt.subplot(2, 4, 1)
        plt.ylabel('Activity proj. with opto trials')
        plt.xlabel('Time')
        
        return None

    def func_plot_mean_and_sem(self, x, y, line_color='b', fill_color='b', sem_option=1, n_std=1):
        
        """
        :param x: 1D numpy array with length m (m features)
        :param y: 2D numpy array with shape (n, m) (n observations, m features)
        :param line_color: line color (default 'b')
        :param fill_color: fill color (default 'b')
        :param sem_option: standard error option (1: sem, 2: std, 3: bootstrapping) (default 1)
        :param n_std: standard deviation multiplier (default 1)
        """
    
        x_line = x
        y_line = np.mean(y, axis=0)
    
        if sem_option == 1:
            y_sem = np.std(y, axis=0) / np.sqrt(y.shape[0])
        elif sem_option == 2:
            y_sem = np.std(y, axis=0)
        elif sem_option == 3:
            y_tmp = np.zeros((1000, y.shape[1]))
            for i in range(1000):
                y_isample = np.random.choice(y.shape[0], size=y.shape[0], replace=True)
                y_tmp[i, :] = np.mean(y[y_isample, :], axis=0)
            y_sem = np.std(y_tmp, axis=0)
        else:
            y_sem = np.std(y, axis=0) / np.sqrt(y.shape[0])

        
        plt.plot(y_line, line_color)

        plt.fill_between(x, y_line - y_sem, 
                            y_line + y_sem,
                            color=[fill_color])

        

    