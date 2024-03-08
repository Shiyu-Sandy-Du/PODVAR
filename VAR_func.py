#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 10:49:01 2023

@author: shiyud
"""

import statsmodels.api as sm
import numpy as np
import scipy
import time

###
# dataSet = T_coeff_raw[0,:,0::2]
# dataSet = dataSet - np.mean(dataSet,axis=0)

def sig_group(dataSet,threshold_lowccf=0.1):
    """group signals according to max[abs(ccf)]"""
    ## input dataSet should be a npl-by-num_sig array
    num_sig = np.shape(dataSet)[1]
    lenccf = 300
    index = [[i] for i in range(num_sig)]
    for i in range(num_sig):
        for j in range(i+1,num_sig):
            maxabsccf1 = np.amax(np.abs(sm.tsa.stattools.ccf(dataSet[:,i],dataSet[:,j])[:lenccf]))
            maxabsccf2 = np.amax(np.abs(sm.tsa.stattools.ccf(dataSet[:,j],dataSet[:,i])[:lenccf]))
            maxabsccf = max(maxabsccf1,maxabsccf2)
            if maxabsccf > threshold_lowccf:
                index[i].append(j)
    ## So far, one signal could appear in several groups,
    ## The following steps merge groups having same signals together   
    for i_index in range(num_sig):#iterate elements in list "index"
        for i_index_sub in index[i_index]: #iterate elements in index[i_index]
            for i_index2 in range(i_index+1,i_index_sub+1): #daparture from the current sub-list to the corresponding one
            # if index[i_index_sub] != [] and i_index_sub != i_index:
                if i_index_sub in index[i_index2]:
                    ## merge the higher order relevant list into lower order
                    for i in range(len(index[i_index2])):
                        if index[i_index2][i] not in index[i_index]:
                            index[i_index].append(index[i_index2][i])
                    index[i_index2] = []
    ## take the non-empty elements in list index
    index_result = []
    for i in range(len(index)):
        if index[i] != []:
            index_result.append(index[i])
            
    return index_result
                
def VAR_orderSelection(f):
    num_sig = np.shape(f)[1]
    t_orderSelectStart = time.time()
    threshold_drop = 0.1   ###### threshold of low correlation
    index_lowccf = []
    integralTimeScaleLag = np.zeros(num_sig,dtype="int")
    for sig1 in range(num_sig):
        for sig2 in range(num_sig):
            ccfrange = 300 ### truncate the ccf
            ccf = sm.tsa.stattools.ccf(f[:,sig1],f[:,sig2])[:ccfrange+1]
            absccf = abs(ccf)
            # flag_min = False
            
            ## find the range of searching
            # lim_search = 0 # for strategy 1-3, search from 0
            
            # ### strategy 1: second zero-crossing
            # flag_zero1 = False
            # for i in range(lim_search,ccfrange):
            #     if flag_zero1 == False and ccf[i]*ccf[i+1] < 0:
            #         flag_zero1 = True
            #     elif flag_zero1 == True and ccf[i]*ccf[i+1] < 0:
            #         lim_search = i
            #         break
                
            # ## strategy 2: second or third zero-crossing, depending on initial tendency
            # num_cross = 0
            # if (ccf[1]-ccf[0])*ccf[0] > 0:
            #     num_cross = 2
            # else:
            #     num_cross = 3
            
            # if num_cross == 2:
            #     flag_zero1 = False
            #     for i in range(lim_search,ccfrange):
            #         if flag_zero1 == False and ccf[i]*ccf[i+1] < 0:
            #             flag_zero1 = True
            #         elif flag_zero1 == True and ccf[i]*ccf[i+1] < 0:
            #             lim_search = i
            #             break
            # elif num_cross == 3:
            #     flag_zero1 = False
            #     flag_zero2 = False
            #     for i in range(lim_search,ccfrange):
            #         if flag_zero1 == False and flag_zero2 == False and ccf[i]*ccf[i+1] < 0:
            #             flag_zero1 = True
            #             # if sig1==sig2==0:
            #             #     print("first",i)
            #         elif flag_zero1 == True and flag_zero2 == False and ccf[i]*ccf[i+1] < 0:
            #             flag_zero2 = True
            #             # if sig1==sig2==0:
            #             #     print("second",i)
            #         elif flag_zero1 == True and flag_zero2 == True and ccf[i]*ccf[i+1] < 0:
            #             lim_search = i
            #             # if sig1==sig2==0:
            #             #     print("third",i)
            #             break
        
            # ## strategy 3: pick first several relevant peaks, and ends at the last one
            # peak_valid = False
            # # bound_l = 0
            # bound_r = 0    
            # for i in range(1,ccfrange):
            #     if absccf[i] > threshold_drop:
            #         peak_valid = True
            #     if peak_valid == True and ccf[i]*ccf[i+1]<0:
            #         # bound_l = bound_r ## move bounds to right
            #         bound_r = i
            #         peak_valid = False ## reset for new peak
            #     elif peak_valid == False and ccf[i]*ccf[i+1]<0:
            #         # print("AAOO:",bound_l)
            #         lim_search = bound_r
            #         break
        
            lim_search1 = 0
            ## strategy 4: end of the last valid peak while keep two zero-crossings
            # flag_peak = False
            # zerocross = ccfrange
            for i in range(ccfrange,0,-1):
                if absccf[i] > threshold_drop:
                    lim_search1 = i
                    break
            for i in range(lim_search1,ccfrange):
                if ccf[i]*ccf[i+1]<0:
                # if absccf[i] < absccf[i+1]:
                    lim_search1 = i
                    break
            # keep two zero-crossings
            #lim_search2 = 0
            #flag_zero1 = False
            #for i in range(ccfrange):
            #    # if flag_zero1 == False and ccf[i]*ccf[i+1] < 0:
            #    if flag_zero1 == False:
            #        if ccf[i]*ccf[i+1] < 0:
            #            flag_zero1 = True
            #    # elif flag_zero1 == True and ccf[i]*ccf[i+1] < 0:
            #    elif flag_zero1 == True and ccf[i]*ccf[i+1]<0:
            #        lim_search2 = i
            #        break
            #lim_search = max(lim_search1,lim_search2)
            lim_search = lim_search1
            ### Set integral time scale for each signal
            if sig1 == sig2:
                # print("ITS for",sig1,":",lim_search)
                integralTimeScaleLag[sig1] = lim_search
            
            ## Check whether there is at least one valid peak
            threshold_overshoot = False 
            for i in range(lim_search):
                if absccf[i] >= threshold_drop:
                    threshold_overshoot = True
                    break
            if threshold_overshoot == False:
                index_lowccf.append(0) ###0 means the abs(ccf) did not exceed the threshold
            else:
                index_lowccf.append(lim_search)
                
    #orderVAR = int(max(integralTimeScaleLag))   ##### choose the order
    orderVAR = int(max(index_lowccf))
    #for i in range(len(index_lowccf)):
        #if index_lowccf[i] > orderVAR:
            #index_lowccf[i] = orderVAR
            
    index_lowccf = np.reshape(np.array(index_lowccf),(num_sig,num_sig))    
    print("order of VAR:",orderVAR)
    t_orderSelectEnd = time.time()
    print("time for order selection: ",t_orderSelectEnd-t_orderSelectStart,"s")
    return orderVAR, integralTimeScaleLag, index_lowccf

def VAR_fitting(f,orderVAR,index_lowccf,constraint=True,sigma_u="biased"):
    """
    Multivariate Least Square: Method introduced in:
        Helmut Lütkepohl, New Introduction to Multiple Time Series Analysis,
        Springer Berlin, Heidelberg, 2005, pp.70-71
    Self-define method is used to constraint some entries in coefficient matrices to be zero
    """
    
    """Matrices formulation"""
    
    npl = np.shape(f)[0]
    K = np.shape(f)[1] ## number of signal
    p = orderVAR
    T = npl-p ## number of series (deducted by the pre-sampled)
    Y = f[p:,:].T
    y = Y.flatten('F')
    Z = np.zeros((K*p+1,T))
    for i_T in range(T):
        Z[0,i_T] = 1
        for i_p in range(p):
            Z[i_p*K+1:(i_p+1)*K+1,i_T] = f[p-i_p+i_T-1,:]
        
    if constraint == True:
        ### with constraint: some entries in beta is zero
        ### Substitute beta_hat into beta in S(beta), given beta = R@beta_hat
        index_nonzeroInBeta = []
        ## Find the indeces for zeros in beta
        for i_sig1 in range(K):
            for i_sig2 in range(K):
                start_zeroccf = index_lowccf[i_sig1,i_sig2]
                for i in range(start_zeroccf):
                    index_nonzeroInBeta.append(K + i*K*K + i_sig2*K + i_sig1)
        index_nonzeroInBeta.sort()
        if len(index_nonzeroInBeta)>1:    
            ## delete the column in R corresponding to zero in beta
            R_csc = scipy.sparse.csc_matrix(scipy.sparse.eye(K*K*p+K))
            R_csc = R_csc[:,index_nonzeroInBeta]
            RT_csc = R_csc.T.tocsc()
            
            ## convert Z and Ik into scipy sparse matrix   
            Z_csc = scipy.sparse.csc_matrix(Z)
            ZZT_csc = scipy.sparse.csc_matrix(Z@Z.T)
            Ik = scipy.sparse.eye(K)
    
            ## get beta_hat from Z, R and y
            ZZT_kron_Ik = scipy.sparse.kron(ZZT_csc,Ik)
            Z_kron_Ik = scipy.sparse.kron(Z_csc,Ik)
    
            RT_ZZTIk_R = RT_csc.dot(ZZT_kron_Ik).dot(R_csc)
            RT_ZZTIk_R_inv = scipy.sparse.linalg.inv(RT_ZZTIk_R)
            beta_hat = Z_kron_Ik.dot(y)
            beta_hat = RT_ZZTIk_R_inv.dot(RT_csc).dot(beta_hat)
    
            ## get beta from beta_hat, beta = R@beta_hat
            beta = R_csc.dot(beta_hat)
            
            """Construct B from beta"""
            B = np.reshape(beta,(K*p+1,K))
            """Uncertainty"""
            U = Y-B.T@Z
            if sigma_u == "unbiased":
                sig_u = U@U.T/(T-K*p-1)
            elif sigma_u == "biased":
                sig_u = U@U.T/T
        else:
            print("noise")
            print("num_sig",K)
            B = np.zeros((2,K))
            if K == 1:
                sig_u = np.var(f).reshape(1,-1)
            else:
                sig_u = np.cov(f.T)
            print("cov",sig_u)
            print("cov size",sig_u.shape)
    else:
        if orderVAR != 0:
            B = (Y@Z.T@np.linalg.inv(Z@Z.T)).T
            """Uncertainty"""
            U = Y-B.T@Z
            if sigma_u == "unbiased":
                sig_u = U@U.T/(T-K*p-1) ### unbiased estimator
            elif sigma_u == "biased":
                sig_u = U@U.T/T   ### biased estimator
    
        else:
            B = np.zeros((2,K))
            if K == 1:
                sig_u = np.var(f).reshape(1,-1)
            else:
                sig_u = np.cov(f.T)
    return B,sig_u

    
def VAR_stable(coeff_mat,orderVAR,num_sig):
    """
    Method introduced in:
        Helmut Lütkepohl, New Introduction to Multiple Time Series Analysis,
        Springer Berlin, Heidelberg, 2005, p.16, Equation 2.1.9
    """
    if orderVAR != 0:
        M = np.zeros((orderVAR*num_sig,orderVAR*num_sig))
        for i in range(orderVAR):
            M[:num_sig,i*num_sig:(i+1)*num_sig] = coeff_mat[i,:,:]
        for i in range(1,orderVAR):
            M[i*num_sig:(i+1)*num_sig,(i-1)*num_sig:i*num_sig] = np.identity(num_sig)
        M_eig,_ = np.linalg.eig(M)
        for i in range(orderVAR*num_sig):
            M_eig[i] = np.abs(M_eig[i])
        return np.amax(M_eig)
    else:
        print("this signal is noise")
        return 0
        
        
def VAR_genSyn(f,step_Emu,VAR_intercept,coeff_mat,sigma_u_noise,orderVAR):
    # print("cov size:",sigma_u_noise.shape)
    num_sig = np.shape(f)[1]       
    # print("num sig:",num_sig)
    ### predict further samples
    if orderVAR != 0:
        f_fcst = f[-orderVAR:,:] #raw data for the first few predictions
    else:
        f_fcst = np.zeros((1,num_sig)) #artificial start (will be discarded)
    for i_step in range(step_Emu):
        f_curr = np.copy(VAR_intercept)
        for i_order in range(orderVAR): ### This for-loop could be parallelized
            f_curr += np.matmul(coeff_mat[i_order,:,:],f_fcst[-i_order-1,:]) #weighted summation
        f_curr += np.random.multivariate_normal(np.zeros((num_sig)), sigma_u_noise) #noise
        f_fcst = np.vstack((f_fcst,f_curr)) #append into the signal series
    if orderVAR != 0:
        f_fcst = f_fcst[orderVAR:,:] #discard the raw data
    else:
        f_fcst = f_fcst[1:,:] #discard the artificial starting point
    return f_fcst
