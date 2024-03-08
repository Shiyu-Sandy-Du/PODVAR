#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  10 15:23:33 2023

@author: shiyud
"""
import h5py
import numpy as np
import pickle
import VAR_func
from mpi4py import MPI
import matplotlib.pyplot as plt
import time
#############################################################################################
#############################################################################################
#######################################################################################
#
VARmodelpath = "/scratch/shiyud/POD_TBL/ReTheta_790/VAR/data/"
VARmodelname = "VARfit_param_sigGrouping_5"

inputpath = "/scratch/shiyud/POD_TBL/ReTheta_790/"
intputname = "t_coeffs_5"

outputpath = "/scratch/shiyud/POD_TBL/ReTheta_790/VAR/data/"
outputname = "T_coeff_Emu0_5_100k_0"

npl = 22900 # number of planes for the input
nz_half = 97 # number of wavenumber for the input
nmodes = 5 # number of POD modes for the input
step_Emu = 100000 # number of emulations
### MPI parameters
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
if rank == 0:
    t_start = time.time() 
nstep = int(nz_half/size)
kz_sc = [] # "sc" for single core
ikz_sc = 0
while ikz_sc*size+rank < nz_half:
    kz_sc.append(ikz_sc*size+rank)
    ikz_sc += 1
kz_size = len(kz_sc)
## take container size in rank 0
if rank > 0:
    comm.send(kz_sc, dest=0, tag=rank)  # send results to process 0
else: # process 0
    kz_container = [kz_sc]  # initialize final results with results from process 0
    for ipro in range(1,size):    
        tmp = comm.recv(source=ipro, tag=ipro)  # receive results from the process
        kz_container.append(tmp) # add the received results to the final results

#%% Load parameters
with open(VARmodelpath + VARmodelname + ".pkl",'rb') as fp:
    param = pickle.load(fp)
VAR_intercept = param['VAR_intercept']
coeff_mat = param['coeff_mat']
sig_u_noise = param['sig_u_noise']
orderVAR = param['orderVAR']
mean_sig = param['mean_sig']
modeSep = param['modeSep']

#%% Load original signals
T_coeff_raw = h5py.File(inputpath+intputname+".mat",'r')
T_coeff_raw = np.array(T_coeff_raw['bb_trunc'])
### shape of T_coeff_raw [nkz,npl,nmode]
### Transpose to npl*nmode*nkz
T_coeff_raw = np.transpose(T_coeff_raw,(1,2,0))

#%% Generate time series
## demean the time series
T_coeff_demean = T_coeff_raw - mean_sig
f_fcst_local = np.zeros((kz_size,step_Emu,2*nmodes))
for ikz in range(kz_size):
    print('kz:',kz_sc[ikz],'rank',rank)
    if kz_sc[ikz] == 0 or kz_sc[ikz] == nz_half-1:
        f = T_coeff_demean[:,0::2,kz_sc[ikz]]
        group_all = modeSep[kz_sc[ikz]]
        for i_group in range(len(group_all)):
            print("group number:",i_group)
            group = group_all[i_group]
            num_sig = len(group)
            f_group = f[:,group]
            f_fcst_currGroup = VAR_func.VAR_genSyn(f_group,step_Emu,VAR_intercept[kz_sc[ikz]][i_group],\
                                                coeff_mat[kz_sc[ikz]][i_group],sig_u_noise[kz_sc[ikz]][i_group],\
                                                orderVAR[kz_sc[ikz]][i_group])
            for i_mode in range(num_sig):
                f_fcst_local[ikz,:,group[i_mode]*2] = f_fcst_currGroup[:,i_mode]
    else:
        f = T_coeff_demean[:,:,kz_sc[ikz]]
        group_all = modeSep[kz_sc[ikz]]
        for i_group in range(len(group_all)):
            group = group_all[i_group]
            num_sig = len(group)
            f_group = f[:,group]
            print("group number:",i_group)
            f_fcst_currGroup = VAR_func.VAR_genSyn(f_group,step_Emu,VAR_intercept[kz_sc[ikz]][i_group],\
                                                coeff_mat[kz_sc[ikz]][i_group],sig_u_noise[kz_sc[ikz]][i_group],\
                                                orderVAR[kz_sc[ikz]][i_group])
            for i_mode in range(num_sig):
                f_fcst_local[ikz,:,group[i_mode]] = f_fcst_currGroup[:,i_mode]                      
## add the mean value back to the signal
for ikz in range(kz_size):
    for j in range(nmodes):
        f_fcst_local[ikz,:,j] += mean_sig[j,kz_sc[ikz]]

## MPI processor communication
if rank > 0:
    comm.Send(f_fcst_local, dest=0, tag=0)  # send results to process 0
else: # process 0
    f_fcst = np.zeros((nz_half,step_Emu,2*nmodes))  # initialize final results with results from process 0
    f_fcst[kz_container[0],:,:] = f_fcst_local
    for ipro in range(1,size):    #kz_container: list[processor][wavenumber]
        tmp = np.empty((len(kz_container[ipro]),step_Emu,2*nmodes))  # create empty array to receive results
        comm.Recv(tmp, source=ipro, tag=0)  # receive results from the process
        f_fcst[kz_container[ipro],:,:] = tmp  # add the received results to the final results

# #%% save
if rank == 0:
    t_end = time.time() 
    f_fcst = np.array(f_fcst,'float')
    print("saving emulated time coefficients ...")
    hf = h5py.File(outputpath+outputname+".mat", 'w')
    hf.create_dataset('T_coeff', data=f_fcst)
    hf.close()        
    print("time cost:", t_end - t_start)         