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
# import matplotlib.pyplot as plt


#%% Load original signals
T_coeff_raw = h5py.File("/scratch/shiyud/POD_TBL/ReTheta_790/VAR/data/T_coeff_Emu_10000.mat",'r')
T = np.array(T_coeff_raw['T_coeff'])
#%%
T = np.squeeze(T)
#%%
hf = h5py.File('/scratch/shiyud/POD_TBL/ReTheta_790/VAR/data/T_coeff_Emu0_10000.mat', 'w')
hf.create_dataset('T_coeff', data=T)
#hf.create_dataset('T_coeff', data=f_fcst_ensemble)
hf.close()

