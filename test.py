import h5py
import numpy as np
import pickle
import VAR_func
intputname = "T_coeff_Emu0_5_100k"
#%% Load original signals
ikz = 6
imode = 1
T_coeff_raw = h5py.File("/scratch/shiyud/POD_TBL/ReTheta_790/t_coeffs_5.mat",'r')
T_coeff_raw = np.array(T_coeff_raw['bb_trunc'])
print(np.shape(T_coeff_raw))

T_coeff_emu = h5py.File("/scratch/shiyud/POD_TBL/ReTheta_790/VAR/data/"+intputname+".mat",'r')
T_coeff_emu = np.array(T_coeff_emu['T_coeff'])

sig_orig = T_coeff_raw[ikz,:,imode]
sig_emu = T_coeff_emu[ikz,:,imode]

import matplotlib.pyplot as plt
plt.plot([i for i in range(len(sig_orig))],sig_orig)
plt.plot([i+len(sig_orig) for i in range(len(sig_emu))],sig_emu)
plt.xlim(len(sig_orig)-len(sig_emu),len(sig_orig)+len(sig_emu))
plt.show()
