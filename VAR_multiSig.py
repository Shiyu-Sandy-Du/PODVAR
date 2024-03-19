import numpy as np
import h5py
import time
import VAR_func
import sys
##############################################################################
#%% set-up
# kind reminder:
# It's cheaper to check the stability from the lambda_z-PSD of the resulting velocified in the end.
# In this way, the kz of which the PODVAR model becomes unstable could be identified...
# Then it is possible to stabilize it either by setting the constraint in VAR_func.VAR_fitting() to be true,
# or by adding that particular kz into kz_tune.
check_stability = False # whether or not check the stability of the VAR model
t_start = time.time()
"""Read the T-coefficient data file""" 
time_series_filepath = ""
time_series_filename = "" ## POD time coefficients
time_series_name_matlab = "" ## name of time series in matlab
timestep = ## number of planes used to fit the VAR model
output_filepath = ""
output_filename = "" ## parameters of VAR model
kz_tune = [0] # kz with a different strong ccf criteria (i.e. critical_ccf_tune)
critical_ccf = 0.1
critical_ccf_tune = 0.3
################################################################################
#%% data reading
T_coeff_raw = h5py.File(time_series_filepath + time_series_filename + ".mat",'r')
T_coeff_raw = np.array(T_coeff_raw[time_series_name_matlab],,dtype=np.float64)

### Transpose to npl*nmode*nkz
T_coeff_raw = np.transpose(T_coeff_raw,(1,2,0))[-timestep:,:,:]
### get the dimensions
npl = np.shape(T_coeff_raw)[0]
nmodes = int(np.shape(T_coeff_raw)[1]/2)
nz_half = np.shape(T_coeff_raw)[2]
T_coeff_r = np.zeros((npl,nmodes,nz_half))
T_coeff_i = np.zeros((npl,nmodes,nz_half))  

###The raw T_coeff matrix store real and imag part separately, and need to be merge into a complex matrix
for i in range(nmodes):
    for j in range(nz_half):
        T_coeff_r[:,i,j] = T_coeff_raw[:,2*i,j]
        T_coeff_i[:,i,j] = T_coeff_raw[:,2*i+1,j]
t_readfinish = time.time()
print("time for reading data: ",t_readfinish-t_start,"s")
print('data shape: ',np.shape(T_coeff_raw))
#%% model fitting
list_VAR_intercept = []
list_coeff_mat = []
list_sig_u_noise = []
list_orderVAR = []
list_modeSep = []

mean_sig = np.mean(T_coeff_raw,axis=0)

for ikz in range(nz_half):
    print("kz:",ikz)
    if ikz == nz_half-1 or ikz == 0:
        f_pre = T_coeff_r[:,:,ikz] - mean_sig[0::2,ikz]
    else: 
        f_pre = T_coeff_raw[:,:,ikz] - mean_sig[:,ikz]
    if ikz in kz_tune:
        group_all = VAR_func.sig_group(f_pre,threshold_lowccf=critical_ccf_tune)
    else:    
        group_all = VAR_func.sig_group(f_pre,threshold_lowccf=critical_ccf)
    num_group = len(group_all)
    countgroup = 0
    list_VAR_intercept_curr = []
    list_coeff_mat_curr = []
    list_sig_u_noise_curr = []
    list_orderVAR_curr = []
    for group_curr in group_all:
        countgroup += 1
        print("group:{:d}/{:d}".format(countgroup,num_group))
        
        num_sig = len(group_curr)
        f = np.empty((npl,num_sig))
        print("number of signals:",num_sig)
        
        """formulate the time series"""
        for index_mode in range(len(group_curr)):
            f[:,index_mode] = f_pre[:,group_curr[index_mode]]
            
        """select VAR order"""   
        orderVAR, integralTSLag, index_lowccf = VAR_func.VAR_orderSelection(f)
        
        """VAR model fitting"""
        t_VARtrainStart = time.time()
        VAR_params, sig_u_noise = VAR_func.VAR_fitting(f,orderVAR, index_lowccf, constraint=False)
        ## re-organize the parameters
        VAR_intercept = VAR_params[0,:]
        VAR_coeff = VAR_params[1:,:]
        coeff_mat = np.zeros((orderVAR,num_sig,num_sig))  
        for i_order in range(orderVAR):
            for i_sig in range(num_sig):
                for i_coe_sig in range(num_sig):
                    coeff_mat[i_order,i_sig,i_coe_sig] = VAR_coeff[i_coe_sig+i_order*num_sig,i_sig]
        t_VARtrainEnd = time.time()
        print("time for VAR modelling: ",t_VARtrainEnd-t_VARtrainStart,"s")
        
        """Check the stability of the VAR model"""
        if check_stability == True:
            t_VARstableStart = time.time()
            eig = np.real(VAR_func.VAR_stable(coeff_mat,orderVAR,num_sig))
            print("largest eigen value's magnitude: ",eig) ### the returned value is already mag
            if eig > 1:
                sys.exit('unstable VAR model')
            t_VARstableEnd = time.time()
            print("time for VAR stability: ",t_VARstableEnd-t_VARstableStart,"s")
        
        list_VAR_intercept_curr.append(VAR_intercept)
        list_coeff_mat_curr.append(coeff_mat)
        list_sig_u_noise_curr.append(sig_u_noise)
        list_orderVAR_curr.append(orderVAR)
        print("")
        
    """Collect params from each kz and real-imag data set"""
    list_VAR_intercept.append(list_VAR_intercept_curr)
    list_coeff_mat.append(list_coeff_mat_curr)
    list_sig_u_noise.append(list_sig_u_noise_curr)
    list_orderVAR.append(list_orderVAR_curr)
    list_modeSep.append(group_all)

#%% saving parameters of the model
VARfit_param = {
    "VAR_intercept": list_VAR_intercept,
    "coeff_mat": list_coeff_mat,
    "sig_u_noise": list_sig_u_noise,
    "orderVAR": list_orderVAR,
    "modeSep": list_modeSep,
    "mean_sig": mean_sig 
}
print("save parameters")
import pickle
with open(output_filepath + output_filename + '.pkl','wb') as fp:
    pickle.dump(VARfit_param,fp)

tfinish = time.time()
print("total time cost:",tfinish-t_start,"s")

















        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
