import h5py
import numpy as np
from mpi4py import MPI
import time

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank==0:
    print('Programme Started:')
    ### Start the time counting
    t_start = time.time()  
time_steps = 20*1000 ### the number of snapshots user want to reconstruct

### Load the data file for spatial modes and time coefficients
nz = # number of points in z-direction
ny = # number of points in y-direction
nmodes = # number of POD modes per wavenumber
nkz = # number of wavenumbers
start = int(rank*nz/size)
stop = int((rank+1)*nz/size)
step = int(nz/size)
######################################################################################
# file path set-up
############################################################################## input
Modes_filepath = ""
Modes_filename = "" ## POD modes
Modes_name_matlab = "" ## name of POD modes in matlab
time_series_filepath = ""
time_series_filename = "" ## POD time coefficients
## name of time series in matlab, 
## if time series is VAR-generated, then by default to be "T_coeff"
time_series_name_matlab = "T_coeff" 
############################################################################## output
outputpath = "" ## path name of the reconstructed field
outputname = "" ## file name of the reconstructed field

######################################################################################    
Modes = h5py.File(Modes_filepath + Modes_filename + '.mat','r')
local_modes = np.array(Modes[Modes_name_matlab][:,:,:,start:stop],dtype=np.float64)
T_coef = h5py.File(time_series_filepath + time_series_filename + ".mat",'r')
T_coef = np.array(T_coef[time_series_name_matlab][:,:time_steps,:],dtype=np.float64)


#########################################################################
### Reconstruct the velocity fields
if rank==0: 
    t_startpara = time.time()
u_rec = np.zeros((time_steps,nz,3*ny))
### Transpose the modes and t-coefficient data structure to enhance efficiency
local_modes = np.transpose(local_modes,(3,2,1,0))
T_coef = np.transpose(T_coef,(1,2,0))
## local_modes -- [step,3*ny,2*nmodes,nkz]
## T_coef -- [npl,2*nmodes,nkz]
#%%
if rank==0:
    print('Parallel Reconstruction Started:')
    print('T_coef size: ',np.shape(T_coef))
    print('modes size: ',np.shape(Modes[Modes_name_matlab]))

"""Parallelled codes for reconstruction"""
t_step_op = time.time()
local_urec = np.tensordot(local_modes,T_coef, axes=([2,3],[1,2]))    

"""Send the result to processor 0"""    
if rank > 0:
    comm.Send(local_urec, dest=0, tag=14)  # send results to process 0
else: # process 0
    final_urec = np.copy(local_urec)  # initialize final results with results from process 0
    for ipro in range(1,size):    
        tmp = np.empty((step,3*ny,time_steps),dtype=np.float64)  # create empty array to receive results
        comm.Recv(tmp, source=ipro, tag=14)  # receive results from the process
        final_urec = np.vstack((final_urec, tmp))  # add the received results to the final results
    u_rec = final_urec
    t_step_ed = time.time()

### End the time counting and report
if rank==0:    
    ### End the time counting and report
    t_end = time.time()
    print('Elapsed Time for Reconstruction: ', t_end-t_start, 's')
     
    ### Save the reconstructed velocity fields
    print("Saving fields")
    t_save_op = time.time()
    hf = h5py.File(outputpath + outputname +'.mat', 'w')
    hf.create_dataset('u1', data=u_rec[:,:ny,:])
    hf.create_dataset('u2', data=u_rec[:,ny:2*ny,:])
    hf.create_dataset('u3', data=u_rec[:,2*ny:3*ny,:])
    hf.close()
    t_save_ed = time.time()
    print("Time cost for saving fields: ",t_save_ed-t_save_op,'s')
