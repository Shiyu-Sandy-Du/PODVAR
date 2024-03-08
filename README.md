# PODVAR
This is a place to the python code to perform model fitting and time series prediction related to Vector Autoregression (VAR) model for Fourier-based Proper Orthogonal Decomposition (POD) time coefficients.
Shape of the array of the Fourier-based POD time coefficients: (nz/2+1 , npl, nmode).
nz: the number of the points in the direction where Fourier transform is performed;
npl: the length of the time series;
nmode: the number of the POD mode truncated.

# Python package needed
statsmodels, numpy, scipy, time, h5py, sys, pickle, mpi4py.

