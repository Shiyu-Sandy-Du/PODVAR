# PODVAR
This is a place to the python code to perform model fitting and time series prediction related to Vector Autoregression (VAR) model for Fourier-based Proper Orthogonal Decomposition (POD) time coefficients.

nz: the number of the points in the direction where Fourier transform is performed;

npl: the length of the time series;

nmode: the number of the POD mode truncated.

# Python package needed
statsmodels, numpy, scipy, time, h5py, sys, pickle, mpi4py.

# Sequence of PODVAR method
1. Perform Fourier-based POD on flow field:
    Shape of the resulting array in matlab:
        POD modes: (nz/2+1 , 2nmode, 3ny, nz)
        Time coefficient: (nz/2+1 , npl, 2nmode)

2. Use VAR_multiSig.py to perform VAR model fitting (more than 1 VAR model for each wavenumber)
    Resulting into a .pkl file with all parameters needed for VAR models
    The model fitting is currently serial...
    Parallel could be intended but the time cost on a single core is acceptable...

3. Use VAR_SigGen.py to generate PODVAR time coefficients as long as you want
    This step is parallel...

4. Use rec_0_20k.py to reconstruct the field

