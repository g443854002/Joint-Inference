# Load SABC from abcpy
from abcpy.inferences import SABC
# Load kernels
from abcpy.perturbationkernel import DefaultKernel
# Load distance measure
from Distance import PenLogReg
# Load simulation model
from model_sim import Mimicree
# Load probability model for prior
from abcpy.continuousmodels import Uniform
from abcpy.discretemodels import DiscreteUniform
# Load summary statistics
from Statistics_new import LogitTransformStat
# Utility packages
import numpy as np
import logging
from itertools import product
logging.basicConfig(level=logging.DEBUG)
# Define the backend
from abcpy.backends import BackendMPI as Backend
backend = Backend()    

"""
This file runs inference given the prior for parameter
,specify kernel, summary statistics and distance measure.

"""



# Defining the prior for selection coefficient
lambdaparam = Uniform([[0], [0.2]], name='lambdaparam')
# Defining the prior for number of selected target
nns = DiscreteUniform([[0], [3]], name='nns')
# list of s values
s_list = [0.02,0.05,0.07,0.09]
# list of number of selected targets
nns_list = [1,2]
# list of replicates number
rep_list = np.linspace(1,20,20,dtype = int).tolist()
# Make a tuple of combination of s,nns and rep
comb = list(product(*[s_list,nns_list,rep_list]))


# Making inference for difference combination of s,nns and rep
for i in comb:
# Transform input s value to string so we can read data and save result easily
    s_str = str(round(i[0],3)).replace('.','')
# Specify data name based on the value given in s, nns and rep
    dataname = 'hamimi_10rep_1000_data_'+str(i[1])+'_'+s_str+'_rep'+str(i[2])+'.npz'
# Load the data
    fakeobs = list(np.load(dataname)['d'])
# Defining graphical model for simulations
    MMC = Mimicree([lambdaparam,nns], name='MMC')
# Specify the summary statistics function
    statcalc = LogitTransformStat(degree=1,cross=False)
# Specify the distance function
    distance_calculator = PenLogReg(statcalc)
# Specify the kernel function           
    kernel = DefaultKernel([lambdaparam,nns])
# SABC inference
    sampler = SABC([MMC], [distance_calculator], backend,kernel,seed=1)
# Specify the parameters of SABC
# We use 25 steps, with beta = 2 , delta = 0.2, v =0.9, starting epsilon set to 1.
# We will obtain 200 samples for our approximation of the posterior.
# Data have 10 replicates, so we specify n_samples_per_param to be 10
    journal_sabc1 = sampler.sample([fakeobs],steps=25,epsilon = 1,n_samples=200, n_samples_per_param=10,
        beta = 2, delta = 0.2, v =0.9, ar_cutoff=0.0001, resample=20, n_update=None,full_output=1)

# Save the result
    file = 'hamimi_10rep_1000_logit_'+s_str+'_'+str(i[1])+'_rep'+str(i[2])+'.jrnl'
#file = 'yeast.jrnl'
    journal_sabc1.save(file)
