from abcpy.probabilisticmodels import ProbabilisticModel, Continuous, InputConnector, Discrete
import numpy as np
import subprocess
import os
import errno
import random
import gzip
import shutil

import string
import time
from pathlib import Path
import pandas as pd
import random
from datetime import datetime

class Mimicree(ProbabilisticModel, Continuous):
    """
    This class is an re-implementation of the `abcpy.continousmodels.Normal` for documentation purposes.
    """

    def __init__(self, parameters, name='Mimicree'):
        # We expect input of type parameters = [pathExecJarFile, haplotypeFile, replicateRuns, snapshots, outputFile]
        if not isinstance(parameters, list):
            raise TypeError('Input of Mimicree model is of type list')

        if len(parameters) != 2:
            raise RuntimeError(
                'Input list must be of length 2, containing [lambda, nns].')
 

        input_connector = InputConnector.from_list(parameters)
        super().__init__(input_connector, name)

    def _check_input(self, input_values):


        return True

    def _check_output(self, values):
        return True

    def get_output_dimension(self):
        return 1

    def forward_simulate(self, input_values, rep, rng=np.random.RandomState()):
        
        lambdaparam = input_values[0]
        nns = input_values[1]

        # Do the actual forward simulation
        vector_of_k_samples = self.mimicrees(lambdaparam, nns, rep)
        # Format the output to obey API
        result = [np.array([x]).reshape(-1, ) for x in vector_of_k_samples]
        return result

    def mimicrees(self, lambdaparam, nns, rep):
        """ (originally mimicrees)
        This function simulate allele frequencies based on some 
        founder haplotype.
        
        Parameters
        ----------
        
        lambdaparam : float
                    selection coefficient
        nns : integer
            number of selected target
        rep : integer
            number of replicates
        

        Returns
        -------
        list of array
        
        This returns a list with k elements, where each element is a numpy array consisting of a time-series
        """
        ###############################################################################################
        ###############################################################################################
        # Create a random files that contains input file
        Random = os.getcwd() + '/files/' + ''.join(
            [random.choice(string.ascii_letters + string.digits) for n in range(32)])
        os.system('mkdir -p ' + Random)
        os.system('cp -r ' + 'input/.' + ' ' + Random)
        # Load the foulder haplotye
        hp_str = pd.read_csv(Random + "/" + "haplotype.txt",header=None, sep='\t').to_numpy()
        # List of 0s to store the value of selection coefficient for selected SNPs
        s = np.zeros(hp_str.shape[0])
        # Calculate the starting allele frequency
        hp0 = hp_str.dot((1/hp_str.shape[1]) * np.ones(hp_str.shape[1]))
        # Select SNPs with starting allele frequencies between 0.1 and 0.9
        # to be the potential candidate for selection.
        filters = np.where((hp0>0.1) & (hp0<0.9))[0].tolist()
        # Randomly select SNP to have selection
        selected = np.random.choice(filters,size=nns,replace=False)
        # Applied selection to that SNP
        s[selected] = lambdaparam
        # Index of selected target
        benef_all = np.array(selected)
        # Number of time points (generations)
        tp = np.r_[0:61:10]
        # effective population size
        Ne = 2000
        
        result = []
        for i in range(rep):
            # main simulation
            freq = Freq(hp_str,Ne,tp).simulate(s=s,benef_all=benef_all)
            result.append(freq)
        #move temorary files.
        shutil.rmtree(Random,ignore_errors=True)

        return result
        

class DiscreteUniform(Discrete, ProbabilisticModel):
    def __init__(self, parameters, name='DiscreteUniform'):
        """This class implements a probabilistic model following a Discrete Uniform distribution.

        Parameters
        ----------
        parameters: list
             A list containing two entries, the upper and lower bound of the range.

        name: string
            The name that should be given to the probabilistic model in the journal file.
        """

        if not isinstance(parameters, list):
            raise TypeError(
                'Input for Discrete Uniform has to be of type list.')
        if len(parameters) != 2:
            raise ValueError(
                'Input for Discrete Uniform has to be of length 2.')

        self._dimension = 1
        input_parameters = InputConnector.from_list(parameters)
        super(DiscreteUniform, self).__init__(input_parameters, name)
        self.visited = False

    def _check_input(self, input_values):
        # Check whether input has correct type or format
        if len(input_values) != 2:
            raise ValueError(
                'Number of parameters of FloorField model must be 2.')

        # Check whether input is from correct domain
        lowerbound = input_values[0]  # Lower bound
        upperbound = input_values[1]  # Upper bound

        if not isinstance(lowerbound, (int, np.int64, np.int32, np.int16)) or not isinstance(upperbound, (int, np.int64, np.int32, np.int16)) or lowerbound >= upperbound:
            return False
        return True

    def _check_output(self, parameters):
        """
        Checks parameter values given as fixed values. Returns False iff it is not an integer
        """
        if not isinstance(parameters[0], (int, np.int32, np.int64)):
            return False
        return True

    def forward_simulate(self, input_values, k, rng=np.random.RandomState()):
        """
        Samples from the Discrete Uniform distribution associated with the probabilistic model.

        Parameters
        ----------
        input_values: list
            List of input parameters, in the same order as specified in the InputConnector passed to the init function
        k: integer
            The number of samples to be drawn.
        rng: random number generator
            The random number generator to be used.

        Returns
        -------
        list: [np.ndarray]
            A list containing the sampled values as np-array.
        """
        result = np.array(rng.randint(
            input_values[0], input_values[1]+1, size=k, dtype=np.int64))
        return [np.array([x]).reshape(-1,) for x in result]

    def get_output_dimension(self):
        return self._dimension

    def pmf(self, input_values, x):
        """Evaluates the probability mass function at point x.

        Parameters
        ----------
        input_values: list
            List of input parameters, in the same order as specified in the InputConnector passed to the init function
        x: float
            The point at which the pmf should be evaluated.

        Returns
        -------
        float:
            The pmf evaluated at point x.
        """
        lowerbound, upperbound = input_values[0], input_values[1]
        if x >= lowerbound and x <= upperbound:
            pmf = 1. / (upperbound - lowerbound + 1)
        else:
            pmf = 0
        self.calculated_pmf = pmf
        return pmf

class Freq:
    """
    Function to forward simulate the allele frequencies data given haplotype struture 
    
    and selection coefficients.
    
    
    """
    def __init__(self, hp_str, Ne, tp, meancov=None, hp0 = None, benef_all=None, s=0, diploid=False):
        
      self.hp_str = hp_str
      self.Ne = Ne
      self.tp = tp
      self.hp0 = hp0
      self.diploid = diploid
      self.meancov = meancov
      """
        Parameters
        ----------
        hp_str : 2D-array
                matrix of 0 and 1 represent haplotype stucture.
                
        Ne : int
            population size
            
        tp: array of number of time point to sample
            Time points 
            
        meancov: int, optional
                mean coverage
                
        hp0: int, optional
            initial haplotype frequencies
            
        benef_all: array or list
                beneficial allele index
                
        s : array of length of number of SNPs
            selection of SNPs
            
        diploid : boolean
                sampling diploid or haploid individual
      """
      # initial haplotype frequency evenly distributed among SNPs if not specified.
      if self.hp0 is None:
          self.hp0 = (1/self.hp_str.shape[1]) * np.ones(self.hp_str.shape[1])
          
    def simulate(self,s,benef_all):
        """

        Parameters
        ----------
        s : array
            Array of selection coefficient for all SNPs.
        benef_all : array
            beneficial allele index.

        Returns
        -------
        Allele Frequencies data (SNPs x Time points)

        """
        max_len = len(self.hp0)
        hp_freq = np.zeros((max_len,len(self.tp)))
        if 0 in self.tp:
            hp_freq[:,0] = self.hp0
        if self.diploid:
          self.Ne = 2*self.Ne
         
        g = 1
        p = self.hp0.copy()
        
        # Simulation Loop
        while g <= np.max(self.tp):
            # Initial fitness
            fitness_hp = np.ones(max_len)
            if np.sum(s != 0) > 0:
                for i in range(self.hp_str.shape[1]):
                    if np.sum(self.hp_str[benef_all,i]) > 0:
                        # Fitness based on the number of shared haplotypes.
                        fitness_hp[i] = fitness_hp[i]+np.sum(s[benef_all][(self.hp_str[benef_all,i] > 0).tolist()])

            
            p = p * fitness_hp
            p = p/p.sum()
            # Multinomial sampling 
            p = np.random.multinomial(self.Ne, p.flatten(), size=1)/self.Ne
            if g in self.tp:
                ind = np.where(self.tp==g)[0][0]
                hp_freq[:,ind] = p
            g += 1
        # Allele freuquencies = Haplotype Structure x Haplotype Frequencies
        Y = self.hp_str.dot(hp_freq)
        # Whether mean coverage is specified
        if self.meancov is not None:
            covMat = np.random.poisson(lam=self.meancov,size=Y.shape)
            Y_err = np.random.binomial(n = covMat, p = Y)/covMat
            return np.insert(np.around(Y_err.flatten(),3),0,len(self.tp))
        else:
            # Return the format of desired
            return np.insert(np.around(Y.flatten(),3),0,len(self.tp))
