from abcpy.probabilisticmodels import ProbabilisticModel, Continuous, InputConnector, Discrete
import numpy as np
import subprocess
import os
import random
import gzip
import string
from pathlib import Path
import pandas as pd
import shutil


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
            #raise RuntimeError('Input list must be of length 5, containing [recratevalues, selcoeffs, recpositions, nspositions, generations].')

 

        input_connector = InputConnector.from_list(parameters)
        super().__init__(input_connector, name)


    def _check_input(self, input_values):

        return True

    def _check_output(self, values):
        return True

    def get_output_dimension(self):
        return 1

    def forward_simulate(self, input_values, k, rng=np.random.RandomState()):
      
        lambdaparam = input_values[0]
        nns = input_values[1]

        # Do the actual forward simulation
        vector_of_k_samples = self.mimicrees(lambdaparam, nns, k)
        # Format the output to obey API
        result = [np.array([x]).reshape(-1, ) for x in vector_of_k_samples]
        return result

    def mimicrees(self, lambdaparam, nns, rep):
        """
        Simulation function using mimiCREE [1]
        [1] Christos Vlachos and Robert Kofler. 
        Mimicree2: Genome-wide forward simulations of evolve and resequencing studies.
        PLoS computational biology, 14(8):e1006413, 2018.
        Parameters
        ----------
        lambdaparam : float
            selection coefficient.
        nns : integer
            number of selected target.
        rep : integer
            number of replicates.


        Returns
        -------
        list of array
        
        This returns a list with k elements, where each element is a numpy array consisting of a time-series

        """
   
        ###############################################################################################
        ###############################################################################################
        
        # Chromosome
        chromosomename = "2L"
        # Number of generations
        generations = 60
        # list of generations e.g. "10,20,30,40,50,60"
        outputmode = ','.join(str(i)
                              for i in range(1, generations+1)if(i % 10 == 0))
        
        
        len_outputmode = len(outputmode.split(","))

        # Creating a folder with random name and copying the haplotype files in there
        Random = os.getcwd() + '/tmp/' + ''.join(
            [random.choice(string.ascii_letters + string.digits) for n in range(32)])
        os.system('mkdir -p ' + Random)
        os.system('cp -r ' + 'input/.' + ' ' + Random)
        
        # Name of the input haplotype file
        haplotypefilename = Random + "/" + "haplotype.txt"  
        # Name of the output file
        outputfilename = Random + '/output.sync'

        # Get SNPs' position 
        data = pd.read_csv(haplotypefilename, delimiter="\t", header=None)
        SNPs = list(data.iloc[:, 1])
        # Write recombination and natural selection rates in the correct file
        recratevalues = 0
        # Create a file to supply recombination rates
        recratefilename = Random + "/" + "rec.txt"
        # Write recombination rate in the correct file
        f = open(recratefilename, 'w')
        # Start by writing lambda on top
        f.write("[cM/Mb]" + "\n")

        f.write(chromosomename + ":" + str(SNPs[0]) + ".." + str(SNPs[-1]) + "	" +
                str(recratevalues) + "\n")

        # # close file
        f.close()

        # Selection coefficient
        selcoeffs = lambdaparam
        # Now simulate the positions where the selection happens
        nspositions = random.sample(SNPs, nns)
        
        # Create selection files and write the selection and position into file.
        selcoefffilename = Random + "/" + "selcoeff.txt"
        
        with open(selcoefffilename, 'w') as f:
            f.write("[s]" + "\n")
            for ind in range(len(nspositions)):
                f.write('2L '+ str(nspositions[ind]) +' A/C '+ str(selcoeffs) + "  " + str(0.5) + "\n")
        # close file
        f.close()
        
        # Create the command providing inputs and outputfilenames for mimiCREE2
        # Details of the arguments refer ot mimiCREE2 sourceforge wiki
        command = "java -jar *.jar w --haploid --haplotypes-g0 " + haplotypefilename  + " --fitness " + selcoefffilename + \
                  " --snapshots " + str(outputmode) + " --replicate-runs " + str(
                      rep) + " --recombination-rate " + recratefilename + " --output-sync " + outputfilename +"  2> out.txt "
                  
        # Run the program
        iterationrun = 0
        run = True
        while run and iterationrun < 100:
            try:

                # Runnning Mimicree2 on command line
                p = subprocess.run(command, shell=True,
                                   cwd=Random)
                #print(Random + ' : Program ran successfully. Time taken :' + str(time.time() - stime) + '\n')
                if Path(outputfilename).is_file() is False:
                    raise ValueError('Output not created')
                else:
                    run = False
            except OSError as e:
                print(Random + ' : Error occurred: ' +
                      str(e.errno) + '. Will try again.' + '\n')
            except subprocess.TimeoutExpired:
                print(Random + ' : Process ran too long. Will try again.' + '\n')
            except (ValueError, IndexError):
                print(Random + ' : Output not created' + '\n')
            iterationrun += 1


        result = [[int(len_outputmode)] for i in range(rep)]
        
        # Read the mimiCREE2 output and calculate the allele frequencies.
        der_list = []
        fh=open(haplotypefilename)
        for line in fh:
            line=line.rstrip()
            a=line.split("\t")
            anc,der=a[3].split("/")
            der_list.append(der)
        # Read Output
        c = 0

        with gzip.open(outputfilename, 'rt') as f:
            for line in f:
                a = line.replace('\n','').split('\t')
                derived = der_list[c]
                c += 1
                for ind in range(rep):
                    base = list(map(int,a[3+len_outputmode*ind].split(':')))
                    
                    tmp = []
                    for indegen in range(len_outputmode):
                        if(derived == 'A'):
                            tmp.append((np.array(list(map(int,a[3+len_outputmode*ind+indegen].split(':'))))/sum(base))[0])
                        elif(derived == 'T'):
                            tmp.append((np.array(list(map(int,a[3+len_outputmode*ind+indegen].split(':'))))/sum(base))[1])
                        elif(derived == 'C'):
                            tmp.append((np.array(list(map(int,a[3+len_outputmode*ind+indegen].split(':'))))/sum(base))[2])
                        else:
                            tmp.append((np.array(list(map(int,a[3+len_outputmode*ind+indegen].split(':'))))/sum(base))[3])
                    result[ind] += tmp

        # Delete the random folder
        shutil.rmtree(Random,ignore_errors=True)

        # We flatten the matrix to satisfy convention of ABCpy, but they can be retrieved by putting them in 6 columns back

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