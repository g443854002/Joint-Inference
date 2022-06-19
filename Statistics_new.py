from abcpy.statistics import Statistics
import numpy as np



class LogitTransformStat(Statistics):
    """
    Applies a linear transformation to the data to get (usually) a lower dimensional statistics. 
    Then you can apply an additional polynomial expansion step.
    
    """

    def __init__(self, gen=60, degree=1, num_snp=1000, diploid = True, cross=False, reference_simulations=None, previous_statistics=None):
        """
        `degree` and `cross` specify the polynomial expansion you want to apply to the statistics.
        If `reference_simulations` are provided, the standard deviation of the different statistics on the set
        of reference simulations is computed and stored; these will then be used to rescale
        the statistics for each new simulation or observation.
        
        If no set of reference simulations are provided, then this is not done.
        `previous_statistics` allows different Statistics object to be pipelined. Specifically, if the final
        statistic to be used is determined by the
        
        composition of two Statistics, you can pass the first here; then, whenever the final statistic is needed, it
        is sufficient to call the `statistics` method of the second one, and that will automatically apply both
        transformations.
        
        Parameters
        ----------
        gen: integer
            number of generations
            
        degree : integer, optional
            Of polynomial expansion. The default value is 2 meaning second order polynomial expansion.
            
        num_snp : integer
                number of SNPs in the dataset
                
        diploid : boolean
                calculate the statistics for diploid or haploid
            
        cross : boolean, optional
            Defines whether to include the cross-product terms. The default value is True, meaning the cross product term
            is included.
            
        reference_simulations: array, optional
            A numpy array with shape (n_samples, output_size) containing a set of reference simulations. If provided,
            statistics are computed at initialization for all reference simulations, and the standard deviation of the
            different statistics is extracted. The standard deviation is then used to standardize the summary
            statistics each time they are compute on a new observation or simulation. Defaults to None, in which case
            standardization is not applied.
            
        previous_statistics : abcpy.statistics.Statistics, optional
            It allows pipelining of Statistics. Specifically, if the final statistic to be used is determined by the
            composition of two Statistics, you can pass the first here; then, whenever the final statistic is needed, it
            is sufficient to call the `statistics` method of the second one, and that will automatically apply both
            transformations.
            
            
        """
        self.gen = gen
        self.num_snp = num_snp
        self.diploid = diploid

        super(LogitTransformStat, self).__init__(degree, cross, reference_simulations, previous_statistics)

    def statistics(self, data):
        """
        Parameters
        ----------
        data: python list
            Contains n data sets with length p.
        Returns
        -------
        numpy.ndarray
            nx(d+degree*d+cross*nchoosek(d,2)) matrix where for each of the n data points with length p you apply the
            linear transformation to get to dimension d, from where (d+degree*d+cross*nchoosek(d,2)) statistics are
            calculated.
        """

        # need to call this first which takes care of calling the previous statistics if that is defined and of properly
        # formatting data
        data = self._preprocess(data)
        # Create matrix with zeros to store the result
        result = np.zeros((data.shape[0], self.num_snp))
        
        # different denominator for diploid and haploid individual
        if self.diploid:
            dom_stat = self.gen/2
        else:
            dom_stat = self.gen
            
        # Loop over replicates
        for ind in range(data.shape[0]):
            # Reshape the data to desired form

            data_tmp = data[ind, 1:].reshape(
                int((len(data[ind, :]) - 1) / data[ind, 0]), int(data[ind, 0]))
            # Convert the value to 0.0001 if it reaches 0, 0.9999 when it reaches 1

            pt = np.minimum(np.maximum(data_tmp[:, -1:], 0.0001), 0.9999)
            p0 = np.minimum(np.maximum(data_tmp[:, :1], 0.0001), 0.9999)
            # Compute the value of logit transformed estimates

            linear_stat = (np.log(pt / (1 - pt)) - np.log(p0 / (1 - p0))) / (dom_stat)
            
            # Store the result

            result[ind, :] = linear_stat.reshape(-1)
            
        # Order the result with the ordering of its mean

        result = result[:, np.argsort(result.mean(0))]
       
        result = self._polynomial_expansion(result)

        # now call the _rescale function which automatically rescales the different statistics using the standard
        # deviation of them on the training set provided at initialization.
        result = self._rescale(result)

        return result