# -*- coding: utf-8 -*-
# Copyright 2007-2011 The Hyperspy developers
#
# This file is part of  Hyperspy.
#
#  Hyperspy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  Hyperspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  Hyperspy.  If not, see <http://www.gnu.org/licenses/>.


from __future__ import division
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import mdp

from hyperspy.misc import utils
from hyperspy.learn.svd_pca import svd_pca
from hyperspy.learn.mlpca import mlpca
from hyperspy.defaults_parser import preferences
from hyperspy import messages
from hyperspy.decorators import auto_replot, do_not_replot
from scipy import linalg
try:
    import sklearn.decomposition
except:
    pass

def centering_and_whitening(X):
    X = X.T
    # Centering the columns (ie the variables)
    X = X - X.mean(axis=-1)[:, np.newaxis]
    # Whitening and preprocessing by PCA
    u, d, _ = linalg.svd(X, full_matrices=False)
    del _
    K = (u / (d/np.sqrt(X.shape[1]))).T
    del u, d
    X1 = np.dot(K, X)
    return X1.T, K
    
class MVA():
    """
    Multivariate analysis capabilities for the Spectrum class.

    """

    def __init__(self):
        if not hasattr(self,'mva_results'):
            self.mva_results = MVA_Results()
        if not hasattr(self.mapped_parameters,'peak_chars'):
            self.mapped_parameters.peak_chars = None

    def _get_target(self, on_peaks):
        if on_peaks:
            target=self.peak_mva_results
        else:
            target=self.mva_results
        return target
    
    @do_not_replot
    def decomposition(self, normalize_poissonian_noise=False,
    algorithm = 'svd', output_dimension=None, navigation_mask=None,
    signal_mask=None, center=False, normalize_variance=False, var_array=None,
    var_func=None, polyfit=None, on_peaks=False, **kwargs):
        """Decomposition with a choice of algorithms

        The results are stored in self.mva_results

        Parameters
        ----------
        normalize_poissonian_noise : bool
            If True, scale the SI to normalize Poissonian noise
            
        algorithm : 'svd' | 'fast_svd' | 'mlpca' | 'fast_mlpca' | 'nmf' |
            'sparse_pca' | 'mini_batch_sparse_pca'
        
        
        output_dimension : None or int
            number of components to keep/calculate
            
        navigation_mask : boolean numpy array
        
        signal_mask : boolean numpy array
        
        center : bool
            Perform energy centering before applying the algorithm
            
        normalize_variance : bool
            Scale the data to have unit variance
            
        var_array : numpy array
            Array of variance for the maximum likelihood PCA algorithm
            
        var_func : function or numpy array
            If function, it will apply it to the dataset to obtain the
            var_array. Alternatively, it can a an array with the coefficients
            of a polynomial.
            
        polyfit :


        See also
        --------
        plot_decomposition_factors, plot_decomposition_scores, plot_lev

        """
        # backup the original data
        if on_peaks:
            if hasattr(self.mapped_parameters,'peak_chars'):
                self._data_before_treatments = \
                    self.mapped_parameters.peak_chars.copy()
            else:
                print """No peak characteristics found.  You must run the 
                         peak_char_stack function to obtain these before 
                         you can run PCA or ICA on them."""
        else:
            self._data_before_treatments = self.data.copy()
            

        if algorithm == 'mlpca':
            if normalize_poissonian_noise is True:
                messages.warning(
                "It makes no sense to do normalize_poissonian_noise with "
                "the MLPCA algorithm. Therefore, "
                "normalize_poissonian_noise is set to False")
                normalize_poissonian_noise = False
            if output_dimension is None:
                messages.warning_exit("With the mlpca algorithm the "
                "output_dimension must be expecified")


        # Apply pre-treatments
        # Centering
        if center is True:
            self.energy_center()
        # Variance normalization
        if normalize_variance is True:
            self.normalize_variance()
        # Transform the data in a line spectrum
        self._unfolded4decomposition = self.unfold_if_multidim()
        # Normalize the poissonian noise
        # Note that this function can change the masks
        if normalize_poissonian_noise is True:
            navigation_mask, signal_mask = \
                self.normalize_poissonian_noise(navigation_mask=navigation_mask,
                                                signal_mask=signal_mask,
                                                return_masks = True)
        if hasattr(navigation_mask, 'ravel'):
            navigation_mask = navigation_mask.ravel()

        messages.information('Performing decomposition analysis')

        if on_peaks:
            dc = self.mapped_parameters.peak_chars
        else:
            # The data must be transposed both for Images and Spectra
            dc = self.data
            
        #set the output target (peak results or not?)
        target = self._get_target(on_peaks)
        
        # Transform the None masks in slices to get the right behaviour
        explained_variance = None
        if navigation_mask is None:
            navigation_mask = slice(None)
        if signal_mask is None:
            signal_mask = slice(None)
        
        if algorithm == 'svd':
            factors, scores, explained_variance = svd_pca(
                dc[:,signal_mask][navigation_mask,:])
            # We recompute the the scores because for some reason otherwise
            # the first pixels get higher variance
            scores = np.dot(dc[:,signal_mask], factors)

        elif algorithm == 'fast_svd':
            factors, scores, explained_variance = svd_pca(
                dc[:,signal_mask][navigation_mask,:],
            fast = True, output_dimension = output_dimension)
            # We recompute the the scores because for some reason otherwise
            # the first pixels get higher variance
            scores = np.dot(dc[:,signal_mask], factors)

        elif algorithm == 'sklearn_pca':    
            pca = sklearn.decomposition.PCA(**kwargs)
            pca.n_components = output_dimension
            pca.fit((dc[:,signal_mask][navigation_mask,:]))
            factors = pca.components_.T
            scores = pca.transform((dc[:,signal_mask]))
            explained_variance = pca.explained_variance_    

        elif algorithm == 'nmf':    
            nmf = sklearn.decomposition.NMF(**kwargs)
            nmf.n_components = output_dimension
            nmf.fit((dc[:,signal_mask][navigation_mask,:]))
            factors = nmf.components_.T
            scores = nmf.transform((dc[:,signal_mask]))
            
        elif algorithm == 'sparse_pca':
            spca = sklearn.decomposition.SparsePCA(output_dimension, **kwargs)
            spca.fit(dc[:,signal_mask][navigation_mask,:])
            factors = spca.components_.T
            scores = spca.transform(dc[navigation_mask,:])
            
        elif algorithm == 'mini_batch_sparse_pca':
            spca = sklearn.decomposition.MiniBatchSparsePCA(output_dimension,
                                                            **kwargs)
            spca.fit(dc[:,signal_mask][navigation_mask,:])
            factors = spca.components_.T
            scores = spca.transform(dc[:,signal_mask])

        elif algorithm == 'mlpca' or algorithm == 'fast_mlpca':
            print "Performing the MLPCA training"
            if output_dimension is None:
                messages.warning_exit(
                "For MLPCA it is mandatory to define the output_dimension")
            if var_array is None and var_func is None:
                messages.information('No variance array provided.'
                'Supposing poissonian data')
                var_array = dc[:,signal_mask][navigation_mask,:]

            if var_array is not None and var_func is not None:
                messages.warning_exit(
                "You have defined both the var_func and var_array keywords"
                "Please, define just one of them")
            if var_func is not None:
                if hasattr(var_func, '__call__'):
                    var_array = var_func(dc[signal_mask,...][:,navigation_mask])
                else:
                    try:
                        var_array = np.polyval(polyfit,dc[signal_mask,
                        navigation_mask])
                    except:
                        messages.warning_exit(
                        'var_func must be either a function or an array'
                        'defining the coefficients of a polynom')
            if algorithm == 'mlpca':
                fast = False
            else:
                fast = True
            target.mlpca_output = mlpca(
                dc[:,signal_mask][navigation_mask,:],
                var_array, output_dimension, fast = fast)
            U,S,V,Sobj, ErrFlag  = target.mlpca_output
            print "Performing PCA projection"
            scores = np.dot(dc[:,signal_mask], V)
            factors = V
            explained_variance = S ** 2
            
        else:
            messages.information('Error: Algorithm not recognised. '
                                 'Nothing done')
            return False

        target.factors = factors
        target.scores = scores
        target.explained_variance = explained_variance
        target.decomposition_algorithm = algorithm
        target.centered = center
        target.poissonian_noise_normalized = \
            normalize_poissonian_noise
        target.output_dimension = output_dimension
        target.unfolded = self._unfolded4decomposition
        target.variance_normalized = normalize_variance
        
        if output_dimension:
            self.crop_decomposition_dimension(output_dimension)
        
        # Delete the unmixing information, because it'll refer to a previous
        # decompositions
        target.unmixing_matrix = None
        target.ica_algorithm = None

        if self._unfolded4decomposition is True:
            target.original_shape = self._shape_before_unfolding

        # Rescale the results if the noise was normalized
        if normalize_poissonian_noise is True:
            target.factors[:] *= self._root_bH.T
            target.scores[navigation_mask,:] *= self._root_aG
            if isinstance(navigation_mask, slice):
                navigation_mask = None
            if isinstance(signal_mask, slice):
                signal_mask = None

        #undo any pre-treatments
        self.undo_treatments(on_peaks)

        # Set the pixels that were not processed to nan
        if navigation_mask is not None and not isinstance(
            navigation_mask, slice):
            target.scores[navigation_mask == False,:] = np.nan
            
        if signal_mask is not None and not isinstance(
            signal_mask, slice):
            factors = np.zeros((dc.shape[-1], target.factors.shape[1]))
            factors[signal_mask == True,:] = target.factors
            factors[signal_mask == False,:] = np.nan
            target.factors = factors

        if self._unfolded4decomposition is True:
            self.fold()
            self._unfolded4decomposition is False
    
    def get_factors_as_spectrum(self):
        from hyperspy.signals.spectrum import Spectrum
        return Spectrum({'data' : self.mva_results.factors.T})
    
    def independent_components_analysis(
        self, number_of_components=None, algorithm='CuBICA', diff_order=1,
        factors=None, comp_list = None, mask = None, on_peaks=False, on_scores=False,
        smoothing = None, **kwargs):
        """Independent components analysis.

        Available algorithms: FastICA, JADE, CuBICA, and TDSEP

        Parameters
        ----------
        number_of_components : int
            number of principal components to pass to the ICA algorithm
        algorithm : {FastICA, JADE, CuBICA, TDSEP}
        diff : bool
        diff_order : int
        factors : numpy array
            externally provided components
        comp_list : boolen numpy array
            choose the components to use by the boolean list. It permits to
            choose non contiguous components.
        mask : numpy boolean array with the same dimension as the PC
            If not None, only the selected channels will be used by the
            algorithm.
        smoothing: dict
        
        Any extra parameter is passed to the ICA algorithm
        """
        target=self._get_target(on_peaks)

        if not hasattr(target, 'factors') or target.factors==None:
            self.decomposition(on_peaks=on_peaks)

        else:
            if factors is None:
                if on_scores:
                    factors = target.scores
                else:
                    factors = target.factors
            bool_index = np.zeros((factors.shape[0]), dtype = 'bool')
            if number_of_components is not None:
                bool_index[:number_of_components] = True
            else:
                if self.output_dimension is not None:
                    number_of_components = self.output_dimension
                    bool_index[:number_of_components] = True

            if comp_list is not None:
                for ifactors in comp_list:
                    bool_index[ifactors] = True
                number_of_components = len(comp_list)
            factors = factors[:,bool_index]
            if diff_order > 0 and smoothing is None:
                factors = np.diff(factors, diff_order, axis = 0)
            if smoothing is not None:
                from hyperspy.signals.spectrum import Spectrum
                sfactors = Spectrum({'data' : factors.T})
                if smoothing['algorithm'] == 'savitzky_golay':
                    sfactors.smooth_savitzky_golay(
                        number_of_points = smoothing['number_of_points'],
                        polynomial_order = smoothing['polynomial_order'],
                        differential_order = diff_order)
                if smoothing['algorithm'] == 'tv':
                    sfactors.smooth_tv(
                        smoothing_parameter= smoothing['smoothing_parameter'],
                        differential_order = diff_order)
                    factors = sfactors.data.T
            
            if mask is not None:
                factors = factors[mask, :]

            # first centers and scales data
            factors,invsqcovmat = centering_and_whitening(factors)
            if algorithm != 'sklearn_fastica':
                to_exec = 'target.ica_node=mdp.nodes.%sNode(' % algorithm
                for key, value in kwargs.iteritems():
                    to_exec += '%s=%s,' % (key, value)
                to_exec += ')'
                exec(to_exec)
                target.ica_node.train(factors)
                unmixing_matrix = target.ica_node.get_recmatrix()
            else:
                target.ica_node = sklearn.decomposition.FastICA(**kwargs)
                target.ica_node.whiten = False
                target.ica_node.fit(factors)
                unmixing_matrix = target.ica_node.unmixing_matrix_
            target.unmixing_matrix = np.dot(unmixing_matrix, invsqcovmat)
            self._unmix_factors(target)
            target.ica_scores = self._get_ica_scores(target)
            target.ica_algorithm = algorithm

    def reverse_ic(self, ic_n, on_peaks = False):
        """Reverse the independent component

        Parameters
        ----------
        ic_n : list or int
            component index/es

        Examples
        -------
        >>> s = load('some_file')
        >>> s.decomposition(True) # perform PCA
        >>> s.independent_components_analysis(3)  # perform ICA on 3 PCs
        >>> s.reverse_ic(1) # reverse IC 1
        >>> s.reverse_ic((0, 2)) # reverse ICs 0 and 2
        """
        if on_peaks:
            target=self.peak_mva_results
        else:
            target=self.mva_results

        for i in [ic_n,]:
            target.ic[:,i] *= -1
            target.unmixing_matrix[i,:] *= -1

    def _unmix_factors(self,target):
        w = target.unmixing_matrix
        n = len(w)
        if target.explained_variance is not None:
            # The output of ICA is not sorted in any way what makes it difficult
            # to compare results from different unmixings. The following code
            # is an experimental attempt to sort them in a more predictable way
            sorting_indexes = np.argsort(np.dot(target.explained_variance[:n],
                np.abs(w.T)))[::-1]
            w[:] = w[sorting_indexes,:]
        target.ic = np.dot(target.factors[:,:n], w.T)
        n_channels = target.ic.shape[0]
        for i in xrange(n):
            neg_channels = np.sum(target.ic[:,i] < 0)
            if neg_channels > n_channels/2.:
                self.reverse_ic(i)
                print("IC %i reversed" % i)

    def _get_ica_scores(self,target):
        """
        Returns the ICA score matrix (formerly known as the recmatrix)
        """
        W = target.scores.T[:target.ic.shape[1],:]
        Q = np.linalg.inv(target.unmixing_matrix.T)
        return np.dot(Q,W)

    @do_not_replot
    def _calculate_recmatrix(self, components = None, mva_type=None,
                             on_peaks=False):
        """
        Rebuilds SIs from selected components

        Parameters
        ------------
        target : target or self.peak_mva_results
        components : None, int, or list of ints
             if None, rebuilds SI from all components
             if int, rebuilds SI from components in range 0-given int
             if list of ints, rebuilds SI from only components in given list
        mva_type : string, currently either 'decomposition' or 'ica'
             (not case sensitive)

        Returns
        -------
        Signal instance
        """

        target=self._get_target(on_peaks)

        if mva_type.lower() == 'decomposition':
            factors = target.factors
            scores = target.scores.T
        elif mva_type.lower() == 'ica':
            factors = target.ic
            scores = self._get_ica_scores(target)
        if components is None:
            a = np.atleast_3d(np.dot(factors,scores))
            signal_name = 'model from %s with %i components' % (
            mva_type,factors.shape[1])
        elif hasattr(components, '__iter__'):
            tfactors = np.zeros((factors.shape[0],len(components)))
            tscores = np.zeros((len(components),scores.shape[1]))
            for i in xrange(len(components)):
                tfactors[:,i] = factors[:,components[i]]
                tscores[i,:] = scores[components[i],:]
            a = np.atleast_3d(np.dot(tfactors, tscores))
            signal_name = 'model from %s with components %s' % (
            mva_type,components)
        else:
            a = np.atleast_3d(np.dot(factors[:,:components],
                                     scores[:components,:]))
            signal_name = 'model from %s with %i components' % (
            mva_type,components)

        self._unfolded4decomposition = self.unfold_if_multidim()

        sc = self.deepcopy()

        import hyperspy.signals.spectrum
        #if self.mapped_parameters.record_by==spectrum:
        sc.data = a.T.squeeze()
        #else:
        #    sc.data = a.squeeze()
        sc.mapped_parameters.title += signal_name
        if self._unfolded4decomposition is True:
            self.fold()
            sc.history = ['unfolded']
            sc.fold()
        return sc

    def get_decomposition_model(self, components=None, on_peaks=False):
        """Return the spectrum generated with the selected number of principal
        components

        Parameters
        ------------
        components : None, int, or list of ints
             if None, rebuilds SI from all components
             if int, rebuilds SI from components in range 0-given int
             if list of ints, rebuilds SI from only components in given list

        Returns
        -------
        Signal instance
        """
        rec=self._calculate_recmatrix(components=components,
                                      mva_type='decomposition',
                                      on_peaks=on_peaks)
        rec.residual=rec.copy()
        rec.residual.data=self.data-rec.data
        return rec

    def get_ica_model(self,components = None, on_peaks=False):
        """Return the spectrum generated with the selected number of
        independent components

        Parameters
        ------------
        components : None, int, or list of ints
             if None, rebuilds SI from all components
             if int, rebuilds SI from components in range 0-given int
             if list of ints, rebuilds SI from only components in given list

        Returns
        -------
        Signal instance
        """
        rec=self._calculate_recmatrix(components=components, mva_type='ica',
                                      on_peaks=on_peaks)
        rec.residual=rec.copy()
        rec.residual.data=self.data-rec.data
        return rec
        
    @auto_replot
    def energy_center(self):
        """Subtract the mean energy pixel by pixel"""
        print "\nCentering the energy axis"
        self._energy_mean = np.mean(self.data, -1)[..., np.newaxis]
        self.data = (self.data - self._energy_mean)
    
    @auto_replot
    def undo_energy_center(self):
        if hasattr(self,'_energy_mean'):
            self.data = (self.data + self._energy_mean)
    
    @auto_replot
    def normalize_variance(self, on_peaks=False):
        if on_peaks:
            d=self.mapped_parameters.peak_chars
        else:
            d=self.data
        self._std = np.std(d, -1)[..., np.newaxis]
        d /= self._std

    @auto_replot
    def undo_normalize_variance(self, on_peaks=False):
        if on_peaks:
            d=self.mapped_parameters.peak_chars
        else:
            d=self.data
        if hasattr(self,'_std'):
            d *= self._std

    def plot_lev(self, n=50, on_peaks=False):
        """Plot the principal components LEV up to the given number

        Parameters
        ----------
        n : int
        """
        target = self._get_target(on_peaks)
        if target.explained_variance==None:
            self.decomposition()
        if n>target.explained_variance.shape[0]:
            n=target.explained_variance.shape[0]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(range(n), target.explained_variance[:n], 'o')
        ax.semilogy()
        ax.set_title('Log(eigenvalues)')
        ax.set_xlabel('Principal component')
        plt.draw()
        plt.show()
        return ax

    def plot_explained_variance_ratio(self,n=50,on_peaks=False):
        """Plot the principal components explained variance up to the given
        number

        Parameters
        ----------
        n : int
        """
        target = self._get_target(on_peaks)
        if n > target.explained_variance.shape[0]:
            n=target.explained_variance.shape[0]
        cumu = np.cumsum(target.explained_variance) / np.sum(
                                                     target.explained_variance)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(range(n), cumu[:n])
        ax.set_xlabel('Principal component')
        ax.set_ylabel('Explained variance ratio')
        plt.draw()
        plt.show()
        return ax

    def als(self, **kwargs):
        """Alternate Least Squares imposing positivity constraints
        to the result of a previous ICA

        Stores in result in self.als_out

        Parameters
        ----------
        thresh : float
            default=.001
        nonnegS : bool
            Impose non-negative constraint of the components. Default: True
        nonnegC : bool
            Impose non-negative constraint of the maps. Default: True

        See also
        -------
        plot_als_ic_maps, plot_als_ic
        """
        shape = (self.data.shape[2], self.data.shape[1],-1)
        if hasattr(self, 'ic') and (self.ic is not None):
            also = utils.ALS(self, **kwargs)
            self.als_ic = also['S']
            self.als_maps = also['CList'].reshape(shape, order = 'C')
            self.als_output = also

    def plot_als_ic_maps(self):
        """Same as plot_ic_maps for the ALS results"""
        return self.plot_independent_components_maps(recmatrix =
        self.als_output['CList'].T, ic = self.als_ic)

    def plot_als_ic(self):
        """Same as plot_independent_componets for the ALS results"""
        self.plot_independent_components(ic = self.als_ic)

    def save_als_ica_results(self, elements = None,
    format = preferences.General.default_file_format, image_format = 'tif'):
        """Same as save_ica_results for the ALS results"""
        self.save_ica_results(elements = elements, image_format = image_format,
        recmatrix = self.als_output['CList'].T, ic = self.als_ic)

    def normalize_poissonian_noise(self, navigation_mask = None,
                                   signal_mask = None, return_masks = False):
        """
        Scales the SI following Surf. Interface Anal. 2004; 36: 203–212 to
        "normalize" the poissonian data for decomposition analysis

        Parameters
        ----------
        navigation_mask : boolen numpy array
        signal_mask  : boolen numpy array
        """
        messages.information(
            "Scaling the data to normalize the (presumably) Poissonian noise")
        refold = self.unfold_if_multidim()
        dc = self.data
        if navigation_mask is None:
            navigation_mask = slice(None)
        else:
            navigation_mask = navigation_mask.ravel()
        if signal_mask is None:
            signal_mask = slice(None)
        # Rescale the data to gaussianize the poissonian noise
        aG = dc[:,signal_mask][navigation_mask,:].sum(1).squeeze()
        bH = dc[:,signal_mask][navigation_mask,:].sum(0).squeeze()
        # Checks if any is negative
        if (aG < 0).any() or (bH < 0).any():
            messages.warning_exit(
            "Data error: negative values\n"
            "Are you sure that the data follow a poissonian distribution?")
        # Update the spatial and energy masks so it does not include rows
        # or colums that sum zero.
        aG0 = (aG == 0)
        bH0 = (bH == 0)
        if aG0.any():
            if isinstance(navigation_mask, slice):
                # Convert the slice into a mask before setting its values
                navigation_mask = np.ones((self.data.shape[0]),dtype = 'bool')
            # Set colums summing zero as masked
            navigation_mask[aG0] = False
            aG = aG[aG0 == False]
        if bH0.any():
            if isinstance(signal_mask, slice):
                # Convert the slice into a mask before setting its values
                signal_mask = np.ones((self.data.shape[1]), dtype = 'bool')
            # Set rows summing zero as masked
            signal_mask[bH0] = False
            bH = bH[bH0 == False]
        self._root_aG = np.sqrt(aG)[:, np.newaxis]
        self._root_bH = np.sqrt(bH)[np.newaxis, :]
        dc[:,signal_mask][navigation_mask,:] = \
            (dc[:,signal_mask][navigation_mask,:] /
                (self._root_aG * self._root_bH))
        if refold is True:
            print "Automatically refolding the SI after scaling"
            self.fold()
        if return_masks is True:
            if isinstance(navigation_mask, slice):
                navigation_mask = None
            if isinstance(signal_mask, slice):
                signal_mask = None
            return navigation_mask, signal_mask

    def undo_treatments(self, on_peaks=False):
        """Undo normalize_poissonian_noise"""
        print "Undoing data pre-treatments"
        if on_peaks:
            self.mapped_parameters.peak_chars=self._data_before_treatments
            del self._data_before_treatments
        else:
            self.data=self._data_before_treatments
            del self._data_before_treatments

    def peak_pca(self,normalize_poissonian_noise = False, algorithm = 'svd', 
                 output_dimension = None, navigation_mask = None, 
                 signal_mask = None, center = False, normalize_variance = False, 
                 var_array = None, var_func = None, polyfit = None):
        """Performs Principal Component Analysis on peak characteristic data.

        Requires that you have run the peak_char_stack function on your stack of
        images.

        Parameters
        ----------

        normalize_poissonian_noise : bool
            If True, scale the SI to normalize Poissonian noise
        algorithm : {'svd', 'fast_svd', 'mlpca', 'fast_mlpca'}
        output_dimension : None or int
            number of PCA to keep
        navigation_mask : boolean numpy array
        signal_mask : boolean numpy array
        center : bool
            Perform energy centering before PCA
        normalize_variance : bool
            Perform whitening before PCA
        var_array : numpy array
            Array of variance for the maximum likelihood PCA algorithm
        var_func : function or numpy array
            If function, it will apply it to the dataset to obtain the
            var_array. Alternatively, it can a an array with the coefficients
            of a polynomial.
        polyfit :

        """
        self.decomposition(on_peaks=True,
               normalize_poissonian_noise = normalize_poissonian_noise,
               algorithm = algorithm, output_dimension = output_dimension, 
               navigation_mask = navigation_mask, signal_mask = signal_mask, 
               center = center, normalize_variance = normalize_variance, 
               var_array = var_array, var_func = var_func, polyfit = polyfit)

    def peak_ica(self, number_of_components, algorithm = 'CuBICA',
                 diff_order = 1, factors=None, comp_list=None, mask=None,
                 on_peaks=False, **kwds):
        """Independent components analysis on peak characteristic data.

        Requires that you have run the peak_char_stack function on your stack of
        images.

        Available algorithms: FastICA, JADE, CuBICA, and TDSEP

        Parameters
        ----------
        number_of_components : int
            number of principal components to pass to the ICA algorithm
        algorithm : {FastICA, JADE, CuBICA, TDSEP}
        diff : bool
        diff_order : int
        factors : numpy array
            externally provided factors
        comp_list : boolen numpy array
            choose the components to use by the boolen list. It permits to
            choose non contiguous components.
        mask : numpy boolean array with the same dimension as the PC
            If not None, only the selected channels will be used by the
            algorithm.
        """
        self.independent_components_analysis(number_of_components=
                                             number_of_components, 
                                             on_peaks=True,algorithm=algorithm, 
                                             diff_order=diff_order,
                                             factors=factors, 
                                             comp_list=comp_list, mask=mask)

class MVA_Results(object):
    def __init__(self):
        self.factors = None
        self.scores = None
        self.explained_variance = None
        self.decomposition_algorithm = None
        self.ica_algorithm = None
        self.centered = None
        self.poissonian_noise_normalized = None
        self.output_dimension = None
        self.unfolded = None
        self.original_shape = None
        self.ica_node=None
        # Demixing matrix
        self.unmixing_matrix = None

    def save(self, filename):
        """Save the result of the decomposition and demixing analysis

        Parameters
        ----------
        filename : string
        """
        np.savez(filename, factors = self.factors, v = self.scores,
        explained_variance=self.explained_variance,
        decomposition_algorithm=self.decomposition_algorithm,
        centered=self.centered, output_dimension=self.output_dimension,
        normalize_variance=self.variance_normalized,
        poissonian_noise_normalized=self.poissonian_noise_normalized,
        unmixing_matrix = self.unmixing_matrix,
        ica_algorithm=self.ica_algorithm)

    def load(self, filename):
        """Load the results of a previous decomposition and demixing analysis
        from a file.

        Parameters
        ----------
        filename : string
        """
        decomposition = np.load(filename)
        for key in decomposition.files:
            exec('self.%s = decomposition[\'%s\']' % (key, key))
        print "\n%s loaded correctly" %  filename

        # For compatibility with old version ##################

        if hasattr(self, 'algorithm'):
            self.decomposition_algorithm = self.algorithm
            del self.algorithm
        if hasattr(self, 'V'):
            self.explained_variance = self.V
            del self.V
        if hasattr(self, 'w'):
            self.unmixing_matrix = self.w
            del self.w
            
        if hasattr(self, 'variance2one'):
            self.variance_normalized = self.variance2one
            del self.variance2one
            
        if hasattr(self, 'pca_algorithm'):
            self.decomposition_algorithm = self.pca_algorithm
            del self.pca_algorithm
            
        if hasattr(self, 'v'):
            self.scores = self.v
            del self.v
            
        if hasattr(self, 'pc'):
            self.scores = self.pc
            del self.pc

        #######################################################
        
        # Output_dimension is an array after loading, convert it to int            
        if hasattr(self, 'output_dimension'):
            try:
                self.output_dimension = int(self.output_dimension)
            except TypeError:
                self.output_dimension = None
            
        defaults = {
        'centered' : False,
        'variance_normalized' : False,
        'poissonian_noise_normalized' : False,
        'output_dimension' : None,
        'decomposition_algorithm' : None
        }
        for attrib in defaults.keys():
            if not hasattr(self, attrib):
                exec('self.%s = %s' % (attrib, defaults[attrib]))
        self.summary()

    def summary(self):
        """Prints a summary of the decomposition and demixing parameters to the 
        stdout
        """
        print
        print "Decomposition parameters:"
        print "-------------------------"
        print "Decomposition algorithm : ", self.decomposition_algorithm
        print "Poissonian noise normalization : %s" % \
        self.poissonian_noise_normalized
        print "Energy centered : %s" % self.centered
        print "Variance normalized : %s" % self.variance_normalized
        print "Output dimension : %s" % self.output_dimension
        
        if self.ica_algorithm is not None:
            print
            print "Demixing parameters:"
            print "---------------------"
            print "ICA algorithm : %s" % self.ica_algorithm
            print "Number of components : %i" % len(self.unmixing_matrix)


    def crop_decomposition_dimension(self, n):
        """
        Crop the score matrix up to the given number.

        It is mainly useful to save memory and reduce the storage size
        """
        print "trimming to %i dimensions" % n
        self.scores = self.scores[:,:n]
        if self.explained_variance is not None:
            self.explained_variance = self.explained_variance[:n]
        self.factors = self.factors[:,:n]
