# -*- coding: utf-8 -*-
# Copyright 2007-2011 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.


from __future__ import division
import types

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
try:
    import mdp
    mdp_installed = True
except:
    mdp_installed = False


from hyperspy.misc.machine_learning import import_sklearn
import hyperspy.misc.io.tools as io_tools
from hyperspy.learn.svd_pca import svd_pca
from hyperspy.learn.mlpca import mlpca
from hyperspy import messages
from hyperspy.decorators import do_not_replot
from scipy import linalg
from hyperspy.misc.machine_learning.orthomax import orthomax


def centering_and_whitening(X):
    X = X.T
    # Centering the columns (ie the variables)
    X = X - X.mean(axis=-1)[:, np.newaxis]
    # Whitening and preprocessing by PCA
    u, d, _ = linalg.svd(X, full_matrices=False)
    del _
    K = (u / (d / np.sqrt(X.shape[1]))).T
    del u, d
    X1 = np.dot(K, X)
    return X1.T, K


class MVA():

    """
    Multivariate analysis capabilities for the Spectrum class.

    """

    def __init__(self):
        if not hasattr(self, 'learning_results'):
            self.learning_results = LearningResults()

    @do_not_replot
    def decomposition(self,
                      normalize_poissonian_noise=False,
                      algorithm='svd',
                      output_dimension=None,
                      centre=None,
                      auto_transpose=True,
                      navigation_mask=None,
                      signal_mask=None,
                      var_array=None,
                      var_func=None,
                      polyfit=None,
                      reproject=None,
                      **kwargs):
        """Decomposition with a choice of algorithms

        The results are stored in self.learning_results

        Parameters
        ----------
        normalize_poissonian_noise : bool
            If True, scale the SI to normalize Poissonian noise

        algorithm : 'svd' | 'fast_svd' | 'mlpca' | 'fast_mlpca' | 'nmf' |
            'sparse_pca' | 'mini_batch_sparse_pca'

        output_dimension : None or int
            number of components to keep/calculate

        centre : None | 'variables' | 'trials'
            If None no centring is applied. If 'variable' the centring will be
            performed in the variable axis. If 'trials', the centring will be
            performed in the 'trials' axis. It only has effect when using the
            svd or fast_svd algorithms

        auto_transpose : bool
            If True, automatically transposes the data to boost performance.
            Only has effect when using the svd of fast_svd algorithms.

        navigation_mask : boolean numpy array
            The navigation locations marked as True are not used in the
            decompostion.

        signal_mask : boolean numpy array
            The signal locations marked as True are not used in the
            decomposition.

        var_array : numpy array
            Array of variance for the maximum likelihood PCA algorithm

        var_func : function or numpy array
            If function, it will apply it to the dataset to obtain the
            var_array. Alternatively, it can a an array with the coefficients
            of a polynomial.

        polyfit :

        reproject : None | signal | navigation | both
            If not None, the results of the decomposition will be projected in
            the selected masked area.


        See also
        --------
        plot_decomposition_factors, plot_decomposition_loadings, plot_lev

        """
        # Check if it is the wrong data type
        if self.data.dtype.char not in ['e', 'f', 'd']:  # If not float
            messages.warning(
                'To perform a decomposition the data must be of the float type.'
                ' You can change the type using the change_dtype method'
                ' e.g. s.change_dtype(\'float64\')\n'
                'Nothing done.')
            return

        if self.axes_manager.navigation_size < 2:
            raise AttributeError("It is not possible to decompose a dataset "
                                 "with navigation_dimension < 2")
        # backup the original data
        self._data_before_treatments = self.data.copy()

        if algorithm == 'mlpca':
            if normalize_poissonian_noise is True:
                messages.warning(
                    "It makes no sense to do normalize_poissonian_noise with "
                    "the MLPCA algorithm. Therefore, "
                    "normalize_poissonian_noise is set to False")
                normalize_poissonian_noise = False
            if output_dimension is None:
                raise ValueError("With the mlpca algorithm the "
                                 "output_dimension must be expecified")

        # Apply pre-treatments
        # Transform the data in a line spectrum
        self._unfolded4decomposition = self.unfold_if_multidim()
        try:
            if hasattr(navigation_mask, 'ravel'):
                navigation_mask = navigation_mask.ravel()

            if hasattr(signal_mask, 'ravel'):
                signal_mask = signal_mask.ravel()

            # Normalize the poissonian noise
            # TODO this function can change the masks and this can cause
            # problems when reprojecting
            if normalize_poissonian_noise is True:
                self.normalize_poissonian_noise(
                    navigation_mask=navigation_mask,
                    signal_mask=signal_mask,)
            messages.information('Performing decomposition analysis')
            # The rest of the code assumes that the first data axis
            # is the navigation axis. We transpose the data if that is not the
            # case.
            dc = (self.data if self.axes_manager[0].index_in_array == 0
                  else self.data.T)
            # set the output target (peak results or not?)
            target = self.learning_results

            # Transform the None masks in slices to get the right behaviour
            if navigation_mask is None:
                navigation_mask = slice(None)
            else:
                navigation_mask = ~navigation_mask
            if signal_mask is None:
                signal_mask = slice(None)
            else:
                signal_mask = ~signal_mask

            # WARNING: signal_mask and navigation_mask values are now their
            # negaties i.e. True -> False and viceversa. However, the
            # stored value (at the end of the method) coincides with the
            # input masks

            # Reset the explained_variance which is not set by all the
            # algorithms
            explained_variance = None
            explained_variance_ratio = None
            mean = None

            if algorithm == 'svd':
                factors, loadings, explained_variance, mean = svd_pca(
                    dc[:, signal_mask][navigation_mask, :], centre=centre,
                    auto_transpose=auto_transpose)

            elif algorithm == 'fast_svd':
                factors, loadings, explained_variance, mean = svd_pca(
                    dc[:, signal_mask][navigation_mask, :],
                    fast=True,
                    output_dimension=output_dimension,
                    centre=centre,
                    auto_transpose=auto_transpose)

            elif algorithm == 'sklearn_pca':
                if import_sklearn.sklearn_installed is False:
                    raise ImportError(
                        'sklearn is not installed. Nothing done')
                sk = import_sklearn.sklearn.decomposition.PCA(**kwargs)
                sk.n_components = output_dimension
                loadings = sk.fit_transform((
                    dc[:, signal_mask][navigation_mask, :]))
                factors = sk.components_.T
                explained_variance = sk.explained_variance_
                mean = sk.mean_
                centre = 'trials'

            elif algorithm == 'nmf':
                if import_sklearn.sklearn_installed is False:
                    raise ImportError(
                        'sklearn is not installed. Nothing done')
                sk = import_sklearn.sklearn.decomposition.NMF(**kwargs)
                sk.n_components = output_dimension
                loadings = sk.fit_transform((
                    dc[:, signal_mask][navigation_mask, :]))
                factors = sk.components_.T

            elif algorithm == 'sparse_pca':
                if import_sklearn.sklearn_installed is False:
                    raise ImportError(
                        'sklearn is not installed. Nothing done')
                sk = import_sklearn.sklearn.decomposition.SparsePCA(
                    output_dimension, **kwargs)
                loadings = sk.fit_transform(
                    dc[:, signal_mask][navigation_mask, :])
                factors = sk.components_.T

            elif algorithm == 'mini_batch_sparse_pca':
                if import_sklearn.sklearn_installed is False:
                    raise ImportError(
                        'sklearn is not installed. Nothing done')
                sk = import_sklearn.sklearn.decomposition.MiniBatchSparsePCA(
                    output_dimension, **kwargs)
                loadings = sk.fit_transform(
                    dc[:, signal_mask][navigation_mask, :])
                factors = sk.components_.T

            elif algorithm == 'mlpca' or algorithm == 'fast_mlpca':
                print "Performing the MLPCA training"
                if output_dimension is None:
                    raise ValueError(
                        "For MLPCA it is mandatory to define the "
                        "output_dimension")
                if var_array is None and var_func is None:
                    messages.information('No variance array provided.'
                                         'Supposing poissonian data')
                    var_array = dc[:, signal_mask][navigation_mask, :]

                if var_array is not None and var_func is not None:
                    raise ValueError(
                        "You have defined both the var_func and var_array "
                        "keywords."
                        "Please, define just one of them")
                if var_func is not None:
                    if hasattr(var_func, '__call__'):
                        var_array = var_func(
                            dc[signal_mask, ...][:, navigation_mask])
                    else:
                        try:
                            var_array = np.polyval(polyfit, dc[signal_mask,
                                                               navigation_mask])
                        except:
                            raise ValueError(
                                'var_func must be either a function or an array'
                                'defining the coefficients of a polynom')
                if algorithm == 'mlpca':
                    fast = False
                else:
                    fast = True
                U, S, V, Sobj, ErrFlag = mlpca(
                    dc[:, signal_mask][navigation_mask, :],
                    var_array, output_dimension, fast=fast)
                loadings = U * S
                factors = V
                explained_variance_ratio = S ** 2 / Sobj
                explained_variance = S ** 2 / len(factors)
            else:
                raise ValueError('Algorithm not recognised. '
                                 'Nothing done')

            # We must calculate the ratio here because otherwise the sum
            # information can be lost if the user call
            # crop_decomposition_dimension
            if explained_variance is not None and \
                    explained_variance_ratio is None:
                explained_variance_ratio = \
                    explained_variance / explained_variance.sum()

            # Store the results in learning_results
            target.factors = factors
            target.loadings = loadings
            target.explained_variance = explained_variance
            target.explained_variance_ratio = explained_variance_ratio
            target.decomposition_algorithm = algorithm
            target.poissonian_noise_normalized = \
                normalize_poissonian_noise
            target.output_dimension = output_dimension
            target.unfolded = self._unfolded4decomposition
            target.centre = centre
            target.mean = mean

            if output_dimension and factors.shape[1] != output_dimension:
                target.crop_decomposition_dimension(output_dimension)

            # Delete the unmixing information, because it'll refer to a previous
            # decompositions
            target.unmixing_matrix = None
            target.bss_algorithm = None

            if self._unfolded4decomposition is True:
                folding = \
                    self.metadata._HyperSpy.Folding
                target.original_shape = folding.original_shape

            # Reproject
            if mean is None:
                mean = 0
            if reproject in ('navigation', 'both'):
                if algorithm not in ('nmf', 'sparse_pca',
                                     'mini_batch_sparse_pca'):
                    loadings_ = np.dot(dc[:, signal_mask] - mean, factors)
                else:
                    loadings_ = sk.transform(dc[:, signal_mask])
                target.loadings = loadings_
            if reproject in ('signal', 'both'):
                if algorithm not in ('nmf', 'sparse_pca',
                                     'mini_batch_sparse_pca'):
                    factors = np.dot(np.linalg.pinv(loadings),
                                     dc[navigation_mask, :] - mean).T
                    target.factors = factors
                else:
                    messages.information("Reprojecting the signal is not yet "
                                         "supported for this algorithm")
                    if reproject == 'both':
                        reproject = 'signal'
                    else:
                        reproject = None

            # Rescale the results if the noise was normalized
            if normalize_poissonian_noise is True:
                target.factors[:] *= self._root_bH.T
                target.loadings[:] *= self._root_aG

            # Set the pixels that were not processed to nan
            if not isinstance(signal_mask, slice):
                # Store the (inverted, as inputed) signal mask
                target.signal_mask = ~signal_mask.reshape(
                    self.axes_manager._signal_shape_in_array)
                if reproject not in ('both', 'signal'):
                    factors = np.zeros((dc.shape[-1], target.factors.shape[1]))
                    factors[signal_mask == True, :] = target.factors
                    factors[signal_mask == False, :] = np.nan
                    target.factors = factors
            if not isinstance(navigation_mask, slice):
                # Store the (inverted, as inputed) navigation mask
                target.navigation_mask = ~navigation_mask.reshape(
                    self.axes_manager._navigation_shape_in_array)
                if reproject not in ('both', 'navigation'):
                    loadings = np.zeros(
                        (dc.shape[0], target.loadings.shape[1]))
                    loadings[navigation_mask == True, :] = target.loadings
                    loadings[navigation_mask == False, :] = np.nan
                    target.loadings = loadings
        finally:
            # undo any pre-treatments
            self.undo_treatments()

            if self._unfolded4decomposition is True:
                self.fold()
                self._unfolded4decomposition is False

    def blind_source_separation(self,
                                number_of_components=None,
                                algorithm='sklearn_fastica',
                                diff_order=1,
                                factors=None,
                                comp_list=None,
                                mask=None,
                                on_loadings=False,
                                pretreatment=None,
                                **kwargs):
        """Blind source separation (BSS) on the result on the
        decomposition.

        Available algorithms: FastICA, JADE, CuBICA, and TDSEP

        Parameters
        ----------
        number_of_components : int
            number of principal components to pass to the BSS algorithm
        algorithm : {FastICA, JADE, CuBICA, TDSEP}
        diff_order : int
            Sometimes it is convenient to perform the BSS on the derivative
            of the signal. If diff_order is 0, the signal is not differentiated.
        factors : numpy.array
            Factors to decompose. If None, the BSS is performed on the result
            of a previous decomposition.
        comp_list : boolen numpy array
            choose the components to use by the boolean list. It permits
             to choose non contiguous components.
        mask : numpy boolean array with the same dimension as the signal
            If not None, the signal locations marked as True (masked) will
            not be passed to the BSS algorithm.
        on_loadings : bool
            If True, perform the BSS on the loadings of a previous
            decomposition. If False, performs it on the factors.
        pretreatment: dict

        **kwargs : extra key word arguments
            Any keyword arguments are passed to the BSS algorithm.

        """
        target = self.learning_results
        if not hasattr(target, 'factors') or target.factors is None:
            raise AttributeError(
                'A decomposition must be performed before blind '
                'source seperation or factors must be provided.')
        else:
            if factors is None:
                if on_loadings:
                    factors = target.loadings
                else:
                    factors = target.factors
            bool_index = np.zeros((factors.shape[0]), dtype='bool')
            if number_of_components is not None:
                bool_index[:number_of_components] = True
            else:
                if target.output_dimension is not None:
                    number_of_components = target.output_dimension
                    bool_index[:number_of_components] = True

            if comp_list is not None:
                for ifactors in comp_list:
                    bool_index[ifactors] = True
                number_of_components = len(comp_list)
            factors = factors[:, bool_index]

            if pretreatment is not None:
                from hyperspy._signals.spectrum import Spectrum
                sfactors = Spectrum(factors.T)
                if pretreatment['algorithm'] == 'savitzky_golay':
                    sfactors.smooth_savitzky_golay(
                        number_of_points=pretreatment[
                            'number_of_points'],
                        polynomial_order=pretreatment[
                            'polynomial_order'],
                        differential_order=diff_order)
                if pretreatment['algorithm'] == 'tv':
                    sfactors.smooth_tv(
                        smoothing_parameter=pretreatment[
                            'smoothing_parameter'],
                        differential_order=diff_order)
                factors = sfactors.data.T
                if pretreatment['algorithm'] == 'butter':
                    b, a = sp.signal.butter(pretreatment['order'],
                                            pretreatment['cutoff'], pretreatment['type'])
                    for i in range(factors.shape[1]):
                        factors[:, i] = sp.signal.filtfilt(b, a,
                                                           factors[:, i])
            elif diff_order > 0:
                factors = np.diff(factors, diff_order, axis=0)

            if mask is not None:
                factors = factors[~mask]

            # first center and scale the data
            factors, invsqcovmat = centering_and_whitening(factors)
            if algorithm == 'orthomax':
                _, unmixing_matrix = orthomax(factors, **kwargs)
                unmixing_matrix = unmixing_matrix.T

            elif algorithm == 'sklearn_fastica':
                # if sklearn_installed is False:
                    # raise ImportError(
                    #'sklearn is not installed. Nothing done')
                if 'tol' not in kwargs:
                    kwargs['tol'] = 1e-10
                target.bss_node = import_sklearn.FastICA(
                    **kwargs)
                target.bss_node.whiten = False
                target.bss_node.fit(factors)
                try:
                    unmixing_matrix = target.bss_node.unmixing_matrix_
                except AttributeError:
                    # unmixing_matrix was renamed to components
                    unmixing_matrix = target.bss_node.components_
            else:
                if mdp_installed is False:
                    raise ImportError(
                        'MDP is not installed. Nothing done')
                to_exec = 'target.bss_node=mdp.nodes.%sNode(' % algorithm
                for key, value in kwargs.iteritems():
                    to_exec += '%s=%s,' % (key, value)
                to_exec += ')'
                exec(to_exec)
                target.bss_node.train(factors)
                unmixing_matrix = target.bss_node.get_recmatrix()

            target.unmixing_matrix = np.dot(unmixing_matrix, invsqcovmat)
            self._unmix_factors(target)
            self._unmix_loadings(target)
            self._auto_reverse_bss_component(target)
            target.bss_algorithm = algorithm

    def normalize_factors(self, which='bss', by='area', sort=True):
        """Normalises the factors and modifies the loadings
        accordingly

        Parameters
        ----------
        which : 'bss' | 'decomposition'
        by : 'max' | 'area'
        sort : bool

        """
        if which == 'bss':
            factors = self.learning_results.bss_factors
            loadings = self.learning_results.bss_loadings
            if factors is None:
                raise UserWarning("This method can only be used after "
                                  "a blind source separation operation")
        elif which == 'decomposition':
            factors = self.learning_results.factors
            loadings = self.learning_results.loadings
            if factors is None:
                raise UserWarning("This method can only be used after"
                                  "a decomposition operation")
        else:
            raise ValueError("what must be bss or decomposition")

        if by == 'max':
            by = np.max
        elif by == 'area':
            by = np.sum
        else:
            raise ValueError("by must be max or mean")

        factors /= by(factors, 0)
        loadings *= by(factors, 0)
        sorting_indices = np.argsort(loadings.max(0))
        factors[:] = factors[:, sorting_indices]
        loadings[:] = loadings[:, sorting_indices]
        loadings[:] = loadings[:, sorting_indices]

    def reverse_bss_component(self, component_number):
        """Reverse the independent component

        Parameters
        ----------
        component_number : list or int
            component index/es

        Examples
        -------
        >>> s = load('some_file')
        >>> s.decomposition(True) # perform PCA
        >>> s.blind_source_separation(3)  # perform ICA on 3 PCs
        >>> s.reverse_bss_component(1) # reverse IC 1
        >>> s.reverse_bss_component((0, 2)) # reverse ICs 0 and 2
        """

        target = self.learning_results

        for i in [component_number, ]:
            target.bss_factors[:, i] *= -1
            target.bss_loadings[:, i] *= -1
            target.unmixing_matrix[i, :] *= -1

    def _unmix_factors(self, target):
        w = target.unmixing_matrix
        n = len(w)
        if target.explained_variance is not None:
            # The output of ICA is not sorted in any way what makes it difficult
            # to compare results from different unmixings. The following code
            # is an experimental attempt to sort them in a more predictable way
            sorting_indices = np.argsort(np.dot(target.explained_variance[:n],
                                                np.abs(w.T)))[::-1]
            w[:] = w[sorting_indices, :]
        target.bss_factors = np.dot(target.factors[:, :n], w.T)

    def _auto_reverse_bss_component(self, target):
        n_components = target.bss_factors.shape[1]
        for i in xrange(n_components):
            minimum = np.nanmin(target.bss_loadings[:, i])
            maximum = np.nanmax(target.bss_loadings[:, i])
            if minimum < 0 and -minimum > maximum:
                self.reverse_bss_component(i)
                print("IC %i reversed" % i)

    def _unmix_loadings(self, target):
        """
        Returns the ICA score matrix (formerly known as the recmatrix)
        """
        W = target.loadings.T[:target.bss_factors.shape[1], :]
        Q = np.linalg.inv(target.unmixing_matrix.T)
        target.bss_loadings = np.dot(Q, W).T

    @do_not_replot
    def _calculate_recmatrix(self, components=None, mva_type=None,):
        """
        Rebuilds SIs from selected components

        Parameters
        ------------
        components : None, int, or list of ints
             if None, rebuilds SI from all components
             if int, rebuilds SI from components in range 0-given int
             if list of ints, rebuilds SI from only components in given list
        mva_type : string, currently either 'decomposition' or 'bss'
             (not case sensitive)

        Returns
        -------
        Signal instance
        """

        target = self.learning_results

        if mva_type.lower() == 'decomposition':
            factors = target.factors
            loadings = target.loadings.T
        elif mva_type.lower() == 'bss':
            factors = target.bss_factors
            loadings = target.bss_loadings.T
        if components is None:
            a = np.dot(factors, loadings)
            signal_name = 'model from %s with %i components' % (
                mva_type, factors.shape[1])
        elif hasattr(components, '__iter__'):
            tfactors = np.zeros((factors.shape[0], len(components)))
            tloadings = np.zeros((len(components), loadings.shape[1]))
            for i in xrange(len(components)):
                tfactors[:, i] = factors[:, components[i]]
                tloadings[i, :] = loadings[components[i], :]
            a = np.dot(tfactors, tloadings)
            signal_name = 'model from %s with components %s' % (
                mva_type, components)
        else:
            a = np.dot(factors[:, :components],
                       loadings[:components, :])
            signal_name = 'model from %s with %i components' % (
                mva_type, components)

        self._unfolded4decomposition = self.unfold_if_multidim()

        sc = self.deepcopy()
        sc.data = a.T.reshape(self.data.shape)
        sc.metadata.General.title += signal_name
        if target.mean is not None:
            sc.data += target.mean
        if self._unfolded4decomposition is True:
            self.fold()
            sc.fold()
        return sc

    def get_decomposition_model(self, components=None):
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
        rec = self._calculate_recmatrix(components=components,
                                        mva_type='decomposition')
        return rec

    def get_bss_model(self, components=None):
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
        rec = self._calculate_recmatrix(components=components, mva_type='bss',)
        rec.residual = rec.copy()
        rec.residual.data = self.data - rec.data
        return rec

    def get_explained_variance_ratio(self):
        """Return the explained variation ratio of the PCA components as a
        Spectrum.

        Returns
        -------
        s : Spectrum
            Explained variation ratio.

        See Also:
        ---------

        `plot_explained_variance_ration`, `decomposition`,
        `get_decomposition_loadings`,
        `get_decomposition_factors`.

        """
        from hyperspy._signals.spectrum import Spectrum
        target = self.learning_results
        if target.explained_variance_ratio is None:
            raise AttributeError("The explained_variance_ratio attribute is "
                                 "`None`, did you forget to perform a PCA "
                                 "decomposition?")
        s = Spectrum(target.explained_variance_ratio)
        s.metadata.General.title = self.metadata.General.title + \
            "\nPCA Scree Plot"
        s.axes_manager[-1].name = 'Principal component index'
        s.axes_manager[-1].units = ''
        return s

    def plot_explained_variance_ratio(self, n=50, log=True):
        """Plot the decomposition explained variance ratio vs index number.

        Parameters
        ----------
        n : int
            Number of components.
        log : bool
            If True, the y axis uses a log scale.

        Returns
        -------
        ax : matplotlib.axes

        See Also:
        ---------

        `get_explained_variance_ration`, `decomposition`,
        `get_decomposition_loadings`,
        `get_decomposition_factors`.

        """
        s = self.get_explained_variance_ratio()
        if n < s.axes_manager[-1].size:
            s = s.isig[:n]
        s.plot()
        ax = s._plot.signal_plot.ax
        # ax.plot(range(n), target.explained_variance_ratio[:n], 'o',
        #         label=label)
        ax.set_ylabel("Explained variance ratio")
        ax.margins(0.05)
        ax.autoscale()
        ax.lines[0].set_marker("o")
        ax.lines[0].set_linestyle("None")
        if log is True:
            ax.semilogy()
        return ax

    def plot_cumulative_explained_variance_ratio(self, n=50):
        """Plot the principal components explained variance up to the
        given number

        Parameters
        ----------
        n : int
        """
        target = self.learning_results
        if n > target.explained_variance.shape[0]:
            n = target.explained_variance.shape[0]
        cumu = np.cumsum(target.explained_variance) / np.sum(
            target.explained_variance)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(range(n), cumu[:n])
        ax.set_xlabel('Principal component')
        ax.set_ylabel('Cumulative explained variance ratio')
        plt.draw()
        plt.show()
        return ax

    def normalize_poissonian_noise(self, navigation_mask=None,
                                   signal_mask=None):
        """
        Scales the SI following Surf. Interface Anal. 2004; 36: 203–212
        to "normalize" the poissonian data for decomposition analysis

        Parameters
        ----------
        navigation_mask : boolen numpy array
        signal_mask  : boolen numpy array
        """
        messages.information(
            "Scaling the data to normalize the (presumably)"
            " Poissonian noise")
        refold = self.unfold_if_multidim()
        # The rest of the code assumes that the first data axis
        # is the navigation axis. We transpose the data if that is not the
        # case.
        dc = (self.data if self.axes_manager[0].index_in_array == 0
              else self.data.T)
        if navigation_mask is None:
            navigation_mask = slice(None)
        else:
            navigation_mask = ~navigation_mask.ravel()
        if signal_mask is None:
            signal_mask = slice(None)
        else:
            signal_mask = ~signal_mask
        # Rescale the data to gaussianize the poissonian noise
        aG = dc[:, signal_mask][navigation_mask, :].sum(1).squeeze()
        bH = dc[:, signal_mask][navigation_mask, :].sum(0).squeeze()
        # Checks if any is negative
        if (aG < 0).any() or (bH < 0).any():
            raise ValueError(
                "Data error: negative values\n"
                "Are you sure that the data follow a poissonian "
                "distribution?")

        self._root_aG = np.sqrt(aG)[:, np.newaxis]
        self._root_bH = np.sqrt(bH)[np.newaxis, :]
        # We first disable numpy's warning when the result of an
        # operation produces nans
        np.seterr(invalid='ignore')
        dc[:, signal_mask][navigation_mask, :] /= (self._root_aG *
                                                   self._root_bH)
        # Enable numpy warning
        np.seterr(invalid=None)
        # Set the nans resulting from 0/0 to zero
        dc[:, signal_mask][navigation_mask, :] = \
            np.nan_to_num(dc[:, signal_mask][navigation_mask, :])

        if refold is True:
            print "Automatically refolding the SI after scaling"
            self.fold()

    def undo_treatments(self):
        """Undo normalize_poissonian_noise"""
        print "Undoing data pre-treatments"
        self.data = self._data_before_treatments
        del self._data_before_treatments


class LearningResults(object):
    # Decomposition
    factors = None
    loadings = None
    explained_variance = None
    explained_variance_ratio = None
    decomposition_algorithm = None
    poissonian_noise_normalized = None
    output_dimension = None
    mean = None
    centre = None
    # Unmixing
    bss_algorithm = None
    unmixing_matrix = None
    bss_factors = None
    bss_loadings = None
    # Shape
    unfolded = None
    original_shape = None
    # Masks
    navigation_mask = None
    signal_mask = None

    def save(self, filename, overwrite=None):
        """Save the result of the decomposition and demixing analysis
        Parameters
        ----------
        filename : string
        overwrite : {True, False, None}
            If True(False) overwrite(don't overwrite) the file if it exists.
            If None (default) ask what to do if file exists.
        """
        kwargs = {}
        for attribute in [
                v for v in dir(self) if not isinstance(getattr(self, v), types.MethodType) and not v.startswith('_')]:
            kwargs[attribute] = self.__getattribute__(attribute)
        # Check overwrite
        if overwrite is None:
            overwrite = io_tools.overwrite(filename)
        # Save, if all went well!
        if overwrite is True:
            np.savez(filename, **kwargs)

    def load(self, filename):
        """Load the results of a previous decomposition and
         demixing analysis from a file.
        Parameters
        ----------
        filename : string
        """
        decomposition = np.load(filename)
        for key, value in decomposition.iteritems():
            if value.dtype == np.dtype('object'):
                value = None
            setattr(self, key, value)
        print "\n%s loaded correctly" % filename
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
            del self.variance2one
        if hasattr(self, 'centered'):
            del self.centered
        if hasattr(self, 'pca_algorithm'):
            self.decomposition_algorithm = self.pca_algorithm
            del self.pca_algorithm
        if hasattr(self, 'ica_algorithm'):
            self.bss_algorithm = self.ica_algorithm
            del self.ica_algorithm
        if hasattr(self, 'v'):
            self.loadings = self.v
            del self.v
        if hasattr(self, 'scores'):
            self.loadings = self.scores
            del self.scores
        if hasattr(self, 'pc'):
            self.loadings = self.pc
            del self.pc
        if hasattr(self, 'ica_scores'):
            self.bss_loadings = self.ica_scores
            del self.ica_scores
        if hasattr(self, 'ica_factors'):
            self.bss_factors = self.ica_factors
            del self.ica_factors
        #######################################################
        # Output_dimension is an array after loading, convert it to int
        if hasattr(self, 'output_dimension') and self.output_dimension \
                is not None:
            self.output_dimension = int(self.output_dimension)
        self.summary()

    def summary(self):
        """Prints a summary of the decomposition and demixing parameters
         to the stdout
        """
        print
        print "Decomposition parameters:"
        print "-------------------------"
        print "Decomposition algorithm : ", self.decomposition_algorithm
        print "Poissonian noise normalization : %s" % \
            self.poissonian_noise_normalized
        print "Output dimension : %s" % self.output_dimension
        print "Centre : %s" % self.centre
        if self.bss_algorithm is not None:
            print
            print "Demixing parameters:"
            print "---------------------"
            print "BSS algorithm : %s" % self.bss_algorithm
            print "Number of components : %i" % len(self.unmixing_matrix)

    def crop_decomposition_dimension(self, n):
        """
        Crop the score matrix up to the given number.
        It is mainly useful to save memory and reduce the storage size
        """
        print "trimming to %i dimensions" % n
        self.loadings = self.loadings[:, :n]
        if self.explained_variance is not None:
            self.explained_variance = self.explained_variance[:n]
        self.factors = self.factors[:, :n]

    def _transpose_results(self):
        (self.factors, self.loadings, self.bss_factors,
            self.bss_loadings) = (self.loadings, self.factors,
                                  self.bss_loadings, self.bss_factors)
