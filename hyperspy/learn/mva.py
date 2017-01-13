# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
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


import types
import logging

import numpy as np
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
from hyperspy.learn.rpca import rpca_godec, orpca
from scipy import linalg
from hyperspy.misc.machine_learning.orthomax import orthomax
from hyperspy.misc.utils import stack, ordinal

_logger = logging.getLogger(__name__)


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


def get_derivative(signal, diff_axes, diff_order):
    if signal.axes_manager.signal_dimension == 1:
        signal = signal.diff(order=diff_order, axis=-1)
    else:
        # n-d signal case.
        # Compute the differences for each signal axis, unfold the
        # signal axes and stack the differences over the signal
        # axis.
        if diff_axes is None:
            diff_axes = signal.axes_manager.signal_axes
            iaxes = [axis.index_in_axes_manager
                     for axis in diff_axes]
        else:
            iaxes = diff_axes
        diffs = [signal.derivative(order=diff_order, axis=i)
                 for i in iaxes]
        for signal in diffs:
            signal.unfold()
        signal = stack(diffs, axis=-1)
        del diffs
    return signal


def _normalize_components(target, other, function=np.sum):
    coeff = function(target, axis=0)
    target /= coeff
    other *= coeff


class MVA():

    """
    Multivariate analysis capabilities for the Signal1D class.

    """

    def __init__(self):
        if not hasattr(self, 'learning_results'):
            self.learning_results = LearningResults()

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
                      return_info=False,
                      **kwargs):
        """Decomposition with a choice of algorithms

        The results are stored in self.learning_results

        Parameters
        ----------
        normalize_poissonian_noise : bool
            If True, scale the SI to normalize Poissonian noise
        algorithm : 'svd' | 'fast_svd' | 'mlpca' | 'fast_mlpca' | 'nmf' |
            'sparse_pca' | 'mini_batch_sparse_pca' | 'RPCA_GoDec' | 'ORPCA'
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
        reproject : None | signal | navigation | both
            If not None, the results of the decomposition will be projected in
            the selected masked area.
        return_info: bool, default False
            The result of the decomposition is stored internally. However, some algorithms generate some extra
            information that is not stored. If True (the default is False) return any extra information if available

        Returns
        -------
        (X, E) : (numpy array, numpy array)
            If 'algorithm' == 'RPCA_GoDec' or 'ORPCA' and 'return_info' is True,
            returns the low-rank (X) and sparse (E) matrices from robust PCA.

        See also
        --------
        plot_decomposition_factors, plot_decomposition_loadings, plot_lev

        """
        to_return = None
        # Check if it is the wrong data type
        if self.data.dtype.char not in ['e', 'f', 'd']:  # If not float
            _logger.warning(
                'To perform a decomposition the data must be of the float '
                'type. You can change the type using the change_dtype method'
                ' e.g. s.change_dtype(\'float64\')\n'
                'Nothing done.')
            return

        if self.axes_manager.navigation_size < 2:
            raise AttributeError("It is not possible to decompose a dataset "
                                 "with navigation_size < 2")
        # backup the original data
        self._data_before_treatments = self.data.copy()
        # set the output target (peak results or not?)
        target = LearningResults()

        if algorithm == 'mlpca':
            if normalize_poissonian_noise is True:
                _logger.warning(
                    "It makes no sense to do normalize_poissonian_noise with "
                    "the MLPCA algorithm. Therefore, "
                    "normalize_poissonian_noise is set to False")
                normalize_poissonian_noise = False
            if output_dimension is None:
                raise ValueError("With the MLPCA algorithm the "
                                 "output_dimension must be specified")
        if algorithm == 'RPCA_GoDec' or algorithm == 'ORPCA':
            if output_dimension is None:
                raise ValueError("With the robust PCA algorithms ('RPCA_GoDec' "
                                 "and 'ORPCA'), the output_dimension "
                                 "must be specified")

        # Apply pre-treatments
        # Transform the data in a line spectrum
        self._unfolded4decomposition = self.unfold()
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
            _logger.info('Performing decomposition analysis')
            # The rest of the code assumes that the first data axis
            # is the navigation axis. We transpose the data if that is not the
            # case.
            dc = (self.data if self.axes_manager[0].index_in_array == 0
                  else self.data.T)

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
                if return_info:
                    to_return = sk

            elif algorithm == 'nmf':
                if import_sklearn.sklearn_installed is False:
                    raise ImportError(
                        'sklearn is not installed. Nothing done')
                sk = import_sklearn.sklearn.decomposition.NMF(**kwargs)
                sk.n_components = output_dimension
                loadings = sk.fit_transform((
                    dc[:, signal_mask][navigation_mask, :]))
                factors = sk.components_.T
                if return_info:
                    to_return = sk

            elif algorithm == 'sparse_pca':
                if import_sklearn.sklearn_installed is False:
                    raise ImportError(
                        'sklearn is not installed. Nothing done')
                sk = import_sklearn.sklearn.decomposition.SparsePCA(
                    output_dimension, **kwargs)
                loadings = sk.fit_transform(
                    dc[:, signal_mask][navigation_mask, :])
                factors = sk.components_.T
                if return_info:
                    to_return = sk

            elif algorithm == 'mini_batch_sparse_pca':
                if import_sklearn.sklearn_installed is False:
                    raise ImportError(
                        'sklearn is not installed. Nothing done')
                sk = import_sklearn.sklearn.decomposition.MiniBatchSparsePCA(
                    output_dimension, **kwargs)
                loadings = sk.fit_transform(
                    dc[:, signal_mask][navigation_mask, :])
                factors = sk.components_.T
                if return_info:
                    to_return = sk

            elif algorithm == 'mlpca' or algorithm == 'fast_mlpca':
                _logger.info("Performing the MLPCA training")
                if output_dimension is None:
                    raise ValueError(
                        "For MLPCA it is mandatory to define the "
                        "output_dimension")
                if var_array is None and var_func is None:
                    _logger.info('No variance array provided.'
                                 'Assuming poissonian data')
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
                            var_array = np.polyval(
                                polyfit, dc[
                                    signal_mask, navigation_mask])
                        except:
                            raise ValueError(
                                'var_func must be either a function or an '
                                'array defining the coefficients of a polynom')
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
            elif algorithm == 'RPCA_GoDec':
                _logger.info("Performing Robust PCA with GoDec")

                X, E, G, U, S, V = rpca_godec(
                    dc[:, signal_mask][navigation_mask, :],
                    rank=output_dimension, fast=True, **kwargs)

                loadings = U * S
                factors = V
                explained_variance = S ** 2 / len(factors)

                if return_info:
                    to_return = (X, E)

            elif algorithm == 'ORPCA':
                _logger.info("Performing Online Robust PCA")

                X, E, U, S, V = orpca(
                    dc[:, signal_mask][navigation_mask, :],
                    rank=output_dimension, fast=True, **kwargs)

                loadings = U * S
                factors = V
                explained_variance = S ** 2 / len(factors)

                if return_info:
                    to_return = (X, E)
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

            # Delete the unmixing information, because it'll refer to a
            # previous decomposition
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
                    _logger.info("Reprojecting the signal is not yet "
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
                    factors[signal_mask, :] = target.factors
                    factors[~signal_mask, :] = np.nan
                    target.factors = factors
            if not isinstance(navigation_mask, slice):
                # Store the (inverted, as inputed) navigation mask
                target.navigation_mask = ~navigation_mask.reshape(
                    self.axes_manager._navigation_shape_in_array)
                if reproject not in ('both', 'navigation'):
                    loadings = np.zeros(
                        (dc.shape[0], target.loadings.shape[1]))
                    loadings[navigation_mask, :] = target.loadings
                    loadings[~navigation_mask, :] = np.nan
                    target.loadings = loadings
        finally:
            if self._unfolded4decomposition is True:
                self.fold()
                self._unfolded4decomposition is False
            self.learning_results.__dict__.update(target.__dict__)
            # undo any pre-treatments
            self.undo_treatments()

        return to_return

    def blind_source_separation(self,
                                number_of_components=None,
                                algorithm='sklearn_fastica',
                                diff_order=1,
                                diff_axes=None,
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
            Sometimes it is convenient to perform the BSS on the derivative of
            the signal. If diff_order is 0, the signal is not differentiated.
        diff_axes : None or list of ints or strings
            If None, when `diff_order` is greater than 1 and `signal_dimension`
            (`navigation_dimension`) when `on_loadings` is False (True) is
            greater than 1, the differences are calculated across all
            signal (navigation) axes. Otherwise the axes can be specified in
            a list.
        factors : Signal or numpy array.
            Factors to decompose. If None, the BSS is performed on the
            factors of a previous decomposition. If a Signal instance the
            navigation dimension must be 1 and the size greater than 1.
        comp_list : boolen numpy array
            choose the components to use by the boolean list. It permits
             to choose non contiguous components.
        mask : bool numpy array or Signal instance.
            If not None, the signal locations marked as True are masked. The
            mask shape must be equal to the signal shape
            (navigation shape) when `on_loadings` is False (True).
        on_loadings : bool
            If True, perform the BSS on the loadings of a previous
            decomposition. If False, performs it on the factors.
        pretreatment: dict

        **kwargs : extra key word arguments
            Any keyword arguments are passed to the BSS algorithm.

        FastICA documentation is here, with more arguments that can be passed as **kwargs:
        http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html

        """
        from hyperspy.signal import BaseSignal

        lr = self.learning_results

        if factors is None:
            if not hasattr(lr, 'factors') or lr.factors is None:
                raise AttributeError(
                    'A decomposition must be performed before blind '
                    'source seperation or factors must be provided.')

            else:
                if on_loadings:
                    factors = self.get_decomposition_loadings()
                else:
                    factors = self.get_decomposition_factors()

        # Check factors
        if not isinstance(factors, BaseSignal):
            raise ValueError(
                "`factors` must be a BaseSignal instance, but an object of type "
                "%s was provided." %
                type(factors))

        # Check factor dimensions
        if factors.axes_manager.navigation_dimension != 1:
            raise ValueError("`factors` must have navigation dimension"
                             "equal one, but the navigation dimension "
                             "of the given factors is %i." %
                             factors.axes_manager.navigation_dimension
                             )
        elif factors.axes_manager.navigation_size < 2:
            raise ValueError("`factors` must have navigation size"
                             "greater than one, but the navigation "
                             "size of the given factors is %i." %
                             factors.axes_manager.navigation_size)

        # Check mask dimensions
        if mask is not None:
            ref_shape, space = (factors.axes_manager.signal_shape,
                                "navigation" if on_loadings else "signal")
            if isinstance(mask, BaseSignal):
                if mask.axes_manager.signal_shape != ref_shape:
                    raise ValueError(
                        "The `mask` signal shape is not equal to the %s shape."
                        " Mask shape: %s\t%s shape:%s" %
                        (space,
                         str(mask.axes_manager.signal_shape),
                         space,
                         str(ref_shape)))

        # Note that we don't check the factor's signal dimension. This is on
        # purpose as an user may like to apply pretreaments that change their
        # dimensionality.

        # The diff_axes are given for the main signal. We need to compute
        # the correct diff_axes for the factors.
        # Get diff_axes index in axes manager
        if diff_axes is not None:
            diff_axes = [1 + axis.index_in_axes_manager for axis in
                         [self.axes_manager[axis] for axis in diff_axes]]
            if not on_loadings:
                diff_axes = [index - self.axes_manager.navigation_dimension
                             for index in diff_axes]
        # Select components to separate
        if number_of_components is not None:
            comp_list = range(number_of_components)
        elif comp_list is not None:
            number_of_components = len(comp_list)
        else:
            if lr.output_dimension is not None:
                number_of_components = lr.output_dimension
                comp_list = range(number_of_components)
            else:
                raise ValueError(
                    "No `number_of_components` or `comp_list` provided.")
        factors = stack([factors.inav[i] for i in comp_list])

        # Apply differences pre-processing if requested.
        if diff_order > 0:
            factors = get_derivative(factors,
                                     diff_axes=diff_axes,
                                     diff_order=diff_order)
            if mask is not None:
                # The following is a little trick to dilate the mask as
                # required when operation on the differences. It exploits the
                # fact that np.diff autimatically "dilates" nans. The trick has
                # a memory penalty which should be low compare to the total
                # memory required for the core application in most cases.
                mask_diff_axes = (
                    [iaxis - 1 for iaxis in diff_axes]
                    if diff_axes is not None
                    else None)
                mask.change_dtype("float")
                mask.data[mask.data == 1] = np.nan
                mask = get_derivative(mask,
                                      diff_axes=mask_diff_axes,
                                      diff_order=diff_order)
                mask.data[np.isnan(mask.data)] = 1
                mask.change_dtype("bool")

        # Unfold in case the signal_dimension > 1
        factors.unfold()
        if mask is not None:
            mask.unfold()
            factors = factors.data.T[np.where(~mask.data)]
        else:
            factors = factors.data.T

        # Center and scale the data
        factors, invsqcovmat = centering_and_whitening(factors)

        # Perform actual BSS
        if algorithm == 'orthomax':
            _, unmixing_matrix = orthomax(factors, **kwargs)
            unmixing_matrix = unmixing_matrix.T

        elif algorithm == 'sklearn_fastica':
            if not import_sklearn.sklearn_installed:
                raise ImportError(
                    "The optional package scikit learn is not installed "
                    "and it is required for this feature.")
            if 'tol' not in kwargs:
                kwargs['tol'] = 1e-10
            lr.bss_node = import_sklearn.FastICA(
                **kwargs)
            lr.bss_node.whiten = False
            lr.bss_node.fit(factors)
            try:
                unmixing_matrix = lr.bss_node.unmixing_matrix_
            except AttributeError:
                # unmixing_matrix was renamed to components
                unmixing_matrix = lr.bss_node.components_
        else:
            if mdp_installed is False:
                raise ImportError(
                    'MDP is not installed. Nothing done')
            temp_function = getattr(mdp.nodes, algorithm + "Node")
            lr.bss_node = temp_function(**kwargs)
            lr.bss_node.train(factors)
            unmixing_matrix = lr.bss_node.get_recmatrix()
        w = np.dot(unmixing_matrix, invsqcovmat)
        if lr.explained_variance is not None:
            # The output of ICA is not sorted in any way what makes it
            # difficult to compare results from different unmixings. The
            # following code is an experimental attempt to sort them in a
            # more predictable way
            sorting_indices = np.argsort(np.dot(
                lr.explained_variance[:number_of_components],
                np.abs(w.T)))[::-1]
            w[:] = w[sorting_indices, :]
        lr.unmixing_matrix = w
        lr.on_loadings = on_loadings
        self._unmix_components()
        self._auto_reverse_bss_component(lr)
        lr.bss_algorithm = algorithm
        lr.bss_node = str(lr.bss_node)

    def normalize_decomposition_components(self, target='factors',
                                           function=np.sum):
        """Normalize decomposition components.

        Parameters
        ----------
        target : {"factors", "loadings"}
        function : numpy universal function, optional, default np.sum
            Each target component is divided by the output of function(target).
            `function` must return a scalar when operating on numpy arrays and
            must have an `axis`.

        """
        if target == 'factors':
            target = self.learning_results.factors
            other = self.learning_results.loadings
        elif target == 'loadings':
            target = self.learning_results.loadings
            other = self.learning_results.factors
        else:
            raise ValueError("target must be \"factors\" or \"loadings\"")
        if target is None:
            raise Exception("This method can only be used after "
                            "decomposition operation.")
        _normalize_components(target=target, other=other, function=function)

    def normalize_bss_components(self, target='factors', function=np.sum):
        """Normalize BSS components.

        Parameters
        ----------
        target : {"factors", "loadings"}
        function : numpy universal function, optional, default np.sum
            Each target component is divided by the output of function(target).
            `function` must return a scalar when operating on numpy arrays and
            must have an `axis`.

        """
        if target == 'factors':
            target = self.learning_results.bss_factors
            other = self.learning_results.bss_loadings
        elif target == 'loadings':
            target = self.learning_results.bss_loadings
            other = self.learning_results.bss_factors
        else:
            raise ValueError("target must be \"factors\" or \"loadings\"")
        if target is None:
            raise Exception("This method can only be used after "
                            "a blind source separation operation.")
        _normalize_components(target=target, other=other, function=function)

    def reverse_decomposition_component(self, component_number):
        """Reverse the decomposition component

        Parameters
        ----------
        component_number : list or int
            component index/es

        Examples
        -------
        >>> s = hs.load('some_file')
        >>> s.decomposition(True) # perform PCA
        >>> s.reverse_decomposition_component(1) # reverse IC 1
        >>> s.reverse_decomposition_component((0, 2)) # reverse ICs 0 and 2
        """

        target = self.learning_results

        for i in [component_number, ]:
            target.factors[:, i] *= -1
            target.loadings[:, i] *= -1

    def reverse_bss_component(self, component_number):
        """Reverse the independent component

        Parameters
        ----------
        component_number : list or int
            component index/es

        Examples
        -------
        >>> s = hs.load('some_file')
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

    def _unmix_components(self):
        lr = self.learning_results
        w = lr.unmixing_matrix
        n = len(w)
        if lr.on_loadings:
            lr.bss_loadings = np.dot(lr.loadings[:, :n], w.T)
            lr.bss_factors = np.dot(lr.factors[:, :n], np.linalg.inv(w))
        else:

            lr.bss_factors = np.dot(lr.factors[:, :n], w.T)
            lr.bss_loadings = np.dot(lr.loadings[:, :n], np.linalg.inv(w))

    def _auto_reverse_bss_component(self, target):
        n_components = target.bss_factors.shape[1]
        for i in range(n_components):
            minimum = np.nanmin(target.bss_loadings[:, i])
            maximum = np.nanmax(target.bss_loadings[:, i])
            if minimum < 0 and -minimum > maximum:
                self.reverse_bss_component(i)
                _logger.info("IC %i reversed" % i)

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
            for i in range(len(components)):
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

        self._unfolded4decomposition = self.unfold()
        try:
            sc = self.deepcopy()
            sc.data = a.T.reshape(self.data.shape)
            sc.metadata.General.title += ' ' + signal_name
            if target.mean is not None:
                sc.data += target.mean
        finally:
            if self._unfolded4decomposition is True:
                self.fold()
                sc.fold()
                self._unfolded4decomposition = False
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
        return rec

    def get_explained_variance_ratio(self):
        """Return the explained variation ratio of the PCA components as a
        Signal1D.

        Returns
        -------
        s : Signal1D
            Explained variation ratio.

        See Also:
        ---------

        `plot_explained_variance_ration`, `decomposition`,
        `get_decomposition_loadings`,
        `get_decomposition_factors`.

        """
        from hyperspy._signals.signal1d import Signal1D
        target = self.learning_results
        if target.explained_variance_ratio is None:
            raise AttributeError("The explained_variance_ratio attribute is "
                                 "`None`, did you forget to perform a PCA "
                                 "decomposition?")
        s = Signal1D(target.explained_variance_ratio)
        s.metadata.General.title = self.metadata.General.title + \
            "\nPCA Scree Plot"
        s.axes_manager[-1].name = 'Principal component index'
        s.axes_manager[-1].units = ''
        return s

    def plot_explained_variance_ratio(self, n=None, log=True, threshold=0,
                                      hline='auto', xaxis_type='index',
                                      xaxis_labeling=None, signal_fmt=None,
                                      noise_fmt=None, fig=None, ax=None,
                                      **kwargs):
        """Plot the decomposition explained variance ratio vs index number
        (Scree Plot).

        Parameters
        ----------
        n : int or None
            Number of components to plot. If None, all components will be plot
        log : bool
            If True, the y axis uses a log scale.
        threshold : float or int
            Threshold used to determine how many components should be
            highlighted as signal (as opposed to noise).
            If a float (between 0 and 1), ``threshold`` will be
            interpreted as a cutoff value, defining the variance at which to
            draw a line showing the cutoff between signal and noise;
            the number of signal components will be automatically determined
            by the cutoff value.
            If an int, ``threshold`` is interpreted as the number of
            components to highlight as signal (and no cutoff line will be
            drawn)
        hline: {'auto', True, False}
            Whether or not to draw a horizontal line illustrating the variance
            cutoff for signal/noise determination. Default is to draw the line
            at the value given in ``threshold`` (if it is a float) and not
            draw in the case  ``threshold`` is an int, or not given.
            If True, (and ``threshold`` is an int), the line will be drawn
            through the last component defined as signal.
            If False, the line will not be drawn in any circumstance.
        xaxis_type : {'index', 'number'}
            Determines the type of labeling applied to the x-axis.
            If ``'index'``, axis will be labeled starting at 0 (i.e.
            "pythonic index" labeling); if ``'number'``, it will start at 1
            (number labeling).
        xaxis_labeling : {'ordinal', 'cardinal', None}
            Determines the format of the x-axis tick labels. If ``'ordinal'``,
            "1st, 2nd, ..." will be used; if ``'cardinal'``, "1, 2,
            ..." will be used. If None, an appropriate default will be
            selected.
        signal_fmt : dict
            Dictionary of matplotlib formatting values for the signal
            components
        noise_fmt : dict
            Dictionary of matplotlib formatting values for the noise
            components
        fig : matplotlib figure or None
            If None, a default figure will be created, otherwise will plot
            into fig
        ax : matplotlib ax (subplot) or None
            If None, a default ax will be created, otherwise will plot into ax
        **kwargs
            remaining keyword arguments are passed to matplotlib.figure()

        Example
        --------
        To generate a scree plot with customized symbols for signal vs.
        noise components and a modified cutoff threshold value:

        >>> s = hs.load("some_spectrum_image")
        >>> s.decomposition()
        >>> s.plot_explained_variance_ratio(n=40,
        >>>                                 threshold=0.005,
        >>>                                 signal_fmt={'marker': 'v',
        >>>                                             's': 150,
        >>>                                             'c': 'pink'}
        >>>                                 noise_fmt={'marker': '*',
        >>>                                             's': 200,
        >>>                                             'c': 'green'})

        Returns
        -------

        ax : matplotlib.axes


        See Also
        --------

        :py:meth:`~.learn.mva.MVA.decomposition`,
        :py:meth:`~.learn.mva.MVA.get_explained_variance_ratio`,
        :py:meth:`~.signal.MVATools.get_decomposition_loadings`,
        :py:meth:`~.signal.MVATools.get_decomposition_factors`

        """
        s = self.get_explained_variance_ratio()

        if n is None:
            n = len(self.learning_results.explained_variance_ratio)

        # Determine right number of components for signal and cutoff value
        if isinstance(threshold, float):
            if not 0 < threshold < 1:
                raise ValueError('Variance threshold should be between 0 and'
                                 ' 1')
            # Catch if the threshold is less than the minimum variance value:
            if threshold < s.data.min():
                n_signal_pcs = n
            else:
                n_signal_pcs = np.where((s < threshold).data)[0][0]
        else:
            n_signal_pcs = threshold
            if n_signal_pcs == 0:
                hline = False

        # Handling hline logic
        if hline == 'auto':
            # Set cutoff to threshold if float
            if isinstance(threshold, float):
                cutoff = threshold
            # Turn off the hline otherwise
            else:
                hline = False
        # If hline is True and threshold is int, set cutoff at value of last
        # signal component
        elif hline:
            if isinstance(threshold, float):
                cutoff = threshold
            elif n_signal_pcs > 0:
                cutoff = s.data[n_signal_pcs - 1]
        # Catches hline==False and hline==True (if threshold not given)
        else:
            hline = False

        # Some default formatting for signal markers
        if signal_fmt is None:
            signal_fmt = {'c': '#C24D52',
                          's': 100,
                          'marker': "^",
                          'zorder': 3}

        # Some default formatting for noise markers
        if noise_fmt is None:
            noise_fmt = {'c': '#4A70B0',
                         's': 100,
                         'marker': 'o',
                         'zorder': 3}

        # Sane defaults for xaxis labeling
        if xaxis_labeling is None:
            xaxis_labeling = 'cardinal' if xaxis_type == 'index' else 'ordinal'

        axes_titles = {'y': "Proportion of variance",
                       'x': "Principal component {}".format(xaxis_type)}

        if n < s.axes_manager[-1].size:
            s = s.isig[:n]

        if fig is None:
            fig = plt.figure(**kwargs)

        if ax is None:
            ax = fig.add_subplot(111)

        if log:
            ax.semilogy()

        if hline:
            ax.axhline(cutoff,
                       linewidth=2,
                       color='gray',
                       linestyle='dashed',
                       zorder=1)

        if n_signal_pcs == n:
            ax.scatter(range(n),
                       s.isig[:n].data,
                       **signal_fmt)
        elif n_signal_pcs > 0:
            ax.scatter(range(n_signal_pcs),
                       s.isig[:n_signal_pcs].data,
                       **signal_fmt)
            ax.scatter(range(n_signal_pcs, n),
                       s.isig[n_signal_pcs:n].data,
                       **noise_fmt)
        else:
            ax.scatter(range(n),
                       s.isig[:n].data,
                       **noise_fmt)

        if xaxis_type == 'index':
            locs = ax.get_xticks()
            if xaxis_labeling == 'ordinal':
                ax.set_xticklabels([ordinal(int(i)) for i in locs])
            else:
                ax.set_xticklabels([int(i) for i in locs])

        if xaxis_type == 'number':
            locs = ax.get_xticks()
            if xaxis_labeling == 'ordinal':
                ax.set_xticklabels([ordinal(int(i + 1)) for i in locs])
            else:
                ax.set_xticklabels([int(i + 1) for i in locs])

        ax.set_ylabel(axes_titles['y'])
        ax.set_xlabel(axes_titles['x'])
        ax.margins(0.05)
        ax.autoscale()
        ax.set_title(s.metadata.General.title, y=1.01)

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
        _logger.info(
            "Scaling the data to normalize the (presumably)"
            " Poissonian noise")
        with self.unfolded():
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

    def undo_treatments(self):
        """Undo normalize_poissonian_noise"""
        _logger.info("Undoing data pre-treatments")
        self.data[:] = self._data_before_treatments
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
            v for v in dir(self) if not isinstance(
                getattr(
                    self,
                    v),
                types.MethodType) and not v.startswith('_')]:
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
        for key, value in decomposition.items():
            if value.dtype == np.dtype('object'):
                value = None
            setattr(self, key, value)
        _logger.info("\n%s loaded correctly" % filename)
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
        #
        # Output_dimension is an array after loading, convert it to int
        if hasattr(self, 'output_dimension') and self.output_dimension \
                is not None:
            self.output_dimension = int(self.output_dimension)
        _logger.info(self._summary())

    def summary(self):
        """Prints a summary of the decomposition and demixing parameters
         to the stdout
        """
        print(self._summary())

    def _summary(self):
        summary_str = (
            "Decomposition parameters:\n"
            "-------------------------\n\n" +
            ("Decomposition algorithm : \t%s\n" %
                self.decomposition_algorithm) +
            ("Poissonian noise normalization : %s\n" %
                self.poissonian_noise_normalized) +
            ("Output dimension : %s\n" % self.output_dimension) +
            ("Centre : %s" % self.centre))
        if self.bss_algorithm is not None:
            summary_str += (
                "\n\nDemixing parameters:\n"
                "------------------------\n" +
                ("BSS algorithm : %s" % self.bss_algorithm) +
                ("Number of components : %i" % len(self.unmixing_matrix)))
        return summary_str

    def crop_decomposition_dimension(self, n):
        """
        Crop the score matrix up to the given number.
        It is mainly useful to save memory and reduce the storage size
        """
        _logger.info("trimming to %i dimensions" % n)
        self.loadings = self.loadings[:, :n]
        if self.explained_variance is not None:
            self.explained_variance = self.explained_variance[:n]
        self.factors = self.factors[:, :n]

    def _transpose_results(self):
        (self.factors, self.loadings, self.bss_factors,
            self.bss_loadings) = (self.loadings, self.factors,
                                  self.bss_loadings, self.bss_factors)
