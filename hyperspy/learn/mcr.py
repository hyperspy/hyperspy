import numpy as np
from pymcr.mcr import McrAR
from pymcr.constraints import ConstraintNonneg, ConstraintNorm
from hyperspy.misc.machine_learning.orthomax import orthomax
from hyperspy.misc.utils import stack
import logging
from contextlib import redirect_stdout
import io

_logger = logging.getLogger(__name__)


def mcrals(self,
           number_of_components=None,
           simplicity='spatial',
           mask=None,
           compute=False,
           verbosity='error',
           c_regr='OLS',
           st_regr='OLS',
           max_iter=100,
           c_constraints=[ConstraintNonneg(), ConstraintNorm()],
           st_constraints=[ConstraintNonneg()],
           tol_increase=1.0,
           tol_n_increase=10,
           tol_err_change=1e-14,
           tol_n_above_min=10):
    """Perform multivariate curve resolution (MCR) on the result on the
    decomposition. MCR is carried out using the PyMCR library, details of
    which can be found at the following link:

    https://github.com/usnistgov/pyMCR

    If spatial simplicity is chosen, a varimax rotation is computed on the
    SVD loadings.  This rotation is applied to the factors, which are then
    fit to the input data via MCR until the convergence criteria are met.
    If spectral simplicity is chosen, the varimax rotation is calculated
    directly on the factors prior to MCR.

    In addition, prior to MCR fitting, the rotated factors are flipped if
    necessary so that they are mostly positive. This is required since a
    non-negativity constraint is applied.

    Parameters
    ----------
    number_of_components : int
        number of principal components to pass to the MCR algorithm
    simplicity : str
        Should be either 'spatial' or 'spectral'.
        Data is rotated to enforce simplicity in either the spatial or
        the spectral domains prior to MCR fitting depending on the value
        provided.
    mask : bool numpy array or Signal instance.
        If not None, the signal locations marked as True are masked. The
        mask shape must be equal to the signal shape
        (navigation shape) when `on_loadings` is False (True).
    compute : bool
        If the decomposition results are lazy, compute the BSS components
        so that they are not lazy.
        Default is False.
    verbosity : str
        One of ``['error', 'warning', 'info', 'debug']``
        Controls the verbosity of the external pyMCR routines. The
        strings provided correspond to levels of the ``logging`` module.
        Default is ``'error'`` (only critical failure output).
    c_regr : str, class
        Regression class (or string) for calculating the C matrix.
        Must be either 'OLS' or 'NNLS'. Default is 'OLS".
    st_regr : str
        Regression class (or string) for calculating the S^T matrix.
        Must be either 'OLS' or 'NNLS'. Default is 'OLS'.
    max_iter : int
        Maximum number of iterations. One iteration calculates both C and S^T.
        Default value is 100
    c_constraints : list
        List of constraints applied to calculation of C matrix
        Default is [ConstraintNonneg(), ConstraintNorm()]
    st_constraints : list
        List of constraints applied to calculation of S^T matrix
        Default is [ConstraintNonneg()]
    tol_increase : float
        Factor increase to allow in err attribute. Set to 0 for no increase
        allowed. E.g., setting to 1.0 means the err can double per iteration.
        Default is 1.0
    tol_n_increase : int
        Number of consecutive iterations for which the err attribute can
        increase
        Default is 10
    tol_err_change : float
        If err changes less than tol_err_change, per iteration, break.
        Default is 1e-14
    tol_n_above_min : int
        Number of half-iterations that can be performed without reaching a
        new error-minimum
        Default is 10
    """
    from hyperspy.signal import BaseSignal

    lr = self.learning_results
    data = self.data

    # should only perform MCR is a previous decomposition was performed (
    # using same check as for ``blind_source_separation()``
    if not hasattr(lr, 'factors') or lr.factors is None:
        raise ValueError("The 'MCR' algorithm operates on a previous "
                         "decomposition output, and this signal's "
                         "`learning_results` was None. Please run an SVD or "
                         "other PCA decomposition with `s.decomposition()` "
                         "prior to using the MCR technique.")

    factors = self.get_decomposition_factors()
    loadings = self.get_decomposition_loadings()

    # Check factors
    if not isinstance(factors, BaseSignal):
        raise ValueError(
            f"`factors` must be a BaseSignal instance, but an object of type "
            f"{type(factors)} was provided.")

    # Check factor dimensions
    if factors.axes_manager.navigation_dimension != 1:
        raise ValueError(f"`factors` must have navigation dimension"
                         f"equal one, but the navigation dimension "
                         f"of the given factors is "
                         f"{factors.axes_manager.navigation_dimension}.")
    elif factors.axes_manager.navigation_size < 2:
        raise ValueError(f"`factors` must have navigation size"
                         f"greater than one, but the navigation "
                         f"size of the given factors is"
                         f"{factors.axes_manager.navigation_size}.")

    # Select components to fit
    if number_of_components is not None:
        comp_list = range(number_of_components)
    else:
        if lr.output_dimension is not None:
            number_of_components = lr.output_dimension
            comp_list = range(number_of_components)
        else:
            raise ValueError(
                "No `number_of_components` or `comp_list` provided.")
    loadings = stack([loadings.inav[i] for i in comp_list])
    factors = stack([factors.inav[i] for i in comp_list])

    # Unfold in case the signal_dimension > 1
    factors.unfold()
    loadings.unfold()
    if mask is not None:
        mask.unfold()
        factors.data = factors.data.T[np.where(~mask.data)]
    else:
        factors.data = factors.data.T

    if self.learning_results.poissonian_noise_normalized is True:
        spec_weight_vec = self._root_bH.T.squeeze()
        im_weight_vec = self._root_aG.squeeze()
    else:
        spec_weight_vec = None
        im_weight_vec = None

    # Set logging level for pyMCR:
    levels = {'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}
    logging.getLogger('pymcr.mcr').setLevel(levels[verbosity])

    # Set MCR constraints from strings if necessary
    for i in range(0, len(c_constraints)):
        if type(c_constraints[i]) is str:
            if c_constraints[i] == 'Nonneg':
                c_constraints[i] = ConstraintNonneg()
            elif c_constraints[i] == 'Norm':
                c_constraints[i] = ConstraintNorm()
            else:
                raise ValueError("C constraint string '%s' not allowed. Must "
                                 "be 'Nonneg' or 'Norm'" % c_constraints[i])

    for i in range(0, len(st_constraints)):
        if type(st_constraints[i]) is str:
            if st_constraints[i] == 'Nonneg':
                st_constraints[i] = ConstraintNonneg()
            elif st_constraints[i] == 'Norm':
                st_constraints[i] = ConstraintNorm()
            else:
                raise ValueError("ST constraint string '%s' not allowed. Must "
                                 "be 'Nonneg' or 'Norm'" % st_constraints[i])

    # Perform MCR
    if simplicity == 'spatial':
        rot_loadings, rotation = orthomax(loadings.data.T, gamma=1)
        rot_loadings = np.array(rot_loadings)
        rotation = np.array(rotation)
        rot_factors = np.dot(factors.data, rotation)
        # Flipping as necessary to make the factors "mostly" positive:
        rot_factors = np.sign(rot_factors.sum(0)) * rot_factors

        rot_factors[rot_factors < 0] = 0
        if self.learning_results.poissonian_noise_normalized is True:
            rot_factors = (rot_factors.T / spec_weight_vec).T
            rot_factors = np.nan_to_num(rot_factors)
            data = (data.T / im_weight_vec).T / spec_weight_vec
            data = np.nan_to_num(data)

        fitmcr = McrAR(max_iter=max_iter,
                       st_regr=st_regr,
                       c_regr=c_regr,
                       c_constraints=c_constraints,
                       st_constraints=st_constraints,
                       tol_increase=tol_increase,
                       tol_n_increase=tol_n_increase,
                       tol_err_change=tol_err_change,
                       tol_n_above_min=tol_n_above_min)
        f = io.StringIO()
        with redirect_stdout(f):
            fitmcr.fit(data.T, C=rot_factors, verbose=False)
        _logger.info(f"PyMCR result: {f.getvalue()}")

        if self.learning_results.poissonian_noise_normalized is True:
            loadings_out = (fitmcr.ST_opt_ * im_weight_vec).T
            factors_out = (fitmcr.C_opt_.T * spec_weight_vec).T
            data = (data.T * im_weight_vec).T / spec_weight_vec
        else:
            loadings_out = fitmcr.ST_opt_.T
            factors_out = fitmcr.C_opt_

    elif simplicity == 'spectral':
        rot_factors, rotation = orthomax(factors.data, gamma=1)
        rot_factors = np.array(rot_factors)
        rotation = np.array(rotation)
        rot_factors = np.sign(rot_factors.sum(0)) * rot_factors
        rot_factors[rot_factors < 0] = 0
        if self.learning_results.poissonian_noise_normalized is True:
            rot_factors = (rot_factors.T / spec_weight_vec).T
            rot_factors = np.nan_to_num(rot_factors)
            data = (data.T / im_weight_vec).T / spec_weight_vec
            data = np.nan_to_num(data)

        fitmcr = McrAR(max_iter=max_iter,
                       st_regr=st_regr,
                       c_regr=c_regr,
                       c_constraints=c_constraints,
                       st_constraints=st_constraints,
                       tol_increase=tol_increase,
                       tol_n_increase=tol_n_increase,
                       tol_err_change=tol_err_change,
                       tol_n_above_min=tol_n_above_min)
        f = io.StringIO()
        with redirect_stdout(f):
            fitmcr.fit(data, ST=rot_factors.T, verbose=False)
        _logger.info(f"PyMCR result: {f.getvalue()}")

        if self.learning_results.poissonian_noise_normalized is True:
            loadings_out = (fitmcr.C_opt_.T * im_weight_vec).T
            factors_out = (fitmcr.ST_opt_ * spec_weight_vec).T
            data = (data.T * im_weight_vec).T / spec_weight_vec
        else:
            loadings_out = fitmcr.C_opt_
            factors_out = fitmcr.ST_opt_.T

    else:
        raise ValueError(f"'simplicity' must be either 'spatial' or"
                         f"'spectral'."
                         f"{str(simplicity)} was provided.")

    # MCR can return NaNs at certain channels in the event a signal channel
    # was zero at all navigation positions, so strip out the NaN values
    # before normalization and raise a warning if they're present:
    if np.isnan(factors_out).sum():
        _logger.warning('NaN values were detected in the MCR factors. They '
                        'have been replaced with zeros, but please check that '
                        'the output is as expected. This is typically caused '
                        'by the presence of uniformly zero-valued signal '
                        'channels.')
        factors_out = np.nan_to_num(factors_out)
    if np.isnan(loadings_out).sum():
        _logger.warning('NaN values were detected in the MCR loadings. They '
                        'have been replaced with zeros, but please check that '
                        'the output is as expected. This is typically caused '
                        'by the presence of uniformly zero-valued signal '
                        'channels.')
        loadings_out = np.nan_to_num(loadings_out)

    factors_out = factors_out / factors_out.sum(0)
    loadings_out = loadings_out / loadings_out.sum(0)

    return factors_out, loadings_out
