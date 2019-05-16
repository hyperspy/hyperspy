import numpy as np
from pymcr.mcr import McrAR
from hyperspy.misc.machine_learning.orthomax import orthomax
from hyperspy.misc.utils import stack
import logging
from contextlib import redirect_stdout
import io

_logger = logging.getLogger(__name__)


def mcrals(self,
           number_of_components=None,
           simplicity='spatial',
           factors=None,
           comp_list=None,
           mask=None,
           compute=False,
           verbosity='error',):
    """Multivariate curve resolution (MCR) on the result on the
    decomposition.

    Available options: Assume simplicity in either the spatial or the
    spectral domain

    Parameters
    ----------
    number_of_components : int
        number of principal components to pass to the MCR algorithm
    simplicity : str
        Data is rotated to enforce simplicity in either the spatial or
        the spectral domains prior to MCR fitting.
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
    compute : bool
        If the decomposition results are lazy, compute the BSS components
        so that they are not lazy.
        Default is False.
    verbosity : str
        One of ``['error', 'warning', 'info', 'debug']``
        Controls the verbosity of the external pyMCR routines. The
        strings provided correspond to levels of the ``logging`` module.
        Default is ``'error'`` (only critical failure output).
    """
    from hyperspy.signal import BaseSignal

    lr = self.learning_results

    if factors is None:
        if not hasattr(lr, 'factors') or lr.factors is None:
            raise AttributeError(
                'A decomposition must be performed before MCR or factors'
                'must be provided.')
        else:
            factors = self.get_decomposition_factors()
            loadings = self.get_decomposition_loadings()

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
                         factors.axes_manager.navigation_size
                         )

    # Select components to fit
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

    # Perform MCR
    if simplicity == 'spatial':
        f = io.StringIO()
        with redirect_stdout(f):
            factors, loadings = mcr(self.data,
                                    loadings.data,
                                    factors.data,
                                    self.learning_results.
                                    poissonian_noise_normalized,
                                    im_weight_vec,
                                    spec_weight_vec,
                                    simplicity='spatial')
    elif simplicity == 'spectral':
        f = io.StringIO()
        with redirect_stdout(f):
            factors, loadings = mcr(self.data,
                                    loadings.data,
                                    factors.data,
                                    self.learning_results.
                                    poissonian_noise_normalized,
                                    im_weight_vec,
                                    spec_weight_vec,
                                    simplicity='spectral')

    _logger.info("PyMCR result: %s" % f.getvalue())
    # self.learning_results.mcr_factors = factors
    # self.learning_results.mcr_loadings = loadings
    return factors, loadings


def mcr(data, concentrations, purespectra, poisson_scale, im_weight_vec,
        spec_weight_vec, simplicity):

    if simplicity == 'spatial':
        rot_conc, rotation = orthomax(concentrations.T, gamma=1)
        rot_conc = np.array(rot_conc)
        rotation = np.array(rotation)
        rot_spec = np.dot(purespectra, rotation)
        rot_spec = np.sign(rot_spec.sum(0)) * rot_spec

        rot_spec[rot_spec < 0] = 0
        if poisson_scale:
            rot_spec = (rot_spec.T / spec_weight_vec).T
            rot_spec = np.nan_to_num(rot_spec)
            data = (data.T / im_weight_vec).T / spec_weight_vec
            data = np.nan_to_num(data)

        fitmcr = McrAR(max_iter=50, tol_err_change=1e-6)
        fitmcr.fit(data.T, C=rot_spec, verbose=False)

        if poisson_scale:
            conc_out = (fitmcr.ST_opt_ * im_weight_vec).T
            spec_out = (fitmcr.C_opt_.T * spec_weight_vec).T
            data = (data.T * im_weight_vec).T / spec_weight_vec
        else:
            conc_out = fitmcr.ST_opt_.T
            spec_out = fitmcr.C_opt_

    elif simplicity == 'spectral':
        rot_spec, rotation = orthomax(purespectra, gamma=1)
        rot_spec = np.array(rot_spec)
        rotation = np.array(rotation)
        rot_spec = np.sign(rot_spec.sum(0)) * rot_spec
        rot_spec[rot_spec < 0] = 0
        if poisson_scale:
            rot_spec = (rot_spec.T / spec_weight_vec).T
            rot_spec = np.nan_to_num(rot_spec)
            data = (data.T / im_weight_vec).T / spec_weight_vec
            data = np.nan_to_num(data)

        fitmcr = McrAR(max_iter=50, tol_err_change=1e-6)
        fitmcr.fit(data, ST=rot_spec.T, verbose=False)

        if poisson_scale:
            conc_out = (fitmcr.C_opt_.T * im_weight_vec).T
            spec_out = (fitmcr.ST_opt_ * spec_weight_vec).T
            data = (data.T * im_weight_vec).T / spec_weight_vec
        else:
            conc_out = fitmcr.C_opt_
            spec_out = fitmcr.ST_opt_.T

    else:
        raise ValueError("'simplicity' must be either 'spatial' or"
                         "'spectral'."
                         "{} was provided.".format(str(simplicity)))

    spec_out = spec_out/spec_out.sum(0)
    conc_out = conc_out/spec_out.sum(0)
    return spec_out, conc_out
