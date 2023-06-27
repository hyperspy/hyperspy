"""
Ported first from the astroML project: https://astroml.org/
Ported again from the astropy project: https://astropy.org/

Tools for working with distributions
"""

import logging

import numpy as np
from scipy.optimize import fmin
from scipy.special import gammaln

from hyperspy.docstrings.signal import HISTOGRAM_MAX_BIN_ARGS


_logger = logging.getLogger(__name__)


def knuth_bin_width(data, return_bins=False, quiet=True, max_num_bins=250):
    r"""Return the optimal histogram bin width using Knuth's rule.

    Knuth's rule is a fixed-width, Bayesian approach to determining
    the optimal bin width of a histogram.

    Parameters
    ----------
    data : array_like, ndim=1
        observed (one-dimensional) data
    return_bins : bool, optional
        if True, then return the bin edges
    quiet : bool, optional
        if True (default) then suppress stdout output from scipy.optimize
    %s

    Returns
    -------
    dx : float
        optimal bin width. Bins are measured starting at the first data point.
    bins : ndarray
        bin edges: returned if ``return_bins`` is True

    Notes
    -----
    The optimal number of bins is the value M which maximizes the function

    .. math::
        F(M|x,I) = n\log(M) + \log\Gamma(\frac{M}{2})
        - M\log\Gamma(\frac{1}{2})
        - \log\Gamma(\frac{2n+M}{2})
        + \sum_{k=1}^M \log\Gamma(n_k + \frac{1}{2})

    where :math:`\Gamma` is the Gamma function, :math:`n` is the number of
    data points, :math:`n_k` is the number of measurements in bin :math:`k`
    [1]_.

    References
    ----------
    .. [1] Knuth, K.H. "Optimal Data-Based Binning for Histograms".
       arXiv:0605197, 2006

    """

    knuthF = _KnuthF(data)

    # First calculate bins according to Freedman-Diaconis rule
    bins0 = np.histogram_bin_edges(data, bins="fd")

    if len(bins0) > max_num_bins:
        # To avoid memory errors such as that detailed in
        # https://github.com/hyperspy/hyperspy/issues/784,
        # we log a warning and cap the number of bins at
        # a sensible value.
        _logger.warning(
            "Initial estimation of number of bins using Freedman-Diaconis "
            f"rule is too large ({len(bins0)}). Capping the number of bins "
            f"at `max_num_bins={max_num_bins}`. Consider using an "
            "alternative method for calculating the bins such as "
            "`bins='scott'`, or increasing the value of the "
            "`max_num_bins` keyword argument."
        )
        bins0 = np.histogram_bin_edges(data, bins=max_num_bins)

    M = fmin(knuthF, len(bins0), disp=not quiet)[0]
    bins = knuthF.bins(M)
    dx = bins[1] - bins[0]

    if return_bins:
        return dx, bins
    else:
        return dx


knuth_bin_width.__doc__ %= HISTOGRAM_MAX_BIN_ARGS


class _KnuthF:
    r"""Class which implements the function minimized by knuth_bin_width

    Parameters
    ----------
    data : array_like, one dimension
        data to be histogrammed

    Notes
    -----
    the function F is given by

    .. math::
        F(M|x,I) = n\log(M) + \log\Gamma(\frac{M}{2})
        - M\log\Gamma(\frac{1}{2})
        - \log\Gamma(\frac{2n+M}{2})
        + \sum_{k=1}^M \log\Gamma(n_k + \frac{1}{2})

    where :math:`\Gamma` is the Gamma function, :math:`n` is the number of
    data points, :math:`n_k` is the number of measurements in bin :math:`k`.

    """

    def __init__(self, data):
        self.data = np.array(data, copy=True)
        if self.data.ndim != 1:
            raise ValueError("data should be 1-dimensional")
        self.data.sort()
        self.n = self.data.size

        # import here rather than globally: scipy is an optional dependency.
        # Note that scipy is imported in the function which calls this,
        # so there shouldn't be any issue importing here.
        from scipy import special

        # create a reference to gammaln to use in self.eval()
        self.gammaln = special.gammaln

    def bins(self, M):
        """Return the bin edges given a width dx"""
        return np.linspace(self.data[0], self.data[-1], int(M) + 1)

    def __call__(self, M):
        return self.eval(M)

    def eval(self, M):
        """Evaluate the Knuth function

        Parameters
        ----------
        dx : float
            Width of bins

        Returns
        -------
        F : float
            evaluation of the negative Knuth likelihood function:
            smaller values indicate a better fit.

        """
        M = int(M)

        if M <= 0:
            return np.inf

        bins = self.bins(M)
        nk, bins = np.histogram(self.data, bins)

        return -(
            self.n * np.log(M)
            + gammaln(0.5 * M)
            - M * gammaln(0.5)
            - gammaln(self.n + 0.5 * M)
            + np.sum(gammaln(nk + 0.5))
        )
