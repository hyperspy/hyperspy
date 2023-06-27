"""
This module implements the Lowess function for nonparametric regression.
Functions:
lowess Fit a smooth nonparametric regression curve to a scatterplot.
For more information, see
William S. Cleveland: "Robust locally weighted regression and smoothing
scatterplots", Journal of the American Statistical Association, December 1979,
volume 74, number 368, pp. 829-836.
William S. Cleveland and Susan J. Devlin: "Locally weighted regression: An
approach to regression analysis by local fitting", Journal of the American
Statistical Association, September 1988, volume 83, number 403, pp. 596-610.
"""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)
#
# https://gist.github.com/agramfort/850437

import numpy as np
from numba import njit


def lowess(y, x, f=2.0 / 3.0, n_iter=3):
    """Lowess smoother (robust locally weighted regression).

    Fits a nonparametric regression curve to a scatterplot.

    Parameters
    ----------
    y, x : np.ndarrays
        The arrays x and y contain an equal number of elements;
        each pair (x[i], y[i]) defines a data point in the
        scatterplot.

    f : float
        The smoothing span. A larger value will result in a
        smoother curve.
    n_iter : int
        The number of robustifying iteration. Thefunction will
        run faster with a smaller number of iterations.

    Returns
    -------
    yest : np.ndarray
        The estimated (smooth) values of y.

    """
    if not y.dtype.isnative:
        y = y.astype(y.dtype.type)
    return _lowess(y, x, f, n_iter)


@njit(cache=True, nogil=True)
def _lowess(y, x, f=2.0 / 3.0, n_iter=3):  # pragma: no cover
    """Lowess smoother requiring native endian datatype (for numba).

    """
    n = len(x)
    r = int(np.ceil(f * n))
    h = np.array([np.sort(np.abs(x - x[i]))[r] for i in range(n)])
    w = np.minimum(1.0, np.maximum(np.abs((x.reshape((-1, 1)) - x.reshape((1, -1))) / h), 0.0))
    w = (1 - w ** 3) ** 3
    yest = np.zeros(n)
    delta = np.ones(n)

    for _ in range(n_iter):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array(
                [
                    [np.sum(weights), np.sum(weights * x)],
                    [np.sum(weights * x), np.sum(weights * x * x)],
                ]
            )

            beta = np.linalg.lstsq(A, b)[0]
            yest[i] = beta[0] + beta[1] * x[i]

        residuals = y - yest
        s = np.median(np.abs(residuals))
        #delta = np.clip(residuals / (6.0 * s), -1.0, 1.0)
        delta = np.minimum(1.0, np.maximum(residuals / (6.0 * s), -1.0))
        delta = (1 - delta ** 2) ** 2

    return yest
