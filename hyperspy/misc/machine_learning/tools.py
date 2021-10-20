import numpy as np


def amari(C, A):
    """Amari test for ICA
    Adapted from the MILCA package http://www.klab.caltech.edu/~kraskov/MILCA/

    Parameters
    ----------
    C : numpy array
    A : numpy array
    """
    b, a = C.shape

    dummy = np.dot(np.linalg.pinv(A), C)
    dummy = np.sum(_ntu(np.abs(dummy)), 0) - 1

    dummy2 = np.dot(np.linalg.pinv(C), A)
    dummy2 = np.sum(_ntu(np.abs(dummy2)), 0) - 1

    out = (np.sum(dummy) + np.sum(dummy2)) / (2 * a * (a - 1))
    return out


def _ntu(C):
    m, n = C.shape
    CN = C.copy() * 0
    for t in range(n):
        CN[:, t] = C[:, t] / np.max(np.abs(C[:, t]))
    return CN
