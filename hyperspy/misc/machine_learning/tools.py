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
    for t in xrange(n):
        CN[:, t] = C[:, t] / np.max(np.abs(C[:, t]))
    return CN


# def ALS(s, thresh =.001, nonnegS = True, nonnegC = True):
#    """Alternate least squares
#
#    Wrapper around the R's ALS package
#
#    Parameters
#    ----------
#    s : Spectrum instance
#    threshold : float
#        convergence criteria
#    nonnegS : bool
#        if True, impose non-negativity constraint on the components
#    nonnegC : bool
#        if True, impose non-negativity constraint on the maps
#
#    Returns
#    -------
#    Dictionary
#    """
#    import_rpy()
# Format
# ic format (channels, components)
# W format (experiment, components)
# s format (experiment, channels)
#
#    nonnegS = 'TRUE' if nonnegS is True else 'FALSE'
#    nonnegC = 'TRUE' if nonnegC is True else 'FALSE'
#    print "Non negative constraint in the sources: ", nonnegS
#    print "Non negative constraint in the mixing matrix: ", nonnegC
#
#    refold = unfold_if_2D(s)
#    W = s._calculate_recmatrix().T
#    ic = np.ones(s.ic.shape)
#    rpy.r.library('ALS')
#    rpy.r('W = NULL')
#    rpy.r('ic = NULL')
#    rpy.r('d1 = NULL')
#    rpy.r['<-']('d1', s.data_cube.squeeze().T)
#    rpy.r['<-']('W', W)
#    rpy.r['<-']('ic', ic)
#    i = 0
# Workaround a bug in python rpy version 1
#    while hasattr(rpy.r, 'test' + str(i)):
#        rpy.r('test%s = NULL' % i)
#        i+=1
#    rpy.r('test%s = als(CList = list(W), thresh = %s, S = ic,\
#     PsiList = list(d1), nonnegS = %s, nonnegC = %s)' %
#     (i, thresh, nonnegS, nonnegC))
#    if refold:
#        s.fold()
#    exec('als_result = rpy.r.test%s' % i)
#    return als_result
