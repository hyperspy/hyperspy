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

import copy
import types
import  math
import glob
import os
import re
from StringIO import StringIO
import string
import codecs
try:
    from collections import OrderedDict
    ordict = True
except ImportError:
    # happens with Python < 2.7
    ordict = False

import numpy as np
import scipy as sp
import scipy.interpolate
import scipy.signal
import scipy.ndimage
from scipy.signal import medfilt

def import_rpy():
    try:
        import rpy
        print "rpy imported..."
    except:
        print "python-rpy is not installed"

import matplotlib.pyplot as plt

from hyperspy import messages
from hyperspy.gui import messages as messagesui
import hyperspy.defaults_parser

def dump_dictionary(file, dic, string = 'root', node_separator = '.',
                    value_separator = ' = '):
    for key in dic.keys():
        if isinstance(dic[key], dict):
            dump_dictionary(file, dic[key], string + node_separator + key)
        else:
            file.write(string + node_separator + key + value_separator +
            str(dic[key]) + '\n')

def sarray2dict(sarray, dictionary = None):
    '''Converts a struct array to an ordered dictionary

    Parameters
    ----------
    sarray: struct array
    dictionary: None or dic
        If dictionary is not None the content of sarray will be appended to the
        given dictonary

    Returns
    -------
    Ordered dictionary

    '''
    if dictionary is None:
        if ordict:
            dictionary = OrderedDict()
        else:
            print("\nWARNING:")
            print("sarray2dict")
            print("OrderedDict is not available, using a standard dictionary.\n")
            dictionary = {}
    for name in sarray.dtype.names:
        dictionary[name] = sarray[name][0] if len(sarray[name]) == 1 \
        else sarray[name]
    return dictionary

def generate_axis(origin,step,N,index=0):
    """Creates an axis given the origin, step and number of channels

    Alternatively, the index of the origin channel can be specified.

    Parameters
    ----------
    origin : float
    step : float
    N : number of channels
    index : int
        index number of the origin

    Returns
    -------
    Numpy array
    """
    return np.linspace(origin-index*step, origin+step*(N-1-index), N)

def check_cube_dimensions(sp1, sp2, warn = True):
    """Checks if the given SIs has the same dimensions.

    Parameters
    ----------
    sp1, sp2 : Spectrum instances
    warn : bool
        If True, produce a warning message if the SIs do not have the same
        dimensions
    Returns
    -------
    Boolean
    """
    if sp1.data_cube.shape == sp2.data_cube.shape:
        return True
    else:
        if warn:
            print \
            "The given SI objects do not have the same cube dimensions"
        return False

def check_energy_dimensions(sp1, sp2, warn = True, sp2_name = None):
    """Checks if sp2 is a single spectrum with the same energy dimension as sp1

    Parameters
    ----------
    sp1, sp2 : Spectrum instances
    warn : bool
        If True, produce a warning message if the SIs do not have the same
        dimensions

    Returns
    -------
    Boolean
    """
    sp2_dim = len(sp2.data_cube.squeeze().shape)
    sp1_Edim = sp1.data_cube.shape[0]
    sp2_Edim = sp2.data_cube.shape[0]

    if sp2_dim == 1:
        if sp1_Edim == sp2_Edim:
            return True
        else:
            if warn:
                print "The spectra must have the same energy dimensions"
            return False
    else:
        if warn:
            print "The %s should be unidimensional" % sp2_name
        return False

def unfold_if_multidim(signal):
    """Unfold the SI if it is 2D

    Parameters
    ----------
    signal : Signal instance

    Returns
    -------

    Boolean. True if the SI was unfolded by the function.
    """
    if len(signal.axes_manager.axes)>2:
        print "Automatically unfolding the SI"
        signal.unfold()
        return True
    else:
        return False
        
def _estimate_gain(ns, cs, weighted = False, higher_than = None, 
    plot_results = False, binning = 0, pol_order = 1):
    if binning > 0:
        factor = 2 ** binning
        remainder = np.mod(ns.shape[1], factor)
        if remainder != 0:
            ns = ns[:,remainder:]
            cs = cs[:,remainder:]
        new_shape = (ns.shape[0], ns.shape[1]/factor)
        ns = rebin(ns, new_shape)
        cs = rebin(cs, new_shape)

    noise = ns - cs
    variance = np.var(noise, 0)
    average = np.mean(cs, 0).squeeze()

    # Select only the values higher_than for the calculation
    if higher_than is not None:
        sorting_index_array = np.argsort(average)
        average_sorted = average[sorting_index_array]
        average_higher_than = average_sorted > higher_than
        variance_sorted = variance.squeeze()[sorting_index_array]
        variance2fit = variance_sorted[average_higher_than]
        average2fit = average_sorted[average_higher_than]
    else:
        variance2fit = variance
        average2fit = average
        
    fit = np.polyfit(average2fit, variance2fit, pol_order)
    if weighted is True:
        from hyperspy.signals.spectrum import Spectrum
        from hyperspy.model import Model
        from hyperspy.components import Line
        s = Spectrum({'data' : varso[avesoh]})
        s.axes_manager.signal_axes[0].axis = aveso[avesoh]
        m = Model(s)
        l = Line()
        l.a.value = fit[1]
        l.b.value = fit[0]
        m.append(l)
        m.fit(weights = True)
        fit[0] = l.b.value
        fit[1] = l.a.value
        
    if plot_results is True:
        plt.figure()
        plt.scatter(average.squeeze(), variance.squeeze())
        plt.xlabel('Counts')
        plt.ylabel('Variance')
        plt.plot(average2fit, np.polyval(fit,average2fit), color = 'red')
    results = {'fit' : fit, 'variance' : variance.squeeze(),
    'counts' : average.squeeze()}
    
    return results

def _estimate_correlation_factor(g0, gk,k):
    a = math.sqrt(g0/gk)
    e = k*(a-1)/(a-k)
    c = (1 - e)**2
    return c    

def estimate_variance_parameters(noisy_signal, clean_signal, mask=None,
    pol_order=1, higher_than=None, return_results=False,
    plot_results=True, weighted=False):
    """Find the scale and offset of the Poissonian noise

    By comparing an SI with its denoised version (i.e. by PCA), this plots an
    estimation of the variance as a function of the number of counts and fits a
    polynomy to the result.

    Parameters
    ----------
    noisy_SI, clean_SI : spectrum.Spectrum instances
    mask : numpy bool array
        To define the channels that will be used in the calculation.
    pol_order : int
        The order of the polynomy.
    higher_than: float
        To restrict the fit to counts over the given value.
        
    return_results : Bool
    
    plot_results : Bool

    Returns
    -------
    Dictionary with the result of a linear fit to estimate the offset and
    scale factor
    """
    fold_back_noisy =  unfold_if_multidim(noisy_signal)
    fold_back_clean =  unfold_if_multidim(clean_signal)
    ns = noisy_signal.data.copy()
    cs = clean_signal.data.copy()

    if mask is not None:
        ns = ns[~mask]
        cs = cs[~mask]

    results0 = _estimate_gain(ns, cs, weighted = weighted, 
        higher_than = higher_than, plot_results = plot_results, binning = 0,
        pol_order = pol_order)
        
    results2 = _estimate_gain(ns, cs, weighted = weighted, 
        higher_than = higher_than, plot_results = False, binning = 2,
        pol_order = pol_order)
        
    c = _estimate_correlation_factor(results0['fit'][0], results2['fit'][0],
        4)
    
    message = ("Gain factor: %.2f\n" % results0['fit'][0] +
               "Gain offset: %.2f\n" % results0['fit'][1] +
               "Correlation factor: %.2f\n" % c )
    is_ok = True
    if hyperspy.defaults_parser.preferences.General.interactive is True:
        is_ok = messagesui.information(message +
                                       "Would you like to store the results?")
    else:
        print message
    if is_ok:
        if not noisy_signal.mapped_parameters.has_item('Variance_estimation'):
            noisy_signal.mapped_parameters.add_node('Variance_estimation')
        noisy_signal.mapped_parameters.Variance_estimation.gain_factor = \
            results0['fit'][0]
        noisy_signal.mapped_parameters.Variance_estimation.gain_offset = \
            results0['fit'][1]
        noisy_signal.mapped_parameters.Variance_estimation.correlation_factor =\
            c
        noisy_signal.mapped_parameters.Variance_estimation.\
        parameters_estimation_method = 'Hyperspy'

    if fold_back_noisy is True:
        noisy_signal.fold()
    if fold_back_clean is True:
        clean_signal.fold()
        
    if return_results is True:
        return results0

def rebin(a, new_shape):
    """Rebin SI

    rebin ndarray data into a smaller ndarray of the same rank whose dimensions
    are factors of the original dimensions. eg. An array with 6 columns and 4
    rows can be reduced to have 6,3,2 or 1 columns and 4,2 or 1 rows.
    example usages:
    >>> a=rand(6,4); b=rebin(a,3,2)
    >>> a=rand(6); b=rebin(a,2)
    Adapted from scipy cookbook

    Parameters
    ----------
    a : numpy array
    new_shape : tuple
        shape after binning

    Returns
    -------
    numpy array
    """
    shape = a.shape
    lenShape = len(shape)
    factor = np.asarray(shape)/np.asarray(new_shape)
    evList = ['a.reshape('] + \
             ['new_shape[%d],factor[%d],'%(i,i) for i in xrange(lenShape)] + \
             [')'] + ['.sum(%d)'%(i+1) for i in xrange(lenShape)]
    return eval(''.join(evList))

def estimate_drift(im1,im2):
    """Estimate the drift  between two images by cross-correlation


    It preprocess the images by applying a median filter (to smooth the images)
    and the solbi edge detection filter.

    Parameters
    ----------
    im1, im2 : Image instances

    Output
    ------
    array with the coordinates of the translation of im2 in respect to im1.
    """
    print "Estimating the spatial drift"
    im1 = im1.data_cube.squeeze()
    im2 = im2.data_cube.squeeze()
    # Apply a "denoising filter" and edge detection filter
    im1 = scipy.ndimage.sobel(scipy.signal.medfilt(im1))
    im2 = scipy.ndimage.sobel(scipy.signal.medfilt(im2))
    # Compute the cross-correlation
    _correlation = scipy.signal.correlate(im1,im2)

    shift = im1.shape - np.ones(2) - \
    np.unravel_index(_correlation.argmax(),_correlation.shape)
    print "The estimated drift is ", shift
    return shift

def savitzky_golay(data, kernel = 11, order = 4):
    """Savitzky-Golay filter

    Adapted from scipy cookbook http://www.scipy.org/Cookbook/SavitzkyGolay

    Parameters
    ----------
    data : 1D numpy array
    kernel : positiv integer > 2*order giving the kernel size - order
    order : order of the polynomial

    Returns
    -------
    returns smoothed data as a numpy array

    Example
    -------
    smoothed = savitzky_golay(<rough>, [kernel = value], [order = value]
    """
    try:
            kernel = abs(int(kernel))
            order = abs(int(order))
    except ValueError, msg:
        raise ValueError("kernel and order have to be of type int (floats will \
        be converted).")
    if kernel % 2 != 1 or kernel < 1:
        raise TypeError("kernel size must be a positive odd number, was: %d"
        % kernel)
    if kernel < order + 2:
        raise TypeError(
        "kernel is to small for the polynomals\nshould be > order + 2")

    # a second order polynomal has 3 coefficients
    order_range = range(order+1)
    N = (kernel -1) // 2
    b = np.mat([[k**i for i in order_range] for k in xrange(-N,
    N+1)])
    # since we don't want the derivative, else choose [1] or [2], respectively
    m = np.linalg.pinv(b).A[0]
    window_size = len(m)
    N = (window_size-1) // 2

    # precompute the offset values for better performance
    offsets = range(-N, N+1)
    offset_data = zip(offsets, m)

    smooth_data = list()

    # temporary data, with padded zeros
    # (since we want the same length after smoothing)
    # temporary data, extended with a mirror image to the left and right
    firstval=data[0]
    lastval=data[len(data)-1]
    #left extension: f(x0-x) = f(x0)-(f(x)-f(x0)) = 2f(x0)-f(x)
    #right extension: f(xl+x) = f(xl)+(f(xl)-f(xl-x)) = 2f(xl)-f(xl-x)
    leftpad=np.zeros(N)+2*firstval
    rightpad=np.zeros(N)+2*lastval
    leftchunk=data[1:1+N]
    leftpad=leftpad-leftchunk[::-1]
    rightchunk=data[len(data)-N-1:len(data)-1]
    rightpad=rightpad-rightchunk[::-1]
    data = np.concatenate((leftpad, data))
    data = np.concatenate((data, rightpad))

#    data = np.concatenate((np.zeros(N), data, np.zeros(N)))
    for i in xrange(N, len(data) - N):
            value = 0.0
            for offset, weight in offset_data:
                value += weight * data[i + offset]
            smooth_data.append(value)
    return np.array(smooth_data)

# Functions to calculates de savitzky-golay filter from
def resub(D, rhs):
    """solves D D^T = rhs by resubstituion.
    D is lower triangle-matrix from cholesky-decomposition
    http://www.procoders.net
    """

    M = D.shape[0]
    x1= np.zeros((M,),float)
    x2= np.zeros((M,),float)

    # resub step 1
    for l in xrange(M):
        sum_ = rhs[l]
        for n in xrange(l):
            sum_ -= D[l,n]*x1[n]
        x1[l] = sum_/D[l,l]

    # resub step 2
    for l in xrange(M-1,-1,-1):
        sum_ = x1[l]
        for n in xrange(l+1,M):
            sum_ -= D[n,l]*x2[n]
        x2[l] = sum_/D[l,l]

    return x2


def calc_coeff(num_points, pol_degree, diff_order=0):
    """Calculates filter coefficients for symmetric savitzky-golay filter.
    http://www.procoders.net
    see: http://www.nrbook.com/a/bookcpdf/c14-8.pdf

    num_points   means that 2*num_points+1 values contribute to the
                 smoother.

    pol_degree   is degree of fitting polynomial

    diff_order   is degree of implicit differentiation.
                 0 means that filter results in smoothing of function
                 1 means that filter results in smoothing the first
                                             derivative of function.
                 and so on ...
    """

    # setup normal matrix
    A = np.zeros((2*num_points+1, pol_degree+1), float)
    for i in xrange(2*num_points+1):
        for j in xrange(pol_degree+1):
            A[i,j] = math.pow(i-num_points, j)

    # calculate diff_order-th row of inv(A^T A)
    ATA = np.dot(A.transpose(), A)
    rhs = np.zeros((pol_degree+1,), float)
    rhs[diff_order] = 1
    D = np.linalg.cholesky(ATA)
    wvec = resub(D, rhs)

    # calculate filter-coefficients
    coeff = np.zeros((2*num_points+1,), float)
    for n in xrange(-num_points, num_points+1):
        x = 0.0
        for m in xrange(pol_degree+1):
            x += wvec[m]*pow(n, m)
        coeff[n+num_points] = x
    return coeff * (-1) ** diff_order
    

def smooth(data, coeff):
    """applies coefficients calculated by calc_coeff() to signal
    http://www.procoders.net
    """
    # temporary data, extended with a mirror image to the left and right
    N = np.size(coeff-1)/2
    firstval=data[0]
    lastval=data[len(data)-1]
#    left extension: f(x0-x) = f(x0)-(f(x)-f(x0)) = 2f(x0)-f(x)
#    right extension: f(xl+x) = f(xl)+(f(xl)-f(xl-x)) = 2f(xl)-f(xl-x)
    leftpad=np.zeros(N)+2*firstval
    rightpad=np.zeros(N)+2*lastval
    leftchunk=data[1:1+N]
    leftpad=leftpad-leftchunk[::-1]
    rightchunk=data[len(data)-N-1:len(data)-1]
    rightpad=rightpad-rightchunk[::-1]
    data = np.concatenate((leftpad, data))
    data = np.concatenate((data, rightpad))
    res = np.convolve(data, coeff)
    return res[N:-N][len(leftpad):-len(rightpad)]

def sg(data, num_points, pol_degree, diff_order=0):
    """Savitzky-Golay filter
    http://www.procoders.net
    """
    coeff = calc_coeff(num_points, pol_degree, diff_order)
    return smooth(data, coeff)

def lowess(x, y, f=2/3., iter=3):
    """lowess(x, y, f=2./3., iter=3) -> yest

    Lowess smoother: Robust locally weighted regression.
    The lowess function fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.

    The smoothing span is given by f. A larger value for f will result in a
    smoother curve. The number of robustifying iterations is given by iter. The
    function will run faster with a smaller number of iterations.

    Code adapted from Biopython:

    Original doc:

    This module implements the Lowess function for nonparametric regression.

    Functions:
    lowess        Fit a smooth nonparametric regression curve to a scatterplot.

    For more information, see

    William S. Cleveland: "Robust locally weighted regression and smoothing
    scatterplots", Journal of the American Statistical Association, December 1979,
    volume 74, number 368, pp. 829-836.

    William S. Cleveland and Susan J. Devlin: "Locally weighted regression: An
    approach to regression analysis by local fitting", Journal of the American
    Statistical Association, September 1988, volume 83, number 403, pp. 596-610.
    """
    n = len(x)
    r = int(np.ceil(f*n))
    h = [np.sort(abs(x-x[i]))[r] for i in xrange(n)]
    w = np.clip(abs(([x]-np.transpose([x]))/h),0.0,1.0)
    w = 1-w*w*w
    w = w*w*w
    yest = np.zeros(n,'d')
    delta = np.ones(n,'d')
    for iteration in xrange(iter):
        for i in xrange(n):
            weights = delta * w[:,i]
            b = np.array([np.sum(weights*y), np.sum(weights*y*x)])
            A = np.array([[np.sum(weights), np.sum(weights*x)],
                     [np.sum(weights*x), np.sum(weights*x*x)]])
            beta = np.linalg.solve(A,b)
            yest[i] = beta[0] + beta[1]*x[i]
        residuals = y-yest
        s = np.median(abs(residuals))
        delta = np.clip(residuals/(6*s),-1,1)
        delta = 1-delta*delta
        delta = delta*delta
    return yest


def wavelet_poissonian_denoising(spectrum):
    """Denoise data with pure Poissonian noise using wavelets

    Wrapper around the R packages EbayesThresh and wavethresh

    Parameters
    ----------
    spectrum : spectrum instance

    Returns
    -------
    Spectrum instance.
    """
    import_rpy()
    rpy.r.library('EbayesThresh')
    rpy.r.library('wavethresh')
    rpy.r['<-']('X',spectrum)
    rpy.r('XHF <- hft(X)')
    rpy.r('XHFwd  <- wd(XHF, bc="symmetric")')
    rpy.r('XHFwdT  <- ebayesthresh.wavelet(XHFwd)')
    rpy.r('XHFdn  <- wr(XHFwdT)')
    XHFest = rpy.r('XHFest <- hft.inv(XHFdn)')
    return XHFest

def wavelet_gaussian_denoising(spectrum):
    """Denoise data with pure Gaussian noise using wavelets

    Wrapper around the R packages EbayesThresh and wavethresh

    Parameters
    ----------
    spectrum : spectrum instance

    Returns
    -------
    Spectrum instance.
    """
    import_rpy()
    rpy.r.library('EbayesThresh')
    rpy.r.library('wavethresh')
    rpy.r['<-']('X',spectrum)
    rpy.r('Xwd  <- wd(X, bc="symmetric")')
    rpy.r('XwdT  <- ebayesthresh.wavelet(Xwd)')
    Xdn = rpy.r('Xdn  <- wr(XwdT)')
    return Xdn

def wavelet_dd_denoising(spectrum):
    """Denoise data with arbitraty noise using wavelets

    Wrapper around the R packages EbayesThresh, wavethresh and DDHFm

    Parameters
    ----------
    spectrum : spectrum instance

    Returns
    -------
    Spectrum instance.
    """
    import_rpy()
    rpy.r.library('EbayesThresh')
    rpy.r.library('wavethresh')
    rpy.r.library('DDHFm')
    rpy.r['<-']('X',spectrum)
    rpy.r('XDDHF <- ddhft.np.2(X)')
    rpy.r('XDDHFwd  <- wd(XDDHF$hft,filter.number = 8, bc="symmetric" )')
    rpy.r('XDDHFwdT  <- ebayesthresh.wavelet(XDDHFwd)')
    rpy.r('XDDHFdn  <- wr(XDDHFwdT)')
    rpy.r('XDDHF$hft  <- wr(XDDHFwdT)')
    XHFest = rpy.r('XHFest <- ddhft.np.inv(XDDHF)')
    return XHFest

def loess(y,x = None, span = 0.2):
    """locally weighted scatterplot smoothing

    Wrapper around the R funcion loess

    Parameters
    ----------
    spectrum : spectrum instance
    span : float
        parameter to control the smoothing

    Returns
    -------
    Spectrum instance.
    """
    import_rpy()
    if x is None:
        x = np.arange(0,len(y))
    rpy.r['<-']('x',x)
    rpy.r['<-']('y',y)
    rpy.r('y.loess <- loess(y ~ x, span = %s, data.frame(x=x, y=y))' % span)
    loess = rpy.r('y.predict <- predict(y.loess, data.frame(x=x))')
    return loess



def ALS(s, thresh =.001, nonnegS = True, nonnegC = True):
    """Alternate least squares

    Wrapper around the R's ALS package

    Parameters
    ----------
    s : Spectrum instance
    threshold : float
        convergence criteria
    nonnegS : bool
        if True, impose non-negativity constraint on the components
    nonnegC : bool
        if True, impose non-negativity constraint on the maps

    Returns
    -------
    Dictionary
    """
    import_rpy()
#    Format
#    ic format (channels, components)
#    W format (experiment, components)
#    s format (experiment, channels)

    nonnegS = 'TRUE' if nonnegS is True else 'FALSE'
    nonnegC = 'TRUE' if nonnegC is True else 'FALSE'
    print "Non negative constraint in the sources: ", nonnegS
    print "Non negative constraint in the mixing matrix: ", nonnegC

    refold = unfold_if_2D(s)
    W = s._calculate_recmatrix().T
    ic = np.ones(s.ic.shape)
    rpy.r.library('ALS')
    rpy.r('W = NULL')
    rpy.r('ic = NULL')
    rpy.r('d1 = NULL')
    rpy.r['<-']('d1', s.data_cube.squeeze().T)
    rpy.r['<-']('W', W)
    rpy.r['<-']('ic', ic)
    i = 0
    # Workaround a bug in python rpy version 1
    while hasattr(rpy.r, 'test' + str(i)):
        rpy.r('test%s = NULL' % i)
        i+=1
    rpy.r('test%s = als(CList = list(W), thresh = %s, S = ic,\
     PsiList = list(d1), nonnegS = %s, nonnegC = %s)' %
     (i, thresh, nonnegS, nonnegC))
    if refold:
        s.fold()
    exec('als_result = rpy.r.test%s' % i)
    return als_result



def amari(C,A):
    """Amari test for ICA
    Adapted from the MILCA package http://www.klab.caltech.edu/~kraskov/MILCA/

    Parameters
    ----------
    C : numpy array
    A : numpy array
    """
    b,a = C.shape

    dummy= np.dot(np.linalg.pinv(A),C)
    dummy = np.sum(_ntu(np.abs(dummy)),0)-1

    dummy2 = np.dot(np.linalg.pinv(C),A)
    dummy2 = np.sum(_ntu(np.abs(dummy2)),0)-1

    out=(np.sum(dummy)+np.sum(dummy2))/(2*a*(a-1))
    return out

def _ntu(C):
    m, n = C.shape
    CN = C.copy() * 0
    for t in xrange(n):
        CN[:,t] = C[:,t] / np.max(np.abs(C[:,t]))
    return CN

def analyze_readout(spectrum):
    """Readout diagnostic tool

    Parameters
    ----------
    spectrum : Spectrum instance

    Returns
    -------
    tuple of float : (variance, mean, normalized mean as a function of time)
    """
    s = spectrum
    # If it is 2D, sum the first axis.
    if s.data_cube.shape[2] > 1:
        dc = s.data_cube.sum(1)
    else:
        dc = s.data_cube.squeeze()
    time_mean = dc.mean(0).squeeze()
    norm_time_mean = time_mean / time_mean.mean()
    corrected_dc = dc * (1/norm_time_mean.reshape((1,-1)))
    channel_mean = corrected_dc.mean(1)
    variance = (corrected_dc - channel_mean.reshape((-1,1))).var(0)
    return variance, channel_mean, norm_time_mean

def multi_readout_analyze(folder, ccd_height = 100., plot = True, freq = None):
    """Analyze several readout measurements in different files for readout
    diagnosys

    The readout files in dm3 format must be contained in a folder, preferentely
    numered in the order of acquisition.

    Parameters
    ----------
    folder : string
        Folder where the dm3 readout files are stored
    ccd_heigh : float
    plot : bool
    freq : float
        Frequency of the camera

    Returns
    -------
    Dictionary
    """
    from spectrum import Spectrum
    files = glob.glob1(folder, '*.nc')
    if not files:
        files = glob.glob1(folder, '*.dm3')
    spectra = []
    variances = []
    binnings = []
    for f in files:
        print os.path.join(folder,f)
        s = Spectrum(os.path.join(folder,f))
        variance, channel_mean, norm_time_mean = analyze_readout(s)
        s.readout_analysis = {}
        s.readout_analysis['variance'] = variance.mean()
        s.readout_analysis['pattern'] = channel_mean
        s.readout_analysis['time'] = norm_time_mean
        if not hasattr(s,'binning'):
            s.binning = float(os.path.splitext(f)[0][1:])
            if freq:
                s.readout_frequency = freq
                s.ccd_height = ccd_height
            s.save(f)
        spectra.append(s)
        binnings.append(s.binning)
        variances.append(variance.mean())
    pixels = ccd_height / np.array(binnings)
    plt.scatter(pixels, variances, label = 'data')
    fit = np.polyfit(pixels, variances,1, full = True)
    if plot:
        x = np.linspace(0,pixels.max(),100)
        y = x*fit[0][0] + fit[0][1]
        plt.plot(x,y, label = 'linear fit')
        plt.xlabel('number of pixels')
        plt.ylabel('variance')
        plt.legend(loc = 'upper left')

    print "Variance = %s * pixels + %s" % (fit[0][0], fit[0][1])
    dictio = {'pixels': pixels, 'variances': variances, 'fit' : fit,
    'spectra' : spectra}
    return dictio

def chrono_align_and_sum(spectrum, energy_range = (None, None),
                         spatial_shape = None):
    """Alignment and sum of a chrono-spim SI

    Parameters
    ----------
    spectrum : Spectrum instance
        Chrono-spim
    energy_range : tuple of floats
        energy interval in which to perform the alignment in energy units
    axis : int
    """
    from spectrum import Spectrum
    dc = spectrum.data_cube
    min_energy_size = dc.shape[0]
#    i = 0
    new_dc = None

    # For the progress bar to work properly we must capture the output of the
    # functions that are called during the alignment process
    import cStringIO
    import sys
    capture_output = cStringIO.StringIO()

    from hyperspy.misc.progressbar import progressbar
    pbar = progressbar(maxval = dc.shape[2] - 1)
    for i in xrange(dc.shape[2]):
        pbar.update(i)
        sys.stdout = capture_output
        s = Spectrum({'calibration': {'data_cube' : dc[:,:,i]}})
        s.get_calibration_from(spectrum)
        s.find_low_loss_origin()
        s.align(energy_range, progress_bar = False)
        min_energy_size = min(s.data_cube.shape[0], min_energy_size)
        if new_dc is None:
            new_dc = s.data_cube.sum(1)
        else:
            new_dc = np.concatenate([new_dc[:min_energy_size],
                                     s.data_cube.sum(1)[:min_energy_size]], 1)
        sys.stdout = sys.__stdout__
    pbar.finish()
    spectrum.data_cube = new_dc
    spectrum.get_dimensions_from_cube()
    spectrum.find_low_loss_origin()
    spectrum.align(energy_range)
    spectrum.find_low_loss_origin()
    if spatial_shape is not None:
        spectrum.data_cube = spectrum.data_cube.reshape(
        [spectrum.data_cube.shape[0]] + list(spatial_shape))
        spectrum.data_cube = spectrum.data_cube.swapaxes(1,2)
        spectrum.get_dimensions_from_cube()

def copy_energy_calibration(from_spectrum, to_spectrum):
    """Copy the energy calibration between two SIs

    Parameters
    ----------
    from_spectrum, to spectrum : Spectrum instances
    """
    f = from_spectrum
    t = to_spectrum
    t.energyscale = f.energyscale
    t.energyorigin = f.energyorigin
    t.energyunits = f.energyunits
    t.get_dimensions_from_cube()
    t.updateenergy_axis()



def str2num(string, **kargs):
    """Transform a a table in string form into a numpy array

    Parameters
    ----------
    string : string

    Returns
    -------
    numpy array
    """
    stringIO = StringIO(string)
    return np.loadtxt(stringIO, **kargs)

def PL_signal_ratio(E, delta = 1000., exponent = -1.96):
    """Ratio between the intensity at E and E+delta in a powerlaw

    Parameters:
    -----------
    E : float or array
    delta : float
    exponent :  float
    """
    return ((E+float(delta))/E)**exponent

def order_of_magnitude(number):
    """Order of magnitude of the given number

    Parameters
    ----------
    number : float

    Returns
    -------
    Float
    """
    return math.floor(math.log10(number))

def bragg_scattering_angle(d, E0 = 100):
    """
    Parameters
    ----------
    d : float
        interplanar distance in m
    E0 : float
        Incident energy in keV

    Returns
    -------
    float : Semiangle of scattering of the first order difracted beam. This is
    two times the bragg angle.
    """

    gamma = 1 + E0 / 511.
    v_rel = np.sqrt(1-1/gamma**2)
    e_lambda = 2*np.pi/(2590e9*(gamma*v_rel)) # m
    print "Lambda = ", e_lambda

    return e_lambda / d

def effective_Z(Z_list, exponent = 2.94):
    """Effective atomic number of a compound or mixture

    exponent = 2.94 for X-ray absorption

    Parameters
    ----------
    Z_list : list
        A list of tuples (f,Z) where f is the number of atoms of the element in
        the molecule and Z its atomic number

    Return
    ------
    float
    """
    exponent = float(exponent)
    temp = 0
    total_e = 0
    for Z in Z_list:
        temp += Z[1]*Z[1]**exponent
        total_e += Z[0]*Z[1]
    print total_e
    return (temp/total_e)**(1/exponent)

def power_law_perc_area(E1,E2, r):
    a = E1
    b = E2
    return 100*((a**r*r-a**r)*(a/(a**r*r-a**r)-(b+a)/((b+a)**r*r-(b+a)**r)))/a

def rel_std_of_fraction(a,std_a,b,std_b,corr_factor = 1):
    rel_a = std_a/a
    rel_b = std_b/b
    return np.sqrt(rel_a**2 +  rel_b**2-2*rel_a*rel_b*corr_factor)

def ratio(edge_A, edge_B):
    a = edge_A.intensity.value
    std_a = edge_A.intensity.std
    b = edge_B.intensity.value
    std_b = edge_B.intensity.std
    ratio = a/b
    ratio_std = ratio * rel_std_of_fraction(a,std_a,b, std_b)
    print "Ratio %s/%s %1.3f +- %1.3f " % (edge_A.name, edge_B.name, a/b,
    1.96*ratio_std )
    return ratio, ratio_std

def iterate_axis(data, axis = -1):
        # We make a copy to guarantee that the data in contiguous, otherwise
        # it will not return a view of the data
#        data = data.copy()
        if axis < 0:
            axis = len(data.shape) + axis
        unfolded_axis = axis - 1
        new_shape = [1] * len(data.shape)
        new_shape[axis] = data.shape[axis]
        new_shape[unfolded_axis] = -1
        data = data.reshape(new_shape)
        for i in xrange(data.shape[unfolded_axis]):
            getitem = [0] * len(data.shape)
            getitem[axis] = slice(None)
            getitem[unfolded_axis] = i
            yield(data[getitem])

#  
#  name: interpolate_1D
#  @param
#  @return
#  
def interpolate_1D(number_of_interpolation_points, data):
    ip = number_of_interpolation_points
    ch = len(data)
    old_ax = np.linspace(0, 100, ch)
    new_ax = np.linspace(0, 100, ch * ip - (ip-1))
    interpolator = sp.interpolate.interp1d(old_ax,data)
    return interpolator(new_ax)
    
_slugify_strip_re = re.compile(r'[^\w\s-]')
_slugify_hyphenate_re = re.compile(r'[-\s]+')
def slugify(value, valid_variable_name=False):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    
    Adapted from Django's "django/template/defaultfilters.py".
    """
    import unicodedata
    if not isinstance(value, unicode):
        value = value.decode('utf8')
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore')
    value = unicode(_slugify_strip_re.sub('', value).strip())
    value = _slugify_hyphenate_re.sub('_', value)
    if valid_variable_name is True:
        if value[:1].isdigit():
            value = u'Number_' + value
    return value
    
class DictionaryBrowser(object):
    """A class to comfortably access some parameters as attributes"""

    def __init__(self, dictionary={}):
        super(DictionaryBrowser, self).__init__()
        self._load_dictionary(dictionary)

    def _load_dictionary(self, dictionary):
        for key, value in dictionary.iteritems():
            self.__setattr__(key, value)
            
    def export(self, filename, encoding = 'utf8'):
        """Export the dictionary to a text file
        
        Parameters
        ----------
        filename : str
            The name of the file without the extension that is
            txt by default
        encoding : valid encoding str
        """
        f = codecs.open(filename, 'w', encoding = encoding)
        f.write(self._get_print_items(max_len=None))
        f.close()

    def _get_print_items(self, padding = '', max_len=20):
        """Prints only the attributes that are not methods"""
        string = ''
        eoi = len(self.__dict__)
        j = 0
        for key_, value in self.__dict__.iteritems():
            if type(key_) != types.MethodType:
                key = ensure_unicode(value['key'])
                value = ensure_unicode(value['value'])
                if isinstance(value, DictionaryBrowser):
                    if j == eoi - 1:
                        symbol = u'└── '
                    else:
                        symbol = u'├── '
                    string += u'%s%s%s\n' % (padding, symbol, key)
                    if j == eoi - 1:
                        extra_padding = u'    '
                    else:
                        extra_padding = u'│   '
                    string += value._get_print_items(
                        padding + extra_padding)
                else:
                    if j == eoi - 1:
                        symbol = u'└── '
                    else:
                        symbol = u'├── '
                    strvalue = unicode(value)
                    if max_len is not None and \
                        len(strvalue) > 2 * max_len:
                        right_limit = min(max_len,
                                          len(strvalue)-max_len)
                        value = u'%s ... %s' % (strvalue[:max_len],
                                              strvalue[-right_limit:])
                    string += u"%s%s%s = %s\n" % (
                                        padding, symbol, key, value)
            j += 1
        return string

    def __repr__(self):
        return self._get_print_items().encode('utf8', errors='ignore')

    def __getitem__(self,key):
        return self.__getattribute__(key)
        
    def __setitem__(self,key, value):
        self.__setattr__(key, value)
        
    def __getattribute__(self,name):
        item = super(DictionaryBrowser,self).__getattribute__(name)
        if isinstance(item, dict) and 'value' in item:
            return item['value']
        else:
            return item
            
    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictionaryBrowser(value)
        super(DictionaryBrowser,self).__setattr__(
                         slugify(key, valid_variable_name=True),
                         {'key' : key, 'value' : value})

    def len(self):
        return len(self.__dict__.keys())

    def keys(self):
        return self.__dict__.keys()

    def as_dictionary(self):
        par_dict = {}
        for key_, item_ in self.__dict__.iteritems():
            if type(item_) != types.MethodType:
                key = item_['key']
                if isinstance(item_['value'], DictionaryBrowser):
                    item = item_['value'].as_dictionary()
                else:
                    item = item_['value']
                par_dict.__setitem__(key, item)
        return par_dict
        
    def has_item(self, item_path):
        """Given a path, return True if it exists
        
        Parameters
        ----------
        item_path : Str
            A string describing the path with each item separated by 
            full stops (periods)
            
        Examples
        --------
        
        >>> dict = {'To' : {'be' : True}}
        >>> dict_browser = DictionaryBrowser(dict)
        >>> dict_browser.has_item('To')
        True
        >>> dict_browser.has_item('To.be')
        True
        >>> dict_browser.has_item('To.be.or')
        False
        
        
        """
        if type(item_path) is str:
            item_path = item_path.split('.')
        else:
            item_path = copy.copy(item_path)
        attrib = item_path.pop(0)
        if hasattr(self, attrib):
            if len(item_path) == 0:
                return True
            else:
                item = self[attrib]
                if isinstance(item, type(self)): 
                    return item.has_item(item_path)
                else:
                    return False
        else:
            return False
            
    def set_item(self, item_path, value):
        """Given the path and value, create the missing nodes in
        the path and assign to the last one the value
        
        Parameters
        ----------
        item_path : Str
            A string describing the path with each item separated by a 
            full stops (periods)
            
        Examples
        --------
        
        >>> dict_browser = DictionaryBrowser({})
        >>> dict_browser.set_item('First.Second.Third', 3)
        >>> dict_browser
        └── First
           └── Second
                └── Third = 3
        
        """
        if not self.has_item(item_path):
            self.add_node(item_path)
        if type(item_path) is str:
            item_path = item_path.split('.')
        if len(item_path) > 1:
            self.__getattribute__(item_path.pop(0)).set_item(
                item_path, value)
        else:
            self.__setattr__(item_path.pop(), value)



    def add_node(self, node_path):
        """Adds all the nodes in the given path if they don't exist.
        
        Parameters
        ----------
        node_path: str
            The nodes must be separated by full stops (periods).
            
        Examples
        --------
        
        >>> dict_browser = DictionaryBrowser({})
        >>> dict_browser.add_node('First.Second')
        >>> dict_browser.First.Second = 3
        >>> dict_browser
        └── First
            └── Second = 3

        """
        keys = node_path.split('.')
        for key in keys:
            if self.has_item(key) is False:
                self[key] = DictionaryBrowser()
            self = self[key]
            
            
def orthomax(A, gamma=1, reltol=1.4901e-07, maxit=256):
    #ORTHOMAX Orthogonal rotation of FA or PCA loadings.
    # Taken from metpy
    
    d,m=A.shape
    B = np.copy(A)
    T = np.eye(m)

    converged = False
    if (0 <= gamma) & (gamma <= 1):
        while converged == False:
#           Use Lawley and Maxwell's fast version
            D = 0
            for k in range(1,maxit+1):
                Dold = D
                tmp11 = np.sum(np.power(B,2),axis=0)
                tmp1 = np.matrix(np.diag(np.array(tmp11).flatten()))
                tmp2 = gamma*B
                tmp3 = d*np.power(B,3)
                L,D,M=np.linalg.svd(np.dot(A.transpose(),tmp3-np.dot(tmp2,tmp1)))
                T = np.dot(L,M)
                D = np.sum(np.diag(D))
                B = np.dot(A,T)
                if (np.abs(D - Dold)/D < reltol):
                    converged = True
                    break
    else:
#       Use a sequence of bivariate rotations
        for iter in range(1,maxit+1):
            print iter
            maxTheta = 0
            for i in range(0,m-1):
                for j in range(i,m):
                    Bi=B[:,i]
                    Bj=B[:,j]
                    u = np.multiply(Bi,Bi)-np.multiply(Bj,Bj)
                    v = 2*np.multiply(Bi,Bj)
                    usum=u.sum()
                    vsum=v.sum()
                    numer = 2*np.dot(u.transpose(),v)-2*gamma*usum*vsum/d
                    denom=np.dot(u.transpose(),u) - np.dot(v.transpose(),v) - \
                                 gamma*(usum**2 - vsum**2)/d
                    theta = np.arctan2(numer,denom)/4
                    maxTheta=max(maxTheta,np.abs(theta))
                    Tij = np.array([[np.cos(theta),-np.sin(theta)],
                                   [np.sin(theta),np.cos(theta)]])
                    B[:,[i,j]] = np.dot(B[:,[i,j]],Tij)
                    T[:,[i,j]] = np.dot(T[:,[i,j]],Tij)
            if (maxTheta < reltol):
                converged = True
                break
    return B,T
    
def append2pathname(filename, to_append):
    """Append a string to a path name
    
    Parameters
    ----------
    filename : str
    to_append : str
    
    """
    pathname, extension = os.path.splitext(filename)
    return pathname + to_append + extension
    
def incremental_filename(filename, i=1):
    """If a file with the same file name exists, returns a new filename that
    does not exists.
    
    The new file name is created by appending `-n` (where `n` is an integer)
    to path name
    
    Parameters
    ----------
    filename : str
    i : int
       The number to be appended.
    """
    
    if os.path.isfile(filename):
        new_filename = append2pathname(filename, '-%s' % i) 
        if os.path.isfile(new_filename):
            return incremental_filename(filename, i + 1)
        else:
            return new_filename
    else:
        return filename
        
def ensure_directory(path):
    """Check if the path exists and if it does not create the directory"""
    directory = os.path.split(path)[0]
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        
def ensure_unicode(stuff, encoding = 'utf8', encoding2 = 'latin-1'):
    if type(stuff) is not str and type(stuff) is not np.string_:
        return stuff
    else:
        string = stuff
    try:
        string = string.decode(encoding)
    except:
        string = string.decode(encoding2, errors = 'ignore')
    return string
    
def one_dim_findpeaks(y, x=None, slope_thresh=0.5, amp_thresh=None,
    medfilt_radius=5, maxpeakn=30000, peakgroup=10, subchannel=True,
    peak_array=None):
    """
    Find peaks along a 1D line.

    Function to locate the positive peaks in a noisy x-y data set.
    
    Detects peaks by looking for downward zero-crossings in the first
    derivative that exceed 'slope_thresh'.
    
    Returns an array containing position, height, and width of each peak.

    'slope_thresh' and 'amp_thresh', control sensitivity: higher values will
    neglect smaller features.

    Parameters
    ---------
    y : array
        1D input array, e.g. a spectrum
        
    x : array (optional)
        1D array describing the calibration of y (must have same shape as y)

    slope_thresh : float (optional)
                   1st derivative threshold to count the peak
                   default is set to 0.5
                   higher values will neglect smaller features.
                   
    amp_thresh : float (optional)
                 intensity threshold above which   
                 default is set to 10% of max(y)
                 higher values will neglect smaller features.
                                  
    medfilt_radius : int (optional)
                     median filter window to apply to smooth the data
                     (see scipy.signal.medfilt)
                     if 0, no filter will be applied.
                     default is set to 5

    peakgroup : int (optional)
                number of points around the "top part" of the peak
                default is set to 10

    maxpeakn : int (optional)
              number of maximum detectable peaks
              default is set to 30000
                
    subchannel : bool (optional)
             default is set to True

    peak_array : array of shape (n, 3) (optional)
             A pre-allocated numpy array to fill with peaks.  Saves memory,
             especially when using the 2D peakfinder.

    Returns
    -------
    P : array of shape (npeaks, 3)
        contains position, height, and width of each peak
    
    Notes
    -----
    Original code from T. C. O'Haver, 1995.
    Version 2  Last revised Oct 27, 2006 Converted to Python by 
    Michael Sarahan, Feb 2011.
    Revised to handle edges better.  MCS, Mar 2011
    """

    if x is None:
        x = np.arange(len(y),dtype=np.int64)
    if not amp_thresh:
        amp_thresh = 0.1 * y.max()
    peakgroup = np.round(peakgroup)
    if medfilt_radius:
        d = np.gradient(medfilt(y,medfilt_radius))
    else:
        d = np.gradient(y)
    n = np.round(peakgroup / 2 + 1)
    if peak_array is None:
        # allocate a result array for 'maxpeakn' peaks
        P = np.zeros((maxpeakn, 3))
    else:
        maxpeakn=peak_array.shape[0]
        P=peak_array
    peak = 0
    for j in xrange(len(y) - 4):
        if np.sign(d[j]) > np.sign(d[j+1]): # Detects zero-crossing
            if np.sign(d[j+1]) == 0: continue
            # if slope of derivative is larger than slope_thresh
            if d[j] - d[j+1] > slope_thresh:
                # if height of peak is larger than amp_thresh
                if y[j] > amp_thresh:  
                    # the next section is very slow, and actually messes
                    # things up for images (discrete pixels),
                    # so by default, don't do subchannel precision in the
                    # 1D peakfind step.
                    if subchannel:
			xx = np.zeros(peakgroup)
			yy = np.zeros(peakgroup)
			s = 0
			for k in xrange(peakgroup): 
			    groupindex = j + k - n + 1 
			    if groupindex < 1:
				xx = xx[1:]
				yy = yy[1:]
				s += 1
				continue
			    elif groupindex > y.shape[0] - 1:
				xx = xx[:groupindex-1]
				yy = yy[:groupindex-1]
				break
			    xx[k-s] = x[groupindex]
			    yy[k-s] = y[groupindex]
			avg = np.average(xx)
			stdev = np.std(xx)
			xxf = (xx - avg) / stdev
			# Fit parabola to log10 of sub-group with
                        # centering and scaling
			coef = np.polyfit(xxf, np.log10(np.abs(yy)), 2)  
			c1 = coef[2]
			c2 = coef[1]
			c3 = coef[0]
			width = np.linalg.norm(
                            stdev * 2.35703 / (np.sqrt(2) * np.sqrt(-1 * c3)))
			# if the peak is too narrow for least-squares
                        # technique to work  well, just use the max value
                        # of y in the sub-group of points near peak.
			if peakgroup < 7:
			    height = np.max(yy)
			    location = xx[np.argmin(np.abs(yy - height))]
			else:
                            location =- ((stdev * c2 / (2 * c3)) - avg)
			    height = np.exp( c1 - c3 * (c2 / (2 * c3))**2)    
                    # Fill results array P. One row for each peak 
                    # detected, containing the
                    # peak position (x-value) and peak height (y-value).
		    else:
			location = x[j]
			height = y[j]
                        # no way to know peak width without
                        # the above measurements.
			width = 0
                    if (location > 0 and not np.isnan(location)
                        and location < x[-1]):
                        P[peak] = np.array([location, height, width])
                        peak = peak + 1
    # return only the part of the array that contains peaks
    # (not the whole maxpeakn x 3 array)
    return P[:peak,:]
    
def strlist2enumeration(lst):
    lst = tuple(lst)
    if not lst:
        return ''
    elif len(lst) == 1:
        return lst[0]
    elif len(lst) == 2:
        return "%s and %s" % lst
    else:
        return "%s, "*(len(lst) - 2) % lst[:-2] + "%s and %s" % lst[-2:]
        
def get_array_memory_size_in_GiB(shape, dtype):
    """Given the size and dtype returns the amount of memory that such
    an array needs to allocate
    
    Parameters
    ----------
    shape: tuple
    dtype : data-type
        The desired data-type for the array.
    """
    if isinstance(dtype, str):
        dtype = np.dtype(dtype)
    return np.array(shape).cumprod()[-1] * dtype.itemsize / 2.**30
    
def symmetrize(a):
    return a + a.swapaxes(0,1) - np.diag(a.diagonal())
    
def antisymmetrize(a):    
    return a - a.swapaxes(0,1)+ np.diag(a.diagonal())
    
def get_linear_interpolation(p1, p2, x):
    '''Given two points in 2D returns y for a given x for y = ax + b
    
    Parameters
    ----------
    p1,p2 : (x, y)
    x : float
    
    Returns
    -------
    y : float
    
    '''  
    x1, y1 =  p1
    x2, y2 = p2
    a = (y2 - y1) / (x2 - x1)
    b = (x2 * y1 - x1 * y2) / (x2-x1)
    y = a*x + b
    return y

