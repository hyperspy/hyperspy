import math

import numpy as np
import scipy.interpolate
import scipy.signal


def find_peaks_ohaver(y, x=None, slope_thresh=0., amp_thresh=None,
                      medfilt_radius=5, maxpeakn=30000, peakgroup=10, subchannel=True,):
    """Find peaks along a 1D line.

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

    Returns
    -------
    P : structured array of shape (npeaks) and fields: position, width, height
        contains position, height, and width of each peak

    Examples
    --------
    >>> x = arange(0,50,0.01)
    >>> y = cos(x)
    >>> one_dim_findpeaks(y, x,0,0)
    array([[  1.68144859e-05,   9.99999943e-01,   3.57487961e+00],
           [  6.28318614e+00,   1.00000003e+00,   3.57589018e+00],
           [  1.25663708e+01,   1.00000002e+00,   3.57600673e+00],
           [  1.88495565e+01,   1.00000002e+00,   3.57597295e+00],
           [  2.51327421e+01,   1.00000003e+00,   3.57590284e+00],
           [  3.14159267e+01,   1.00000002e+00,   3.57600856e+00],
           [  3.76991124e+01,   1.00000002e+00,   3.57597984e+00],
           [  4.39822980e+01,   1.00000002e+00,   3.57591479e+00]])

    Notes
    -----
    Original code from T. C. O'Haver, 1995.
    Version 2  Last revised Oct 27, 2006 Converted to Python by
    Michael Sarahan, Feb 2011.
    Revised to handle edges better.  MCS, Mar 2011

    """

    if x is None:
        x = np.arange(len(y), dtype=np.int64)
    if not amp_thresh:
        amp_thresh = 0.1 * y.max()
    peakgroup = np.round(peakgroup)
    if medfilt_radius:
        d = np.gradient(scipy.signal.medfilt(y, medfilt_radius))
    else:
        d = np.gradient(y)
    n = np.round(peakgroup / 2 + 1)
    peak_dt = np.dtype([('position', np.float),
                        ('width', np.float),
                        ('height', np.float)])
    P = np.array([], dtype=peak_dt)
    peak = 0
    for j in xrange(len(y) - 4):
        if np.sign(d[j]) > np.sign(d[j + 1]):  # Detects zero-crossing
            if np.sign(d[j + 1]) == 0:
                continue
            # if slope of derivative is larger than slope_thresh
            if d[j] - d[j + 1] > slope_thresh:
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
                                xx = xx[:groupindex - 1]
                                yy = yy[:groupindex - 1]
                                break
                            xx[k - s] = x[groupindex]
                            yy[k - s] = y[groupindex]
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
                            position = xx[np.argmin(np.abs(yy - height))]
                        else:
                            position = - ((stdev * c2 / (2 * c3)) - avg)
                            height = np.exp(c1 - c3 * (c2 / (2 * c3)) ** 2)
                    # Fill results array P. One row for each peak
                    # detected, containing the
                    # peak position (x-value) and peak height (y-value).
                    else:
                        position = x[j]
                        height = y[j]
                        # no way to know peak width without
                        # the above measurements.
                        width = 0
                    if (position > 0 and not np.isnan(position)
                            and position < x[-1]):
                        P = np.hstack((P,
                                       np.array([(position, height, width)],
                                                dtype=peak_dt)))
                        peak = peak + 1
    # return only the part of the array that contains peaks
    # (not the whole maxpeakn x 3 array)
    return P


def savitzky_golay(data, kernel=11, order=4):
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

    Examples
    --------
    smoothed = savitzky_golay(<rough>, [kernel = value], [order = value]

    """
    try:
        kernel = abs(int(kernel))
        order = abs(int(order))
    except ValueError:
        raise ValueError("kernel and order have to be of type int (floats will \
        be converted).")
    if kernel % 2 != 1 or kernel < 1:
        raise TypeError("kernel size must be a positive odd number, was: %d"
                        % kernel)
    if kernel < order + 2:
        raise TypeError(
            "kernel is to small for the polynomals\nshould be > order + 2")

    # a second order polynomal has 3 coefficients
    order_range = range(order + 1)
    N = (kernel - 1) // 2
    b = np.mat([[k ** i for i in order_range] for k in xrange(-N,
                                                              N + 1)])
    # since we don't want the derivative, else choose [1] or [2], respectively
    m = np.linalg.pinv(b).A[0]
    window_size = len(m)
    N = (window_size - 1) // 2

    # precompute the offset values for better performance
    offsets = range(-N, N + 1)
    offset_data = zip(offsets, m)

    smooth_data = list()

    # temporary data, with padded zeros
    # (since we want the same length after smoothing)
    # temporary data, extended with a mirror image to the left and right
    firstval = data[0]
    lastval = data[len(data) - 1]
    # left extension: f(x0-x) = f(x0)-(f(x)-f(x0)) = 2f(x0)-f(x)
    # right extension: f(xl+x) = f(xl)+(f(xl)-f(xl-x)) = 2f(xl)-f(xl-x)
    leftpad = np.zeros(N) + 2 * firstval
    rightpad = np.zeros(N) + 2 * lastval
    leftchunk = data[1:1 + N]
    leftpad = leftpad - leftchunk[::-1]
    rightchunk = data[len(data) - N - 1:len(data) - 1]
    rightpad = rightpad - rightchunk[::-1]
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
    x1 = np.zeros((M,), float)
    x2 = np.zeros((M,), float)

    # resub step 1
    for l in xrange(M):
        sum_ = rhs[l]
        for n in xrange(l):
            sum_ -= D[l, n] * x1[n]
        x1[l] = sum_ / D[l, l]

    # resub step 2
    for l in xrange(M - 1, -1, -1):
        sum_ = x1[l]
        for n in xrange(l + 1, M):
            sum_ -= D[n, l] * x2[n]
        x2[l] = sum_ / D[l, l]

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
    A = np.zeros((2 * num_points + 1, pol_degree + 1), float)
    for i in xrange(2 * num_points + 1):
        for j in xrange(pol_degree + 1):
            A[i, j] = math.pow(i - num_points, j)

    # calculate diff_order-th row of inv(A^T A)
    ATA = np.dot(A.transpose(), A)
    rhs = np.zeros((pol_degree + 1,), float)
    rhs[diff_order] = 1
    D = np.linalg.cholesky(ATA)
    wvec = resub(D, rhs)

    # calculate filter-coefficients
    coeff = np.zeros((2 * num_points + 1,), float)
    for n in xrange(-num_points, num_points + 1):
        x = 0.0
        for m in xrange(pol_degree + 1):
            x += wvec[m] * pow(n, m)
        coeff[n + num_points] = x
    return coeff * (-1) ** diff_order


def smooth(data, coeff):
    """applies coefficients calculated by calc_coeff() to signal
    http://www.procoders.net
    """
    # temporary data, extended with a mirror image to the left and right
    N = np.size(coeff - 1) // 2
    firstval = data[0]
    lastval = data[len(data) - 1]
#    left extension: f(x0-x) = f(x0)-(f(x)-f(x0)) = 2f(x0)-f(x)
#    right extension: f(xl+x) = f(xl)+(f(xl)-f(xl-x)) = 2f(xl)-f(xl-x)
    leftpad = np.zeros(N) + 2 * firstval
    rightpad = np.zeros(N) + 2 * lastval
    leftchunk = data[1:1 + N]
    leftpad = leftpad - leftchunk[::-1]
    rightchunk = data[len(data) - N - 1:len(data) - 1]
    rightpad = rightpad - rightchunk[::-1]
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


def interpolate1D(number_of_interpolation_points, data):
    ip = number_of_interpolation_points
    ch = len(data)
    old_ax = np.linspace(0, 100, ch)
    new_ax = np.linspace(0, 100, ch * ip - (ip - 1))
    interpolator = scipy.interpolate.interp1d(old_ax, data)
    return interpolator(new_ax)

# def lowess(x, y, f=2/3., iter=3):
#    """lowess(x, y, f=2./3., iter=3) -> yest
#
#    Lowess smoother: Robust locally weighted regression.
#    The lowess function fits a nonparametric regression curve to a scatterplot.
#    The arrays x and y contain an equal number of elements; each pair
#    (x[i], y[i]) defines a data point in the scatterplot. The function returns
#    the estimated (smooth) values of y.
#
#    The smoothing span is given by f. A larger value for f will result in a
#    smoother curve. The number of robustifying iterations is given by iter. The
#    function will run faster with a smaller number of iterations.
#
#    Code adapted from Biopython:
#
#    Original doc:
#
#    This module implements the Lowess function for nonparametric regression.
#
#    Functions:
#    lowess        Fit a smooth nonparametric regression curve to a scatterplot.
#
#    For more information, see
#
#    William S. Cleveland: "Robust locally weighted regression and smoothing
#    scatterplots", Journal of the American Statistical Association, December 1979,
#    volume 74, number 368, pp. 829-836.
#
#    William S. Cleveland and Susan J. Devlin: "Locally weighted regression: An
#    approach to regression analysis by local fitting", Journal of the American
#    Statistical Association, September 1988, volume 83, number 403, pp. 596-610.
#    """
#    n = len(x)
#    r = int(np.ceil(f*n))
#    h = [np.sort(abs(x-x[i]))[r] for i in xrange(n)]
#    w = np.clip(abs(([x]-np.transpose([x]))/h),0.0,1.0)
#    w = 1-w*w*w
#    w = w*w*w
#    yest = np.zeros(n,'d')
#    delta = np.ones(n,'d')
#    for iteration in xrange(iter):
#        for i in xrange(n):
#            weights = delta * w[:,i]
#            b = np.array([np.sum(weights*y), np.sum(weights*y*x)])
#            A = np.array([[np.sum(weights), np.sum(weights*x)],
#                     [np.sum(weights*x), np.sum(weights*x*x)]])
#            beta = np.linalg.solve(A,b)
#            yest[i] = beta[0] + beta[1]*x[i]
#        residuals = y-yest
#        s = np.median(abs(residuals))
#        delta = np.clip(residuals/(6*s),-1,1)
#        delta = 1-delta*delta
#        delta = delta*delta
#    return yest
#
# def wavelet_poissonian_denoising(spectrum):
#    """Denoise data with pure Poissonian noise using wavelets
#
#    Wrapper around the R packages EbayesThresh and wavethresh
#
#    Parameters
#    ----------
#    spectrum : spectrum instance
#
#    Returns
#    -------
#    Spectrum instance.
#    """
#    import_rpy()
#    rpy.r.library('EbayesThresh')
#    rpy.r.library('wavethresh')
#    rpy.r['<-']('X',spectrum)
#    rpy.r('XHF <- hft(X)')
#    rpy.r('XHFwd  <- wd(XHF, bc="symmetric")')
#    rpy.r('XHFwdT  <- ebayesthresh.wavelet(XHFwd)')
#    rpy.r('XHFdn  <- wr(XHFwdT)')
#    XHFest = rpy.r('XHFest <- hft.inv(XHFdn)')
#    return XHFest
#
# def wavelet_gaussian_denoising(spectrum):
#    """Denoise data with pure Gaussian noise using wavelets
#
#    Wrapper around the R packages EbayesThresh and wavethresh
#
#    Parameters
#    ----------
#    spectrum : spectrum instance
#
#    Returns
#    -------
#    Spectrum instance.
#    """
#    import_rpy()
#    rpy.r.library('EbayesThresh')
#    rpy.r.library('wavethresh')
#    rpy.r['<-']('X',spectrum)
#    rpy.r('Xwd  <- wd(X, bc="symmetric")')
#    rpy.r('XwdT  <- ebayesthresh.wavelet(Xwd)')
#    Xdn = rpy.r('Xdn  <- wr(XwdT)')
#    return Xdn
#
# def wavelet_dd_denoising(spectrum):
#    """Denoise data with arbitraty noise using wavelets
#
#    Wrapper around the R packages EbayesThresh, wavethresh and DDHFm
#
#    Parameters
#    ----------
#    spectrum : spectrum instance
#
#    Returns
#    -------
#    Spectrum instance.
#    """
#    import_rpy()
#    rpy.r.library('EbayesThresh')
#    rpy.r.library('wavethresh')
#    rpy.r.library('DDHFm')
#    rpy.r['<-']('X',spectrum)
#    rpy.r('XDDHF <- ddhft.np.2(X)')
#    rpy.r('XDDHFwd  <- wd(XDDHF$hft,filter.number = 8, bc="symmetric" )')
#    rpy.r('XDDHFwdT  <- ebayesthresh.wavelet(XDDHFwd)')
#    rpy.r('XDDHFdn  <- wr(XDDHFwdT)')
#    rpy.r('XDDHF$hft  <- wr(XDDHFwdT)')
#    XHFest = rpy.r('XHFest <- ddhft.np.inv(XDDHF)')
#    return XHFest

# def loess(y,x = None, span = 0.2):
#    """locally weighted scatterplot smoothing
#
#    Wrapper around the R funcion loess
#
#    Parameters
#    ----------
#    spectrum : spectrum instance
#    span : float
#        parameter to control the smoothing
#
#    Returns
#    -------
#    Spectrum instance.
#    """
#    import_rpy()
#    if x is None:
#        x = np.arange(0,len(y))
#    rpy.r['<-']('x',x)
#    rpy.r['<-']('y',y)
#    rpy.r('y.loess <- loess(y ~ x, span = %s, data.frame(x=x, y=y))' % span)
#    loess = rpy.r('y.predict <- predict(y.loess, data.frame(x=x))')
#    return loess
