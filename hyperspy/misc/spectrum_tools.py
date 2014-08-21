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


def interpolate1D(number_of_interpolation_points, data):
    ip = number_of_interpolation_points
    ch = len(data)
    old_ax = np.linspace(0, 100, ch)
    new_ax = np.linspace(0, 100, ch * ip - (ip - 1))
    interpolator = scipy.interpolate.interp1d(old_ax, data)
    return interpolator(new_ax)
