import math

import numpy as np
import matplotlib.pyplot as plt

from hyperspy.misc.array_tools import rebin
from hyperspy.misc.utils import unfold_if_multidim
from hyperspy.gui import messages as messagesui
import hyperspy.defaults_parser

def _estimate_gain(ns, cs,
                   weighted=False,
                   higher_than=None, 
                   plot_results=False,
                   binning=0,
                   pol_order=1):
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
        s = Spectrum(variance2fit)
        s.axes_manager.signal_axes[0].axis = average2fit
        m = Model(s)
        l = Line()
        l.a.value = fit[1]
        l.b.value = fit[0]
        m.append(l)
        m.fit(weights=True)
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

def estimate_variance_parameters(
    noisy_signal,
    clean_signal,
    mask=None,
    pol_order=1,
    higher_than=None,
    return_results=False,
    plot_results=True,
    weighted=False):
    """Find the scale and offset of the Poissonian noise

    By comparing an SI with its denoised version (i.e. by PCA), 
    this plots an
    estimation of the variance as a function of the number of counts 
    and fits a
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
    Dictionary with the result of a linear fit to estimate the offset 
    and scale factor
    
    """
    fold_back_noisy =  unfold_if_multidim(noisy_signal)
    fold_back_clean =  unfold_if_multidim(clean_signal)
    ns = noisy_signal.data.copy()
    cs = clean_signal.data.copy()

    if mask is not None:
        _slice = [slice(None),] * len(ns.shape)
        _slice[noisy_signal.axes_manager.signal_axes[0].index_in_array]\
         = ~mask
        ns = ns[_slice]
        cs = cs[_slice]

    results0 = _estimate_gain(ns, cs, weighted=weighted, 
        higher_than=higher_than, plot_results=plot_results, binning=0,
        pol_order=pol_order)
        
    results2 = _estimate_gain(ns, cs, weighted=weighted, 
        higher_than=higher_than, plot_results=False, binning=2,
        pol_order=pol_order)
        
    c = _estimate_correlation_factor(results0['fit'][0],
                                     results2['fit'][0], 4)
    
    message = ("Gain factor: %.2f\n" % results0['fit'][0] +
               "Gain offset: %.2f\n" % results0['fit'][1] +
               "Correlation factor: %.2f\n" % c )
    is_ok = True
    if hyperspy.defaults_parser.preferences.General.interactive is True:
        is_ok = messagesui.information(
            message + "Would you like to store the results?")
    else:
        print message
    if is_ok:
        if not noisy_signal.mapped_parameters.has_item(
            'Variance_estimation'):
            noisy_signal.mapped_parameters.add_node(
                'Variance_estimation')
        noisy_signal.mapped_parameters.Variance_estimation.gain_factor = \
            results0['fit'][0]
        noisy_signal.mapped_parameters.Variance_estimation.gain_offset = \
            results0['fit'][1]
        noisy_signal.mapped_parameters.Variance_estimation.correlation_factor = c
        noisy_signal.mapped_parameters.Variance_estimation.\
        parameters_estimation_method = 'Hyperspy'

    if fold_back_noisy is True:
        noisy_signal.fold()
    if fold_back_clean is True:
        clean_signal.fold()
        
    if return_results is True:
        return results0
        
def power_law_perc_area(E1,E2, r):
    a = E1
    b = E2
    return 100*((a**r*r-a**r)*(a/(a**r*r-a**r)-(b+a)/((b+a)**r*r-(b+a)**r)))/a

def rel_std_of_fraction(a, std_a, b, std_b, corr_factor=1):
    rel_a = std_a/a
    rel_b = std_b/b
    return np.sqrt(rel_a**2 + rel_b**2 -
                   2 * rel_a * rel_b * corr_factor)

def ratio(edge_A, edge_B):
    a = edge_A.intensity.value
    std_a = edge_A.intensity.std
    b = edge_B.intensity.value
    std_b = edge_B.intensity.std
    ratio = a/b
    ratio_std = ratio * rel_std_of_fraction(a, std_a,b, std_b)
    print "Ratio %s/%s %1.3f +- %1.3f " % (
        edge_A.name,
        edge_B.name,
        a/b,
        1.96*ratio_std)
    return ratio, ratio_std

#def analyze_readout(spectrum):
#    """Readout diagnostic tool
#
#    Parameters
#    ----------
#    spectrum : Spectrum instance
#
#    Returns
#    -------
#    tuple of float : (variance, mean, normalized mean as a function of time)
#    """
#    s = spectrum
#    # If it is 2D, sum the first axis.
#    if s.data_cube.shape[2] > 1:
#        dc = s.data_cube.sum(1)
#    else:
#        dc = s.data_cube.squeeze()
#    time_mean = dc.mean(0).squeeze()
#    norm_time_mean = time_mean / time_mean.mean()
#    corrected_dc = dc * (1/norm_time_mean.reshape((1,-1)))
#    channel_mean = corrected_dc.mean(1)
#    variance = (corrected_dc - channel_mean.reshape((-1,1))).var(0)
#    return variance, channel_mean, norm_time_mean
#
#def multi_readout_analyze(folder, ccd_height = 100., plot = True, freq = None):
#    """Analyze several readout measurements in different files for readout
#    diagnosys
#
#    The readout files in dm3 format must be contained in a folder, preferentely
#    numered in the order of acquisition.
#
#    Parameters
#    ----------
#    folder : string
#        Folder where the dm3 readout files are stored
#    ccd_heigh : float
#    plot : bool
#    freq : float
#        Frequency of the camera
#
#    Returns
#    -------
#    Dictionary
#    """
#    from spectrum import Spectrum
#    files = glob.glob1(folder, '*.nc')
#    if not files:
#        files = glob.glob1(folder, '*.dm3')
#    spectra = []
#    variances = []
#    binnings = []
#    for f in files:
#        print os.path.join(folder,f)
#        s = Spectrum(os.path.join(folder,f))
#        variance, channel_mean, norm_time_mean = analyze_readout(s)
#        s.readout_analysis = {}
#        s.readout_analysis['variance'] = variance.mean()
#        s.readout_analysis['pattern'] = channel_mean
#        s.readout_analysis['time'] = norm_time_mean
#        if not hasattr(s,'binning'):
#            s.binning = float(os.path.splitext(f)[0][1:])
#            if freq:
#                s.readout_frequency = freq
#                s.ccd_height = ccd_height
#            s.save(f)
#        spectra.append(s)
#        binnings.append(s.binning)
#        variances.append(variance.mean())
#    pixels = ccd_height / np.array(binnings)
#    plt.scatter(pixels, variances, label = 'data')
#    fit = np.polyfit(pixels, variances,1, full = True)
#    if plot:
#        x = np.linspace(0,pixels.max(),100)
#        y = x*fit[0][0] + fit[0][1]
#        plt.plot(x,y, label = 'linear fit')
#        plt.xlabel('number of pixels')
#        plt.ylabel('variance')
#        plt.legend(loc = 'upper left')
#
#    print "Variance = %s * pixels + %s" % (fit[0][0], fit[0][1])
#    dictio = {'pixels': pixels, 'variances': variances, 'fit' : fit,
#    'spectra' : spectra}
#    return dictio

# 
# def chrono_align_and_sum(spectrum, energy_range = (None, None),
                         # spatial_shape = None):
    # """Alignment and sum of a chrono-spim SI
# 
    # Parameters
    # ----------
    # spectrum : Spectrum instance
        # Chrono-spim
    # energy_range : tuple of floats
        # energy interval in which to perform the alignment in energy units
    # axis : int
    # """
    # from spectrum import Spectrum
    # dc = spectrum.data_cube
    # min_energy_size = dc.shape[0]
# #    i = 0
    # new_dc = None
# 
    # # For the progress bar to work properly we must capture the output of the
    # # functions that are called during the alignment process
    # import cStringIO
    # import sys
    # capture_output = cStringIO.StringIO()
# 
    # from hyperspy.misc.progressbar import progressbar
    # pbar = progressbar(maxval = dc.shape[2] - 1)
    # for i in xrange(dc.shape[2]):
        # pbar.update(i)
        # sys.stdout = capture_output
        # s = Spectrum({'calibration': {'data_cube' : dc[:,:,i]}})
        # s.get_calibration_from(spectrum)
        # s.find_low_loss_origin()
        # s.align(energy_range, progress_bar = False)
        # min_energy_size = min(s.data_cube.shape[0], min_energy_size)
        # if new_dc is None:
            # new_dc = s.data_cube.sum(1)
        # else:
            # new_dc = np.concatenate([new_dc[:min_energy_size],
                                     # s.data_cube.sum(1)[:min_energy_size]], 1)
        # sys.stdout = sys.__stdout__
    # pbar.finish()
    # spectrum.data_cube = new_dc
    # spectrum.get_dimensions_from_cube()
    # spectrum.find_low_loss_origin()
    # spectrum.align(energy_range)
    # spectrum.find_low_loss_origin()
    # if spatial_shape is not None:
        # spectrum.data_cube = spectrum.data_cube.reshape(
        # [spectrum.data_cube.shape[0]] + list(spatial_shape))
        # spectrum.data_cube = spectrum.data_cube.swapaxes(1,2)
        # spectrum.get_dimensions_from_cube()