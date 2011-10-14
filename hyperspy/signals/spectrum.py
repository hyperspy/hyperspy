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

from hyperspy.signal import Signal
from hyperspy.misc import progressbar
from hyperspy.misc import utils
from hyperspy.misc import utils_varia
from hyperspy.gui.tools import (SpectrumCalibration, SmoothingSavitzkyGolay,
    SmoothingLowess, )
from hyperspy.gui.egerton_quantification import BackgroundRemoval

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

class Spectrum(Signal):
    """
    """
    def __init__(self, *args, **kwargs):
        Signal.__init__(self, *args, **kwargs)
        self.axes_manager.set_view('hyperspectrum')

    def correct_bad_pixels(self, indexes, axis = -1):
        """Substitutes the value of a given pixel by the average of the
        adjencent pixels

        Parameters
        ----------
        indexes : tuple of int
        axis : -1
        """
        axis = self._get_positive_axis_index_index(axis)
        data = self.data
        for pixel in indexes:
            data[(slice(None),)*axis + (pixel, Ellipsis)] = \
            (data[(slice(None),)*axis + (pixel - 1, Ellipsis)] + \
            data[(slice(None),)*axis + (pixel + 1, Ellipsis)]) / 2.
        self._replot()


    def align_with_array_1D(self, shift_array, axis = -1,
                            interpolation_method = 'linear'):
        """Shift each one dimensional object by the amount specify by a given
        array

        Parameters
        ----------
        shift_map : numpy array
            The shift is specify in the units of the selected axis
        interpolation_method : str or int
            Specifies the kind of interpolation as a string ('linear',
            'nearest', 'zero', 'slinear', 'quadratic, 'cubic') or as an integer
            specifying the order of the spline interpolator to use.
        """

        axis = self._get_positive_axis_index_index(axis)
        coord = self.axes_manager.axes[axis]
        offset = coord.offset
        _axis = coord.axis.copy()
        maxval = np.cumprod(shift_array.shape)[-1] - 1
        pbar = progressbar.progressbar(maxval = maxval)
        i = 0
        for dat, shift in zip(self.iterate_axis(axis),
                              utils.iterate_axis(shift_array, axis)):
                si = sp.interpolate.interp1d(_axis ,dat,
                                             bounds_error = False,
                                             fill_value = 0.,
                                             kind = interpolation_method)
                coord.offset = offset - shift[0]
                dat[:] = si(coord.axis)
                pbar.update(i)
                i += 1
        coord.offset = offset

        # Cropping time
        mini, maxi = shift_array.min(), shift_array.max()
        if mini < 0:
            self.crop_in_units(axis, offset - mini)
        if maxi > 0:
            self.crop_in_units(axis, None, _axis[-1] - maxi)

    def interpolate_in_index_1D(self, axis, i1, i2, delta = 3, **kwargs):
        axis = self.axes_manager.axes[axis]
        i0 = int(np.clip(i1 - delta, 0, np.inf))
        i3 = int(np.clip(i2 + delta, 0, axis.size))
        for dat in self.iterate_axis(axis.index_in_array):
            dat_int = sp.interpolate.interp1d(range(i0,i1) + range(i2,i3),
                dat[i0:i1].tolist() + dat[i2:i3].tolist(),
                **kwargs)
            dat[i1:i2] = dat_int(range(i1,i2))

    def interpolate_in_units_1D(self, axis, u1, u2, delta = 3, **kwargs):
        axis = self.axes_manager.axes[axis]
        i1 = axis.value2index(u1)
        i2 = axis.value2index(u2)
        self.interpolate_in_index_1D(axis.index_in_array, i1, i2, delta,
                                     **kwargs)

    def estimate_shift_in_index_1D(self, irange = (None,None), axis = -1,
                                   reference_indexes = None, max_shift = None,
                                   interpolate = True,
                                   number_of_interpolation_points = 5):
        """Estimate the shifts in a given axis using cross-correlation

        This method can only estimate the shift by comparing unidimensional
        features that should not change the position in the given axis. To
        decrease the memory usage, the time of computation and the accuracy of
        the results it is convenient to select the feature of interest setting
        the irange keyword.

        By default interpolation is used to obtain subpixel precision.

        Parameters
        ----------
        axis : int
            The axis in which the analysis will be performed.
        irange : tuple of ints (i1, i2)
             Define the range of the feature of interest. If i1 or i2 are None,
             the range will not be limited in that side.
        reference_indexes : tuple of ints or None
            Defines the coordinates of the spectrum that will be used as a
            reference. If None the spectrum of 0 coordinates will be used.
        max_shift : int

        interpolate : bool

        number_of_interpolation_points : int
            Number of interpolation points. Warning: making this number too big
            can saturate the memory

        Return
        ------
        An array with the result of the estimation
        """

        ip = number_of_interpolation_points + 1
        axis = self.axes_manager.axes[axis]
        if reference_indexes is None:
            reference_indexes = [0,] * (len(self.axes_manager.axes) - 1)
        else:
            reference_indexes = list(reference_indexes)
        reference_indexes.insert(axis.index_in_array, slice(None))
        i1, i2 = irange
        array_shape = [axis.size for axis in self.axes_manager.axes]
        array_shape[axis.index_in_array] = 1
        shift_array = np.zeros(array_shape)
        ref = self.data[reference_indexes][i1:i2]
        if interpolate is True:
            ref = utils.interpolate_1D(ip, ref)
        for dat, shift in zip(self.iterate_axis(axis.index_in_array),
                              utils.iterate_axis(shift_array,
                                                 axis.index_in_array)):
            dat = dat[i1:i2]
            if interpolate is True:
                dat = utils.interpolate_1D(ip, dat)
            shift[:] = np.argmax(np.correlate(ref, dat,'full')) - len(ref) + 1

        if max_shift is not None:
            if interpolate is True:
                max_shift *= ip
            shift_array.clip(a_min = -max_shift, a_max = max_shift)
        if interpolate is True:
            shift_array /= ip
        shift_array *= axis.scale
        return shift_array

    def estimate_shift_in_units_1D(self, range_in_units = (None,None),
                                   axis = -1, reference_indexes = None,
                                   max_shift = None, interpolate = True,
                                   number_of_interpolation_points = 5):
        """Estimate the shifts in a given axis using cross-correlation. The
        values are given in the units of the selected axis.

        This method can only estimate the shift by comparing unidimensional
        features that should not change the position in the given axis. To
        decrease the memory usage, the time of computation and the accuracy of
        the results it is convenient to select the feature of interest setting
        the irange keyword.

        By default interpolation is used to obtain subpixel precision.

        Parameters
        ----------
        axis : int
            The axis in which the analysis will be performed.
        range_in_units : tuple of floats (f1, f2)
             Define the range of the feature of interest in the units of the
             selected axis. If f1 or f2 are None, thee range will not be limited
             in that side.
        reference_indexes : tuple of ints or None
            Defines the coordinates of the spectrum that will be used as a
            reference. If None the spectrum of 0 coordinates will be used.
        max_shift : float

        interpolate : bool

        number_of_interpolation_points : int
            Number of interpolation points. Warning: making this number too big
            can saturate the memory

        Return
        ------
        An array with the result of the estimation. The shift will be

        See also
        --------
        estimate_shift_in_index_1D, align_with_array_1D and align_1D
        """
        axis = self.axes_manager.axes[axis]
        i1 = axis.value2index(range_in_units[0])
        i2 = axis.value2index(range_in_units[1])
        if max_shift is not None:
            max_shift = int(round(max_shift / axis.scale))

        return self.estimate_shift_in_index_1D(axis = axis.index_in_array,
                                   irange = (i1, i2),
                                   reference_indexes = reference_indexes,
                                   max_shift = max_shift,
                                   number_of_interpolation_points =
                                   number_of_interpolation_points)

    def align_1D(self, range_in_units = (None,None), axis = -1,
                 reference_indexes = None, max_shift = None, interpolate = True,
                 number_of_interpolation_points = 5, also_align = None):
        """Estimates the shifts in a given axis using cross-correlation and uses
         the estimation to align the data over that axis.

        This method can only estimate the shift by comparing unidimensional
        features that should not change the position in the given axis. To
        decrease the memory usage, the time of computation and the accuracy of
        the results it is convenient to select the feature of interest setting
        the irange keyword.

        By default interpolation is used to obtain subpixel precision.

        It is possible to align several signals using the shift map estimated
        for this signal using the also_align keyword.

        Parameters
        ----------
        axis : int
            The axis in which the analysis will be performed.
        range_in_units : tuple of floats (f1, f2)
             Define the range of the feature of interest in the units of the
             selected axis. If f1 or f2 are None, thee range will not be limited
             in that side.
        reference_indexes : tuple of ints or None
            Defines the coordinates of the spectrum that will be used as a
            reference. If None the spectrum of 0 coordinates will be used.
        max_shift : float

        interpolate : bool

        number_of_interpolation_points : int
            Number of interpolation points. Warning: making this number too big
            can saturate the memory

        also_align : list of signals
            A list of Signal instances that has exactly the same dimensions
            as this one and that will be aligned using the shift map estimated
            using the this signal.

        Return
        ------
        An array with the result of the estimation. The shift will be

        See also
        --------
        estimate_shift_in_units_1D, estimate_shift_in_index_1D
        """

        shift_array = self.estimate_shift_in_units_1D(axis = axis,
                             range_in_units = range_in_units, 
                             reference_indexes = reference_indexes,
                             max_shift = max_shift, interpolate = interpolate,
                             number_of_interpolation_points = number_of_interpolation_points)
        if also_align is None:
            also_align = list()
        also_align.append(self)
        for signal in also_align:
            signal.align_with_array_1D(shift_array = shift_array, axis = axis)

    def peakfind_1D(self, xdim=None,slope_thresh=0.5, amp_thresh=None, subchannel=True,
                    medfilt_radius=5, maxpeakn=30000, peakgroup=10):
        """Find peaks along a 1D line (peaks in spectrum/spectra).

        Function to locate the positive peaks in a noisy x-y data set.

        Detects peaks by looking for downward zero-crossings in the first
        derivative that exceed 'slope_thresh'.

        Returns an array containing position, height, and width of each peak.

        'slope_thresh' and 'amp_thresh', control sensitivity: higher values will
        neglect smaller features.

        peakgroup is the number of points around the top peak to search around

        Parameters
        ---------

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
                   default is set to 5000

        subpix : bool (optional)
                 default is set to True

        Returns
        -------
        P : array of shape (npeaks, 3)
            contains position, height, and width of each peak
        """
        from peak_char import one_dim_findpeaks
        if len(self.data.shape)==1:
            # preallocate a large array for the results
            self.peaks=one_dim_findpeaks(self.data, slope_thresh=slope_thresh, amp_thresh=amp_thresh,
                                         medfilt_radius=medfilt_radius, maxpeakn=maxpeakn,
                                         peakgroup=peakgroup, subchannel=subchannel)
        elif len(self.data.shape)==2:
            pbar=progressbar.ProgressBar(maxval=self.data.shape[1]).start()
            # preallocate a large array for the results
            # the third dimension is for the number of rows in your data.
            # assumes spectra are rows of the 2D array, each col is a channel.
            self.peaks=np.zeros((maxpeakn,3,self.data.shape[0]))
            for i in xrange(self.data.shape[1]):
                tmp=one_dim_findpeaks(self.data[i], slope_thresh=slope_thresh,
                                                    amp_thresh=amp_thresh,
                                                    medfilt_radius=medfilt_radius,
                                                    maxpeakn=maxpeakn,
                                                    peakgroup=peakgroup,
                                                    subchannel=subchannel)
                self.peaks[:tmp.shape[0],:,i]=tmp
                pbar.update(i+1)
            # trim any extra blank space
            # works by summing along axes to compress to a 1D line, then finds
            # the first 0 along that line.
            trim_id=np.min(np.nonzero(np.sum(np.sum(self.peaks,axis=2),
                                             axis=1)==0))
            pbar.finish()
            self.peaks=self.peaks[:trim_id,:,:]
        elif len(self.data.shape)==3:
            # preallocate a large array for the results
            self.peaks=np.zeros((maxpeakn,3,self.data.shape[0],
                                 self.data.shape[1]))
            pbar=progressbar.ProgressBar(maxval=self.data.shape[0]).start()
            for i in xrange(self.data.shape[0]):
                for j in xrange(self.data.shape[1]):
                    tmp=one_dim_findpeaks(self.data[i,j], slope_thresh=slope_thresh, amp_thresh=amp_thresh,
                                         medfilt_radius=medfilt_radius, maxpeakn=maxpeakn,
                                         peakgroup=peakgroup, subchannel=subchannel)
                    self.peaks[:tmp.shape[0],:,i,j]=tmp
                pbar.update(i+1)
            # trim any extra blank space
            # works by summing along axes to compress to a 1D line, then finds
            # the first 0 along that line.
            trim_id=np.min(np.nonzero(np.sum(np.sum(np.sum(self.peaks,axis=3),axis=2),axis=1)==0))
            pbar.finish()
            self.peaks=self.peaks[:trim_id,:,:,:]

    def remove_spikes(self, threshold = 2200, subst_width = 5,
                      coordinates = None):
        """Remove the spikes in the SI.

        Detect the spikes above a given threshold and fix them by interpolating
        in the give interval. If coordinates is given, it will only remove the
        spikes for the specified spectra.

        Parameters:
        ------------
        threshold : float
            A suitable threshold can be determined with
            Spectrum.spikes_diagnosis
        subst_width : tuple of int or int
            radius of the interval around the spike to substitute with the
            interpolation. If a tuple, the dimension must be equal to the
            number of spikes in the threshold. If int the same value will be
            applied to all the spikes.

        See also
        --------
        Spectrum.spikes_diagnosis, Spectrum.plot_spikes
        """
        from scipy.interpolate import UnivariateSpline

        int_window = 20
        dc = self.data
        # differentiate the last axis
        der = np.diff(dc,1)
        E_ax = self.axes_manager.axes[-1].axis
        n_ch = E_ax.size
        index = 0
        if coordinates is None:
            if len(dc.shape)==3:
                for i in xrange(dc.shape[0]):
                    for j in xrange(dc.shape[1]):
                        if der[i,j,:].max() >= threshold:
                            print "Spike detected in (%s, %s)" % (i, j)
                            argmax = der[i,j,:].argmax()
                            if hasattr(subst_width, '__iter__'):
                                subst__width = subst_width[index]
                            else:
                                subst__width = subst_width
                            lp1 = np.clip(argmax - int_window, 0, n_ch)
                            lp2 = np.clip(argmax - subst__width, 0, n_ch)
                            rp2 = np.clip(argmax + int_window, 0, n_ch)
                            rp1 = np.clip(argmax + subst__width, 0, n_ch)
                            x = np.hstack((E_ax[lp1:lp2], E_ax[rp1:rp2]))
                            y = np.hstack((dc[i,j,lp1:lp2], dc[i,j,rp1:rp2]))
                            # The weights were commented because the can produce nans
                            # Maybe it should be an option?
                            intp =UnivariateSpline(x,y) #,w = 1/np.sqrt(y))
                            x_int = E_ax[lp2:rp1+1]
                            dc[i,j,lp2:rp1+1] = intp(x_int)
                            index += 1
            elif len(dc.shape)==2:
                for i in xrange(dc.shape[0]):
                        if der[i,:].max() >= threshold:
                            print "Spike detected in (%s)" % (i)
                            argmax = der[i,:].argmax()
                            if hasattr(subst_width, '__iter__'):
                                subst__width = subst_width[index]
                            else:
                                subst__width = subst_width
                            lp1 = np.clip(argmax - int_window, 0, n_ch)
                            lp2 = np.clip(argmax - subst__width, 0, n_ch)
                            rp2 = np.clip(argmax + int_window, 0, n_ch)
                            rp1 = np.clip(argmax + subst__width, 0, n_ch)
                            x = np.hstack((E_ax[lp1:lp2], E_ax[rp1:rp2]))
                            y = np.hstack((dc[i,lp1:lp2], dc[i,rp1:rp2]))
                            # The weights were commented because the can produce nans
                            # Maybe it should be an option?
                            intp =UnivariateSpline(x,y) #,w = 1/np.sqrt(y))
                            x_int = E_ax[lp2:rp1+1]
                            dc[i,lp2:rp1+1] = intp(x_int)
                            index += 1

        else:
            for spike_spectrum in coordinates:
                i, j = spike_spectrum
                print "Spike detected in (%s, %s)" % (i, j)
                argmax = der[:,i,j].argmax()
                if hasattr(subst_width, '__iter__'):
                    subst__width = subst_width[index]
                else:
                    subst__width = subst_width
                lp1 = np.clip(argmax - int_window, 0, n_ch)
                lp2 = np.clip(argmax - subst__width, 0, n_ch)
                rp2 = np.clip(argmax + int_window, 0, n_ch)
                rp1 = np.clip(argmax + subst__width, 0, n_ch)
                x = np.hstack((E_ax[lp1:lp2], E_ax[rp1:rp2]))
                y = np.hstack((dc[lp1:lp2,i,j], dc[rp1:rp2,i,j]))
                # The weights were commented because the can produce nans
                # Maybe it should be an option?
                intp =UnivariateSpline(x,y) # ,w = 1/np.sqrt(y))
                x_int = E_ax[lp2:rp1+1]
                dc[lp2:rp1+1,i,j] = intp(x_int)
                index += 1
        self.data=dc

    def to_image(self):
        from hyperspy.signals.image import Image
        dic = self._get_signal_dict()
        dic['mapped_parameters']['record_by'] = 'image'
        dic['data'] = np.rollaxis(dic['data'], -1, 0)
        dic['axes'] = utils_varia.rollelem(dic['axes'],-1,0)
        i = 0
        for axis in dic['axes']:
            axis['index_in_array'] = i
            i += 1
        return Image(dic)

    def to_EELS(self):
        from hyperspy.signals.eels import EELSSpectrum
        dic = self._get_signal_dict()
        dic['mapped_parameters']['signal'] = 'EELS'
        return EELSSpectrum(dic)

    def calibrate(self):
        '''Calibrate the spectral dimension using a gui

        It displays a window where the new calibration can be set by:
        * Setting the offset, units and scale directly
        * Selection a range by dragging the mouse on the spectrum figure and
        setting the new values for the given range limits

        Notes
        -----
        For this method to work the output_dimension must be 1. Set the view
        accordingly
        '''
        calibration = SpectrumCalibration(self)
        calibration.edit_traits()

    def smooth_savitzky_golay(self):
        '''Savitzky-Golay data smoothing using a gui'''
        sg = SmoothingSavitzkyGolay(self)
        sg.edit_traits()

    def smooth_lowess(self):
        '''Lowess data smoothing using a gui'''
        lw = SmoothingLowess(self)
        lw.edit_traits()

    def remove_background(self):
        '''Remove the background using a gui'''
        br = BackgroundRemoval(self)
        br.edit_traits()

    def plot_principal_components(self, n = None):
        """Plot the principal components up to the given number

        Parameters
        ----------
        n : int
            number of principal components to plot.
        """
        if n is None:
            n = self.mva_results.pc.shape[1]
        for i in xrange(n):
            plt.figure()
            plt.plot(self.axes_manager.axes[-1].axis, self.mva_results.pc[:,i])
            plt.xlabel('Energy (eV)')
            plt.title('Principal component %s' % i)


    def plot_independent_components(self, ic=None, same_window=False):
        """Plot the independent components.

        Parameters
        ----------
        ic : numpy array (optional)
             externally provided independent components array
             The shape of 'ic' must be (channels, n_components),
             so that e.g. ic[:, 0] is the first independent component.

        same_window : bool (optional)
                    if 'True', the components will be plotted in the
                    same window. Default is 'False'.
        """
        if ic is None:
            ic = self.mva_results.ic
            x = self.axes_manager.axes[-1].axis
            x = ic.shape[1]     # no way that we know the calibration

        n = ic.shape[1]

        if not same_window:
            for i in xrange(n):
                plt.figure()
                plt.plot(x, ic[:, i])
                plt.xlabel('Energy (eV)')
                plt.title('Independent component %s' % i)

        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for i in xrange(n):
                # ic = ic / ic.sum(axis=0) # normalize
                lbl = 'IC %i' % i
                # print 'plotting %s' % lbl
                ax.plot(x, ic[:, i], label=lbl)
            col = (ic.shape[1]) // 2
            ax.legend(ncol=col, loc='best')
            ax.set_xlabel('Energy (eV)')
            ax.set_title('Independent components')
            plt.draw()
            plt.show()

    def plot_maps(self, components, mva_type=None, scores=None, factors=None,
                  cmap=plt.cm.gray, no_nans=False, with_components=True,
                  plot=True, on_peaks=False, directory = None):
        """
        Plot component maps for the different MSA types

        Parameters
        ----------
        components : None, int, or list of ints
            if None, returns maps of all components.
            if int, returns maps of components with ids from 0 to given int.
            if list of ints, returns maps of components with ids in given list.
        mva_type: string, currently either 'pca' or 'ica'
        scores: numpy array, the array of score maps
        factors: numpy array, the array of components, with each column as a component.
        cmap: matplotlib colormap instance
        no_nans: bool,
        with_components: bool,
        plot: bool,
        """
        from hyperspy.signals.image import Image
        from hyperspy.signals.spectrum import Spectrum

        if scores is None or (factors is None and with_components is True):
            print "Either recmatrix or components were not provided."
            print "Loading existing values from object."
            if mva_type is None:
                print "Neither scores nor analysis type specified.  Cannot proceed."
                return

            elif mva_type.lower() == 'pca':
                scores=self.mva_results.v.T
                factors=self.mva_results.pc
            elif mva_type.lower() == 'ica':
                scores = self._get_ica_scores(self.mva_results)
                factors=self.mva_results.ic
                if no_nans:
                    print 'Removing NaNs for a visually prettier plot.'
                    scores = np.nan_to_num(scores) # remove ugly NaN pixels
            else:
                print "No scores provided and analysis type '%s' unrecognized. Cannot proceed."%mva_type
                return

#        if len(self.axes_manager.axes)==2:
#            shape=self.data.shape[0],1
#        else:
#            shape=self.data.shape[0],self.data.shape[1]
        im_list = []

        if components is None:
            components=xrange(factors.shape[1])

        elif type(components).__name__!='list':
            components=xrange(components)

        for i in components:
            if plot is True:
                figure = plt.figure()
                if with_components:
                    ax = figure.add_subplot(121)
                    ax2 = figure.add_subplot(122)
                else:
                    ax = figure.add_subplot(111)
            if self.axes_manager.navigation_dimension == 2:
                toplot = scores[i,:].reshape(self.axes_manager.navigation_shape)
                im_list.append(Image({'data' : toplot,
                    'axes' : self.axes_manager._get_non_slicing_axes_dicts()}))
                if plot is True:
                    mapa = ax.matshow(toplot, cmap = cmap)
                    if with_components:
                        ax2.plot(self.axes_manager.axes[-1].axis, factors[:,i])
                        ax2.set_title('%s component %i' % (mva_type.upper(),i))
                        ax2.set_xlabel('Energy (eV)')
                    figure.colorbar(mapa)
                    figure.canvas.draw()
                    #pointer = widgets.DraggableSquare(self.coordinates)
                    #pointer.add_axes(ax)
            elif self.axes_manager.navigation_dimension == 1:
                toplot = scores[i]
                im_list.append(Spectrum({"data" : toplot,
                    'axes' : self.axes_manager._get_non_slicing_axes_dicts()}))
                im_list[-1].get_dimensions_from_data()
                if plot is True:
                    ax.step(range(len(toplot)), toplot)

                    if with_components:
                        ax2.plot(self.axes_manager.axes[-1].axis, factors[:,i])
                        ax2.set_title('%s component %s' % (mva_type.upper(),i))
                        ax2.set_xlabel('Energy (eV)')
            else:
                messages.warning_exit('View not supported')
            if plot is True:
                ax.set_title('%s component number %s map' % (mva_type.upper(),i))
                figure.canvas.draw()
                if directory is not None:
                    if not os.path.isdir(directory):
                        os.makedirs(directory)
                    figure.savefig(os.path.join(directory, '%s-map-%i.png' % (mva_type.upper(),i)),
                              dpi = 600)
        return im_list

    def plot_principal_components_maps(self, comp_ids=None, cmap=plt.cm.gray,
                                       recmatrix=None, with_pc=True,
                                       plot=True, pc=None, on_peaks=False):
        """Plot the map associated to each independent component

        Parameters
        ----------
        comp_ids : None, int, or list of ints
            if None, returns maps of all components.
            if int, returns maps of components with ids from 0 to given int.
            if list of ints, returns maps of components with ids in given list.
        cmap : plt.cm object
        recmatrix : numpy array
            externally suplied recmatrix
        with_ic : bool
            If True, plots also the corresponding independent component in the
            same figure
        plot : bool
            If True it will plot the figures. Otherwise it will only return the
            images.
        ic : numpy array
            externally supplied independent components
        no_nans : bool (optional)
             whether substituting NaNs with zeros for a visually prettier plot
             (default is False)

        Returns
        -------
        List with the maps as MVA instances
        """
        return self.plot_maps(components=comp_ids,mva_type='pca',cmap=cmap,
                              scores=recmatrix, with_components=with_pc,
                              plot=plot, factors=pc, on_peaks=on_peaks)

    def plot_independent_components_maps(self, comp_ids=None, cmap=plt.cm.gray,
                                         recmatrix=None, with_ic=True,
                                         plot=True, ic=None, no_nans=False,
                                         on_peaks=False, directory = None):
        """Plot the map associated to each independent component

        Parameters
        ----------
        cmap : plt.cm object
        recmatrix : numpy array
            externally suplied recmatrix
        comp_ids : int or list of ints
            if None, returns maps of all components.
            if int, returns maps of components with ids from 0 to given int.
            if list of ints, returns maps of components with ids in given list.
        with_ic : bool
            If True, plots also the corresponding independent component in the
            same figure
        plot : bool
            If True it will plot the figures. Otherwise it will only return the
            images.
        ic : numpy array
            externally supplied independent components
        no_nans : bool (optional)
             whether substituting NaNs with zeros for a visually prettier plot
             (default is False)
        Returns
        -------
        List with the maps as MVA instances
        """
        return self.plot_maps(components=comp_ids,mva_type='ica',cmap=cmap,
                              scores=recmatrix, with_components=with_ic,
                              plot=plot, factors=ic, no_nans=no_nans,
                              on_peaks=on_peaks, directory = directory)


    def save_principal_components(self, n, spectrum_prefix = 'pc',
    image_prefix = 'im', spectrum_format = 'msa', image_format = 'tif',
                                  on_peaks=False):
        """Save the `n` first principal components  and score maps
        in the specified format

        Parameters
        ----------
        n : int
            Number of principal components to save_als_ica_results
        image_prefix : string
            Prefix for the image file names
        spectrum_prefix : string
            Prefix for the spectrum file names
        spectrum_format : string
        image_format : string

        """
        from spectrum import Spectrum
        im_list = self.plot_principal_components_maps(n, plot = False)
        axis_dict = self.axes_manager._slicing_axes[0].get_axis_dictionary()
        axis_dict['index_in_array'] = 0
        s = Spectrum({'data' : self.mva_results.pc[:,0],
                      'axes' : [axis_dict,]})
        for i in xrange(n):
            s.data_cube = self.mva_results.pc[:,i]
            s.save('%s-%i.%s' % (spectrum_prefix, i, spectrum_format))
            im_list[i].save('%s-%i.%s' % (image_prefix, i, image_format))

    def save_independent_components(self, elements=None,
                                    spectrum_format='msa',
                                    image_format='tif',
                                    recmatrix=None, ic=None,
                                    on_peaks=False):
        """Saves the result of the ICA in image and spectrum format.
        Note that to save the image, the NaNs in the map will be converted
        to zeros.

        Parameters
        ----------
        elements : None or tuple of strings
            a list of names (normally an element) to be assigned to IC. If not
            the will be name ic-0, ic-1 ...
        image_format : string
        spectrum_format : string
        recmatrix : None or numpy array
            externally supplied recmatrix
        ic : None or numpy array
            externally supplied IC
        """
        from hyperspy.signals.spectrum import Spectrum
        pl = self.plot_independent_components_maps(plot=False,
                                                   recmatrix=recmatrix,
                                                   ic=ic,
                                                   no_nans=True)
        if ic is None:
            ic = self.mva_results.ic
        if self.data.shape[2] > 1:
            maps = True
        else:
            maps = False
        for i in xrange(ic.shape[1]):
            axes = (self.axes_manager._slicing_axes[0].get_axis_dictionary(),)
            axes[0]['index_in_array'] = 0
            spectrum = Spectrum({'data' : ic[:,i], 'axes' : axes})
            spectrum.data_cube = ic[:,i].reshape((-1,1,1))

            if elements is None:
                spectrum.save('ic-%s.%s' % (i, spectrum_format))
                if maps is True:
                    pl[i].save('map_ic-%s.%s' % (i, image_format))
                else:
                    pl[i].save('profile_ic-%s.%s' % (i, spectrum_format))
            else:
                element = elements[i]
                spectrum.save('ic-%s.%s' % (element, spectrum_format))
                if maps:
                    pl[i].save('map_ic-%s.%s' % (element, image_format))
                else:
                    pl[i].save('profile_ic-%s.%s' % (element, spectrum_format))
    
