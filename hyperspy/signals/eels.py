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


import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import traits.api as t


from hyperspy.signals.spectrum import Spectrum
from hyperspy.signals.image import Image
from hyperspy.misc.eels.elements import elements
import hyperspy.axes
from hyperspy.gui.egerton_quantification import SpikesRemoval
from hyperspy.decorators import only_interactive
from hyperspy.gui.eels import TEMParametersUI
from hyperspy.defaults_parser import preferences
import hyperspy.gui.messages as messagesui
from hyperspy.misc.progressbar import progressbar
from hyperspy.components.power_law import PowerLaw
import hyperspy.misc.utils as utils


class EELSSpectrum(Spectrum):
    
    def __init__(self, *args, **kwards):
        Spectrum.__init__(self, *args, **kwards)
        # Attributes defaults
        self.subshells = set()
        self.elements = set()
        self.edges = list()
        if hasattr(self.mapped_parameters, 'Sample') and \
        hasattr(self.mapped_parameters.Sample, 'elements'):
            print('Elemental composition read from file')
            self.add_elements(self.mapped_parameters.Sample.elements)

    def add_elements(self, elements, include_pre_edges=False):
        """Declare the elemental composition of the sample.
        
        The ionisation edges of the elements present in the current 
        energy range will be added automatically.
        
        Parameters
        ----------
        elements : tuple of strings
            The symbol of the elements.
        include_pre_edges : bool
            If True, the ionization edges with an onset below the lower 
            energy limit of the SI will be incluided
            
        Examples
        --------
        
        >>> s = signals.EELSSpectrum({'data' : np.arange(1024)})
        >>> s.add_elements(('C', 'O'))
        Adding C_K subshell
        Adding O_K subshell
        
        """
        for element in elements:
            if element in elements:
                self.elements.add(element)
            else:
                print(
                    "%s is not a valid symbol of an element" % element)
        if not hasattr(self.mapped_parameters, 'Sample'):
            self.mapped_parameters.add_node('Sample')
        self.mapped_parameters.Sample.elements = list(self.elements)
        if self.elements:
            self.generate_subshells(include_pre_edges)
        
    def generate_subshells(self, include_pre_edges=False):
        """Calculate the subshells for the current energy range for the 
        elements present in self.elements
         
        Parameters
        ----------
        include_pre_edges : bool
            If True, the ionization edges with an onset below the lower 
            energy limit of the SI will be incluided
            
        """
        Eaxis = self.axes_manager.signal_axes[0].axis
        if not include_pre_edges:
            start_energy = Eaxis[0]
        else:
            start_energy = 0.
        end_energy = Eaxis[-1]
        for element in self.elements:
            e_shells = list()
            for shell in elements[element]['subshells']:
                if shell[-1] != 'a':
                    if start_energy <= \
                    elements[element]['subshells'][shell][
                        'onset_energy'] \
                    <= end_energy :
                        subshell = '%s_%s' % (element, shell)
                        if subshell not in self.subshells:
                            print "Adding %s subshell" % (subshell)
                            self.subshells.add(
                                '%s_%s' % (element, shell))
                            e_shells.append(subshell)
                    
    def estimate_zero_loss_peak_centre(self, also_apply_to=None):
        """Calculates the position of the zero loss origin as the 
        average of the position of the maximum of all the spectra and 
         calibrates energy axis.
        
        Parameters
        ----------
        also_apply_to : None or list of EELSSPectrum
            If a list of signals is provided, the same offset
            transformation is applied to all the other signals.
            
        """
        axis = self.axes_manager.signal_axes[0] 
        old_offset = axis.offset
        imax = np.mean(np.argmax(self.data,axis.index_in_array))
        axis.offset = hyperspy.axes.generate_axis(0, axis.scale, 
            axis.size, imax)[0]
        print('Energy offset applied: %f %s' % ((
                axis.offset - old_offset), axis.units))
        if also_apply_to:
            for sync_signal in also_apply_to:
                saxis = sync_signal.axes_manager.axes[
                    axis.index_in_array]
                saxis.offset += axis.offset - old_offset
    
    def estimate_elastic_scattering_intensity(self, threshold=None):
        """Rough estimation of the elastic scattering signal by 
        truncation of a EELS low-loss spectrum.
        
        Parameters
        ----------
        threshold : {None, array}
            Truncation energy to estimate the intensity of the 
            elastic scattering. If None the threshold is calculated for 
            each spectrum as the first minimum after the ZLP centre. The
            threshold can be provided as a signal of the same dimension 
            as the input spectrum navigation space containing the 
            estimated threshold. If the navigation size is 0 a float 
            number must be provided.
            
        Returns
        -------
        The elastic scattering intensity. If the navigation size is 0 
        returns a float. Otherwise it returns a Spectrum, Image or a 
        Signal, depending on the currenct spectrum navigation 
        dimensions.
            
        """
        I0 = self._get_navigation_signal()
        axis = self.axes_manager.signal_axes[0]
        # Use the data from the current location to estimate
        # the threshold as the position of the first maximum
        # after the ZLP
        maxval = self.axes_manager.navigation_size
        pbar = hyperspy.misc.progressbar.progressbar(maxval=maxval)
        for i, s in enumerate(self):
            if threshold is None:
                # No threshold has been specified, we calculate it
                data = s()
                index = data.argmax()
                while data[index] > data[index + 1]:
                    index += 1
                del data
            elif hasattr(threshold,'data') is False: 
                # No data attribute
                index = axis.value2index(threshold)
            else:
                # Threshold specified, by an image instance 
                cthreshold = threshold.data[self.axes_manager.coordinates]
                index = axis.value2index(cthreshold)
            
            if I0 is None:
                # Case1: no navigation signal 
                I0 = s.data[0:index].sum()
                pbar.finish()
                return I0
            else:
                # Case2: navigation signal present
                I0.data[self.axes_manager.coordinates] = \
                    s.data[0:index].sum()
                
            pbar.update(i)
            
        pbar.finish()
                
        I0.mapped_parameters.title = (
            self.mapped_parameters.title + ' elastic intensity')
        if self.tmp_parameters.has_item('filename'):
            I0.tmp_parameters.filename = (
                self.tmp_parameters.filename +
                '_elastic_intensity')
            I0.tmp_parameters.folder = self.tmp_parameters.folder
            I0.tmp_parameters.extension = \
                self.tmp_parameters.extension
        return I0
    
    def estimate_elastic_scattering_threshold(self, window=20, npoints=5,
                                              tol = 0.1):
        """Estimation of the elastic scattering signal by truncation of 
        a EELS low-loss spectrum. Calculates the inflexion of the ZLP 
        derivative within a window using a specified tolerance. It 
        previously smoothes the data using a Savitzky-Golay algorithm 
        (can be turned off). 
        
        Notice that if a window parameter lower than the threshold is 
        set, this routine will be unable to find the right point. In
        these cases, the window parameter will be returned as threshold 
        #
        Parameters
        ----------
           
        window : {None, float}
            If None, the search for the local minimum is performed 
            using the full energy range. A positive float will restrict
            the search to the (0,window] energy window.
        npoints : int
            If not zero performs order three Savitzky-Golay smoothing 
            to the data to avoid falling in local minima caused by 
            the noise.
        tol : float
            The threshold tolerance for the derivative. If not provided 
            it is set to 0.1
            
        Returns
        -------
        threshold : {Signal instance, float}
            A Signal of the same dimension as the input spectrum 
            navigation space containing the estimated threshold. In the 
            case of a single spectrum a float is returned.
            
        """
        # Create threshold with the same shape as the navigation dims.
        threshold = self._get_navigation_signal()
        # Progress Bar
        maxval = self.axes_manager.navigation_size
        pbar = hyperspy.misc.progressbar.progressbar(maxval=maxval)
        axis = self.axes_manager.signal_axes[0]
        max_index = axis.value2index(window)
        for i, s in enumerate(self):
            zlpi = s.data.argmax() + 1
            if max_index -zlpi < 10:
                max_index += (10 + zlpi - max_index)
            data = s()[zlpi:max_index].copy()
            if npoints:
                data = np.abs(utils.sg(data, npoints, 1, diff_order=1))
            imin = (data < tol).argmax() + zlpi
            cthreshold = axis.index2value(imin)
            if (cthreshold == 0): cthreshold = window 
            del data 
            # If single spectrum, stop and return value
            if threshold is None:
                threshold = float(cthreshold)
                pbar.finish()
                return threshold
            else:
                threshold.data[self.axes_manager.coordinates] = \
                    cthreshold
                pbar.update(i)
                
        # Create spectrum image, stop and return value
        threshold.mapped_parameters.title = (
            self.mapped_parameters.title + 
            ' ZLP threshold')
        if self.tmp_parameters.has_item('filename'):
            threshold.tmp_parameters.filename = (
                self.tmp_parameters.filename +
                '_ZLP_threshold')
            threshold.tmp_parameters.folder = self.tmp_parameters.folder
            threshold.tmp_parameters.extension = \
                self.tmp_parameters.extension
        pbar.finish()
        return threshold
        
    def estimate_thickness(self, threshold=None, zlp=None,):
        """Estimates the thickness (relative to the mean free path) 
        of a sample using the log-ratio method.
        
        The current EELS spectrum must be a low-loss spectrum containing
        the zero-loss peak. The hyperspectrum must be well calibrated 
        and aligned. 
        
        Parameters
        ----------
        zlp : {None, EELSSpectrum}
            If None estimates the zero-loss peak by integrating the
            intensity of the spectrum up to the value defined in 
            threshold (i.e. by truncation). Otherwise the zero-loss
            peak intensity is calculated from the ZLP spectrum
            supplied.
        threshold : {None, float, Signal instance}
            Truncation energy to estimate the intensity of the 
            elastic scattering. Notice that in the case of a Signal the
            instance has to be of the same dimension as the input 
            spectrum navigation space containing the estimated 
            threshold. In the case of a single spectrum a float is good
            enough.If None the threshold is taken as the first minimum 
            after the ZLP centre.
            
        Returns
        -------
        The thickness relative to the MFP. If it is a single spectrum 
        returns a float. Otherwise it returns a Spectrum, Image or a 
        Signal, depending on the currenct spectrum navigation 
        dimensions.
            
        Notes
        -----        
        For details see: Egerton, R. Electron Energy-Loss 
        Spectroscopy in the Electron Microscope. Springer-Verlag, 2011.
        
        """       
        
        dc = self.data
        axis = self.axes_manager.signal_axes[0]
        total_intensity = dc.sum(axis.index_in_array)
        if zlp is not None:
            I0 = zlp.data.sum(axis.index_in_array)            
        else:
            I0 = self.estimate_elastic_scattering_intensity(
                                                threshold=threshold)
            if self.axes_manager.navigation_size > 0:
                I0 = I0.data
        t_over_lambda = np.log(total_intensity / I0)
        s = self._get_navigation_signal()
        if s is None:
            return t_over_lambda
        else:
            s.data = t_over_lambda
            s.mapped_parameters.title = (self.mapped_parameters.title + 
                ' $\\frac{t}{\\lambda}$')
            if self.tmp_parameters.has_item('filename'):
                s.tmp_parameters.filename = (
                    self.tmp_parameters.filename +
                    '_relative_thickness')
                s.tmp_parameters.folder = self.tmp_parameters.folder
                s.tmp_parameters.extension = \
                    self.tmp_parameters.extension
            return s
                
                
    def estimate_FWHM(self, factor=0.5, energy_range=(-10.,10.),
                      der_roots=False):
        """Use a third order spline interpolation to estimate the FWHM 
        of a peak at the current position.
        
        Parameters:
        -----------
        factor : float < 1
            By default is 0.5 to give FWHM. Choose any other float to 
            give find the position of a different fraction of the peak.
        energy_range : tuple of floats
            energy interval containing the peak.
        der_roots: bool
            If True, compute the roots of the first derivative
            (2 times slower).  
        
        Returns:
        --------
        dictionary. Keys:
            'FWHM' : float
                 width, at half maximum or other fraction as choosen by
            `factor`. 
            'FWHM_E' : tuple of floats
                Coordinates in energy units of the FWHM points.
            'der_roots' : tuple
                Position in energy units of the roots of the first
                derivative if der_roots is True (False by default)
                
        """
        axis = self.axes_manager.signal_axes[0]
        i0, i1 = (axis.value2index(energy_range[0]), 
                  axis.value2index(energy_range[1]))
        data = self()[i0:i1]
        x = axis.axis[i0:i1]
        height = np.max(data)
        spline_fwhm = scipy.interpolate.UnivariateSpline(
                            x, data - factor * height)
        pair_fwhm = spline_fwhm.roots()[0:2]
        fwhm = pair_fwhm[1] - pair_fwhm[0]
        
        if der_roots:
            der_x = np.arange(x[0], x[-1] + 1, (x[1] - x[0]) * 0.2)
            derivative = spline_fwhm(der_x, 1)
            spline_der = scipy.interpolate.UnivariateSpline(der_x,
                derivative)
            return {'FWHM' : fwhm,
                     'pair' : pair_fwhm, 
                     'der_roots': spline_der.roots()}
        else:
            return {'FWHM' : fwhm,
                     'FWHM_E' : pair_fwhm}
                     
    def splice_zero_loss_peak(self, threshold=None, mode=None):
        """ Tool to crop the zero loss peak from the rest of the 
        low-loss EELSSpectrum provided, using the inflexion points 
        calculated from estimate_elastic_scattering_threshold, or any 
        other threshold specified.   
        
        Parameters
        ----------
        threshold : {None, float, Signal instance}
            Truncation energy to estimate the intensity of the 
            elastic scattering. Notice that in the case of a Signal the
            instance has to be of the same dimension as the input 
            spectrum navigation space containing the estimated 
            threshold. In the case of a single spectrum a float is good
            enough.If None the threshold is taken as the first minimum 
            after the ZLP centre.
        mode : {None, 'smooth'}
            Method for smoothing the right-hand-side of the cropped ZLP
	   
            None: No method is applied.
            smooth: Hanning window is applied.
        
        Returns
        -------
        zlp : EELSSpectrum
            This instance should contain the spliced zero loss peak 
            EELSSpectrum.
        """       
        zlp=self.deepcopy()
        axes=zlp.axes_manager
        Eaxis=axes.signal_axes[0]
        # Progress Bar
        maxval = axes.navigation_size
        pbar = hyperspy.misc.progressbar.progressbar(maxval=maxval)
        for i, s in enumerate(zlp):
            if threshold is None:
                # No threshold has been specified, we calculate it
                data = s()
                ith = data.argmax()
                while data[ith] > data[ith + 1]:
                    ith += 1
                del data
            elif hasattr(threshold,'data') is False: 
                # No data attribute
                ith = Eaxis.value2index(threshold)
            else:
                # Threshold specified, by an image instance 
                ith = Eaxis.value2index(threshold.data[axes.coordinates])

            # This does the threshold-selective cropping
            s.data[ith:] = 0
            if mode is 'smooth':
                tmp_s = s[:ith].copy()
                tmp_s.hanning_taper(side='right')
                s.data[:ith] = tmp_s.data
            pbar.update(i)
        pbar.finish()
        
        return zlp                 
    
    def splice_zero_loss_peak_flog(self, threshold=None, window_s=100, background=0.0):
        """ Alternative Tool to crop the zero loss peak (ZLP) from the 
        rest of the low-loss EELSSpectrum provided, using the inflexion 
        points calculated from estimate_elastic_scattering_threshold, or
        any other threshold specified.   
        
        Special preparation for fourier log-deconvolution in which the 
        input EELSSpectrum is also be modified, according to Egerton (2011).
        Both EELSSpectra (input and output ZLP) are expanded to the next 
        power of two, and the left side of the ZLP moved to the right 
        end. The discontinuities are smoothed using Hann window, 
        preserving the value of the integral below the spectra. Input 
        EELSSpectrum will be extrapolated to the right using 
        power_law_extrapolation.
        
        This method returns 2 EELSSpectrum instances, see below.

        Parameters
        ----------
        threshold : {None, float, Signal instance}
            Truncation energy to estimate the intensity of the 
            elastic scattering. Notice that in the case of a Signal the
            instance has to be of the same dimension as the input 
            spectrum navigation space containing the estimated 
            threshold. In the case of a single spectrum a float is good
            enough.If None the threshold is taken as the first minimum 
            after the ZLP centre.
        window_s : int, float
            Number of channels from the right end of the spectrum that are 
            used for the power_law_extrapolation. If specified by a float 
            value, the program will calculate the number of channels that 
            correspond to that value using the signal axis scale.
        background: float
            a background general level to be subtracted from the spectrum.
            
        Returns
        -------
        j : EELSSpectrum
            This instance should contain the extended input EELSSpectrum,
            prepared for fourier_log_deconvolution
        zlp : EELSSpectrum
            This instance should contain the extended zero loss peak 
            EELSSpectrum, prepared for fourier_log_deconvolution
        
        Mode of use
        -----------
        Use: j, zlp = splice_zero_loss_peak_flog(self, thr, ...)

        Notes
        -----        
        For details see: Egerton, R. Electron Energy-Loss 
        Spectroscopy in the Electron Microscope. Springer-Verlag, 2011.
        """ 
        zlp=self.deepcopy()
        axes=zlp.axes_manager
        Eaxis=axes.signal_axes[0]
        # Progress Bar
        maxval = axes.navigation_size
        pbar = hyperspy.misc.progressbar.progressbar(maxval=maxval)
        # Calculate the next Power of 2
        old = Eaxis.size
        npot = int(2 ** np.ceil(np.log2(old)))
        new = npot + (npot-old)
        # Extend the spectrum
        new_shape = list(self.data.shape)
        new_shape[Eaxis.index_in_array] += new
        zlp.data =  np.zeros((new_shape))
        zlp.get_dimensions_from_data()
        
        zlp_index=Eaxis.value2index(0)
        # Slice the zlp left-side to the right!
        slicer = lambda x: zlp.axes_manager._get_data_slice([(Eaxis.index_in_array, slice(x[0],x[1])),])
        zlp.data[slicer((None,old-zlp_index))] = self.data[slicer((zlp_index,old))]
        zlp.data[slicer((-zlp_index,None))] = self.data[slicer((None,zlp_index))]
        
        for i, s in enumerate(zlp):
            if threshold is None:
                # No threshold has been specified, we calculate it
                data = s()
                ith = data.argmax()
                while data[ith] > data[ith + 1]:
                    ith += 1
                del data
            elif hasattr(threshold,'data') is False: 
                # No data attribute
                ith = Eaxis.value2index(threshold)
            else:
                # Threshold specified, by an image instance 
                ith = Eaxis.value2index(threshold.data[axes.coordinates])

            # This does the threshold-selective cropping
            s.data[ith:old] = 0
            # Switch sides, using Hann window
            s.data[:ith] -= s.data[ith-1] * np.hanning((ith)*4)[:ith] 
            s.data[ith:2*ith] = s.data[ith-1] * np.hanning((ith)*4)[-ith:] 
            pbar.update(i)
        
        # Zero loss peak is prepared for fourier log deconvolution
        zlp.data = zlp.data.astype('float32')
        # Now, we prepare the input spectrum
        s = self.deepcopy()
        Eaxis = s.axes_manager.signal_axes[0]
        # Signal data
        self_size = Eaxis.size
        zlp_index = Eaxis.value2index(0)
        zlp_size = zlp.axes_manager.signal_axes[0].size
        # Move E=0 to first channel
        s.crop_in_units(Eaxis.index_in_array, 0, Eaxis.high_value)
        s.data = s.data - background
        s_size = Eaxis.size
        # Compute next "Power of 2" size
        size = int(2 ** np.ceil(np.log2(2*self_size-1)))
        size_new = size - s_size
        # Extrapolate s using a power law
        if type(window_s) is float:
            window_s = Eaxis.value2index(window_s)
        _s=s.power_law_extrapolation(window_size=window_s,
                                    extrapolation_size=size_new,
                                    fix_neg_r=True)
        # Subtract Hann window to the power law 
        slicer = lambda x: _s.axes_manager._get_data_slice([(Eaxis.index_in_array, slice(x[0],x[1])),])
        new_shape = list(self.data.shape)
        new_shape[Eaxis.index_in_array] = size_new
        cbell = 0.5*(1-np.cos(np.pi*np.arange(0,size_new)/(size_new-zlp_index-1)))
        dext = _s.data[slicer((size-1,size))]
        _s.data[slicer((s_size,None))] -= np.ones(new_shape)*cbell*dext
        # Finally, paste left zlp to the right side
        _s.data[slicer((-zlp_index,None))] = zlp.data[slicer((-zlp_index,None))]
        # Sign the work and leave...
        _s.data = _s.data.astype('float32')
        _s.mapped_parameters.title = (_s.mapped_parameters.title + 
                                         ' prepared for Fourier-log deconvolution')
        zlp.mapped_parameters.title = (zlp.mapped_parameters.title + 
                                         ' prepared for Fourier-log deconvolution')
        pbar.finish()
        return _s, zlp

    def fourier_log_deconvolution(self, zlp, prepared=False, 
                                    add_zlp=False, crop=False):
        """Performs fourier-log deconvolution.
        
        Parameters
        ----------
        zlp : EELSSpectrum
            The corresponding zero-loss peak.
        prepared : bool
            Use True if the input EELSSpectra have been preparated with
            split_zero_loss_peak_flog.
        add_zlp : bool
            If True, adds the ZLP to the deconvolved spectrum
        crop : bool
            If True crop the spectrum to leave out the channels that
            have been modified to decay smoothly to zero at the sides 
            of the spectrum.
        
        Returns
        -------
        An EELSSpectrum containing the current data deconvolved.
        
        Notes
        -----        
        For details see: Egerton, R. Electron Energy-Loss 
        Spectroscopy in the Electron Microscope. Springer-Verlag, 2011.
        
        """
        s = self.deepcopy()
        tapped_channels = 0
        
        zlp_size = zlp.axes_manager.signal_axes[0].size 
        self_size = self.axes_manager.signal_axes[0].size
        if prepared is False:
            tapped_channels = s.hanning_taper()
            # Conservative new size to solve the wrap-around problem 
            size = zlp_size + self_size -1
            # Increase to the closest multiple of two to enhance the FFT 
            # performance
            size = int(2 ** np.ceil(np.log2(size)))
        elif self_size == zlp_size:
            tapped_channels = 0
            size = self_size
        axis = self.axes_manager.signal_axes[0]
        z = np.fft.rfft(zlp.data, n=size, axis=axis.index_in_array).astype('complex64')
        j = np.fft.rfft(s.data, n=size, axis=axis.index_in_array).astype('complex64')
        j1 = z * np.nan_to_num(np.log(j / z))
        sdata = np.fft.irfft(j1, axis=axis.index_in_array).astype('float32')
        s.data = sdata[s.axes_manager._get_data_slice(
            [(axis.index_in_array, slice(None,self_size)),])]
        if add_zlp is True:
            if self_size >= zlp_size:
                s.data[s.axes_manager._get_data_slice(
                    [(axis.index_in_array, slice(None,zlp_size)),])
                    ] += zlp.data
            else:
                s.data += zlp.data[s.axes_manager._get_data_slice(
                    [(axis.index_in_array, slice(None,self_size)),])]
                    
        s.mapped_parameters.title = (s.mapped_parameters.title + 
                                     ' after Fourier-log deconvolution')
        if s.tmp_parameters.has_item('filename'):
                s.tmp_parameters.filename = (
                    self.tmp_parameters.filename +
                    '_after_fourier_log_deconvolution')
        if crop is True:
            s.crop_in_pixels(axis.index_in_array,
                             None, -tapped_channels)
        return s

    def fourier_ratio_deconvolution(self, ll, fwhm=None,
                                    threshold=None,
                                    extrapolate_lowloss=True,
                                    extrapolate_coreloss=True):
        """Performs Fourier-ratio deconvolution.
        
        The core-loss should have the background removed. To reduce
         the noise amplication the result is convolved with a
        Gaussian function.        
        
        Parameters
        ----------
        ll: EELSSpectrum
            The corresponding low-loss (ll) EELSSpectrum.
            
        fwhm : float or None
            Full-width half-maximum of the Gaussian function by which 
            the result of the deconvolution is convolved. It can be 
            used to select the final SNR and spectral resolution. If 
            None, the FWHM of the zero-loss peak of the low-loss is
            estimated and used.
        threshold : {None, float}
            Truncation energy to estimate the intensity of the 
            elastic scattering. If None the threshold is taken as the
             first minimum after the ZLP centre.
        extrapolate_lowloss, extrapolate_coreloss : bool
            If True the signals are extrapolated using a power law,
            
        Notes
        -----        
        For details see: Egerton, R. Electron Energy-Loss 
        Spectroscopy in the Electron Microscope. Springer-Verlag, 2011.
        
        """
        orig_cl_size = self.axes_manager.signal_axes[0].size
        if extrapolate_coreloss is True:
            cl = self.power_law_extrapolation(
                window_size=20,
                extrapolation_size=100)
        else:
            cl = self.deepcopy()
            
        if extrapolate_lowloss is True:
            ll = ll.power_law_extrapolation(
                window_size=100,
                extrapolation_size=100)
        else:
            ll = ll.deepcopy()
        
        ll.hanning_taper()
        cl.hanning_taper()

        ll_size = ll.axes_manager.signal_axes[0].size 
        cl_size = self.axes_manager.signal_axes[0].size
        # Conservative new size to solve the wrap-around problem 
        size = ll_size + cl_size -1
        # Increase to the closest multiple of two to enhance the FFT 
        # performance
        size = int(2 ** np.ceil(np.log2(size)))
        
        axis = ll.axes_manager.signal_axes[0]
        if fwhm is None:
            fwhm = ll.estimate_FWHM()['FWHM']
            print("FWHM = %1.2f" % fwhm) 

        I0 = ll.estimate_elastic_scattering_intensity(
                                                threshold=threshold)
        if ll.axes_manager.navigation_size > 0:
            I0 = I0.data
            I0_shape = list(I0.shape)
            I0_shape.insert(axis.index_in_array,1)
            I0 = I0.reshape(I0_shape)
            
        from hyperspy.components.gaussian import Gaussian
        g = Gaussian()
        g.sigma.value = fwhm / 2.3548
        g.A.value = 1
        g.centre.value = 0
        zl = g.function(
                np.linspace(axis.offset,
                            axis.offset + axis.scale * (size - 1),
                            size))
        z = np.fft.rfft(zl)
        jk = np.fft.rfft(cl.data, n=size,axis=axis.index_in_array)
        jl = np.fft.rfft(ll.data, n=size, axis=axis.index_in_array)
        zshape = [1,] * len(cl.data.shape)
        zshape[axis.index_in_array] = jk.shape[axis.index_in_array]
        cl.data = np.fft.irfft(z.reshape(zshape) * jk / jl,
                             axis=axis.index_in_array)
        cl.data *= I0
        cl.crop_in_pixels(-1,None,orig_cl_size)
        cl.mapped_parameters.title = (self.mapped_parameters.title + 
            ' after Fourier-ratio deconvolution')
        if cl.tmp_parameters.has_item('filename'):
                cl.tmp_parameters.filename = (
                    self.tmp_parameters.filename +
                    'after_fourier_ratio_deconvolution')
        return cl
            
    def richardson_lucy_deconvolution(self,  psf, iterations=15, 
                                      mask=None):
        """1D Richardson-Lucy Poissonian deconvolution of 
        the spectrum by the given kernel.
    
        Parameters:
        -----------
        iterations: int
            Number of iterations of the deconvolution. Note that 
            increasing the value will increase the noise amplification.
        psf: EELSSpectrum
            It must have the same signal dimension as the current 
            spectrum and a spatial dimension of 0 or the same as the 
            current spectrum.
            
        Notes:
        -----
        For details on the algorithm see Gloter, A., A. Douiri, 
        M. Tence, and C. Colliex. “Improving Energy Resolution of 
        EELS Spectra: An Alternative to the Monochromator Solution.” 
        Ultramicroscopy 96, no. 3–4 (September 2003): 385–400.
        
        """

        ds = self.deepcopy()
        ds.data = ds.data.copy()
        ds.mapped_parameters.title += (
            ' after Richardson-Lucy deconvolution %i iterations' % 
                iterations)
        if ds.tmp_parameters.has_item('filename'):
                ds.tmp_parameters.filename += (
                    '_after_R-L_deconvolution_%iiter' % iterations)
        psf_size = psf.axes_manager.signal_axes[0].size
        kernel = psf()
        imax = kernel.argmax()
        j = 0
        maxval = self.axes_manager.navigation_size
        if maxval > 0:
            pbar = progressbar(maxval=maxval)
        for D in self:
            D = D.data.copy()
            if psf.axes_manager.navigation_dimension != 0:
                kernel = psf(axes_manager=self.axes_manager)
                imax = kernel.argmax()

            s = ds(axes_manager=self.axes_manager)
            mimax = psf_size -1 - imax
            O = D.copy()
            for i in xrange(iterations):
                first = np.convolve(kernel, O)[imax: imax + psf_size]
                O = O * (np.convolve(kernel[::-1], 
                         D / first)[mimax: mimax + psf_size])
            s[:] = O
            j += 1
            if maxval > 0:
                pbar.update(j)
        if maxval > 0:
            pbar.finish()
        
        return ds

            
    def _spikes_diagnosis(self, signal_mask=None, 
                         navigation_mask=None):
        """Plots a histogram to help in choosing the threshold for 
        spikes removal.
        
        Parameters
        ----------
        signal_mask: boolean array
            Restricts the operation to the signal locations not marked 
            as True (masked)
        navigation_mask: boolean array
            Restricts the operation to the navigation locations not 
            marked as True (masked).
        
        See also
        --------
        spikes_removal_tool
        
        """
        dc = self.data
        axis = self.axes_manager.signal_axes[0]
        if signal_mask is not None:
            dc = dc[..., ~signal_mask]
        if navigation_mask is not None:
            dc = dc[~navigation_mask, :]
        der = np.abs(np.diff(dc, 1, -1))
        plt.figure()
        plt.hist(np.ravel(der.max(-1)),100)
        plt.xlabel('Threshold')
        plt.ylabel('Counts')
        plt.draw()
        
        
    def spikes_removal_tool(self,signal_mask=None, 
                            navigation_mask=None):
        """Graphical interface to remove spikes from EELS spectra.

        Parameters
        ----------
        signal_mask: boolean array
            Restricts the operation to the signal locations not marked 
            as True (masked)
        navigation_mask: boolean array
            Restricts the operation to the navigation locations not 
            marked as True (masked)

        See also
        --------
        _spikes_diagnosis, 

        """
        sr = SpikesRemoval(self,navigation_mask=navigation_mask,
                           signal_mask=signal_mask)
        sr.edit_traits()
        return sr
        
    def _are_microscope_parameters_missing(self):
        """Check if the EELS parameters necessary to calculate the GOS
        are defined in mapped_parameters. If not, in interactive mode 
        raises an UI item to fill the values"""
        must_exist = (
            'TEM.convergence_angle', 
            'TEM.beam_energy',
            'TEM.EELS.collection_angle',)
        missing_parameters = []
        for item in must_exist:
            exists = self.mapped_parameters.has_item(item)
            if exists is False:
                missing_parameters.append(item)
        if missing_parameters:
            if preferences.General.interactive is True:
                par_str = "The following parameters are missing:\n"
                for par in missing_parameters:
                    par_str += '%s\n' % par
                par_str += 'Please set them in the following wizard'
                is_ok = messagesui.information(par_str)
                if is_ok:
                    self._set_microscope_parameters()
                else:
                    return True
            else:
                return True
        else:
            return False
                
    def set_microscope_parameters(self, beam_energy=None, 
            convergence_angle=None, collection_angle=None):
        """Set the microscope parameters that are necessary to calculate
        the GOS.
        
        If not all of them are defined, raises in interactive mode 
        raises an UI item to fill the values
        
        beam_energy: float
            The energy of the electron beam in keV
        convengence_angle : float
            In mrad.
        collection_angle : float
            In mrad.
        """
        if self.mapped_parameters.has_item('TEM') is False:
            self.mapped_parameters.add_node('TEM')
        if self.mapped_parameters.has_item('TEM.EELS') is False:
            self.mapped_parameters.TEM.add_node('EELS')
        mp = self.mapped_parameters
        if beam_energy is not None:
            mp.TEM.beam_energy = beam_energy
        if convergence_angle is not None:
            mp.TEM.convergence_angle = convergence_angle
        if collection_angle is not None:
            mp.TEM.EELS.collection_angle = collection_angle
        
        self._are_microscope_parameters_missing()
                
            
    @only_interactive            
    def _set_microscope_parameters(self):
        if self.mapped_parameters.has_item('TEM') is False:
            self.mapped_parameters.add_node('TEM')
        if self.mapped_parameters.has_item('TEM.EELS') is False:
            self.mapped_parameters.TEM.add_node('EELS')
        tem_par = TEMParametersUI()
        mapping = {
            'TEM.convergence_angle' : 'tem_par.convergence_angle',
            'TEM.beam_energy' : 'tem_par.beam_energy',
            'TEM.EELS.collection_angle' : 'tem_par.collection_angle',}
        for key, value in mapping.iteritems():
            if self.mapped_parameters.has_item(key):
                exec('%s = self.mapped_parameters.%s' % (value, key))
        tem_par.edit_traits()
        mapping = {
            'TEM.convergence_angle' : tem_par.convergence_angle,
            'TEM.beam_energy' : tem_par.beam_energy,
            'TEM.EELS.collection_angle' : tem_par.collection_angle,}
        for key, value in mapping.iteritems():
            if value != t.Undefined:
                exec('self.mapped_parameters.%s = %s' % (key, value))
        self._are_microscope_parameters_missing()
        
            
    def power_law_extrapolation(self, window_size=20,
                                extrapolation_size=1024,
                                add_noise=False,
                                fix_neg_r=False):
        """Extrapolate the spectrum to the right using a powerlaw
        
        
        Parameters
        ----------
        window_size : int
            The number of channels from the right side of the 
            spectrum that are used to estimate the power law 
            parameters.        
        extrapolation_size : int
            Size of the extrapolation in number of channels
        add_noise : bool
            If True, add poissonian noise to the extrapolated spectrum.
        fix_neg_r : bool
            If True, the negative values for the "components.PowerLaw" 
            parameter r will be substituted by zero. The spectrum will 
            be extended, but with a constant value instead of the power 
            law.
        
        Returns
        -------
        A new spectrum, with the extrapolation.
            
        """
        axis = self.axes_manager.signal_axes[0]
        s = self.deepcopy()
        s.mapped_parameters.title += (
            ' %i channels extrapolated' % 
                extrapolation_size)
        if s.tmp_parameters.has_item('filename'):
                s.tmp_parameters.filename += (
                    '_%i_channels_extrapolated' % extrapolation_size)
        new_shape = list(self.data.shape)
        new_shape[axis.index_in_array] += extrapolation_size 
        s.data = np.zeros((new_shape))
        s.get_dimensions_from_data()
        s.data[...,:axis.size] = self.data
        pl = PowerLaw()
        pl.estimate_parameters(
            s, axis.index2value(axis.size - window_size),
            axis.index2value(axis.size - 1))
        if fix_neg_r is True:
            _val = pl.r.map['values']
            _val[_val<=0] = 0
            pl.r.map['values'] = _val
        s.data[...,axis.size:] = (
            pl.A.map['values'][...,np.newaxis]*
            s.axes_manager.signal_axes[0].axis[np.newaxis,axis.size:]**(
            -pl.r.map['values'][...,np.newaxis]))
        return s
        

        
 
        
                        
                      
#    def build_SI_from_substracted_zl(self,ch, taper_nch = 20):
#        """Modify the SI to have fit with a smoothly decaying ZL
#        
#        Parameters
#        ----------
#        ch : int
#            channel index to start the ZL decay to 0
#        taper_nch : int
#            number of channels in which the ZL will decay to 0 from `ch`
#        """
#        sp = copy.deepcopy(self)
#        dc = self.zl_substracted.data_cube.copy()
#        dc[0:ch,:,:] *= 0
#        for i in xrange(dc.shape[1]):
#            for j in xrange(dc.shape[2]):
#                dc[ch:ch+taper_nch,i,j] *= np.hanning(2 * taper_nch)[:taper_nch]
#        sp.zl_substracted.data_cube = dc.copy()
#        dc += self.zero_loss.data_cube
#        sp.data_cube = dc.copy()
#        return sp
#        

#        
#    def correct_dual_camera_step(self, show_lev = False, mean_interval = 3, 
#                                 pca_interval = 20, pcs = 2, 
#                                 normalize_poissonian_noise = False):
#        """Correct the gain difference in a dual camera using PCA.
#        
#        Parameters
#        ----------
#        show_lev : boolen
#            Plot PCA lev
#        mean_interval : int
#        pca_interval : int
#        pcs : int
#            number of principal components
#        normalize_poissonian_noise : bool
#        """ 
#        # The step is between pixels 1023 and 1024
#        pw = pca_interval
#        mw = mean_interval
#        s = copy.deepcopy(self)
#        s.energy_crop(1023-pw, 1023 + pw)
#        s.decomposition(normalize_poissonian_noise)
#        if show_lev:
#            s.plot_lev()
#            pcs = int(raw_input('Number of principal components? '))
#        sc = s.get_decomposition_model(pcs)
#        step = sc.data_cube[(pw-mw):(pw+1),:,:].mean(0) - \
#        sc.data_cube[(pw+1):(pw+1+mw),:,:].mean(0)
#        self.data_cube[1024:,:,:] += step.reshape((1, step.shape[0], 
#        step.shape[1]))
#        self._replot()
#        return step
