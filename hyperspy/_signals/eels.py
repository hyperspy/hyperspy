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
import matplotlib.pyplot as plt
import traits.api as t


from hyperspy._signals.spectrum import Spectrum
from hyperspy.misc.eels.elements import elements as elements_db
import hyperspy.axes
from hyperspy.gui.egerton_quantification import SpikesRemoval
from hyperspy.decorators import only_interactive
from hyperspy.gui.eels import TEMParametersUI
from hyperspy.defaults_parser import preferences
import hyperspy.gui.messages as messagesui
from hyperspy.misc.progressbar import progressbar
from hyperspy.components import PowerLaw
from hyperspy.misc.utils import isiterable, underline
from hyperspy.misc.utils import without_nans


class EELSSpectrum(Spectrum):
    _signal_type = "EELS"
    
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
            The symbol of the elements. Note this input must always be
            in the form of a tuple. Meaning: add_elements(('C',)) will
            work, while add_elements(('C')) will NOT work.
        include_pre_edges : bool
            If True, the ionization edges with an onset below the lower 
            energy limit of the SI will be incluided
            
        Examples
        --------
        
        >>> s = signals.EELSSpectrum(np.arange(1024))
        >>> s.add_elements(('C', 'O'))
        Adding C_K subshell
        Adding O_K subshell
        
        Raises
        ------
        ValueError
        
        """
        if not isiterable(elements) or isinstance(elements, basestring):
            raise ValueError(
            "Input must be in the form of a tuple. For example, "
            "if `s` is the variable containing this EELS spectrum:\n "
            ">>> s.add_elements(('C',))\n"
            "See the docstring for more information.")

        for element in elements:
            if element in elements_db:
                self.elements.add(element)
            else:
                raise ValueError(
                    "%s is not a valid symbol of a chemical element" 
                    % element)
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
            for shell in elements_db[element]['subshells']:
                if shell[-1] != 'a':
                    if start_energy <= \
                    elements_db[element]['subshells'][shell][
                        'onset_energy'] \
                    <= end_energy :
                        subshell = '%s_%s' % (element, shell)
                        if subshell not in self.subshells:
                            print "Adding %s subshell" % (subshell)
                            self.subshells.add(
                                '%s_%s' % (element, shell))
                            e_shells.append(subshell)
                    
    def estimate_zero_loss_peak_centre(self, mask=None):
        """Estimate the posision of the zero-loss peak.
        
        This function provides just a coarse estimation of the position
        of the zero-loss peak centre by computing the position of the maximum
        of the spectra. For subpixel accuracy use `estimate_shift1D`.
        
        Parameters
        ----------
        mask : Signal of bool data type.
            It must have signal_dimension = 0 and navigation_shape equal to the
            current signal. Where mask is True the shift is not computed 
            and set to nan.

        Returns
        -------
        zlpc : Signal subclass
            The estimated position of the maximum of the ZLP peak.
        
        Notes
        -----
        This function only works when the zero-loss peak is the most
        intense feature in the spectrum. If it is not in most cases
        the spectrum can be cropped to meet this criterium.
        Alternatively use `estimate_shift1D`.    
        
        See Also
        --------
        estimate_shift1D, align_zero_loss_peak

        """
        self._check_signal_dimension_equals_one()
        self._check_navigation_mask(mask)
        zlpc = self.valuemax(-1)
        if self.axes_manager.navigation_dimension == 1:
            zlpc = zlpc.as_spectrum(0)
        elif self.axes_manager.navigation_dimension > 1:
            zlpc = zlpc.as_image((0, 1))
        if mask is not None:
            zlpc.data[mask.data] = np.nan
        return zlpc

    def align_zero_loss_peak(
            self,
            calibrate=True,
            also_align=[],
            print_stats=True,
            subpixel=True,
            mask=None,
            **kwargs):
        """Align the zero-loss peak.

        This function first aligns the spectra using the result of
        `estimate_zero_loss_peak_centre` and afterward, if subpixel is True,
        proceeds to align with subpixel accuracy using `align1D`. The offset 
        is automatically correct if `calibrate` is True.
        
        Parameters
        ----------
        calibrate : bool
            If True, set the offset of the spectral axis so that the 
            zero-loss peak is at position zero.
        also_align : list of signals
            A list containing other spectra of identical dimensions to 
            align using the shifts applied to the current spectrum.
            If `calibrate` is True, the calibration is also applied to
            the spectra in the list.
        print_stats : bool
            If True, print summary statistics the ZLP maximum before
            the aligment.
        subpixel : bool
            If True, perform the alignment with subpixel accuracy 
            using cross-correlation.
        mask : Signal of bool data type.
            It must have signal_dimension = 0 and navigation_shape equal to the
            current signal. Where mask is True the shift is not computed 
            and set to nan.

        See Also
        --------
        estimate_zero_loss_peak_centre, align1D, estimate_shift1D.

        Notes
        -----
        Any extra keyword arguments are passed to `align1D`. For
        more information read its docstring.

        """
        def substract_from_offset(value, signals):
            for signal in signals: 
                signal.axes_manager[-1].offset -= value

        zlpc = self.estimate_zero_loss_peak_centre(mask=mask)
        mean_ = without_nans(zlpc.data).mean()
        if print_stats is True:
            print
            print(underline("Initial ZLP position statistics"))
            zlpc.print_summary_statistics()

        for signal in also_align + [self]:
            signal.shift1D(-zlpc.data + mean_)

        if calibrate is True:
            zlpc = self.estimate_zero_loss_peak_centre(mask=mask)
            substract_from_offset(without_nans(zlpc.data).mean(),
                                  also_align + [self])
        
        if subpixel is False: return
        left, right = -3., 3.
        if calibrate is False:
            mean_ = without_nans(self.estimate_zero_loss_peak_centre(
                mask=mask).data).mean()
            left += mean_
            right += mean_
            
        left = (left if left > self.axes_manager[-1].axis[0]
                    else self.axes_manager[-1].axis[0]) 
        right = (right if right < self.axes_manager[-1].axis[-1]
                    else self.axes_manager[-1].axis[-1]) 
        self.align1D(left, right, also_align=also_align, **kwargs)
        zlpc = self.estimate_zero_loss_peak_centre(mask=mask)
        if calibrate is True:
            substract_from_offset(without_nans(zlpc.data).mean(),
                                  also_align + [self])

    def estimate_elastic_scattering_intensity(self,
                                              threshold=None,):
        """Rough estimation of the elastic scattering intensity by 
        truncation of a EELS low-loss spectrum.
        
        Parameters
        ----------
        threshold : {Signal, float, int}
            Truncation energy to estimate the intensity of the 
            elastic scattering. The
            threshold can be provided as a signal of the same dimension 
            as the input spectrum navigation space containing the 
            threshold value in the energy units. Alternatively a constant 
            threshold can be specified in energy/index units by passing 
            float/int.
            
        Returns
        -------
        I0: Signal
            The elastic scattering intensity. If the navigation size is 0 
            returns a float. Otherwise it returns a Spectrum, Image or a 
            Signal, depending on the currenct spectrum navigation 
            dimensions.
            
        See Also
        --------
        estimate_elastic_scattering_threshold
            
        """
        # TODO: Write units tests
        self._check_signal_dimension_equals_one()
        
        if isinstance(threshold, float):
            I0 = self.isig[:threshold].integrate_simpson(-1)
            I0.axes_manager.set_signal_dimension(
                                min(2, self.axes_manager.navigation_dimension))
        
        else:
            bk_threshold_navigate = (
                threshold.axes_manager._get_axis_attribute_values('navigate'))
            threshold.axes_manager.set_signal_dimension(0)
            I0 = self._get_navigation_signal()
            bk_I0_navigate = (
                I0.axes_manager._get_axis_attribute_values('navigate'))
            I0.axes_manager.set_signal_dimension(0)
            pbar = hyperspy.misc.progressbar.progressbar(
                                    maxval=self.axes_manager.navigation_size)
            for i, s in enumerate(self):
                threshold_ = threshold[self.axes_manager.indices].data[0]
                if np.isnan(threshold_):
                    I0[self.axes_manager.indices] = np.nan
                else:
                    I0[self.axes_manager.indices].data[:] = (
                        s[:threshold_].integrate_simpson(-1).data)
                pbar.update(i)
            pbar.finish()
            threshold.axes_manager._set_axis_attribute_values(
                    'navigate',
                    bk_threshold_navigate)
            I0.axes_manager._set_axis_attribute_values(
                    'navigate',
                    bk_I0_navigate)
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
    
    def estimate_elastic_scattering_threshold(self,
                                              window=10.,
                                              tol=None,
                                              number_of_points=5,
                                              polynomial_order=3,
                                              start=1.):
        """Calculates the first inflexion point of the spectrum derivative 
        within a window using a specified tolerance.
        
        It previously smoothes the data using a Savitzky-Golay algorithm 
        (can be turned off). This method assumes that the zero-loss peak is 
        located at position zero in all the spectra.
        
        Parameters
        ----------
           
        window : {None, float}
            If None, the search for the local minimum is performed 
            using the full energy range. A positive float will restrict
            the search to the (0,window] energy window, where window is given
            in the axis units. If no inflexion point is found in this
            spectral range the window value is returned instead.
        tol : {None, float}
            The threshold tolerance for the derivative. If "auto" it is 
            automatically calculated as the minimum value that guarantees 
            finding an inflexion point in all the spectra in given energy
            range.
        number_of_points : int
            If non zero performs order three Savitzky-Golay smoothing 
            to the data to avoid falling in local minima caused by 
            the noise.
        polynomial_order : int
            Savitzky-Golay filter polynomial order.
        start : float
            Position from the zero-loss peak centre from where to start
            looking for the inflexion point.

            
        Returns
        -------
        threshold : Signal
            A Signal of the same dimension as the input spectrum 
            navigation space containing the estimated threshold.
            
        See Also
        --------
        align1D
            
        """
        self._check_signal_dimension_equals_one()
        # Create threshold with the same shape as the navigation dims.
        threshold = self._get_navigation_signal()
        threshold.axes_manager.set_signal_dimension(0)

        # Progress Bar
        axis = self.axes_manager.signal_axes[0]
        max_index = min(axis.value2index(window), axis.size - 1)
        min_index = max(0, axis.value2index(start))
        if max_index < min_index + 10:
            raise ValueError("Please select a bigger window")
        s = self[..., min_index: max_index].deepcopy()
        if number_of_points:
            s.smooth_savitzky_golay(polynomial_order=polynomial_order,
                                    number_of_points=number_of_points,
                                    differential_order=1)
        else:
            s = s.diff(-1)
        if tol == None:
            tol = np.max(np.abs(s.data).min(axis.index_in_array))
        saxis = s.axes_manager[-1]
        inflexion = (np.abs(s.data) <= tol).argmax(saxis.index_in_array)
        threshold.data[:] = saxis.offset + saxis.scale * inflexion
        threshold.data[inflexion==0] = np.nan
        del s 
 
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
        threshold.axes_manager.set_signal_dimension(
                                min(2, self.axes_manager.navigation_dimension))
        return threshold
        
    def estimate_thickness(self,
                           zlp=None,
                           threshold=None,):
        """Estimates the thickness (relative to the mean free path) 
        of a sample using the log-ratio method.
        
        The current EELS spectrum must be a low-loss spectrum containing
        the zero-loss peak. The hyperspectrum must be well calibrated 
        and aligned. 
        
        Parameters
        ----------
        zlp : {None, EELSSpectrum}
            If not None the zero-loss
            peak intensity is calculated from the ZLP spectrum
            supplied by integration using Simpson's rule. If None estimates 
            the zero-loss peak intensity using 
            `estimate_elastic_scattering_intensity` by truncation.
            
        threshold : {Signal, float, int}
            Truncation energy to estimate the intensity of the 
            elastic scattering. The
            threshold can be provided as a signal of the same dimension 
            as the input spectrum navigation space containing the 
            threshold value in the energy units. Alternatively a constant 
            threshold can be specified in energy/index units by passing 
            float/int.
            
            
        Returns
        -------
        s : Signal
            The thickness relative to the MFP. It returns a Spectrum, 
            Image or a Signal, depending on the currenct spectrum navigation 
            dimensions.
            
        Notes
        -----        
        For details see: Egerton, R. Electron Energy-Loss 
        Spectroscopy in the Electron Microscope. Springer-Verlag, 2011.
        
        """
        # TODO: Write units tests
        self._check_signal_dimension_equals_one()
        axis = self.axes_manager.signal_axes[0]
        total_intensity = self.integrate_simpson(axis.index_in_array).data
        if zlp is not None:
            I0 = zlp.integrate_simpson(axis.index_in_array).data 
        else:
            I0 = self.estimate_elastic_scattering_intensity(
                                    threshold=threshold,).data

        t_over_lambda = np.log(total_intensity / I0)
        s = self._get_navigation_signal()
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
                
    def fourier_log_deconvolution(self,
                                  zlp,
                                  add_zlp=False,
                                  crop=False):
        """Performs fourier-log deconvolution.
        
        Parameters
        ----------
        zlp : EELSSpectrum
            The corresponding zero-loss peak.

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
        self._check_signal_dimension_equals_one()
        s = self.deepcopy()        
        zlp_size = zlp.axes_manager.signal_axes[0].size 
        self_size = self.axes_manager.signal_axes[0].size
        tapped_channels = s.hanning_taper()
        # Conservative new size to solve the wrap-around problem 
        size = zlp_size + self_size -1
        # Increase to the closest multiple of two to enhance the FFT 
        # performance
        size = int(2 ** np.ceil(np.log2(size)))

        axis = self.axes_manager.signal_axes[0]
        z = np.fft.rfft(zlp.data, n=size, axis=axis.index_in_array)
        j = np.fft.rfft(s.data, n=size, axis=axis.index_in_array)
        j1 = z * np.nan_to_num(np.log(j / z))
        sdata = np.fft.irfft(j1, axis=axis.index_in_array)

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
            s.crop(axis.index_in_axes_manager,
                             None, int(-tapped_channels))
        return s

    def fourier_ratio_deconvolution(self, ll,
                                    fwhm=None,
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
        self._check_signal_dimension_equals_one()
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
            fwhm = float(ll.get_current_signal().estimate_peak_width()())
            print("FWHM = %1.2f" % fwhm) 

        I0 = ll.estimate_elastic_scattering_intensity(
                                                threshold=threshold)
        if ll.axes_manager.navigation_size > 0:
            I0 = I0.data
            I0_shape = list(I0.shape)
            I0_shape.insert(axis.index_in_array,1)
            I0 = I0.reshape(I0_shape)
            
        from hyperspy.components import Gaussian
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
        cl.crop(-1,None,int(orig_cl_size))
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
    
        Parameters
        ----------
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
        self._check_signal_dimension_equals_one()
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
        self._check_signal_dimension_equals_one()
        dc = self.data
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
        self._check_signal_dimension_equals_one()
        sr = SpikesRemoval(self,
                           navigation_mask=navigation_mask,
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
            parameter r will be flagged and the extrapolation will be 
            done with a constant zero-value.
        
        Returns
        -------
        A new spectrum, with the extrapolation.
            
        """
        self._check_signal_dimension_equals_one()
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
        pl._axes_manager = self.axes_manager
        pl.estimate_parameters(
            s, axis.index2value(axis.size - window_size),
            axis.index2value(axis.size - 1))
        if fix_neg_r is True:
            _r = pl.r.map['values']
            _A = pl.A.map['values']
            _A[_r<=0] = 0
            pl.A.map['values'] = _A
        s.data[...,axis.size:] = (
            pl.A.map['values'][...,np.newaxis]*
            s.axes_manager.signal_axes[0].axis[np.newaxis,axis.size:]**(
            -pl.r.map['values'][...,np.newaxis]))
        return s
        

