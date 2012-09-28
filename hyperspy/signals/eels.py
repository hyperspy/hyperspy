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
        threshold : {None, float}
            Truncation energy to estimate the intensity of the 
            elastic scattering. If None the threshold is taken as the
            first minimum after the ZLP centre.
            
        Returns
        -------
        The elastic scattering intensity. If the navigation size is 0 
        returns a float. Otherwise it returns a Spectrum, Image or a 
        Signal, depending on the currenct spectrum navigation 
        dimensions.
            
        """
        axis = self.axes_manager.signal_axes[0]
        if threshold is None:
            # Use the data from the current location to estimate
            # the threshold as the position of the first maximum
            # after the ZLP
            data = self()
            index = data.argmax()
            while data[index] > data[index + 1]:
                index += 1
            threshold = axis.index2value(index)
            print("Threshold = %1.2f" % threshold)
            del data
        I0 = self.data[
        (slice(None),) * axis.index_in_array + (
            slice(None, axis.value2index(threshold)), 
            Ellipsis,)].sum(axis.index_in_array)
            
        s = self._get_navigation_signal()
        if s is None:
            return I0
        else:
            s.data = I0
            s.mapped_parameters.title = (self.mapped_parameters.title + 
                ' elastic intensity')
            if self.tmp_parameters.has_item('filename'):
                s.tmp_parameters.filename = (
                    self.tmp_parameters.filename +
                    '_elastic_intensity')
                s.tmp_parameters.folder = self.tmp_parameters.folder
                s.tmp_parameters.extension = \
                    self.tmp_parameters.extension
            return s
    
                
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
        threshold : {None, float}
            Truncation energy to estimate the intensity of the 
            elastic scattering. If None the threshold is taken as the
             first minimum after the ZLP centre.
            
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

    def fourier_log_deconvolution(self, zlp, add_zlp=True):
        """Performs fourier-log deconvolution.
        
        Parameters
        ----------
        zlp : EELSSpectrum
            The corresponding zero-loss peak. Note that 
            it must have exactly the same shape as the current spectrum.
        add_zlp : bool
            If True, adds the ZLP to the deconvolved spectrum
        
        Returns
        -------
        An EELSSpectrum containing the current data deconvolved.
        
        Notes
        -----        
        For details see: Egerton, R. Electron Energy-Loss 
        Spectroscopy in the Electron Microscope. Springer-Verlag, 2011.
        
        """
        s = self.deepcopy()
        s.hanning_taper()
        axis = self.axes_manager.signal_axes[0]
        z = np.fft.fft(zlp.data, axis=axis.index_in_array)
        j = np.fft.fft(s.data, axis=axis.index_in_array)
        j1 = z*np.log(j/z)
        s.data = np.fft.ifft(j1, axis=axis.index_in_array).real
        if add_zlp is True:
            s.data += zlp.data
        s.mapped_parameters.title = (s.mapped_parameters.title + 
            ' after Fourier-log deconvolution')
        if s.tmp_parameters.has_item('filename'):
                s.tmp_parameters.filename = (
                    self.tmp_parameters.filename +
                    '_after_fourier_log_deconvolution')
        return s

    def fourier_ratio_deconvolution(self, ll, fwhm=None,
                                    threshold=None):
        """Performs Fourier-ratio deconvolution.
        
        The core-loss should have the background removed. To reduce
         the noise amplication the result is convolved with a
        Gaussian function.        
        
        Parameters
        ----------
        ll: EELSSpectrum
            The corresponding low-loss (ll) EELSSpectrum. Note that 
            it must have exactly the same shape as the current spectrum
            
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
            
        Notes
        -----        
        For details see: Egerton, R. Electron Energy-Loss 
        Spectroscopy in the Electron Microscope. Springer-Verlag, 2011.
        
        """

        ll = ll.power_law_extrapolation(
            window_size=100,
            extrapolation_size=1024)
        cl = self.power_law_extrapolation(
            window_size=20,
            extrapolation_size=1024)

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
        zl = g.function(axis.axis)
        ll.hanning_taper()
        cl.hanning_taper()
        z = np.fft.fft(zl)
        jk = np.fft.fft(cl.data, axis=axis.index_in_array)
        jl = np.fft.fft(ll.data, axis=axis.index_in_array)
        zshape = [1,] * len(cl.data.shape)
        zshape[axis.index_in_array] = axis.size
        s = cl.deepcopy()
        s.data = np.fft.ifft(z.reshape(zshape) * jk / jl,
                             axis=axis.index_in_array).real
        s.data *= I0
        s.crop_in_pixels(-1,None,self.axes_manager.signal_axes[0].size)
        s.mapped_parameters.title = (self.mapped_parameters.title + 
            ' after Fourier-ratio deconvolution')
        if s.tmp_parameters.has_item('filename'):
                s.tmp_parameters.filename = (
                    self.tmp_parameters.filename +
                    'after_fourier_ratio_deconvolution')
        return s
            
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
                                add_noise=False):
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
