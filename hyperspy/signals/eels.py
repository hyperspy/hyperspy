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

from hyperspy.signals.spectrum import Spectrum
from hyperspy.signals.image import Image

from hyperspy.components.eels_cl_edge import edges_dict
import hyperspy.axes

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
#        self.readout = None
#        self.dark_current = None
#        self.gain_correction = None

        # Perform treatments if pretreatments is True
#        if apply_treatments:
#            # Corrects the readout if the readout file is provided
#            if dark_current is not None:
#                self.dark_current = Spectrum(dark_current, 
#                apply_treatments = False)
#                self._process_dark_current()
#                self.dark_current_correction()
#            
#            if readout is not None:
#                self.readout = Spectrum(readout, 
#                dark_current = dark_current)
#                self._process_readout()
#                self.readout_correction()
#
#            if gain is not None:
#                self.gain_correction = Spectrum(gain, apply_treatments = False)
#                self.correct_gain()
            # Corrects the gain of the acquisition system

#    def extract_zero_loss(self, zl = None,right = 0.2,around = 0.05):
#        """
#        Zero loss extraction by the reflected-tail or fingerprinting methods.
#        
#        Creates a new spectrum instance self.zero_loss with the zero loss 
#        extracted by the reflected-tail method if no zero loss in the vacuum is 
#        provided. Otherwise it use the zero loss fingerprinting method.
#
#        Parameters
#        ----------
#        zl : str
#            name of the zero loss in vacuum file for the fingerprinting method
#        right : float
#            maximum channel in energy units to use to fit the zero loss in the 
#            vacuum. Only has effect for the fingerprinting method.
#        around : float
#            interval around the origin to remove from the fit for the of the 
#            zero loss in the vacuum. Only has effect for the fingerprinting 
#            method.
#        
#        Notes
#        -----
#        It is convenient to align the SI and correct the baseline before using
#        this method.
#        """
#
#        print "Extracting the zero loss"
#        if zl is None: # reflected-tail
#            # Zero loss maximum
#            i0 = self.data_cube[:,0,0].argmax(0)
#            # FWQM from the first spectrum in channels
#            # Search only 2eV around the max to avoid counting the plasmons
#            # in thick samples
#            i_range = int(round(2. / self.energyscale))
#            fwqm_bool = self.data_cube[i0-i_range:i0+i_range,0,0] > \
#            0.25 * self.data_cube[i0,0,0]
#            ch_fwqm = len(fwqm_bool[fwqm_bool])
#            self.zero_loss = copy.deepcopy(self)
#            data = self.zero_loss.data_cube
#            canvas = np.zeros(data.shape)
#            # Reflect the tail
#            width = int(round(1.5 * ch_fwqm))
#            canvas[i0 + width : 2 * i0 + 1,:,:] = \
#            data[i0 - width::-1,:,:]
#            # Remove the "background" = mean of first 4 channels and reflects the
#            # tail
#            bkg = np.mean(data[0:4])
#            canvas -= bkg
#            # Scale the extended tail with the ratio obtained from
#            # 2 overlapping channels
#            ch = i0 + width
#            ratio = np.mean(data[ch: ch + 2] / canvas[ch: ch + 2], 0)
#            for ix in xrange(data.shape[1]):
#                for iy in xrange(data.shape[2]):
#                    canvas[:,ix,iy] *= ratio[ix,iy]
#            # Copy the extension
#            data[i0 + width:] = canvas[i0 + width:]
#        else:
#            import components
#            fp = components.ZL_Fingerprinting(zl)
#            m = Model(self,False)
#            m.append(fp)
#            m.set_data_region(None,right)
#            m.remove_data_range(-around,around)
#            m.multifit()
#            self.zero_loss = copy.deepcopy(self)
#            self.zero_loss.data_cube = m.model_cube
#            self.zl_substracted = copy.deepcopy(self)
#            self.zl_substracted.data_cube -= self.zero_loss.data_cube
#        self._replot()
#        
#    def _process_gain_correction(self):
#        gain = self.gain_correction
#        # Check if the spectrum has the same number of channels:
#        if self.data_cube.shape[0] != gain.data_cube.shape[0]:
#            print 
#            messages.warning_exit(
#            "The gain and spectrum don't have the same number of channels")
#        dc = gain.data_cube.copy()
#        dc = dc.sum(1).mean(1)
#        dc /= dc.mean()
#        gain.normalized_gain = dc
#
#    def _process_readout(self):
#        """Readout conditioning
#        
#        Checks if the readout file provided contains more than one spectrum.
#        If that is the case, it makes the average and produce a single spectrum
#        Spectrum object to feed the correct spectrum function"""
#        channels = self.readout.data_cube.shape[0]
#        if self.readout.data_cube.shape[1:]  > (1, 1):
#            self.readout.data_cube = np.average(
#            np.average(self.readout.data_cube,1),1).reshape(channels, 1, 1)
#            self.readout.get_dimensions_from_cube()
#            self.readout.set_new_calibration(0,1)
#            if self.readout.dark_current:
#                self.readout._process_dark_current()
#                self.readout.dark_current_correction()
#
#    def _process_dark_current(self):
#        """Dark current conditioning.
#        
#        Checks if the readout file provided contains more than one spectrum.
#        If that is the case, it makes the average and produce a single spectrum
#        Spectrum object to feed the correct spectrum function. If 
#        a readout correction is provided, it corrects the readout in the dark
#        current spim."""
#        if self.dark_current.data_cube.shape[1:]  > (1, 1):
#            self.dark_current.data_cube = np.average(
#            np.average(self.dark_current.data_cube,1),1).reshape((-1, 1, 1))
#            self.dark_current.get_dimensions_from_cube()
#            self.dark_current.set_new_calibration(0,1)
#            
#    # Elements _________________________________________________________________
    def add_elements(self, elements, include_pre_edges = False):
        """Declare the elemental composition of the sample.
        
        The ionisation edges of the elements present in the current energy range
        will be added automatically.
        
        Parameters
        ----------
        elements : tuple of strings
            The strings must represent a chemical element.
        include_pre_edges : bool
            If True, the ionization edges with an onset below the lower energy 
            limit of the SI will be incluided
        """
        for element in elements:
            self.elements.add(element)
        if not hasattr(self.mapped_parameters, 'Sample'):
            self.mapped_parameters.add_node('Sample')
        self.mapped_parameters.Sample.elements = list(self.elements)
        self.generate_subshells(include_pre_edges)
        
    def generate_subshells(self, include_pre_edges = False):
        """Calculate the subshells for the current energy range for the elements
         present in self.elements
         
        Parameters
        ----------
        include_pre_edges : bool
            If True, the ionization edges with an onset below the lower energy 
            limit of the SI will be incluided
        """
        Eaxis = self.axes_manager._slicing_axes[0].axis
        if not include_pre_edges:
            start_energy = Eaxis[0]
        else:
            start_energy = 0.
        end_energy = Eaxis[-1]
        for element in self.elements:
            e_shells = list()
            for shell in edges_dict[element]['subshells']:
                if shell[-1] != 'a':
                    if start_energy <= \
                    edges_dict[element]['subshells'][shell]['onset_energy'] \
                    <= end_energy :
                        subshell = '%s_%s' % (element, shell)
                        if subshell not in self.subshells:
                            print "Adding %s subshell" % (subshell)
                            self.subshells.add('%s_%s' % (element, shell))
                            e_shells.append(subshell)

    
#            
#    def remove_background(self, start_energy = None, mask = None):
#        """Removes the power law background of the EELS SI if the present 
#        elements were defined.
#        
#        It stores the background in self.background.
#        
#        Parameters
#        ----------
#        start_energy : float
#            The starting energy for the fitting routine
#        mask : boolean numpy array
#        """
#        from spectrum import Spectrum
#        if mask is None:
#            mask = np.ones(self.data_cube.shape[1:], dtype = 'bool')
#        m = Model(self)
#        m.fit_background(startenergy = start_energy, type = 'multi', 
#        mask = mask)
#        m.model_cube[:, mask == False] *= 0
#        self.background = Spectrum()
#        self.background.data_cube = m.model_cube
#        self.background.get_dimensions_from_cube()
#        utils.copy_energy_calibration(self, self.background)
##        self.background.get_calibration_from()
#        print "Background stored in self.background"
#        self.__new_cube(self.data_cube[:] - m.model_cube[:], 
#        'background removal')
#        self._replot()
#        
#    def calculate_I0(self, threshold = None):
#        """Estimates the integral of the ZLP from a LL SI
#        
#        The value is stored in self.I0 as an Image.
#        
#        Parameters
#        ----------
#        thresh : float or None
#            If float, it estimates the intensity of the ZLP as the sum 
#            of all the counts of the SI until the threshold. If None, it 
#            calculates the sum of the ZLP previously stored in 
#            self.zero_loss
#        """
#        if threshold is None:
#            if self.zero_loss is None:
#                messages.warning_exit(
#                "Please, provide a threshold value of define the " 
#                "self.zero_loss attribute by, for example, using the "
#                "extract_zero_loss method")
#            else:
#                self.I0 = Image(dc = self.zero_loss.sum(0))
#        else:
#            threshold = self.energy2index(threshold)
#            self.I0 = Image(dc = self.data_cube[:threshold,:,:].sum(0)) 
#        
#    def correct_gain(self):
#        """Apply the gain correction stored in self.gain_correction
#        """
#        if not self.treatments.gain:
#            self._process_gain_correction()
#            gain = self.gain_correction
#            print "Applying gain correction"
#            # Gain correction
#            data = np.zeros(self.data_cube.shape)
#            for ix in xrange(0, self.xdimension):
#                for iy self.energy_in xrange(0, self.ydimension):
#                    np.divide(self.data_cube[:,ix,iy], 
#                    gain.normalized_gain, 
#                    data[:,ix,iy])
#            self.__new_cube(data, 'gain correction')
#            self.treatments.gain = 1
#            self._replot()
#        else:
#            print "Nothing done, the SI was already gain corrected"
#
#    def correct_baseline(self, kind = 'pixel', positive2zero = True, 
#    averaged = 10, fix_negative = True):
#        """Set the minimum value to zero
#        
#        It can calculate the correction globally or pixel by pixel.
#        
#        Parameters
#        ----------
#        kind : {'pixel', 'global'}
#            if 'pixel' it calculates the correction pixel by pixel.
#            If 'global' the correction is calculated globally.
#        positive2zero : bool
#            If False it will only set the baseline to zero if the 
#            minimum is negative
#        averaged : int
#            If > 0, it will only find the minimum in the first and last 
#            given channels
#        fix_negative : bool
#            When averaged, it will take the abs of the data_cube to assure
#            that no value is negative.
#        
#        """
#        data = copy.copy(self.data_cube)
#        print "Correcting the baseline of the low loss spectrum/a"
#        if kind == 'global':
#            if averaged == 0:
#                minimum = data.min()
#            else:
#                minimum = np.vstack(
#                (data[:averaged,:,:], data[-averaged:,:,:])).min()
#            if minimum < 0. or positive2zero is True:
#                data -= minimum
#        elif kind == 'pixel':
#            if averaged == 0:
#                minimum = data.min(0).reshape(
#            (1,data.shape[1], data.shape[2]))
#            else:
#                minimum = np.vstack((data[:averaged,:,:], data[-averaged:,:,:])
#                ).min(0).reshape((1,data.shape[1], data.shape[2]))
#            mask = np.ones(data.shape[1:], dtype = 'bool')
#            if positive2zero is False:
#                mask[minimum.squeeze() > 0] = False
#            data[:,mask] -= minimum[0,mask]
#        else:
#            messages.warning_exit(
#            "Wrong kind keyword. Possible values are pixel or global")
#        
#        if fix_negative:
#            data = np.abs(data)
#        self.__new_cube(data, 'baseline correction')
#        self._replot()
#
#    def readout_correction(self):
#        if not self.treatments.readout:
#            if hasattr(self, 'readout'):
#                data = copy.copy(self.data_cube)
#                print "Correcting the readout"
#                for ix in xrange(0,self.xdimension):
#                    for iy in xrange(0,self.ydimension):
#                        data[:, ix, iy] -= self.readout.data_cube[:,0,0]
#                self.__new_cube(data, 'readout correction')
#                self.treatments.readout = 1
#                self._replot()
#            else:
#                print "To correct the readout, please define the readout attribute"
#        else:
#            print "Nothing done, the SI was already readout corrected"
#
#    def dark_current_correction(self):
#        """Apply the dark_current_correction stored in self.dark_current"""
#        if self.treatments.dark_current:
#            print "Nothing done, the dark current was already corrected"
#        else:
#            ap = self.acquisition_parameters
#            if hasattr(self, 'dark_current'):
#                if (ap.exposure is not None) and \
#                (self.dark_current.acquisition_parameters.exposure):
#                    if (ap.readout_frequency is not None) and \
#                    (ap.blanking is not None):
#                        if not self.acquisition_parameters.blanking:
#                            exposure = ap.exposure + self.data_cube.shape[0] * \
#                            ap.ccd_height / (ap.binning * ap.readout_frequency)
#                            ap.effective_exposure = exposure
#                        else:
#                            exposure = ap.exposure
#                    else:
#                        print \
#    """Warning: no information about binning and readout frequency found. Please 
#    define the following attributes for a correct dark current correction:
#    exposure, binning, readout_frequency, ccd_height, blanking
#    The correction proceeds anyway
#    """
#                            
#                        exposure = self.acquisition_parameters.exposure
#                    data = copy.copy(self.data_cube)
#                    print "Correcting the dark current"
#                    self.dark_current.data_cube[:,0,0] *= \
#                    (exposure / self.dark_current.acquisition_parameters.exposure)
#                    data -= self.dark_current.data_cube
#                    self.__new_cube(data, 'dark current correction')
#                    self.treatments.dark_current = 1
#                    self._replot()
#                else:
#                    
#                    messages.warning_exit(
#                    "Please define the exposure attribute of the spectrum"
#                    "and its dark_current")
#            else:
#                messages.warning_exit(
#               "To correct the readout, please define the dark_current " \
#                "attribute")
#                
    def find_low_loss_centre(self, sync_signal = None):
        """Calculate the position of the zero loss origin as the average of the 
        postion of the maximum of all the spectra"""
        axis = self.axes_manager._slicing_axes[0] 
        old_offset = axis.offset
        imax = np.mean(np.argmax(self.data,axis.index_in_array))
        axis.offset = hyperspy.axes.generate_axis(0, axis.scale, 
            axis.size, imax)[0]
        print('Energy offset applied: %f %s' % ((axis.offset - old_offset), 
              axis.units))
        if sync_signal is not None:
            saxis = sync_signal.axes_manager.axes[axis.index_in_array]
            saxis.offset += axis.offset - old_offset

    def fourier_log_deconvolution(self):
        """Performs fourier-log deconvolution of the full SI.
        
        The zero-loss can be specified by defining the parameter 
        self.zero_loss that must be an instance of Spectrum. 
        """
        axis = self.axes_manager._slicing_axes[0]
        z = np.fft.fft(self.zero_loss.data, axis = axis.index_in_array)
        j = np.fft.fft(self.data, axis = axis.index_in_array)
        j1 = z*np.log(j/z)
        self.data = np.fft.ifft(j1, axis = 0).real
        
    def calculate_thickness(self, method = 'threshold', threshold = 3, 
    factor = 1):
        """Calculates the thickness from a LL SI.
        
        The resulting thickness map is stored in self.thickness as an image 
        instance. To visualize it: self.thickness.plot()
        
        Parameters
        ----------
        method : {'threshold', 'zl'}
            If 'threshold', it will extract the zero loss by just splittin the 
            spectrum at the threshold value. If 'zl', it will use the 
            self.zero_loss SI (if defined) to perform the calculation.
        threshold : float
            threshold value.
        factor : float
            factor by which to multiple the ZLP
        """
        print "Calculating the thickness"
        # Create the thickness array
        dc = self.data
        axis = self.axes_manager._slicing_axes[0]
        integral = dc.sum(axis.index_in_array)
        if method == 'zl':
            if self.zero_loss is None:
                hyperspy.messages.warning_exit('To use this method the zero_loss'
                'attribute must be defined')
            zl = self.zero_loss.data
            zl_int = zl.sum(axis.index_in_array)            
        elif method == 'threshold':
            ti = axis.value2index(threshold)
            zl_int = self.data[
            (slice(None),) * axis.index_in_array + (slice(None, ti), Ellipsis,)
            ].sum(axis.index_in_array) * factor 
        self.thickness = \
        Image({'data' : np.log(integral / zl_int)})
                
    def calculate_FWHM(self, factor = 0.5, energy_range = (-2,2), der_roots = False):
        """Use a third order spline interpolation to estimate the FWHM of 
        the zero loss peak.
        
        Parameters:
        -----------
        factor : float < 1
            By default is 0.5 to give FWHM. Choose any other float to give
            find the position of a different fraction of the peak.
        channels : int
            radius of the interval around the origin were the algorithm will 
            perform the estimation.
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
        axis = self.axes_manager._slicing_axes[0]
        i0, i1 = axis.value2index(energy_range[0]), axis.value2index(
        energy_range[1])
        data = self()[i0:i1]
        x = axis.axis[i0:i1]
        height = np.max(data)
        spline_fwhm = scipy.interpolate.UnivariateSpline(x, 
                                                         data - factor * height)
        pair_fwhm = spline_fwhm.roots()[0:2]
        fwhm = pair_fwhm[1] - pair_fwhm[0]
        if der_roots:
            der_x = np.arange(x[0], x[-1] + 1, (x[1] - x[0]) * 0.2)
            derivative = spline_fwhm(der_x, 1)
            spline_der = scipy.interpolate.UnivariateSpline(der_x, derivative)
            return {'FWHM' : fwhm, 'pair' : pair_fwhm, 
            'der_roots': spline_der.roots()}
        else:
            return {'FWHM' : fwhm, 'FWHM_E' : pair_fwhm}
#            
#    def power_law_extension(self, interval, new_size = 1024, 
#                            to_the = 'right'):
#        """Extend the SI with a power law.
#        
#        Fit the SI with a power law in the given interval and use the result 
#        to extend it to the left of to the right.
#        
#        Parameters
#        ----------
#        interval : tuple
#            Interval to perform the fit in energy units        
#        new_size : int
#            number of channel after the extension.
#        to_the : {'right', 'left'}
#            extend the SI to the left or to the right
#        """
#        left, right = interval
#        s = self.data_cube.shape
#        original_size = s[0]
#        if original_size >= new_size:
#            print "The new size (in channels) must be bigger than %s" % \
#            original_size
#        new_cube = np.zeros((new_size, s[1], s[2]))
#        iright = self.energy2index(right)
#        new_cube[:iright,:,:] = self.data_cube[:iright,:,:]
#        self.data_cube = new_cube
#        self.get_dimensions_from_cube()
#        m = Model(self, False, auto_add_edges = False)
#        pl = PowerLaw()
#        m.append(pl)
#        m.set_data_region(left,right)
#        m.multifit(grad = True)
#        self.data_cube[iright:,:,:] = m.model_cube[iright:,:,:]
#        
#    def hanning_taper(self, side = 'left', channels = 20,offset = 0):
#        """Hanning taper
#        
#        Parameters
#        ----------
#        side : {'left', 'right'}
#        channels : int
#        offset : int
#        """        
#        dc = self.data_cube
#        if side == 'left':
#            dc[offset:channels+offset,:,:] *= \
#            (np.hanning(2*channels)[:channels]).reshape((-1,1,1))
#            dc[:offset,:,:] *= 0. 
#        if side == 'right':
#            dc[-channels-offset:-offset,:,:] *= \
#            (np.hanning(2*channels)[-channels:]).reshape((-1,1,1))
#            dc[-offset:,:,:] *= 0. 
#        
    def remove_spikes(self, threshold = 2200, subst_width = 5, 
                      coordinates = None, energy_range = None, add_noise = True):
        """Remove the spikes in the SI.
        
        Detect the spikes above a given threshold and fix them by interpolating 
        in the give interval. If coordinates is given, it will only remove the 
        spikes for the specified spectra.
        
        Paramerters:
        ------------
        threshold : float
            A suitable threshold can be determined with 
            Spectrum.spikes_diagnosis
        subst_width : tuple of int or int
            radius of the interval around the spike to substitute with the 
            interpolation. If a tuple, the dimension must be equal to the 
            number of spikes in the threshold. If int the same value will be 
            applied to all the spikes.
        energy_range: List
            Restricts the operation to the energy range given in units
            
        add_noise: Bool
            If True, Poissonian noise will be added to the region that has been
            interpolated to remove the spikes
        
        See also
        --------
        Spectrum.spikes_diagnosis, Spectrum.plot_spikes
        """
        axis = self.axes_manager._slicing_axes[0]
        int_window = 20
        dc = self.data
        if energy_range is not None:
            ileft, iright = (axis.value2index(energy_range[0]),
                             axis.value2index(energy_range[1]))
            _slice = slice(ileft, iright)
        else:
            _slice = slice(None)
            ileft = 0
            iright = len(axis.axis)

        der = np.diff(dc[...,_slice], 1, axis.index_in_array)
        E_ax = axis.axis
        n_ch = len(E_ax)
        i = 0
        if coordinates is None:
            coordinates = []
            for index in np.ndindex(tuple(self.axes_manager.navigation_shape)):
                coordinates.append(index)
        for index in coordinates:
            lindex = list(index)
            lindex.insert(axis.index_in_array, slice(None))
            if der[lindex].max() >= threshold:
                print "Spike detected in ", index
                argmax = ileft + der[lindex].argmax()
                if hasattr(subst_width, '__iter__'):
                    subst__width = subst_width[index]
                else:
                    subst__width = subst_width
                lp1 = np.clip(argmax - int_window, 0, n_ch)
                lp2 = np.clip(argmax - subst__width, 0, n_ch)
                rp2 = np.clip(argmax + int_window, 0, n_ch)
                rp1 = np.clip(argmax + subst__width, 0, n_ch)
                x = np.hstack((E_ax[lp1:lp2], E_ax[rp1:rp2]))
                nlindex1 = list(index)
                nlindex1.insert(axis.index_in_array, 
                                  slice(lp1, lp2))
                nlindex2 = list(index)
                nlindex2.insert(axis.index_in_array, 
                                  slice(rp1, rp2))                
                
                y = np.hstack((dc[nlindex1], dc[nlindex2])) 
                # The weights were commented because the can produce 
                # nans, maybe it should be an option?
                intp = scipy.interpolate.UnivariateSpline(x, y) 
                #,w = 1/np.sqrt(y))
                x_int = E_ax[lp2:rp1 + 1]
                nlindex3 = list(index)
                nlindex3.insert(axis.index_in_array, slice(lp2, rp1 + 1))
                new_data = intp(x_int)
                if add_noise is True:
                    new_data = np.random.poisson(np.clip(new_data, 0, np.inf))
                dc[nlindex3]  = new_data
                i += 1
                
    def spikes_diagnosis(self, energy_range = None):
        """Plots a histogram to help in choosing the threshold for spikes
        removal.
        
        Parameters
        ----------
        energy_range: List
            Restricts the operation to the energy range given in units
        
        See also
        --------
        Spectrum.remove_spikes, Spectrum.plot_spikes
        """
        dc = self.data
        axis = self.axes_manager._slicing_axes[0]
        if energy_range is not None:
            dc = dc[..., axis.value2index(energy_range[0]):
                    axis.value2index(energy_range[1])]
        der = np.diff(dc, 1, axis.index_in_array)
        plt.figure()
        plt.hist(np.ravel(der.max(axis.index_in_array)),100)
        plt.xlabel('Threshold')
        plt.ylabel('Counts')
        plt.draw()
        
    def plot_spikes(self, threshold = 2200, energy_range = None, plot = True):
        """Plot the spikes in the given threshold
        
        Parameters
        ----------
        threshold : float
        energy_range: List
            Restricts the operation to the energy range given in units
        
        Returns
        -------
        list of spikes coordinates
        
        See also
        --------
        Spectrum.remove_spikes, Spectrum.spikes_diagnosis
        """
        dc = self.data
        axis = self.axes_manager._slicing_axes[0]
        if energy_range is not None:
            dc = dc[..., axis.value2index(energy_range[0]):
                    axis.value2index(energy_range[1])]
        der = np.diff(dc,1,axis.index_in_array)
        i = 0
        spikes = []
        for index in np.ndindex(tuple(self.axes_manager.navigation_shape)):
            lindex = list(index)
            lindex.insert(axis.index_in_array, slice(None))
            if der[lindex].max() >= threshold:
                print "Spike detected in ", index
                spikes.append(index)
                argmax = der[lindex].argmax()
                nlindex = list(index)
                i1 = np.clip(argmax-100,0, dc.shape[axis.index_in_array]-1)
                i2 = np.clip(argmax+100,0, dc.shape[axis.index_in_array]-1)
                nlindex.insert(axis.index_in_array, slice(i1, i2))
                toplot = dc[nlindex]
                if plot is True:
                    plt.figure()
                    plt.step(range(len(toplot)), toplot)
                    plt.title(str(index))
                i += 1
        return spikes
#                        
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
#    def jump_ratio(self, left_interval, right_interval):
#        """Returns the jump ratio in the given intervals
#        
#        Parameters
#        ----------
#        left_interval : tuple of floats
#            left interval in energy units
#        right_interval : tuple of floats
#            right interval in energy units
#            
#        Returns
#        -------
#        float
#        """
#        ilt1 = self.energy2index(left_interval[0])
#        ilt2 = self.energy2index(left_interval[1])
#        irt1 = self.energy2index(right_interval[0])
#        irt2 = self.energy2index(right_interval[1])
#        jump_ratio = (self.data_cube[irt1:irt2,:,:].sum(0) \
#        / self.data_cube[ilt1:ilt2,:,:].sum(0))
#        return jump_ratio
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
