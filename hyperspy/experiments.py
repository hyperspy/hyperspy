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
import sys

import numpy as np

import matplotlib.pyplot as plt
from hyperspy.misc.utils import generate_axis, check_cube_dimensions, check_energy_dimensions
from hyperspy.signals.spectrum import Spectrum
from hyperspy.misc.utils import estimate_drift
from hyperspy.io import load
from hyperspy.misc.progressbar import progressbar

class Experiments:
    def __init__(self, hl, ll=None):
        if isinstance(hl, Spectrum):
            self.hl = hl
        else:
            print "\nLoading the high loss spectrum..."
            self.hl = load(hl)
        if ll is not None:
            if isinstance(ll, Spectrum):
                self.ll = ll
            else:
                print "\nLoading the low loss spectrum..."
                self.ll = load(ll)
        else:
            self.ll = None
        self.shape = self.hl.data_cube.shape
        self.convolution_axis = None
    def set_convolution_axis(self):
        """
        Creates an axis to use to generate the data of the model in the precise
        scale to obtain the correct axis and origin after convolution with the
        lowloss spectrum.
        Produces a warning if the powerlaw extension is unsafe.
        For multispectrum you must align the lowloss spectra before using this
        function.
        """
        dimension = self.hl.energydimension+self.ll.energydimension-1
        paso = self.hl.energyscale
        #TODO Warning if tries to run the function with ll not aligned
        knot_position = self.ll.energydimension-np.argmax(
        self.ll.data_cube[:,0,0])-1
        self.convolution_axis=generate_axis(self.hl.energyorigin, paso, 
        dimension, knot_position)
    
    def treat_low_loss(self, sync_with_cl = False, energy_step = None):
        if not sync_with_cl:
            self.ll.find_low_loss_origin()
            self.ll.align((-3,3))
            self.ll.find_low_loss_origin()
        else:
            if energy_step:
                delta = self.hl.energyorigin - self.ll.energyorigin
                self.ll.set_new_calibration(0, energy_step)
                self.hl.set_new_calibration(delta, energy_step)
            self.ll.align(sync_SI = self.hl)
            self.ll.find_low_loss_origin(sync_SI = self.hl)

        self.ll.baseline()
        self.ll.normalize()
        self.ll.calculate_thickness()
        self.__set_convolution_axis()

    def add_elements(self, elements):
        if self.hl is not None:
            print "\nAdding subshells to the high loss spim..."
            self.hl.add_elements(elements)
        if self.ll is not None:
            print "\nAdding subshells to the low loss spim..."
            self.ll.add_elements(elements)
            
    def correct_intensity_loss(self, plot = False):
        print "Calculating the beam intensity drift correction"
        self.intensity_correction = self.hl.HADF_array / self.ll.HADF_array
        if plot == True:
            plt.figure()
            plt.plot(np.ravel(np.transpose(self.intensity_correction)))
            plt.title('Intensity drift correction')
            plt.xlabel('Pixel')
            plt.ylabel('Correction factor')

    def fourier_ratio_deconvolution(self, fwhm = None):
        """Performs Fourier-ratio deconvolution
        
        The kernel is eds.hl and the psf is defined in eds.psf.
        
        Parameters
        ----------
        fwhm : float or None
            If None, the zero loss stored in eds.ll.zero_loss is used if it 
            exists, otherwise it is calculated using the extract_zero_loss 
            Spectrum method. Otherwise if float, a gaussian of the given FWHM 
            will be used.
        """
        
        if fwhm is None:
            if self.ll.zero_loss is None:
                self.ll.extract_zero_loss()
            zl = self.ll.zero_loss
        else:
            from components.gaussian import Gaussian
            g = Gaussian()
            g.sigma.value = fwhm / 2.3548
            g.A.value = 1
            g.origin.value = 0
            zl = np.zeros(self.ll.data_cube.shape)
            zl[:] = g.function(self.ll.energy_axis).reshape(-1,1,1)
        z = np.fft.fft(zl, axis=0)
        jk = np.fft.fft(self.hl.data_cube, axis=0)
        jl = np.fft.fft(self.ll.data_cube, axis=0)
        self.hl._Spectrum__new_cube(np.fft.ifft(z*jk/jl, axis=0).real, 
        'fourier-ratio deconvolution')
            
    def richardson_lucy_deconvolution(self, iterations = 15, 
    to_dec = 'self.hl', kernel = 'self.psf', mask = None):
        """
    Performs 1D Richardson-Lucy Poissonian deconvolution of the to_dec by
    the psf. The deconvolved spim will be stored in the to_dec as a new cube.
    
    Parameters:
    -----------
    iterations: Number of iterations of the deconvolution. Note that increasing
    the value will increase the noise amplification.
    
    to_dec: a spectrum object.
    psf: a 1D spectrum object containing the Point Spread Function
        """
        if to_dec == 'self.hl':
            to_dec = self.hl
        if kernel == 'self.psf':
            kernel = self.psf
        if check_cube_dimensions(to_dec, kernel, warn = False):
            kernel_data = kernel.data_cube
        elif check_energy_dimensions(to_dec, kernel, warn = False):
            shape = to_dec.data_cube.shape
            kernel_data = np.zeros(shape).swapaxes(0,2)
            kernel_data[...] = kernel.data_cube[:,0,0]
            kernel_data = kernel_data.swapaxes(0,2)
        else:
            print """
    The kernel must have either the same dimensions as the to_dec or be an
    unidimensional spectrum with the same number of channels"""
            sys.exit()
        to_dec_data = copy.copy(to_dec.data_cube)
        length = kernel_data.shape[0]
        print "\nPerfoming Richardson-Lucy iterative deconvolution"
        dcshape = to_dec.data_cube.shape
        pbar = progressbar(maxval = (dcshape[1] * dcshape[2]))
        index = 0
        for ix in xrange(to_dec.data_cube.shape[1]):
            for iy in xrange(to_dec.data_cube.shape[2]):
                if mask is None or mask[ix,iy]:
#                    print "\n(%s, %s)" % (ix, iy)
                    kernel_ = kernel_data[:, ix, iy]
                    imax = kernel_.argmax()
                    mimax = length -1 - imax
                    D = to_dec.data_cube[:, ix, iy]
                    O = copy.copy(D)
                    for i in xrange(iterations):
                        first = np.convolve(kernel_,O)[imax: imax + length]
                        O = O * (np.convolve(kernel_[::-1], 
                        D / first)[mimax: mimax + length])
                    to_dec_data[:, ix, iy] = O
                index += 1
                pbar.update(index)
        pbar.finish()
        to_dec._Spectrum__new_cube(to_dec_data, 
        'poissonian R-L deconvolution %i iterations' % iterations)
        
    def correct_spatial_drift(self):
        """Corrects the spatial drift between the CL and LL. It estimates 
        the drift by cross-correlation and crops the ll and cl SI and images to 
        the overlapping area
        """
        
        if not self.hl.HADF or not self.ll.HADF:
            print "To correct the spatial drift, an (hadf) image of hl and ll \
            must be provided"
            sys.exit()
        drift = estimate_drift(self.hl.HADF, self.ll.HADF)
        if drift[0] == 0 and drift[1] == 0:
            return
        if drift[0] > 0:
            lldx = [drift[0], None]
            hldx = [None, -drift[0]]
        elif drift[0] == 0:
            lldx = [None, None]
            hldx = [None, None]
        else:
            lldx = [None, drift[0]]
            hldx = [-drift[0], None]
        
        if drift[1] > 0:
            lldy = [drift[1], None]
            hldy = [None, -drift[1]]
        elif drift[1] == 0:
            lldy = [None, None]
            hldy = [None, None]
        else:
            lldy = [None, drift[1]]
            hldy = [-drift[1], None]
        
        ix1, ix2, iy1, iy2 = hldx + hldy
        self.hl.spatial_crop(ix1, iy1, ix2, iy2)
        ix1, ix2, iy1, iy2 = lldx + lldy
        self.ll.spatial_crop(ix1, iy1, ix2, iy2)
        
    def rebin(self, new_shape):
        self.hl.rebin(new_shape)
        self.ll.rebin(new_shape)
        
    def spatial_crop(self, ix1, iy1, ix2, iy2):
        self.hl.spatial_crop(ix1, iy1, ix2, iy2)
        self.ll.spatial_crop(ix1, iy1, ix2, iy2)
        
    def energy_crop(self,from_channel = None,to_channel = None
    ,in_energy = False):
        if in_energy:
            from_channel = self.hl.energy2index(from_channel)
            to_channel = self.hl.energy2index(to_channel)
        self.hl.energy_crop(from_channel, to_channel, False)
        if from_channel == None:
            from_channel = 0
        if to_channel == None:
            to_channel = self.ll.data_cube.shape[0] - 1
        inc = self.ll.data_cube.shape[0] - (to_channel - from_channel)
        self.ll.energy_crop(None,-inc,False)
        self.__set_convolution_axis()

    def correct_bad_pixels(self, channels_list):
        """Substitutes the offending channels values by the average of the 
        adjencent channels"""
        
        self.hl.correct_bad_pixels(channels_list)
        self.ll.correct_bad_pixels(channels_list)
        
    def plot(self):
        self.hl.plot()
        self.ll.plot()
       
