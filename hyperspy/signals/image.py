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

import numpy as np
import scipy as sp
import scipy.ndimage
import scipy.signal
from scipy.fftpack import fftn, ifftn
import matplotlib.pyplot as plt

from hyperspy.signal import Signal

from hyperspy.misc import utils_varia
from hyperspy import messages

def hanning2d(M, N):
    """
    A 2D hanning window created by outer product.
    """
    return np.outer(np.hanning(M),np.hanning(N))


def sobel_filter(im):
    sx = sp.ndimage.sobel(im, axis=0, mode='constant')
    sy = sp.ndimage.sobel(im, axis=1, mode='constant')
    sob = np.hypot(sx, sy)
    return sob
    
def fft_phase_correlation(in1, in2):
    """Phase correlation of two N-dimensional arrays using FFT.
    
    Adapted from scipy's fftconvolve.

    """
    s1 = np.array(in1.shape)
    s2 = np.array(in2.shape)
    size = s1 + s2 - 1
    # Use 2**n-sized FFT
    fsize = 2 ** np.ceil(np.log2(size))
    IN1 = fftn(in1, fsize)
    IN1 *= fftn(in2, fsize).conjugate()
    fslice = tuple([slice(0, int(sz)) for sz in size])
    ret = ifftn(IN1/np.absolute(IN1)).real.copy()
    del IN1
    return ret

def estimate_image_shift(ref, image, roi=None, sobel=False,
                         medfilter=False, hanning=False, plot=False):
    # Make a copy of the image so the don't get modified
    ref = ref.copy()
    image = image.copy()
    if roi is not None:
            top, bottom, left, right = roi
    else:
        top, bottom, left, right = [None,] * 4
        
    # Select region of interest
    ref = ref[top:bottom, left:right]
    image = image[top:bottom, left:right]
    
    # Apply filters
    for im in (ref,image):
        if hanning is True:
            im *= hanning2d(*im.shape) 
        if medfilter is True:
            im[:] = sp.signal.medfilt(im)
        if sobel is True:
            im[:] = sobel_filter(im)
    
    
    phase_correlation = fft_phase_correlation(ref, image)
    
    # Estimate the shift by getting the coordinates of the maximum
    argmax = np.unravel_index(np.argmax(phase_correlation),
                              phase_correlation.shape)
    threshold = (phase_correlation.shape[0]/2 - 1,
                phase_correlation.shape[1]/2 - 1)
    shift0 = argmax[0] if argmax[0] < threshold[0] else  \
        argmax[0] - phase_correlation.shape[0] 
    shift1 = argmax[1] if argmax[1] < threshold[1] else \
        argmax[1] - phase_correlation.shape[1]
    
    # Plot on demand
    if plot is True:
        plt.matshow(ref)
        plt.matshow(image)
        plt.matshow(phase_correlation)
        plt.show()
    return shift0, shift1

class Image(Signal):
    """
    """    
    def __init__(self, *args, **kw):
        super(Image,self).__init__(*args, **kw)
        self.axes_manager.set_view('image')
                
    def to_spectrum(self):
        from hyperspy.signals.spectrum import Spectrum
        dic = self._get_signal_dict()
        dim = len(self.data.shape)
        dic['mapped_parameters']['record_by'] = 'spectrum'
        dic['data'] = np.rollaxis(dic['data'], 0, dim)
        dic['axes'] = utils_varia.rollelem(dic['axes'],0, dim)
        i = 0
        for axis in dic['axes']:
            axis['index_in_array'] = i
            i += 1
        sp = Spectrum(dic)
        sp.axes_manager._set_axes_index_in_array_from_position()
        if hasattr(self, 'learning_results'):
            sp.learning_results = copy.deepcopy(self.learning_results)
            sp.learning_results._transpose_results()
            sp.learning_results.original_shape = self.data.shape
        return sp
        
    def align2D(self, roi=None, sobel=False,
                medfilter=False, hanning=False, plot=False,):
        """Estimate the shifts in a image using phase correlation

        This method can only estimate the shift by comparing 
        bidimensional features that should not change the position 
        in the given axis. To decrease the memory usage, the time of 
        computation and the accuracy of the results it is convenient 
        to select a region of interest by setting the roi keyword.

        Parameters
        ----------
        
        roi : tuple of ints (top, bottom, left, right)
             Define the region of interest
        sobel : bool
            apply a sobel filter for edge enhancement 
        medfilter :  bool
            apply a median filter for noise reduction
        hanning : bool
            Apply a 2d hanning filter
        plot : bool
            If True plots the images after applying the filters and
            the phase correlation
            
        Returns
        -------
        
        list of applied shifts
        
        """

        axes = self.axes_manager._slicing_axes
        reference_indexes = [0,] * (len(self.axes_manager.axes)
            - len(axes))
        reference_indexes.insert(axes[0].index_in_array,
            slice(None))
        reference_indexes.insert(axes[1].index_in_array,
            slice(None))
        ref = self.data[reference_indexes]
        shifts = []
        for im in self._iterate_signal():
            shift = estimate_image_shift(ref, im, roi=roi,
                         sobel=sobel, medfilter=medfilter,
                         hanning=hanning, plot=plot)
            shifts.append(shift)
        shifts = np.array(shifts)
        ylimits = np.array((shifts[:,0].min() if shifts[:,0].min() < 0 
                            else 0, shifts[:,0].max()))
        xlimits = np.array((shifts[:,1].min() if shifts[:,1].min() < 0 
                            else 0, shifts[:,1].max()))

        def get_slices(shift):
            if shift > 0:
                    slice1 = slice(shift,None)
                    slice2 = slice(None, -shift)
            elif shift < 0:
                slice1 = slice(None, shift)
                slice2 = slice(-shift, None)
            else:
                slice1, slice2 = slice(None), slice(None)
            return slice1, slice2
                    
            
        for im, shift in zip(self._iterate_signal(), shifts.tolist()):
            if np.any(shift):
                slicey1, slicey2 = get_slices(shift[0])
                slicex1, slicex2 = get_slices(shift[1])
                im[slicey1, slicex1] = im[slicey2, slicex2]
        # Crop the image to the valid size
        top,bottom = ylimits[1], ref.shape[0] + ylimits[0]
        left,right = xlimits[1], ref.shape[1] + xlimits[0]
        self.crop2D_in_pixels(top, bottom, left, right)
        return shifts
        
    def crop2D_in_pixels(self,top=None, bottom=None,
                         left=None, right=None):
        """Crops an image according to the values given in pixels
        
        top : int
        bottom : int
        left : int
        right : int


        See also:
        ---------
        crop_in_units, crop_in_pixels
        
        """
        vaxis, haxis = self.axes_manager._slicing_axes
        self.crop_in_pixels(vaxis.index_in_array, top, bottom)
        self.crop_in_pixels(haxis.index_in_array, left, right)
        
          
        
