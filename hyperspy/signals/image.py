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
import numpy.ma as ma

from hyperspy.signal import Signal
from hyperspy.misc import utils_varia
from hyperspy.misc.image_utils import (shift_image, hanning2d,
    sobel_filter, fft_correlation, estimate_image_shift)
from hyperspy import messages
from hyperspy.misc.progressbar import progressbar
from hyperspy.misc.utils import symmetrize, antisymmetrize




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
        
    def estimate_2D_translation(self, reference='current',
                                correlation_threshold=None,
                                chunk_size=30,
                                roi=None,
                                normalize_corr=False,
                                sobel=True,
                                medfilter=True,
                                hanning=True,
                                plot=False,
                                dtype='float',):
        """Estimate the shifts in a image using phase correlation

        This method can only estimate the shift by comparing 
        bidimensional features that should not change position 
        between frames. To decrease the memory usage, the time of 
        computation and the accuracy of the results it is convenient 
        to select a region of interest by setting the roi keyword.

        Parameters
        ----------
        
        reference : {'current', 'cascade' ,'stat'}
            If 'current' (default) the image at the current 
            coordinates is taken as reference. If 'cascade' each image
            is aligned with the previous one. If 'stat' the translation
            of every image with all the rest is estimated and by 
            performing statistical analysis on the result the 
            translation is estimated.
        correlation_threshold : {None, 'auto', float}
            This parameter is only relevant when `reference` is 'stat'.
            If float, the shift estimations with a maximum correlation
            value lower than the given value are not used to compute
            the estimated shifts. If 'auto' the threshold is calculated
            automatically as the minimum maximum correlation value
            of the automatically selected reference image.
        chunk_size: {None, int}
            If int and `reference`=='stat' the number of images used
            as reference are limited to the given value.
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
        dtype : str or dtype
            Typecode or data-type in which the calculations must be
            performed.
            
        Returns
        -------
        
        list of applied shifts
        
        Notes
        -----
        
        The statistical analysis approach to the translation estimation
        when using `reference`='stat' roughly follows [1]_ . If you use 
        it please cite their article.
        
        References
        ----------
        
        .. [1] Schaffer, Bernhard, Werner Grogger, and Gerald 
        Kothleitner. “Automated Spatial Drift Correction for EFTEM 
        Image Series.” 
        Ultramicroscopy 102, no. 1 (December 2004): 27–36.
        
        """

        axes = self.axes_manager.signal_axes
        ref = None if reference == 'cascade' else \
            self.__call__().copy()
        shifts = []
        nrows = None
        images_number = self.axes_manager._max_index + 1
        if reference == 'stat':
            nrows = images_number if chunk_size is None else \
                min(images_number, chunk_size)
            pcarray = ma.zeros((nrows, self.axes_manager._max_index + 1,
                                ),
                                dtype=np.dtype([('max_value', np.float),
                                                ('shift', np.int32,
                                                (2,))]))
            nshift, max_value = estimate_image_shift(
                          self(),
                          self(),
                          roi=roi,
                          sobel=sobel,
                          medfilter=medfilter,
                          hanning=hanning,
                          normalize_corr=normalize_corr,
                          plot=plot,
                          dtype=dtype)
            np.fill_diagonal(pcarray['max_value'], max_value)
            pbar = progressbar(maxval=nrows*images_number).start()
        else:
            pbar = progressbar(maxval=images_number).start()
            
            
        # Main iteration loop. Fills the rows of pcarray when reference 
        # is stat
        for i1, im in enumerate(self._iterate_signal(copy=False)):
            if reference in ['current', 'cascade']:
                if ref is None:
                    ref = im.copy()
                    shift = np.array([0,0])
                nshift, max_val = estimate_image_shift(ref,
                                      im,
                                      roi=roi,
                                      sobel=sobel,
                                      medfilter=medfilter,
                                      hanning=hanning,
                                      plot=plot,
                                      normalize_corr=normalize_corr,
                                      dtype=dtype)
                if reference == 'cascade':
                    shift += nshift
                    ref = im.copy()
                else:
                    shift = nshift
                shifts.append(shift.copy())
                pbar.update(i1+1)
            elif reference == 'stat':
                if i1 == nrows:
                    break
                # Iterate to fill the columns of pcarray
                for i2, im2 in enumerate(
                                    self._iterate_signal(copy=False)):
                    if i2 > i1:
                        nshift, max_value = estimate_image_shift(
                                      im,
                                      im2,
                                      roi=roi,
                                      sobel=sobel,
                                      medfilter=medfilter,
                                      hanning=hanning,
                                      normalize_corr=normalize_corr,
                                      plot=plot,
                                      dtype=dtype)
                        
                        pcarray[i1,i2] = max_value, nshift
                    del im2
                    pbar.update(i2 + images_number*i1 + 1)
                del im
        if reference == 'stat':
            # Select the reference image as the one that has the
            # higher max_value in the row
            sqpcarr = pcarray[:,:nrows]
            sqpcarr['max_value'][:] = symmetrize(sqpcarr['max_value'])
            sqpcarr['shift'][:] = antisymmetrize(sqpcarr['shift'])
            ref_index = np.argmax(pcarray['max_value'].min(1))
            self.ref_index = ref_index
            shifts = (pcarray['shift']  + 
                pcarray['shift'][ref_index,:nrows][:,np.newaxis])
            if correlation_threshold is not None:
                if correlation_threshold == 'auto':
                    correlation_threshold = \
                        (pcarray['max_value'].min(0)).max()
                    print("Correlation threshold = %1.2f" % 
                                correlation_threshold)
                shifts[pcarray['max_value'] < \
                    correlation_threshold] = ma.masked
                shifts.mask[ref_index,:] = False
                
            std_ = shifts.std(0)
            shifts = shifts.mean(0)
        else:
            shifts = np.array(shifts)
            del ref
        return shifts
        
    def align2D(self, crop=True, fill_value=np.nan, shifts=None,
                roi=None,
                sobel=True,
                medfilter=True,
                hanning=True,
                plot=False,
                normalize_corr=False,
                reference='current',
                dtype='float',
                correlation_threshold=None, 
                chunk_size=30):
        """Align the images using user provided shifts or by 
        estimating the shifts. 
        
        Please, see `estimate_2D_translation` docstring for details
        on the rest of the parameters not documented in the following
        section
        
        Parameters
        ----------
        crop : bool
            If True, the data will be cropped not to include regions
            with missing data
        fill_value : int, float, nan
            The areas with missing data are filled with the given value.
            Default is nan.
        shifts : None or list of tuples
            If None the shifts are estimated using 
            `estimate_2D_translation`.
            
        Returns
        -------
        shifts : np.array
            The shifts are returned only if `shifts` is None
            
        Notes
        -----
        
        The statistical analysis approach to the translation estimation
        when using `reference`='stat' roughly follows [1]_ . If you use 
        it please cite their article.
        
        References
        ----------
        
        .. [1] Schaffer, Bernhard, Werner Grogger, and Gerald 
        Kothleitner. “Automated Spatial Drift Correction for EFTEM 
        Image Series.” 
        Ultramicroscopy 102, no. 1 (December 2004): 27–36.
            
        """
        axes = self.axes_manager.signal_axes
        if shifts is None:
            shifts = self.estimate_2D_translation(
                roi=roi,sobel=sobel, medfilter=medfilter,
                hanning=hanning, plot=plot,reference=reference,
                dtype=dtype, correlation_threshold=
                correlation_threshold,
                normalize_corr=normalize_corr,
                chunk_size=chunk_size)
            return_shifts = True
        else:
            return_shifts = False
        # Translate with sub-pixel precision if necesary 
        for im, shift in zip(self._iterate_signal(copy=False),
                              shifts):
            if np.any(shift):
                shift_image(im, -shift,
                    fill_value=fill_value)
                del im
                    
        # Crop the image to the valid size
        if crop is True:
            shifts = -shifts
            bottom, top = (np.floor(shifts[:,0].min()) if 
                                    shifts[:,0].min() < 0 else None,
                           np.ceil(shifts[:,0].max()) if 
                                    shifts[:,0].max() > 0 else 0)
            right, left = (np.floor(shifts[:,1].min()) if 
                                    shifts[:,1].min() < 0 else None,
                           np.ceil(shifts[:,1].max()) if 
                                    shifts[:,1].max() > 0 else 0)
            self.crop_image(top, bottom, left, right)
            shifts = -shifts
        if return_shifts:
            return shifts
        
    def crop_image(self,top=None, bottom=None,
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
        vaxis, haxis = self.axes_manager.signal_axes
        self.crop_in_pixels(vaxis.index_in_array, top, bottom)
        self.crop_in_pixels(haxis.index_in_array, left, right)
        
          
        
