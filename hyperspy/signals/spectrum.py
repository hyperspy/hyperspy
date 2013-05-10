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

from hyperspy.signal import Signal
from hyperspy.misc import progressbar
from hyperspy.misc import utils_varia

from hyperspy.misc.utils import one_dim_findpeaks
            
class Spectrum(Signal):
    """
    """
    _default_record_by = 'spectrum'
    
    def __init__(self, *args, **kwargs):
        Signal.__init__(self, *args, **kwargs)
        self.axes_manager.set_signal_dimension(1)


    def peakfind_1D(self, xdim=None,slope_thresh=0.5, amp_thresh=None, 
                    subchannel=True, medfilt_radius=5, maxpeakn=30000, 
                    peakgroup=10):
        """Find peaks along a 1D line (peaks in spectrum/spectra).

        Function to locate the positive peaks in a noisy x-y data set.

        Detects peaks by looking for downward zero-crossings in the 
        first
        derivative that exceed 'slope_thresh'.

        Returns an array containing position, height, and width of each 
        peak.

        'slope_thresh' and 'amp_thresh', control sensitivity: higher 
        values will
        neglect smaller features.

        peakgroup is the number of points around the top peak to search 
        around

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
            
        Raises
        ------
        SignalDimensionError if the signal dimension is not 1.
            
        """
        # TODO: generalize the code
        self._check_signal_dimension_equals_one()
        if len(self.data.shape)==1:
            # preallocate a large array for the results
            self.peaks=one_dim_findpeaks(self.data,
                slope_thresh=slope_thresh,
                amp_thresh=amp_thresh,
                medfilt_radius=medfilt_radius,
                maxpeakn=maxpeakn,
                peakgroup=peakgroup,
                subchannel=subchannel)
                
        elif len(self.data.shape)==2:
            pbar=progressbar.ProgressBar(
                    maxval=self.data.shape[1]).start()
            # preallocate a large array for the results
            # the third dimension is for the number of rows in your 
            # data.
            # assumes spectra are rows of the 2D array, each col is a 
            # channel.
            self.peaks=np.zeros((maxpeakn,3,self.data.shape[0]))
            for i in xrange(self.data.shape[1]):
                tmp=one_dim_findpeaks(
                                    self.data[i],
                                    slope_thresh=slope_thresh,
                                    amp_thresh=amp_thresh,
                                    medfilt_radius=medfilt_radius,
                                    maxpeakn=maxpeakn,
                                    peakgroup=peakgroup,
                                    subchannel=subchannel)
                self.peaks[:tmp.shape[0],:,i]=tmp
                pbar.update(i + 1)
            # trim any extra blank space
            # works by summing along axes to compress to a 1D line, then
            # finds
            # the first 0 along that line.
            trim_id=np.min(np.nonzero(np.sum(np.sum(self.peaks,axis=2),
                                             axis=1)==0))
            pbar.finish()
            self.peaks=self.peaks[:trim_id,:,:]
        elif len(self.data.shape)==3:
            # preallocate a large array for the results
            self.peaks=np.zeros((maxpeakn,3,self.data.shape[0],
                                 self.data.shape[1]))
            pbar=progressbar.ProgressBar(
                maxval=self.data.shape[0]).start()
            for i in xrange(self.data.shape[0]):
                for j in xrange(self.data.shape[1]):
                    tmp=one_dim_findpeaks(self.data[i,j],
                            slope_thresh=slope_thresh,
                            amp_thresh=amp_thresh,
                            medfilt_radius=medfilt_radius,
                            maxpeakn=maxpeakn,
                            peakgroup=peakgroup,
                            subchannel=subchannel)
                    self.peaks[:tmp.shape[0],:,i,j]=tmp
                pbar.update(i + 1)
            # trim any extra blank space
            # works by summing along axes to compress to a 1D line, 
            # then finds
            # the first 0 along that line.
            trim_id=np.min(np.nonzero(np.sum(np.sum(np.sum(
                            self.peaks,axis=3),axis=2),axis=1)==0))
            pbar.finish()
            self.peaks=self.peaks[:trim_id,:,:,:]

    def to_image(self, signal_to_index=0):
        """Spectrum to image

        Parameters
        ----------
        signal_to_index : integer
            Position to move the signal axis.        
            
        Examples
        --------        
        >>> s = signals.Spectrum({'data' : np.ones((3,4,5,6))})
        >>> s
        <Spectrum, title: , dimensions: (3L, 4L, 5L, 6L)>

        >>> s.to_image()
        <Image, title: , dimensions: (6L, 3L, 4L, 5L)>

        >>> s.to_image(1)
        <Image, title: , dimensions: (3L, 6L, 4L, 5L)>
        
        """
        from hyperspy.signals.image import Image
        dic = self._get_signal_dict()
        dic['mapped_parameters']['record_by'] = 'image'
        dic['data'] = np.rollaxis(dic['data'], -1, signal_to_index)
        dic['axes'] = utils_varia.rollelem(dic['axes'],-1,signal_to_index)
        im = Image(dic)
        
        if hasattr(self, 'learning_results'):
            if (signal_to_index != 0 and 
                self.learning_results.loadings is not None):
                print("The learning results won't be transfered correctly")
            else:
                im.learning_results = copy.deepcopy(
                    self.learning_results)
                im.learning_results._transpose_results()
                im.learning_results.original_shape = self.data.shape

        im.tmp_parameters = self.tmp_parameters.deepcopy()
        return im

    def to_EELS(self):
        from hyperspy.signals.eels import EELSSpectrum
        dic = self._get_signal_dict()
        dic['mapped_parameters']['signal_type'] = 'EELS'
        eels = EELSSpectrum(dic)
        if hasattr(self, 'learning_results'):
            eels.learning_results = copy.deepcopy(self.learning_results)
        eels.tmp_parameters = self.tmp_parameters.deepcopy()
        return eels
        
    def to_simulation(self):
        from hyperspy.signals.spectrum_simulation import (
                SpectrumSimulation)
        dic = self._get_signal_dict()
        signal_type = self.mapped_parameters.signal_type
        dic['mapped_parameters']['signal_type'] = \
            signal_type + '_simulation'
        simu = SpectrumSimulation(dic)
        if hasattr(self, 'learning_results'):
            simu.learning_results = copy.deepcopy(self.learning_results)
        simu.tmp_parameters = self.tmp_parameters.deepcopy()
        return simu