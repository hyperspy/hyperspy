# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np
from hyperspy.component import Component
from hyperspy._components.gaussian import _estimate_gaussian_parameters
from hyperspy.misc.math.filter.snip import snip_method
from hyperspy.docstrings.parameters import FUNCTION_ND_DOCSTRING

class Snip(Component):

    """Snip background component to find background of a signal
       Uses an agressive filtering smoothing to approach to remove peaks and
       create a background.  Can be applied to any 1D data set with peaks.
       
    References
    ----------

    [1] C.G. Ryan etc, "SNIP, a statistics-sensitive background
           treatment for the quantitative analysis of PIXE spectra in
           geoscience applications", Nuclear Instruments and Methods in
           Physics Research Section B, vol. 34, 1998.       
       
    """

    def __init__(self, width=12,iterations=16):
        Component.__init__(self,('width','iterations',))
        self._whitelist['width'] = ('init', width)        
        self._whitelist['iterations'] = ('init', iterations)        
        
        self.width.free = False
        self.width.value = width

        self.iterations.free = False
        self.iterations.value = iterations

        self.isbackground = True
        self.convolved = False

    def function(self, x):
        return self._function(x, self.width.value,self.iterations.value)
    
    
    def get_snip_width(self):
        return self.width.value
    
    def get_snip_iterations(self):
        return self.iterations.value
    
    def _function(self,x,width,iterations):
        if self.model is None:
            # assume x is data
            component_array = snip_method(x,width=width,
                                            iterations=int(iterations))
        else:        
            # extract the data from the model
            data = self.model.signal().data
            component_array = snip_method(data,width=width,
                                            iterations=int(iterations))
            
        return component_array
    

    def estimate_parameters(self, signal, x1, x2, only_current=False):
        """Estimate the parameters by fitting a gaussian to a sub-region

        Parameters
        ----------
        signal : Signal1D instance
        x1 : float
            Defines the left limit of the spectral range to use for the
            estimation.
        x2 : float
            Defines the right limit of the spectral range to use for the
            estimation.

        only_current : bool
            If False estimates the parameters for the full dataset.

        Returns
        -------
        bool

        Notes
        -----

        Examples
        --------

        >>> g = hs.model.components1D.Snip()
        >>> x = np.arange(-10, 10, 0.01)
        >>> data = np.zeros((32, 32, 2000))
        >>> data[:] = g.function(x).reshape((1, 1, 2000))
        >>> s = hs.signals.Signal1D(data)
        >>> s.axes_manager._axes[-1].offset = -10
        >>> s.axes_manager._axes[-1].scale = 0.01
        >>> g.estimate_parameters(s, -10, 10, False)

        """
        super(Snip, self)._estimate_parameters(signal)
        centre, height, sigma = \
                 _estimate_gaussian_parameters(signal, x1, x2, only_current)
        # in case the gaussian fit goes wrong
        axis = signal.axes_manager.signal_axes[0]
        axis_size = axis.size
        axis_scale = axis.scale
        window = int((1.17*sigma)/axis_scale)
        if window < 1:
            window = 3
        if window > axis_size/2:
            window = int(axis_size/3)                 
        if only_current is True:
             self.width.value = window
             self.iterations.value = self.iterations.value
             return True
        else:
             if self.width.map is None:
                 self._create_arrays()
             self.width.map['values'][:] = window
             self.fetch_stored_values()
             return True


    def function_nd(self, axis):
        """%s

        """
        x = axis[np.newaxis, :]
        width = self.width.map["values"][..., np.newaxis]
        iterations = self.iterations.map["values"][..., np.newaxis]
        return self._function(x, width,iterations)

    function_nd.__doc__ %= FUNCTION_ND_DOCSTRING
   