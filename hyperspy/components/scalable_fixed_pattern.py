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


from hyperspy.component import Component
from scipy.interpolate import interp1d

class ScalableFixedPattern(Component):
    """Fixed pattern component with interpolation support.
    
        f(x) = A*p(x-x0)
    
    +------------+-----------+
    | Parameter  | Attribute |
    +------------+-----------+
    +------------+-----------+
    |     A      | intensity |
    +------------+-----------+
    |    x0      |  origin   |
    +------------+-----------+
    
    
    The fixed pattern is defined by an array which must be provided to the 
    FixedPattern constructor, e.g.:
    
    .. code-block:: ipython

        In [1]: my_fixed_pattern = components.FixedPattern(np.array([1,2,3,4,5,6,7,8]))
    
    The array must have the same spectral dimension as the data that is being
    analysed if interpolation is not used. When interpolation is not used 
    the origin parameter is always fixed and its value is zero.
    
    To enable interpolation use the :py:meth:`prepare_interpolator`
    method and set the :py:attr:`interpolate` attribute to True, e.g.:
    
    .. code-block:: ipython
    
        In [2]: # First provide the spectral axis of the fixed pattern
        In [3]: my_fixed_pattern.prepare_interpolator(
                        np.array((0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08])))
        In [4]: # Then enable interpolation
        In [5]: my_fixed_pattern.interpolate = True
    
    See Also
    --------
    ScalableFixedPattern : another component which permit 
        "stretching" the fixed pattern in the spectral dimension.
    """

    def __init__(self, spectrum):
    
        Component.__init__(self, ['intensity', 'xscale', 'origin', 'offset'])
        self.name = 'ScalableFixedPattern'
        self.spectrum = spectrum
        self.intensity.free = True
        self.intensity.value = 1.
        self.xscale.value = 1.
        self.offset.value = 0.
        self.origin.value = 0.
        # Options
        self.isbackground = True
        self.convolved = False
#        self.intensity.grad = self.grad_intensity
        self.f = interp1d(spectrum.energy_axis, spectrum.data_cube.squeeze(), 
        bounds_error = False, fill_value = 0.)
        
    def function(self, x):
        return self.offset.value + self.intensity.value * self.f(
                                    x * self.xscale.value - self.origin.value)
    
