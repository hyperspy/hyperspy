# -*- coding: utf-8 -*-
# Copyright 2007-2015 The HyperSpy developers
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


from hyperspy.component import Component
from scipy.interpolate import interp2d


class ScalableFixedPattern2D(Component):
    #TODO: updated docstring
    """Fixed pattern component with interpolation support.

        f(x,y) = a*s(b*x-x0) + c

    +------------+-----------+
    | Parameter  | Attribute |
    +------------+-----------+
    +------------+-----------+
    |     a      |  yscale   |
    +------------+-----------+
    |     b      |  xscale   |
    +------------+-----------+
    |    x0      |  shift    |
    +------------+-----------+


    The fixed two-dimensional pattern is defined by a single image which must 
    be passed to the ScalableFixedPattern2D constructor, e.g.:

    .. code-block:: ipython

        In [1]: s = load('my_spectrum.hdf5')
        In [2] : my_fixed_pattern = components.ScalableFixedPattern(s))

    Attributes
    ----------

    yscale, xscale, shift : Float
    interpolate : Bool
        If False no interpolation is performed and only a y-scaled spectrum is
        returned.

    Methods
    -------

    prepare_interpolator : method to fine tune the interpolation

    """

    def __init__(self, spectrum):

        Component.__init__(self, ['zscale', 'xscale', 'yscale', 'xshift', 'yshift'])

        self._position = self.shift
        self.image = image
        self.zscale.free = True
        self.zscale.value = 1.
        self.xscale.value = 1.
        self.yscale.value = 1.
        self.xshift.value = 0.
        self.yshift.value = 0.

        self.prepare_interpolator()
        # Options
        self.isbackground = True
        self.convolved = False
        self.interpolate = True

    def prepare_interpolator(self, kind='cubic', fill_value=0, **kwargs):
        """Prepare interpolation.

        Parameters
        ----------
        x : array
            The spectral axis of the fixed pattern
        kind: str or int, optional
            Specifies the kind of interpolation as a string
            ('linear','nearest', 'zero', 'slinear', 'quadratic, 'cubic')
            or as an integer specifying the order of the spline interpolator
            to use. Default is 'linear'.

        fill_value : float, optional
            If provided, then this value will be used to fill in for requested
            points outside of the data range. If not provided, then the default
            is NaN.

        Notes
        -----
        Any extra keyword argument is passed to `scipy.interpolate.interp2d`

        """

        self.f = interp2d(
            self.image.axes_manager.signal_axes[0].axis,
            self.image.axes_manager.signal_axes[1].axis,
            self.image.data.squeeze(),
            kind=kind,
            bounds_error=False,
            fill_value=fill_value,
            **kwargs)

    def function(self, x, y):
        if self.interpolate is True:
            result = self.zscale.value * self.f(
                x * self.xscale.value - self.xshift.value,
                y * self.yscale.value - self.yshift.value)
        else:
            result = self.zscale.value * self.image.data
        return result
