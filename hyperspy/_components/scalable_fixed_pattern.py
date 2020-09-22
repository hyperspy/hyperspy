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


from scipy.interpolate import interp1d
from hyperspy.component import Component


class ScalableFixedPattern(Component):

    """Fixed pattern component with interpolation support.

        f(x) = a*s(b*x-x0) + c

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


    The fixed pattern is defined by a single spectrum which must be provided to
    the ScalableFixedPattern constructor, e.g.:

    .. code-block:: ipython

        In [1]: s = load('my_spectrum.hdf5')
        In [2]: my_fixed_pattern = components.ScalableFixedPattern(s))

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

    def __init__(self, signal1D):

        Component.__init__(self, ['yscale', 'xscale', 'shift'])

        self._position = self.shift
        self._whitelist['signal1D'] = ('init,sig', signal1D)
        self.signal = signal1D
        self.yscale.free = True
        self.yscale.value = 1.
        self.xscale.value = 1.
        self.shift.value = 0.

        self.prepare_interpolator()
        # Options
        self.isbackground = True
        self.convolved = False
        self.interpolate = True

    def prepare_interpolator(self, kind='linear', fill_value=0, **kwargs):
        """Prepare interpolation.

        Parameters
        ----------
        x : array
            The spectral axis of the fixed pattern
        kind : str or int, optional
            Specifies the kind of interpolation as a string
            ('linear', 'nearest', 'zero', 'slinear', 'quadratic, 'cubic')
            or as an integer specifying the order of the spline interpolator
            to use. Default is 'linear'.

        fill_value : float, optional
            If provided, then this value will be used to fill in for requested
            points outside of the data range. If not provided, then the default
            is NaN.

        Notes
        -----
        Any extra keyword argument is passed to `scipy.interpolate.interp1d`

        """

        self.f = interp1d(
            self.signal.axes_manager.signal_axes[0].axis,
            self.signal.data.squeeze(),
            kind=kind,
            bounds_error=False,
            fill_value=fill_value,
            **kwargs)

    def function(self, x):
        if self.interpolate is True:
            result = self.yscale.value * self.f(
                x * self.xscale.value - self.shift.value)
        else:
            result = self.yscale.value * self.signal.data
        if self.signal.metadata.Signal.binned is True:
            return result / self.signal.axes_manager.signal_axes[0].scale
        else:
            return result

    def grad_yscale(self, x):
        return self.function(x) / self.yscale.value

    def notebook_interaction(self, display=True):
        from ipywidgets import Checkbox
        from traitlets import TraitError as TraitletError
        from IPython.display import display as ip_display

        try:
            container = super(ScalableFixedPattern,
                              self).notebook_interaction(display=False)
            interpolate = Checkbox(description='interpolate',
                                   value=self.interpolate)

            def on_interpolate_change(change):
                self.interpolate = change['new']

            interpolate.observe(on_interpolate_change, names='value')

            container.children = (container.children[0], interpolate) + \
                container.children[1:]

            if not display:
                return container
            ip_display(container)
        except TraitletError:
            if display:
                print('This function is only avialable when running in a'
                      ' notebook')
            else:
                raise
    notebook_interaction.__doc__ = Component.notebook_interaction.__doc__
