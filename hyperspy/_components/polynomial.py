# -*- coding: utf-8 -*-
# Copyright 2007-2011 The HyperSpy developers
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
from hyperspy.misc.utils import ordinal


class Polynomial(Component):

    """n-order polynomial component.

    Polynomial component defined by the coefficients parameters which is an
    array of len the order of the polynomial.

    For example, the [1,2,3] coefficients define the following 3rd order
    polynomial: f(x) = 1x² + 2x + 3

    Attributes
    ----------

    coeffcients : array

    """

    def __init__(self, order=2):
        Component.__init__(self, ['coefficients', ])
        self.coefficients._number_of_elements = order + 1
        self.coefficients.value = np.zeros((order + 1,))
        self.coefficients.grad = self.grad_coefficients

    def get_polynomial_order(self):
        return len(self.coefficients.value) - 1

    def function(self, x):
        return np.polyval(self.coefficients.value, x)

    def compute_grad_coefficients(self):
        to_mult = np.arange(self.get_polynomial_order(), -1, -1)
        coeff = self.coefficients.value * to_mult
        self.diff_coefficients = coeff[:-1]

    def grad_coefficients(self, x):
        self.compute_grad_coefficients()
        if self.diff_coefficients is not None:
            return np.polyval(self.diff_coefficients, x)
        else:
            return x * 0

    def __repr__(self):
        return u'Component <%s order polynomial>'.encode('utf8') % (
            ordinal(self.get_polynomial_order()))

    def estimate_parameters(self, signal, x1, x2, only_current=False):
        """Estimate the parameters by the two area method

        Parameters
        ----------
        signal : Signal instance
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

        """
        axis = signal.axes_manager.signal_axes[0]
        binned = signal.metadata.Signal.binned
        i1, i2 = axis.value_range_to_indices(x1, x2)
        if only_current is True:
            estimation = np.polyfit(axis.axis[i1:i2],
                                    signal()[i1:i2],
                                    self.get_polynomial_order())
            if binned is True:
                self.coefficients.value = estimation / axis.scale
            else:
                self.coefficients.value = estimation
            return True
        else:
            if self.coefficients.map is None:
                self._create_arrays()
            nav_shape = signal.axes_manager._navigation_shape_in_array
            signal.unfold()
            dc = signal.data
            # For polyfit the spectrum goes in the first axis
            if axis.index_in_array > 0:
                dc = np.rollaxis(dc, axis.index_in_array, 0)
            cmaps = np.polyfit(axis.axis[i1:i2],
                               dc[i1:i2, :], self.get_polynomial_order()).reshape([
                self.get_polynomial_order() + 1, ] + nav_shape)
            self.coefficients.map['values'][:] = np.rollaxis(cmaps, 0,
                                                             axis.index_in_array)
            if binned is True:
                self.coefficients.map["values"] /= axis.scale
            self.coefficients.map['is_set'][:] = True
            signal.fold()
            self.fetch_stored_values()
            return True
