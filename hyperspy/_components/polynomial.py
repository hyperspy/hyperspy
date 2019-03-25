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

from hyperspy._components.expression import Expression
from hyperspy.misc.utils import ordinal


class Polynomial(Expression):

    """n-order polynomial component.

    Polynomial component consisting of order + 1 parameters.
    The parameters are named "a" followed by the corresponding order, 
    i.e.
    
    .. math::

        f(x) = a_{2} x^{2} + a_{1} x^{1} + a_{0}

    Zero padding is used for polynomial of order > 10.

    Parameters
    ----------
    order : int
        Order of the polynomial.
    **kwargs
        Keyword arguments can be used to initialise the value of the
        parameters, i.e. a2=2, a1=3, a0=1.

    """

    def __init__(self, order=2, module="numexpr", **kwargs):
        coeff_list = ['{}'.format(o).zfill(len(list(str(order)))) for o in
                      range(order, -1, -1)]
        expr = "+".join(["a{}*x**{}".format(c, o) for c, o in 
                         zip(coeff_list, range(order, -1, -1))])
        name = "{} order Polynomial".format(ordinal(order))
        super(Polynomial, self).__init__(
            expression=expr, 
            name=name,
            module=module,
            autodoc=False,
            **kwargs)
    
    def get_polynomial_order(self):
        return len(self.parameters) - 1

    def estimate_parameters(self, signal, x1, x2, only_current=False):
        """Estimate the parameters by the two area method

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

        """
        super(Polynomial, self)._estimate_parameters(signal)
        axis = signal.axes_manager.signal_axes[0]
        i1, i2 = axis.value_range_to_indices(x1, x2)
        if only_current is True:
            estimation = np.polyfit(axis.axis[i1:i2],
                                    signal()[i1:i2],
                                    self.get_polynomial_order())
            if self.binned:
                for para, estim in zip(self.parameters[::-1], estimation):
                    para.value = estim / axis.scale
            else:
                for para, estim in zip(self.parameters[::-1], estimation):
                    para.value = estim
            return True
        else:
            if self.a0.map is None:
                self._create_arrays()
            nav_shape = signal.axes_manager._navigation_shape_in_array
            with signal.unfolded():
                data = signal.data
                # For polyfit the spectrum goes in the first axis
                if axis.index_in_array > 0:
                    data = data.T             # Unfolded, so simply transpose
                fit = np.polyfit(axis.axis[i1:i2], data[i1:i2, ...],
                                   self.get_polynomial_order())
                if axis.index_in_array > 0:
                    fit = fit.T       # Transpose back if needed
                # Shape needed to fit coefficients.map:

                cmap_shape = nav_shape + (self.get_polynomial_order() + 1, )
                fit = fit.reshape(cmap_shape)

                if self.binned:
                    for para, i in zip(self.parameters[::-1], range(fit.shape[-1])):
                        para.map['values'][:] = fit[...,i] / axis.scale
                        para.map['is_set'][:] = True
                else:
                    for para, i in zip(self.parameters[::-1], range(fit.shape[-1])):
                        para.map['values'][:] = fit[...,i]
                        para.map['is_set'][:] = True
            self.fetch_stored_values()
            return True
