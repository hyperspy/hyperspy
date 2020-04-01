# -*- coding: utf-8 -*-
# Copyright 2007-2020 The HyperSpy developers
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

from hyperspy.docstrings.parameters import FUNCTION_ND_DOCSTRING
from hyperspy._components.expression import Expression


class DoublePowerLaw(Expression):

    r"""Double power law component for EELS spectra.

    .. math::

        f(x) = A \cdot [s_r \cdot (x - x_0 - x_s)^{-r} + (x - x_0)^{-r}]

    ============= =============
     Variable      Parameter
    ============= =============
     :math:`A`     A
     :math:`r`     r
     :math:`x_0`   origin
     :math:`x_s`   shift
     :math:`s_r`   ratio
    ============= =============

    Parameters
    ----------
    A : float
        Height parameter.
    r : float
        Power law coefficient.
    origin : float
        Location parameter.
    shift : float
        Offset of second power law.
    ratio : float
        Height ratio of the two power law components.
    **kwargs
        Extra keyword arguments are passed to the ``Expression`` component.

    The `left_cutoff` parameter can be used to set a lower threshold from which
    the component will return 0.
    """

    def __init__(self, A=1e-5, r=3., origin=0., shift=20., ratio=1., 
                 left_cutoff=0.0, module="numexpr", compute_gradients=False, 
                 **kwargs):
        super(DoublePowerLaw, self).__init__(
            expression="where(x > left_cutoff, \
                        A * (ratio * (x - origin - shift) ** -r \
                        + (x - origin) ** -r), 0)",
            name="DoublePowerLaw",
            A=A,
            r=r,
            origin=origin,
            shift=shift,
            ratio=ratio,
            left_cutoff=left_cutoff,
            position="origin",
            autodoc=False,
            module=module,
            compute_gradients=compute_gradients,
            **kwargs,
        )

        self.origin.free = False
        self.shift.value = 20.
        self.shift.free = False

        # Boundaries
        self.A.bmin = 0.
        self.A.bmax = None
        self.r.bmin = 1.
        self.r.bmax = 5.

        self.isbackground = True
        self.convolved = False

    def function_nd(self, axis):
        """%s

        """
        return super().function_nd(axis)

    function_nd.__doc__ %= FUNCTION_ND_DOCSTRING

    # Define gradients
    def grad_A(self, x):
        return self.function(x) / self.A.value

    def grad_r(self, x):
        return np.where(x > self.left_cutoff.value, -self.A.value * 
                        self.ratio.value * (x - self.origin.value -
                        self.shift.value) ** (-self.r.value) *
                        np.log(x - self.origin.value - self.shift.value) -
                        self.A.value * (x - self.origin.value) ** 
                        (-self.r.value) * np.log(x - self.origin.value), 0)

    def grad_origin(self, x):
        return np.where(x > self.left_cutoff.value, self.A.value * self.r.value
                        * self.ratio.value * (x - self.origin.value - self.shift.value) 
                        ** (-self.r.value - 1) + self.A.value * self.r.value
                        * (x - self.origin.value) ** (-self.r.value - 1), 0)

    def grad_shift(self, x):
        return np.where(x > self.left_cutoff.value, self.A.value * self.r.value
                        * self.ratio.value * (x - self.origin.value - 
                        self.shift.value) ** (-self.r.value - 1), 0)

    def grad_ratio(self, x):
        return np.where(x > self.left_cutoff.value, self.A.value *
                        (x - self.origin.value - self.shift.value) ** 
                        (-self.r.value), 0)
