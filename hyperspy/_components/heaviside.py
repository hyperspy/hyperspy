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

from hyperspy._components.expression import Expression


class HeavisideStep(Expression):

    r"""The Heaviside step function.

    .. math::

        f(x) =
        \\begin{cases}
        0 & x<n\\\\
        A & x>=n
        \\end{cases}

    Parameters
    -----------
    n : float
        Location parameter defining the x position of the step.
    A : float
        Height parameter for x>=n.
    **kwargs
        Extra keyword arguments are passed to the ``Expression`` component.
    """

    def __init__(self, A=1., n=0., module="numpy", compute_gradients=False,
                 **kwargs):
        super(HeavisideStep, self).__init__(
            expression="where(x < n, 0, A)",
            name="HeavisideStep",
            A=A,
            n=n,
            position="n",
            module=module,
            autodoc=False,
            compute_gradients=compute_gradients,
            **kwargs)

        self.isbackground = True
        self.convolved = False

        # Gradients
        self.A.grad = self.grad_A

    def grad_A(self, x):
        x = np.asanyarray(x)
        return np.ones(x.shape)

    def grad_n(self, x):
        x = np.asanyarray(x)
        return np.where(x < self.n.value, 0, 1)
