# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.


from hyperspy._components.expression import Expression


class HeavisideStep(Expression):

    r"""The Heaviside step function.

    Based on the corresponding `numpy function
    <https://numpy.org/doc/stable/reference/generated/numpy.heaviside.html>`_
    using the half maximum definition for the central point:

    .. math::

        f(x) =
        \begin{cases}
        0 & x<n\\
        A/2 & x=n\\
        A & x>n
        \end{cases}


    ============== =============
    Variable        Parameter
    ============== =============
    :math:`n`       centre
    :math:`A`       height
    ============== =============


    Parameters
    -----------
    n : float
        Location parameter defining the x position of the step.
    A : float
        Height parameter for x>n.
    **kwargs
        Extra keyword arguments are passed to the
        :py:class:`~._components.expression.Expression` component.
    """

    def __init__(self, A=1., n=0., module="numpy", compute_gradients=True,
                 **kwargs):
        super().__init__(
            expression="A*heaviside(x-n,0.5)",
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
