# -*- coding: utf-8 -*-
# Copyright 2007-2023 The exspy developers
#
# This file is part of exspy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# exspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with exspy. If not, see <https://www.gnu.org/licenses/#GPL>.


from hyperspy._components.expression import Expression


class EELSArctan(Expression):

    r"""Arctan function component for EELS (with minimum at zero).

    .. math::

        f(x) = A \cdot \left( \frac{\pi}{2} +
               \arctan \left[ k \left( x-x_0 \right) \right] \right)


    ============ =============
    Variable      Parameter
    ============ =============
    :math:`A`     A
    :math:`k`     k
    :math:`x_0`   x0
    ============ =============


    Parameters
    -----------
    A : float
        Amplitude parameter. :math:`\lim_{x\to -\infty}f(x)=0` and
        :math:`\lim_{x\to\infty}f(x)=2A`
    k : float
        Slope (steepness of the step). The larger :math:`k`, the sharper the
        step.
    x0 : float
        Center parameter (:math:`f(x_0)=A`).

    """

    def __init__(self, A=1., k=1., x0=1., module=["numpy", "scipy"], **kwargs):
        # To be able to still read old file versions that contain this argument
        if "minimum_at_zero" in kwargs:
            del kwargs["minimum_at_zero"]
        super().__init__(
            expression="A * (pi /2 + arctan(k * (x - x0)))",
            name="EELSArctan",
            A=A,
            k=k,
            x0=x0,
            position="x0",
            module=module,
            autodoc=False,
            **kwargs,
        )

        self.isbackground = False
        self.convolved = True
