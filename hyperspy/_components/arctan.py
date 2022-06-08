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


class Arctan(Expression):

    r"""Arctan function component.

    .. math::

        f(x) = A \cdot \arctan \left[ k \left( x-x_0 \right) \right]


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
        Amplitude parameter. :math:`\lim_{x\to -\infty}f(x)=-A` and
        :math:`\lim_{x\to\infty}f(x)=A`
    k : float
        Slope (steepness of the step). The larger :math:`k`, the sharper the
        step.
    x0 : float
        Center parameter (position of zero crossing :math:`f(x_0)=0`).

    """

    def __init__(self, A=1., k=1., x0=1., module=["numpy", "scipy"], **kwargs):
        # Not to break scripts once we remove the legacy Arctan
        if "minimum_at_zero" in kwargs:
            del kwargs["minimum_at_zero"]
        super().__init__(
            expression="A * arctan(k * (x - x0))",
            name="Arctan",
            A=A,
            k=k,
            x0=x0,
            position="x0",
            module=module,
            autodoc=False,
            **kwargs,
        )
