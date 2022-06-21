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
from packaging.version import Version
import sympy


class Erf(Expression):

    r"""Error function component.

    .. math::

        f(x) = \frac{A}{2}~\mathrm{erf}\left[\frac{(x - x_0)}{\sqrt{2}
            \sigma}\right]


    ============== =============
    Variable        Parameter
    ============== =============
    :math:`A`       A
    :math:`\sigma`  sigma
    :math:`x_0`     origin
    ============== =============

    Parameters
    ----------
    A : float
        The min/max values of the distribution are -A/2 and A/2.
    sigma : float
        Width of the distribution.
    origin : float
        Position of the zero crossing.
    **kwargs
        Extra keyword arguments are passed to the
        :py:class:`~._components.expression.Expression` component.
    """

    def __init__(self, A=1., sigma=1., origin=0., module=["numpy", "scipy"],
                 **kwargs):
        if Version(sympy.__version__) < Version("1.3"):
            raise ImportError("The `ErrorFunction` component requires "
                              "SymPy >= 1.3")
        super().__init__(
            expression="A * erf((x - origin) / sqrt(2) / sigma) / 2",
            name="Erf",
            A=A,
            sigma=sigma,
            origin=origin,
            module=module,
            autodoc=False,
            **kwargs,
        )

        # Boundaries
        self.A.bmin = 0.

        self.isbackground = False
        self.convolved = True
