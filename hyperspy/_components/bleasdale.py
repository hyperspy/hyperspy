# -*- coding: utf-8 -*-
# Copyright 2007-2024 The HyperSpy developers
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

import numpy as np

from hyperspy._components.expression import Expression


class Bleasdale(Expression):
    r"""Bleasdale function component.

    Also called the Bleasdale-Nelder function. Originates from
    the description of the yield-density relationship in crop growth.

    .. math::

        f(x) = \left(a+b\cdot x\right)^{-1/c}

    Parameters
    ----------
    a : float, default=1.0
        The value of Parameter a.
    b : float, default=1.0
        The value of Parameter b.
    c : float, default=1.0
        The value of Parameter c.
    **kwargs
        Extra keyword arguments are passed to
        :class:`~.api.model.components1D.Expression`.

    Notes
    -----
    For :math:`(a+b\cdot x)\leq0`, the component will be set to 0.

    """

    def __init__(self, a=1.0, b=1.0, c=1.0, module=None, **kwargs):
        super().__init__(
            expression="where((a + b * x) > 0, pow(a + b * x, -1 / c), 0)",
            name="Bleasdale",
            a=a,
            b=b,
            c=c,
            module=module,
            autodoc=False,
            compute_gradients=False,
            linear_parameter_list=["b"],
            check_parameter_linearity=False,
            **kwargs,
        )
        module = self._whitelist["module"][1]
        if module in ("numpy", "scipy"):
            # Issue with calculating component at 0...
            raise ValueError(
                f"{module} is not supported for this component, use numexpr instead."
            )

    def grad_a(self, x):
        """
        Returns d(function)/d(parameter_1)
        """
        a = self.a.value
        b = self.b.value
        c = self.c.value

        return np.where((a + b * x) > 0, -((a + b * x) ** (-1 / c - 1)) / c, 0)

    def grad_b(self, x):
        """
        Returns d(function)/d(parameter_1)
        """
        a = self.a.value
        b = self.b.value
        c = self.c.value

        return np.where((a + b * x) > 0, -x * (a + b * x) ** (-1 / c - 1) / c, 0)

    def grad_c(self, x):
        """
        Returns d(function)/d(parameter_1)
        """
        a = self.a.value
        b = self.b.value
        c = self.c.value
        return np.where(
            (a + b * x) > 0, np.log(a + b * x) / (c**2.0 * (b * x + a) ** (1.0 / c)), 0
        )
