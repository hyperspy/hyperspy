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

from hyperspy._components.expression import Expression


class RC(Expression):
    r"""
    RC function component (based on the time-domain capacitor voltage response
    of an RC-circuit)

    .. math::

        f(x) = V_\mathrm{0} + V_\mathrm{max} \left[1 - \mathrm{exp}\left(
            -\frac{x}{\tau}\right)\right]

    ====================== =============
    Variable                Parameter
    ====================== =============
    :math:`V_\mathrm{max}`  Vmax
    :math:`V_\mathrm{0}`    V0
    :math:`\tau`            tau
    ====================== =============


    Parameters
    ----------
    Vmax : float
        maximum voltage, asymptote of the function for
        :math:`\mathrm{lim}_{x\to\infty}`
    V0 : float
        vertical offset
    tau : float
        tau=RC is the RC circuit time constant (voltage rise time)
    **kwargs
        Extra keyword arguments are passed to the
        :class:`~.api.model.components1D.Expression` component.

    """

    def __init__(self, Vmax=1.0, V0=0.0, tau=1.0, module=None, **kwargs):
        super().__init__(
            expression="V0 + Vmax * (1 - exp(-x / tau))",
            name="RC",
            Vmax=Vmax,
            V0=V0,
            tau=tau,
            module=module,
            autodoc=False,
            **kwargs,
        )

        self.isbackground = False
