# -*- coding: utf-8 -*-
# Copyright 2007-2024 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of  GNU General Public License as published by
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

import math

import numpy as np

from hyperspy._components.expression import Expression
from hyperspy._components.gaussian import _estimate_gaussian_parameters
from hyperspy.component import _get_scaling_factor

sqrt2pi = math.sqrt(2 * math.pi)


tiny = np.finfo(float).eps


class Doniach(Expression):
    r"""Doniach Sunjic lineshape component.

    .. math::
        :nowrap:

        \[
        f(x) = \frac{A \cos[ \frac{{\pi\alpha}}{2}+
        (1-\alpha)\tan^{-1}(\frac{x-centre+dx}{\sigma})]}
        {(\sigma^2 + (x-centre+dx)^2)^{\frac{(1-\alpha)}{2}}}
        \]


        \[
        dx = \frac{2.354820\sigma}{2 tan[\frac{\pi}{2-\alpha}]}
        \]


    =============== ===========
    Variable         Parameter
    =============== ===========
    :math:`A`        A
    :math:`\sigma`   sigma
    :math:`\alpha`   alpha
    :math:`centre`   centre
    =============== ===========

    Parameters
    ----------
    A : float
        Height
    sigma : float
        Variance parameter of the distribution
    alpha : float
        Tail or asymmetry parameter
    centre : float
        Location of the maximum (peak position).
    **kwargs
        Extra keyword arguments are passed to the
        :class:`~.api.model.components1D.Expression` component.

    Notes
    -----
    This is an asymmetric lineshape, originially design for xps but generally
    useful for fitting peaks with low side tails
    See Doniach S. and Sunjic M., J. Phys. 4C31, 285 (1970)
    or http://www.casaxps.com/help_manual/line_shapes.htm for a more detailed
    description

    """

    def __init__(
        self,
        centre=0.0,
        A=1.0,
        sigma=1.0,
        alpha=0.5,
        module=["numpy", "scipy"],
        **kwargs,
    ):
        super().__init__(
            expression="A*cos(0.5*pi*alpha+\
            ((1.0 - alpha) * arctan( (x-centre+offset)/sigma) ) )\
            /(sigma**2 + (x-centre+offset)**2)**(0.5 * (1.0 - alpha));\
            offset = 2.354820*sigma / (2 * tan(pi / (2 - alpha)))",
            name="Doniach",
            centre=centre,
            A=A,
            sigma=sigma,
            alpha=alpha,
            module=module,
            autodoc=False,
            **kwargs,
        )
        #
        self.sigma.bmin = 1.0e-8
        self.alpha.bmin = 1.0e-8
        self.isbackground = False
        self.convolved = True

    def estimate_parameters(self, signal, x1, x2, only_current=False):
        """Estimate the Donach by calculating the median (centre) and the
        variance parameter (sigma).

        Note that an insufficient range will affect the accuracy of this
        method and that this method doesn't estimate the asymmetry parameter
        (alpha).

        Parameters
        ----------
        signal : :class:`~.api.signals.Signal1D`
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
            Returns True when the parameters estimation is successful

        Examples
        --------

        >>> g = hs.model.components1D.Lorentzian()
        >>> x = np.arange(-10, 10, 0.01)
        >>> data = np.zeros((32, 32, 2000))
        >>> data[:] = g.function(x).reshape((1, 1, 2000))
        >>> s = hs.signals.Signal1D(data)
        >>> s.axes_manager[-1].offset = -10
        >>> s.axes_manager[-1].scale = 0.01
        >>> g.estimate_parameters(s, -10, 10, False)
        True
        """

        super()._estimate_parameters(signal)
        axis = signal.axes_manager.signal_axes[0]
        centre, height, sigma = _estimate_gaussian_parameters(
            signal, x1, x2, only_current
        )
        scaling_factor = _get_scaling_factor(signal, axis, centre)

        if only_current is True:
            self.centre.value = centre
            self.sigma.value = sigma
            self.A.value = height * 1.3
            if axis.is_binned:
                self.A.value /= scaling_factor
            return True
        else:
            if self.A.map is None:
                self._create_arrays()
            self.A.map["values"][:] = height * 1.3
            if axis.is_binned:
                self.A.map["values"][:] /= scaling_factor
            self.A.map["is_set"][:] = True
            self.sigma.map["values"][:] = sigma
            self.sigma.map["is_set"][:] = True
            self.centre.map["values"][:] = centre
            self.centre.map["is_set"][:] = True
            self.fetch_stored_values()
            return True
