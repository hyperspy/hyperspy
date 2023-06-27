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

import math
from packaging.version import Version
import sympy

from hyperspy.component import _get_scaling_factor
from hyperspy._components.expression import Expression
from hyperspy._components.gaussian import _estimate_gaussian_parameters
from hyperspy.misc.utils import is_binned # remove in v2.0


sqrt2pi = math.sqrt(2 * math.pi)
sigma2fwhm = 2 * math.sqrt(2 * math.log(2))


class Voigt(Expression):

    r"""Voigt component.

    Symmetric peak shape based on the convolution of a Lorentzian and Normal
    (Gaussian) distribution:

    .. math::
        f(x) = G(x) \cdot L(x)

    where :math:`G(x)` is the Gaussian function and :math:`L(x)` is the
    Lorentzian function. In this case using an approximate formula by David
    (see Notes). This approximation improves on the pseudo-Voigt function
    (linear combination instead of convolution of the distributions) and is,
    to a very good approximation, equivalent to a Voigt function:

    .. math::
        z(x) &= \frac{x + i \gamma}{\sqrt{2} \sigma} \\
        w(z) &= \frac{e^{-z^2} \text{erfc}(-i z)}{\sqrt{2 \pi} \sigma} \\
        f(x) &= A \cdot \Re\left\{ w \left[ z(x - x_0) \right] \right\}


    ============== =============
    Variable        Parameter
    ============== =============
    :math:`x_0`     centre
    :math:`A`       area
    :math:`\gamma`  gamma
    :math:`\sigma`  sigma
    ============== =============


    Parameters
    -----------
    centre : float
        Location of the maximum of the peak.
    area : float
        Intensity below the peak.
    gamma : float
        :math:`\gamma` = HWHM of the Lorentzian distribution.
    sigma: float
        :math:`2 \sigma \sqrt{(2 \log(2))}` = FWHM of the Gaussian distribution.
    **kwargs
        Extra keyword arguments are passed to the
        :py:class:`~._components.expression.Expression` component.

    Notes
    -----
    For convenience the `gwidth` and `lwidth` attributes can also be used to
    set and get the FWHM of the Gaussian and Lorentzian parts of the
    distribution, respectively. For backwards compatability, `FWHM` is another
    alias for the Gaussian width.

    W.I.F. David, J. Appl. Cryst. (1986). 19, 63-64,
    doi:10.1107/S0021889886089999
    """

    def __init__(self, centre=10., area=1., gamma=0.2, sigma=0.1,
                 module=["numpy", "scipy"], **kwargs):
        # Not to break scripts once we remove the legacy Voigt
        if "legacy" in kwargs:
            del kwargs["legacy"]
        if Version(sympy.__version__) < Version("1.3"):
            raise ImportError("The `Voigt` component requires "
                              "SymPy >= 1.3")
        # We use `gamma_` internally to workaround the use of the `gamma`
        # function in sympy
        super().__init__(
            expression="area * real(V); \
                V = wofz(z) / (sqrt(2.0 * pi) * sigma); \
                z = (x - centre + 1j * gamma_) / (sigma * sqrt(2.0))",
            name="Voigt",
            centre=centre,
            area=area,
            gamma=gamma,
            sigma=sigma,
            position="centre",
            module=module,
            autodoc=False,
            rename_pars={"gamma_": "gamma"},
            **kwargs,
        )

        # Boundaries
        self.area.bmin = 0.
        self.gamma.bmin = 0.
        self.sigma.bmin = 0.

        self.isbackground = False
        self.convolved = True

    def estimate_parameters(self, signal, x1, x2, only_current=False):
        """Estimate the Voigt function by calculating the momenta of the
        Gaussian.

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
         : bool
            Exit status required for the :meth:`remove_background` function.

        Notes
        -----
        Adapted from https://scipy-cookbook.readthedocs.io/items/FittingData.html

        Examples
        --------

        >>> g = hs.model.components1D.Voigt(legacy=False)
        >>> x = np.arange(-10, 10, 0.01)
        >>> data = np.zeros((32, 32, 2000))
        >>> data[:] = g.function(x).reshape((1, 1, 2000))
        >>> s = hs.signals.Signal1D(data)
        >>> s.axes_manager[-1].offset = -10
        >>> s.axes_manager[-1].scale = 0.01
        >>> g.estimate_parameters(s, -10, 10, False)

        """
        super()._estimate_parameters(signal)
        axis = signal.axes_manager.signal_axes[0]
        centre, height, sigma = _estimate_gaussian_parameters(signal, x1, x2,
                                                              only_current)
        scaling_factor = _get_scaling_factor(signal, axis, centre)

        if only_current is True:
            self.centre.value = centre
            self.sigma.value = sigma
            self.area.value = height * sigma * sqrt2pi
            if is_binned(signal):
            # in v2 replace by
            #if axis.is_binned:
                self.area.value /= scaling_factor
            return True
        else:
            if self.area.map is None:
                self._create_arrays()
            self.area.map['values'][:] = height * sigma * sqrt2pi
            if is_binned(signal):
            # in v2 replace by
            #if axis.is_binned:
                self.area.map['values'][:] /= scaling_factor
            self.area.map['is_set'][:] = True
            self.sigma.map['values'][:] = sigma
            self.sigma.map['is_set'][:] = True
            self.centre.map['values'][:] = centre
            self.centre.map['is_set'][:] = True
            self.fetch_stored_values()
            return True

    @property
    def gwidth(self):
        return self.sigma.value * sigma2fwhm

    @gwidth.setter
    def gwidth(self, value):
        self.sigma.value = value / sigma2fwhm

    @property
    def FWHM(self):
        return self.sigma.value * sigma2fwhm

    @FWHM.setter
    def FWHM(self, value):
        self.sigma.value = value / sigma2fwhm

    @property
    def lwidth(self):
        return self.gamma.value * 2

    @lwidth.setter
    def lwidth(self, value):
        self.gamma.value = value / 2
