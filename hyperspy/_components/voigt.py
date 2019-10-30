# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
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

import math
import numpy as np
import sympy

from hyperspy._components.expression import Expression
from distutils.version import LooseVersion

sqrt2pi = math.sqrt(2 * math.pi)
sigma2fwhm = 2 * math.sqrt(2 * math.log(2))

class Voigt(Expression):

    r"""Voigt component.
    
    Symmetric peak shape based on the convolution of a Lorentzian and Normal 
    (Gaussian) distribution. In this case using an approximate formula by David 
    (see Notes).
    
    .. math:: 
        z(x) &= \frac{x + i \gamma}{\sqrt{2} \sigma} \\
        w(z) &= \frac{e^{-z^2} \text{erf}(-i z)}{\sqrt{2 \pi} \sigma} \\
        f(x) &= A \Re\left\{ w \left[ z(x-x_0) \right] \right\}
    
    
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
    lwidth : float
        HWHM=:math:`\gamma` of the Lorentzian distribution.
    gwidth: float
        FWHM=:math:`2 \sigma \sqrt(2 \log(2))` of the Gaussian distribution.
    
    For convenience the `sigma` and `gamma` attributes can also be used to set 
    set and get the width of the Gaussian and Lorentzian parts of the 
    distribution, respectively. The `height` attribute can be used to get and 
    set the height of the distribution.
        
    Notes
    -----
    W.I.F. David, J. Appl. Cryst. (1986). 19, 63-64
    """

    def __init__(self, centre=0., area=1., lwidth=0.2, gwidth=0.05, 
                 module="scipy", **kwargs):
        # Not to break scripts once we remove the legacy Polynomial
        if "legacy" in kwargs:
            del kwargs["legacy"]
        if LooseVersion(sympy.__version__) < LooseVersion("1.3"):
            raise ImportError("The `Voigt` component requires "
                              "SymPy >= 1.3")
        super(Voigt, self).__init__(
            expression="area * real(V); \
                V = wofz(z) / (sqrt(2.0 * pi) * sigma); \
                z = (x - centre + 1j * lwidth) / (sigma * sqrt(2.0)); \
                sigma = gwidth / (2 * sqrt(2 * log(2)))",
            name="Voigt",
            centre=centre,
            area=area,
            lwidth=lwidth,
            gwidth=gwidth,
            module=module,
            autodoc=False,
            **kwargs,
        )

        # Boundaries
        self.area.bmin = 0.
        self.lwidth.bmin = 0.
        self.gwidth.bmin = 0.

        self.isbackground = False
        self.convolved = True


    def estimate_parameters(self, signal, E1, E2, only_current=False):
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
        bool

        Notes
        -----
        Adapted from http://www.scipy.org/Cookbook/FittingData

        Examples
        --------

        >>> g = hs.model.components1D.Voigt()
        >>> x = np.arange(-10,10, 0.01)
        >>> data = np.zeros((32,32,2000))
        >>> data[:] = g.function(x).reshape((1,1,2000))
        >>> s = hs.signals.Signal1D({'data' : data})
        >>> s.axes_manager.axes[-1].offset = -10
        >>> s.axes_manager.axes[-1].scale = 0.01
        >>> g.estimate_parameters(s, -10,10, False)

        """
        super(Voigt, self)._estimate_parameters(signal)
        axis = signal.axes_manager.signal_axes[0]

        energy2index = axis._get_index
        i1 = energy2index(E1) if energy2index(E1) else 0
        i2 = energy2index(E2) if energy2index(E2) else len(axis.axis) - 1
        X = axis.axis[i1:i2]
        if only_current is True:
            data = signal()[i1:i2]
            X_shape = (len(X),)
            i = 0
            center_shape = (1,)
        else:
            # TODO: write the rest of the code to estimate the parameters of
            # the full dataset
            i = axis.index_in_array
            data_gi = [slice(None), ] * len(signal.data.shape)
            data_gi[axis.index_in_array] = slice(i1, i2)
            data = signal.data[data_gi]
            X_shape = [1, ] * len(signal.data.shape)
            X_shape[axis.index_in_array] = data.shape[i]
            center_shape = list(data.shape)
            center_shape[i] = 1

        center = np.sum(X.reshape(X_shape) * data, i
                        ) / np.sum(data, i)

        sigma = np.sqrt(np.abs(np.sum((X.reshape(X_shape) - center.reshape(
            center_shape)) ** 2 * data, i) / np.sum(data, i)))
        height = data.max(i)
        if only_current is True:
            self.centre.value = center
            self.gwidth.value = sigma
            self.area.value = height * sigma * sqrt2pi
            return True
        else:
            if self.area.map is None:
                self.create_arrays(signal.axes_manager.navigation_shape)
            self.area.map['values'][:] = height * sigma * sqrt2pi
            self.area.map['is_set'][:] = True
            self.gwidth.map['values'][:] = sigma
            self.gwidth.map['is_set'][:] = True
            self.centre.map['values'][:] = center
            self.centre.map['is_set'][:] = True
            return True

    @property
    def sigma(self):
        return self.gwidth.value / sigma2fwhm

    @sigma.setter
    def sigma(self, value):
        self.gwidth.value = value * sigma2fwhm

    @property
    def gamma(self):
        return self.lwidth.value

    @gamma.setter
    def gamma(self, value):
        self.lwidth.value = value
        
    @property
    def height(self):
        return self.area.value / (self.gwidth.value * sqrt2pi)

    @height.setter
    def height(self, value):
        self.area.value = value * self.gwidth.value * sqrt2pi
