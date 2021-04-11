# -*- coding: utf-8 -*-
# Copyright 2007-2021 The HyperSpy developers
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
import dask.array as da

from hyperspy._components.expression import Expression
from distutils.version import LooseVersion
import sympy

sqrt2pi = np.sqrt(2 * np.pi)


def _estimate_skewnormal_parameters(signal, x1, x2, only_current):
    axis = signal.axes_manager.signal_axes[0]
    i1, i2 = axis.value_range_to_indices(x1, x2)
    X = axis.axis[i1:i2]
    if only_current is True:
        data = signal()[i1:i2]
        X_shape = (len(X),)
        i = 0
        x0_shape = (1,)
    else:
        i = axis.index_in_array
        data_gi = [slice(None), ] * len(signal.data.shape)
        data_gi[axis.index_in_array] = slice(i1, i2)
        data = signal.data[tuple(data_gi)]
        X_shape = [1, ] * len(signal.data.shape)
        X_shape[axis.index_in_array] = data.shape[i]
        x0_shape = list(data.shape)
        x0_shape[i] = 1

    a1 = np.sqrt(2 / np.pi)
    b1 = (4 / np.pi - 1) * a1
    m1 = np.sum(X.reshape(X_shape) * data, i) / np.sum(data, i)
    m2 = np.abs(np.sum((X.reshape(X_shape) - m1.reshape(x0_shape)) ** 2 * data, i)
              / np.sum(data, i))
    m3 = np.abs(np.sum((X.reshape(X_shape) - m1.reshape(x0_shape)) ** 3 * data, i)
              / np.sum(data, i))

    x0 = m1 - a1 * (m3 / b1) ** (1 / 3)
    scale = np.sqrt(m2 + a1 ** 2 * (m3 / b1) ** (2 / 3))
    delta = np.sqrt(1 / (a1**2 + m2 * (b1 / m3) ** (2 / 3)))
    shape = delta / np.sqrt(1 - delta**2)

    iheight = np.argmin(np.abs(X.reshape(X_shape) - x0.reshape(x0_shape)), i)
    # height is the value of the function at x0, shich has to be computed
    # differently for dask array (lazy) and depending on the dimension
    if isinstance(data, da.Array):
        x0, iheight, scale, shape = da.compute(x0, iheight, scale, shape)
        if only_current is True or signal.axes_manager.navigation_dimension == 0:
            height = data.vindex[iheight].compute()
        elif signal.axes_manager.navigation_dimension == 1:
            height = data.vindex[np.arange(signal.axes_manager.navigation_size),
                                 iheight].compute()
        else:
            height = data.vindex[(*np.indices(signal.axes_manager.navigation_shape),
                                  iheight)].compute()
    else:
        if only_current is True or signal.axes_manager.navigation_dimension == 0:
            height = data[iheight]
        elif signal.axes_manager.navigation_dimension == 1:
            height = data[np.arange(signal.axes_manager.navigation_size),
                          iheight]
        else:
            height = data[(*np.indices(signal.axes_manager.navigation_shape),
                           iheight)]

    return x0, height, scale, shape


class SkewNormal(Expression):

    r"""Skew normal distribution component.

    |  Asymmetric peak shape based on a normal distribution.
    |  For definition see
       https://en.wikipedia.org/wiki/Skew_normal_distribution
    |  See also http://azzalini.stat.unipd.it/SN/
    |

    .. math::

        f(x) &= 2 A \phi(x) \Phi(x) \\
        \phi(x) &= \frac{1}{\sqrt{2\pi}}\mathrm{exp}{\left[
                   -\frac{t(x)^2}{2}\right]} \\
        \Phi(x) &= \frac{1}{2}\left[1 + \mathrm{erf}\left(\frac{
                   \alpha~t(x)}{\sqrt{2}}\right)\right] \\
        t(x) &= \frac{x-x_0}{\omega}


    ============== =============
    Variable        Parameter
    ============== =============
    :math:`x_0`     x0
    :math:`A`       A
    :math:`\omega`  scale
    :math:`\alpha`  shape
    ============== =============


    Parameters
    -----------
    x0 : float
        Location of the peak position (not maximum, which is given by
        the `mode` property).
    A : float
        Height parameter of the peak.
    scale : float
        Width (sigma) parameter.
    shape: float
        Skewness (asymmetry) parameter. For shape=0, the normal
        distribution (Gaussian) is obtained. The distribution is
        right skewed (longer tail to the right) if shape>0 and is
        left skewed if shape<0.


    The properties `mean` (position), `variance`, `skewness` and `mode`
    (=position of maximum) are defined for convenience.
    """

    def __init__(self, x0=0., A=1., scale=1., shape=0.,
                 module=['numpy', 'scipy'], **kwargs):
        if LooseVersion(sympy.__version__) < LooseVersion("1.3"):
            raise ImportError("The `SkewNormal` component requires "
                              "SymPy >= 1.3")
        # We use `_shape` internally because `shape` is already taken in sympy
        # https://github.com/sympy/sympy/pull/20791
        super(SkewNormal, self).__init__(
            expression="2 * A * normpdf * normcdf;\
                normpdf = exp(- t ** 2 / 2) / sqrt(2 * pi);\
                normcdf = (1 + erf(_shape * t / sqrt(2))) / 2;\
                t = (x - x0) / scale",
            name="SkewNormal",
            x0=x0,
            A=A,
            scale=scale,
            shape=shape,
            module=module,
            autodoc=False,
            rename_pars={"_shape": "shape"},
            **kwargs,
        )

        # Boundaries
        self.A.bmin = 0.

        self.scale.bmin = 0

        self.isbackground = False
        self.convolved = True

    def estimate_parameters(self, signal, x1, x2, only_current=False):
        """Estimate the skew normal distribution by calculating the momenta.

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
        Adapted from Lin, Lee and Yen, Statistica Sinica 17, 909-927 (2007)
        https://www.jstor.org/stable/24307705

        Examples
        --------

        >>> g = hs.model.components1D.SkewNormal()
        >>> x = np.arange(-10, 10, 0.01)
        >>> data = np.zeros((32, 32, 2000))
        >>> data[:] = g.function(x).reshape((1, 1, 2000))
        >>> s = hs.signals.Signal1D(data)
        >>> s.axes_manager._axes[-1].offset = -10
        >>> s.axes_manager._axes[-1].scale = 0.01
        >>> g.estimate_parameters(s, -10, 10, False)
        """

        super(SkewNormal, self)._estimate_parameters(signal)
        axis = signal.axes_manager.signal_axes[0]
        x0, height, scale, shape = _estimate_skewnormal_parameters(signal, x1,
                                                                   x2, only_current)
        if only_current is True:
            self.x0.value = x0
            self.A.value = height * sqrt2pi
            self.scale.value = scale
            self.shape.value = shape
            if self.binned:
                self.A.value /= axis.scale
            return True
        else:
            if self.A.map is None:
                self._create_arrays()
            self.A.map['values'][:] = height * sqrt2pi

            if self.binned:
                self.A.map['values'] /= axis.scale
            self.A.map['is_set'][:] = True
            self.x0.map['values'][:] = x0
            self.x0.map['is_set'][:] = True
            self.scale.map['values'][:] = scale
            self.scale.map['is_set'][:] = True
            self.shape.map['values'][:] = shape
            self.shape.map['is_set'][:] = True
            self.fetch_stored_values()
            return True

    @property
    def mean(self):
        delta = self.shape.value / np.sqrt(1 + self.shape.value**2)
        return self.x0.value + self.scale.value * delta * np.sqrt(2 / np.pi)

    @property
    def variance(self):
        delta = self.shape.value / np.sqrt(1 + self.shape.value**2)
        return self.scale.value**2 * (1 - 2 * delta**2 / np.pi)

    @property
    def skewness(self):
        delta = self.shape.value / np.sqrt(1 + self.shape.value**2)
        return (4 - np.pi)/2 * (delta * np.sqrt(2/np.pi))**3 / (1 -
                                                                2 * delta**2 / np.pi)**(3/2)

    @property
    def mode(self):
        delta = self.shape.value / np.sqrt(1 + self.shape.value**2)
        muz = np.sqrt(2 / np.pi) * delta
        sigmaz = np.sqrt(1 - muz**2)
        if self.shape.value == 0:
            return self.x0.value
        else:
            m0 = muz - self.skewness * sigmaz / 2 - np.sign(self.shape.value) \
                / 2 * np.exp(- 2 * np.pi / np.abs(self.shape.value))
            return self.x0.value + self.scale.value * m0
