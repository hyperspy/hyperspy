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

import logging

import numpy as np

from hyperspy._components.expression import Expression

from hyperspy.misc.utils import get_numpy_kwargs

_logger = logging.getLogger(__name__)


class PowerLaw(Expression):

    r"""Power law component.

    .. math::

        f(x) = A\cdot(x-x_0)^{-r}

    ============= =============
     Variable      Parameter
    ============= =============
     :math:`A`     A
     :math:`r`     r
     :math:`x_0`   origin
    ============= =============


    Parameters
    ----------
    A : float
        Height parameter.
    r : float
        Power law coefficient.
    origin : float
        Location parameter.
    **kwargs
        Extra keyword arguments are passed to the
        :py:class:`~._components.expression.Expression` component.

    Attributes
    ----------
    left_cutoff : float
        For x <= left_cutoff, the function returns 0. Default value is 0.0.
    """

    def __init__(self, A=10e5, r=3., origin=0., left_cutoff=0.0,
                 module="numexpr", compute_gradients=False, **kwargs):
        super().__init__(
            expression="where(left_cutoff<x, A*(-origin + x)**-r, 0)",
            name="PowerLaw",
            A=A,
            r=r,
            origin=origin,
            left_cutoff=left_cutoff,
            position="origin",
            module=module,
            autodoc=False,
            compute_gradients=compute_gradients,
            linear_parameter_list=['A'],
            check_parameter_linearity=False,
            **kwargs,
        )

        self.origin.free = False
        self.left_cutoff.free = False

        # Boundaries
        self.A.bmin = 0.
        self.A.bmax = None
        self.r.bmin = 1.
        self.r.bmax = 5.

        self.isbackground = True
        self.convolved = False

    def estimate_parameters(self, signal, x1, x2, only_current=False,
                            out=False):
        """Estimate the parameters for the power law component by the two area
        method.

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
            If False, estimates the parameters for the full dataset.
        out : bool
            If True, returns the result arrays directly without storing in the
            parameter maps/values. The returned order is (A, r).

        Returns
        -------
        {bool, tuple of values}

        """
        super()._estimate_parameters(signal)
        axis = signal.axes_manager.signal_axes[0]
        i1, i2 = axis.value_range_to_indices(x1, x2)
        if not (i2 + i1) % 2 == 0:
            i2 -= 1
        if i2 == i1:
            i2 += 2
        i3 = (i2 + i1) // 2
        x1 = axis.index2value(i1)
        x2 = axis.index2value(i2)
        x3 = axis.index2value(i3)
        if only_current is True:
            s = signal.get_current_signal()
        else:
            s = signal
        if s._lazy:
            I1 = s.isig[i1:i3].integrate1D(2j).data
            I2 = s.isig[i3:i2].integrate1D(2j).data
        else:
            from hyperspy.signal import BaseSignal
            shape = s.data.shape[:-1]
            kw = get_numpy_kwargs(s.data)
            I1_s = BaseSignal(np.empty(shape, dtype='float', **kw))
            I2_s = BaseSignal(np.empty(shape, dtype='float', **kw))
            # Use the `out` parameters to avoid doing the deepcopy
            s.isig[i1:i3].integrate1D(2j, out=I1_s)
            s.isig[i3:i2].integrate1D(2j, out=I2_s)
            I1 = I1_s.data
            I2 = I2_s.data
        with np.errstate(divide='raise'):
            try:
                r = 2 * (np.log(I1) - np.log(I2)) / (np.log(x2) - np.log(x1))
                k = 1 - r
                A = k * I2 / (x2 ** k - x3 ** k)
                if s._lazy:
                    r = r.map_blocks(np.nan_to_num)
                    A = A.map_blocks(np.nan_to_num)
                else:
                    r = np.nan_to_num(r)
                    A = np.nan_to_num(A)
            except (RuntimeWarning, FloatingPointError):
                _logger.warning('Power-law parameter estimation failed '
                                'because of a "divide-by-zero" error.')
                return False

        if only_current is True:
            self.r.value = r
            self.A.value = A
            return True

        if out:
            return A, r
        else:
            self.A.map['values'][:] = A
            self.A.map['is_set'][:] = True
            self.r.map['values'][:] = r
            self.r.map['is_set'][:] = True
            self.fetch_stored_values()
            return True

    def grad_A(self, x):
        return self.function(x) / self.A.value

    def grad_r(self, x):
        return np.where(x > self.left_cutoff.value, -self.A.value *
                        np.log(x - self.origin.value) *
                        (x - self.origin.value) ** (-self.r.value), 0)

    def grad_origin(self, x):
        return np.where(x > self.left_cutoff.value, self.r.value *
                        (x - self.origin.value) ** (-self.r.value - 1) *
                        self.A.value, 0)
