# -*- coding: utf-8 -*-
# Copyright 2007-2020 The HyperSpy developers
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

from hyperspy._components.expression import Expression
import numpy as np
import logging

_logger = logging.getLogger(__name__)


class Exponential(Expression):

    r"""Exponential function component.

    .. math::

        f(x) = A\cdot\exp\left(-\frac{x}{\tau}\right)

    ============= =============
    Variable       Parameter
    ============= =============
    :math:`A`      A
    :math:`\tau`   tau
    ============= =============


    Parameters
    -----------
    A: float
        Maximum intensity
    tau: float
        Scale parameter (time constant)
    **kwargs
        Extra keyword arguments are passed to the ``Expression`` component.
    """

    def __init__(self, A=1., tau=1., module="numexpr", **kwargs):
        super(Exponential, self).__init__(
            expression="A * exp(-x / tau)",
            name="Exponential",
            A=A,
            tau=tau,
            module=module,
            autodoc=False,
            **kwargs,
        )

        self.isbackground = False

    def estimate_parameters(self, signal, x1, x2, only_current=False):
        """Estimate the parameters for the exponential component by splitting
        the signal window into two regions and using their geometric means

        Parameters
        ----------
        signal : BaseSignal instance
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

        """
        super(Exponential, self)._estimate_parameters(signal)
        axis = signal.axes_manager.signal_axes[0]
        i1, i2 = axis.value_range_to_indices(x1, x2)
        if i1 + 1 == i2:
            if i2 < axis.high_index:
                i2 += 1
            elif i1 > axis.low_index:
                i1 -= 1
        i_mid = (i1 + i2) // 2
        x_start = axis.index2value(i1)
        x_mid = axis.index2value(i_mid)
        x_end = axis.index2value(i2)

        if only_current is True:
            s = signal.get_current_signal()
        else:
            s = signal

        if s._lazy:
            import dask.array as da
            exp = da.exp
            log = da.log
        else:
            exp = np.exp
            log = np.log

        with np.errstate(divide='raise', invalid='raise'):
            try:
                # use log and exp to compute geometric mean to avoid overflow
                a1 = s.isig[i1:i_mid].data
                b1 = log(a1)
                a2 = s.isig[i_mid:i2].data
                b2 = log(a2)
                geo_mean1 = exp(b1.mean(axis=-1))
                geo_mean2 = exp(b2.mean(axis=-1))
                x1 = (x_start + x_mid) / 2
                x2 = (x_mid + x_end) / 2

                A = exp((log(geo_mean1) - (x1 / x2) * log(geo_mean2)) /
                        (1 - x1 / x2))
                t = -x2 / (log(geo_mean2) - log(A))

                if s._lazy:
                    A = A.map_blocks(np.nan_to_num)
                    t = t.map_blocks(np.nan_to_num)
                else:
                    A = np.nan_to_num(A)
                    t = np.nan_to_num(t)

            except (FloatingPointError):
                if i1 == i2:
                    _logger.warning('Exponential parameters estimation failed '
                                'because signal range includes only one '
                                'point.')
                else:
                    _logger.warning('Exponential parameters estimation failed '
                                'with a "divide by zero" error (likely log of '
                                'a zero or negative value).')
                return False

            if self.binned:
                A /= axis.scale
            if only_current is True:
                self.A.value = A
                self.tau.value = t
                return True

            if self.A.map is None:
                self._create_arrays()
            self.A.map['values'][:] = A
            self.A.map['is_set'][:] = True
            self.tau.map['values'][:] = t
            self.tau.map['is_set'][:] = True
            self.fetch_stored_values()

            return True
