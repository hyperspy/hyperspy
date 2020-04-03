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
        """Estimate the parameters using the endpoints of the signal window

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
        x1, x2 = axis.index2value(i1), axis.index2value(i2)

        if only_current is True:
            s = signal.get_current_signal()
            y1, y2 = s.isig[i1].data[0], s.isig[i2].data[0]
            with np.errstate(divide='raise'):
                try:
                    A = np.exp((np.log(y1)-x1/x2*np.log(y2))/(1-x1/x2))
                    t = -x2/(np.log(y2)-np.log(A))
                except:
                    _logger.warning('Exponential paramaters estimation failed '
                                'because of a "divide by zero" error.')
                    return False
            if self.binned:
                A /= axis.scale
            self.A.value = np.nan_to_num(A)
            self.tau.value = np.nan_to_num(t)
            return True
        else:
            if self.A.map is None:
                self._create_arrays()
            Y1, Y2 = signal.isig[i1].data, signal.isig[i2].data
            with np.errstate(divide='raise'):
                try:
                    A = np.exp((np.log(Y1)-x1/x2*np.log(Y2))/(1-x1/x2))
                    t = -x2/(np.log(Y2)-np.log(A))
                except:
                    _logger.warning('Exponential paramaters estimation failed '
                                'because of a "divide by zero" error.')
                    return False
            if self.binned:
                A /= axis.scale
            A = np.nan_to_num(A/axis.scale)
            t = np.nan_to_num(t)

            self.A.map['values'][:] = A
            self.A.map['is_set'][:] = True
            self.tau.map['values'][:] = t
            self.tau.map['is_set'][:] = True
            self.fetch_stored_values()

            return True

