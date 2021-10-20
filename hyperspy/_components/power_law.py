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

import numpy as np
import logging

from hyperspy.docstrings.parameters import FUNCTION_ND_DOCSTRING
from hyperspy._components.expression import Expression


_logger = logging.getLogger(__name__)


class PowerLaw(Expression):

    """Power law component

    f(x) = A*(x-origin)^-r

    The left_cutoff parameter can be used to set a lower threshold from which
    the component will return 0.


    """

    def __init__(self, A=10e5, r=3., origin=0., module="numexpr", **kwargs):
        super(PowerLaw, self).__init__(
            expression="A *( x - origin) ** -r",
            name="PowerLaw",
            A=A,
            r=r,
            origin=origin,
            position="origin",
            module=module,
            autodoc=False,
            **kwargs,
        )

        self.origin.free = False
        self.left_cutoff = 0.

        # Boundaries
        self.A.bmin = 0.
        self.A.bmax = None
        self.r.bmin = 1.
        self.r.bmax = 5.

        self.isbackground = True
        self.convolved = False

    def function(self, x):
        return np.where(x > self.left_cutoff, super().function(x), 0)

    def function_nd(self, axis):
        """%s

        """
        return np.where(axis > self.left_cutoff, super().function_nd(axis), 0)

    function_nd.__doc__ %= FUNCTION_ND_DOCSTRING

    def estimate_parameters(self, signal, x1, x2, only_current=False,
                            out=False):
        """Estimate the parameters by the two area method

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
        super(PowerLaw, self)._estimate_parameters(signal)
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
            import dask.array as da
            log = da.log
            I1 = s.isig[i1:i3].integrate1D(2j).data
            I2 = s.isig[i3:i2].integrate1D(2j).data
        else:
            from hyperspy.signal import BaseSignal
            shape = s.data.shape[:-1]
            I1_s = BaseSignal(np.empty(shape, dtype='float'))
            I2_s = BaseSignal(np.empty(shape, dtype='float'))
            # Use the `out` parameters to avoid doing the deepcopy
            s.isig[i1:i3].integrate1D(2j, out=I1_s)
            s.isig[i3:i2].integrate1D(2j, out=I2_s)
            I1 = I1_s.data
            I2 = I2_s.data
            log = np.log
        with np.errstate(divide='raise'):
            try:
                r = 2 * log(I1 / I2) / log(x2 / x1)
                k = 1 - r
                A = k * I2 / (x2 ** k - x3 ** k)
                if s._lazy:
                    r = r.map_blocks(np.nan_to_num)
                    A = A.map_blocks(np.nan_to_num)
                else:
                    r = np.nan_to_num(r)
                    A = np.nan_to_num(A)
            except (RuntimeWarning, FloatingPointError):
                _logger.warning('Power law paramaters estimation failed '
                                'because of a "divide by zero" error.')
                return False
        if only_current is True:
            self.r.value = r
            self.A.value = A
            return True
        if out:
            return A, r
        else:
            if self.A.map is None:
                self._create_arrays()
            self.A.map['values'][:] = A
            self.A.map['is_set'][:] = True
            self.r.map['values'][:] = r
            self.r.map['is_set'][:] = True
            self.fetch_stored_values()
            return True
