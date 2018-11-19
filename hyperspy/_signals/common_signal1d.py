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


from hyperspy.exceptions import DataDimensionError
from hyperspy.docstrings.signal import OPTIMIZE_ARG


class CommonSignal1D(object):

    """Common functions for 1-dimensional signals."""

    def to_signal2D(self, optimize=True):
        """Returns the one dimensional signal as a two dimensional signal.

        By default ensures the data is stored optimally, hence often making a
        copy of the data. See `transpose` for a more general method with more
        options.

        %s

        See Also
        --------
        transpose, as_signal1D, as_signal2D, hs.transpose

        Raises
        ------
        DataDimensionError: when data.ndim < 2


        """
        if self.data.ndim < 2:
            raise DataDimensionError(
                "A Signal dimension must be >= 2 to be converted to Signal2D")
        nat = self.axes_manager._get_axes_in_natural_order()
        im = self.transpose(signal_axes=nat[:2], navigation_axes=nat[2:],
                            optimize=optimize)
        return im
    to_signal2D.__doc__ %= (OPTIMIZE_ARG.replace('False', 'True'))
