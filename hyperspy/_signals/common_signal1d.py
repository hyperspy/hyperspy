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


from hyperspy.signal import BaseSignal
from hyperspy.exceptions import DataDimensionError


class CommonSignal1D(object):

    """Common functions for 1-dimensional signals."""

    def to_signal2D(self):
        """Returns the one dimensional signal as a two dimensional signal.

        See Also
        --------
        as_signal2D : a method for the same purpose with more options.
        signals.Signal1D.to_signal2D : performs the inverse operation on images.

        Raises
        ------
        DataDimensionError: when data.ndim < 2

        """
        if self.data.ndim < 2:
            raise DataDimensionError(
                "A Signal dimension must be >= 2 to be converted to Signal2D")
        im = self.rollaxis(-1 + 3j, 0 + 3j)
        im.axes_manager.set_signal_dimension(2)
        im._assign_subclass()
        return im
