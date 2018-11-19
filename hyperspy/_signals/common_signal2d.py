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


from hyperspy.docstrings.signal import OPTIMIZE_ARG


class CommonSignal2D(object):

    """Common functions for 2-dimensional signals."""

    def to_signal1D(self, optimize=True):
        """Returns the image as a spectrum.

        %s

        See Also
        --------
        as_signal1D : a method for the same purpose with more options.
        signals.Signal1D.to_signal1D : performs the inverse operation on one
        dimensional signals.

        as_signal2D, transpose, hs.transpose

        """
        return self.as_signal1D(0 + 3j, optimize=optimize)
    to_signal1D.__doc__ %= (OPTIMIZE_ARG.replace('False', 'True'))
