# -*- coding: utf-8 -*-
# Copyright 2007-2011 The HyperSpy developers
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


from hyperspy.signal import Signal


class Image(Signal):

    """
    """
    _record_by = "image"

    def __init__(self, *args, **kw):
        super(Image, self).__init__(*args, **kw)
        self.axes_manager.set_signal_dimension(2)

    def to_spectrum(self):
        """Returns the image as a spectrum.

        See Also
        --------
        as_spectrum : a method for the same purpose with more options.
        signals.Image.to_spectrum : performs the inverse operation on images.

        """
        return self.as_spectrum(0 + 3j)
