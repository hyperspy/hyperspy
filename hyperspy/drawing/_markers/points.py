# -*- coding: utf-8 -*-
# Copyright 2007-2023 The HyperSpy developers
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

from hyperspy.drawing.markers import Markers
from hyperspy.docstrings.markers import OFFSET_DOCSTRING

class Points(Markers):
    """A set of Points for faster plotting.
    """
    def __init__(self,
                 offsets,
                 **kwargs):
        """ Initialize the set of points Markers.

        Parameters
        ----------
        %s
        """
        super().__init__(collection_class=None,
                         offsets=offsets,
                         **kwargs)
        self.name = "Points"

    __init__.__doc__ %= OFFSET_DOCSTRING
