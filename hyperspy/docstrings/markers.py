# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
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

"""Common docstrings to Markers"""

OFFSET_DOCSTRING = """offsets : array-like
            The positions [x, y] of the center of the marker. If the offsets are
            not provided, the marker will be placed at the current navigation
            position.
        """
WIDTHS_DOCSTRING = """widths: array-like
            The lengths of the first axes (e.g., major axis lengths).
        """

HEIGHTS_DOCSTRING = """heights: array-like
             The lengths of the second axes.
        """

ANGLES_DOCSTRING = """angles : array-like
        The angles of the first axes, degrees CCW from the x-axis.
        """

UNITS_DOCSTRING = """units : {``"points"``, ``"inches"``, ``"dots"``, ``"width"``", ``"height"``, ``"x"``, ``"y"``, ``"xy"``}
            The units in which majors and minors are given; ``"width"`` and
            ``"height"`` refer to the dimensions of the axes, while ``"x"`` and ``"y"``
            refer to the *offsets* data units. ``"xy"`` differs from all others in
            that the angle as plotted varies with the aspect ratio, and equals
            the specified angle only when the aspect ratio is unity.  Hence
            it behaves the same as the :class:`matplotlib.patches.Ellipse` with
            ``axes.transData`` as its transform.
            """
