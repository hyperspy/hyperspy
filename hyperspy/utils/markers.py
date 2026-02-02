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

"""Markers that can be added to :class:`~.api.signals.BaseSignal` plots.

Examples
--------

>>> import scipy
>>> im = hs.signals.Signal2D(scipy.datasets.face())
>>> m = hs.plot.markers.Rectangles(
...    offsets=[10, 15],
...    widths=(5,),
...    heights=(7,),
...    angles=(0,),
...    color="red",
...    )
>>> im.add_marker(m)

"""

import importlib

# ruff: noqa: F822

__all__ = [
    "Arrows",
    "Circles",
    "Ellipses",
    "HorizontalLines",
    "Lines",
    "Markers",
    "Points",
    "Polygons",
    "Rectangles",
    "Squares",
    "Texts",
    "VerticalLines",
]

_import_mapping = {
    "Arrows": "_markers.arrows",
    "Circles": "_markers.circles",
    "Ellipses": "_markers.ellipses",
    "HorizontalLines": "_markers.horizontal_lines",
    "Lines": "_markers.lines",
    "Markers": "markers",
    "Points": "_markers.points",
    "Polygons": "_markers.polygons",
    "Rectangles": "_markers.rectangles",
    "Squares": "_markers.squares",
    "Texts": "_markers.texts",
    "VerticalLines": "_markers.vertical_lines",
}


def __dir__():
    return sorted(__all__)


def __getattr__(name):
    if name in __all__:
        import_path = f"hyperspy.drawing.{_import_mapping.get(name)}"
        return getattr(importlib.import_module(import_path), name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
