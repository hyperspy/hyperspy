# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
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


from hyperspy.extensions import EXTENSIONS as _EXTENSIONS
import importlib


__all__ = [
    component for component, specs_ in _EXTENSIONS["components2D"].items()
    ]


def __dir__():
    return sorted(__all__)


def __getattr__(name):
    if name in __all__:
        spec = _EXTENSIONS["components2D"][name]
        return getattr(importlib.import_module(spec['module']), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_base_docstring = """

Components that can be used to define a 2D model for e.g. curve fitting.

There are some components that are only useful for one particular kind of signal
and therefore their name are preceded by the signal name: eg. eels_cl_edge.

Writing a new template is easy: see the user guide documentation on creating
components.

For more details see each component docstring.
==============================================
"""


def _generate_docstring(base_docstring):
    # Generate the documentation
    for name in __dir__():
        # get the component class
        component = __getattr__(name)
        spec = _EXTENSIONS["components2D"][name]
        path = spec['module'].replace('hyperspy', '~')
        line1 = f":py:class:`{path}.{name}`" + "\n"
        component_doc = component.__doc__ or "..."
        # Get the first line only
        component_doc = component_doc.split('\n')[0]
        line2 = "    " + component_doc + "\n\n"
        base_docstring += line1 + line2

    return base_docstring


__doc__ = _generate_docstring(_base_docstring)
