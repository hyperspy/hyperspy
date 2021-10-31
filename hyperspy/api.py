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
# along with HyperSpy. If not, see <http://www.gnu.org/licenses/>.

from hyperspy.api_nogui import (
    datasets,
    set_log_level,
    preferences,
    signals,
    eds,
    material,
    model,
    plot,
    roi,
    samfire,
    interactive,
    stack,
    transpose,
    print_known_signal_types,
    load,
    __version__,
    get_configuration_directory_path,
    __doc__,
    )


import importlib

import logging
_logger = logging.getLogger(__name__)

try:
    # Register ipywidgets by importing the module
    import hyperspy_gui_ipywidgets
except ImportError:  # pragma: no cover
    from hyperspy.defaults_parser import preferences
    if preferences.GUIs.warn_if_guis_are_missing:
        _logger.warning(
            "The ipywidgets GUI elements are not available, probably because the "
            "hyperspy_gui_ipywidgets package is not installed.")
try:
    # Register traitui UI elements by importing the module
    import hyperspy_gui_traitsui
except ImportError:  # pragma: no cover
    from hyperspy.defaults_parser import preferences
    if preferences.GUIs.warn_if_guis_are_missing:
        _logger.warning(
            "The traitsui GUI elements are not available, probably because the "
            "hyperspy_gui_traitsui package is not installed.")


__all__ = [
    'datasets',
    'set_log_level',
    'preferences',
    'signals',
    'eds',
    'material',
    'model',
    'plot',
    'roi',
    'samfire',
    'interactive',
    'stack',
    'transpose',
    'print_known_signal_types',
    'load',
    '__version__',
    'get_configuration_directory_path',
    '__doc__',
    ]


def __getattr__(name):
    if name in __all__:
        if name in _import_mapping.keys():
            import_path = 'hyperspy' + _import_mapping.get(name)
            return getattr(importlib.import_module(import_path), name)
        else:
            return importlib.import_module("." + name, 'hyperspy')
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
