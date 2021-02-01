# -*- coding: utf-8 -*-
# Copyright 2007-2021 The HyperSpy developers
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

from hyperspy.api_nogui import *
import importlib
from lazyasd import lazyobject
import logging
_logger = logging.getLogger(__name__)

__doc__ = hyperspy.api_nogui.__doc__

# Register ipywidgets by importing the module
module = 'hyperspy_gui_ipywidgets'
spam_loader = importlib.util.find_spec(module)
if spam_loader is not None:
    @lazyobject
    def hyperspy_gui_traitsui():
        return importlib.import_module(module)
else:
    from hyperspy.defaults_parser import preferences
    if preferences.GUIs.warn_if_guis_are_missing:
        _logger.warning(
            "The ipywidgets GUI elements are not available, probably because the "
            "hyperspy_gui_ipywidgets package is not installed.")

# Register traitui UI elements by importing the module
module = 'hyperspy_gui_traitsui'
spam_loader = importlib.util.find_spec(module)
if spam_loader is not None:
    @lazyobject
    def hyperspy_gui_traitsui():
        return importlib.import_module(module)
else:
    from hyperspy.defaults_parser import preferences
    if preferences.GUIs.warn_if_guis_are_missing:
        _logger.warning(
            "The traitsui GUI elements are not available, probably because the "
            "hyperspy_gui_traitsui package is not installed.")
