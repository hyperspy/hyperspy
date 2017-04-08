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

import logging

import traitsui  # Just to raise an import error at this point if not installed
import matplotlib
from traits.etsconfig.api import ETSConfig


_logger = logging.getLogger(__name__)


def set_ets_toolkit(toolkit):
    try:
        ETSConfig.toolkit = toolkit
        _logger.debug('Current ETS toolkit set to: %s', toolkit)
        # Register the UI elements
        if toolkit != "null":
            import hyperspy.gui_traitsui.axes
            import hyperspy.gui_traitsui.model
            import hyperspy.gui_traitsui.tools
            import hyperspy.gui_traitsui.preferences
            import hyperspy.gui_traitsui.microscope_parameters
            import hyperspy.gui_traitsui.messages
        else:
            _logger.warning(
                "The %s matplotlib backend is not supported by the installed "
                "traitsui version. Hence, the traitsui GUI elements are not "
                "available." % matplotlib.get_backend())

    except ValueError:
        _logger.debug("Setting ETS toolkit to %s failed" % toolkit)
        set_ets_toolkit("null")

# Get the backend from matplotlib
backend = matplotlib.rcParams["backend"]
_logger.debug('Loading hyperspy.gui')
_logger.debug('Current MPL backend: %s', backend)
if "WX" in backend:
    set_ets_toolkit("wx")
elif "Qt4" in backend:
    set_ets_toolkit("qt4")
elif "Qt5" in backend:
    set_ets_toolkit("qt5")
else:
    set_ets_toolkit("null")
