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

import matplotlib.pyplot as plt
import traits.etsconfig.etsconfig

from hyperspy.defaults_parser import current_toolkit
from hyperspy.misc.ipython_tools import get_ipython


def set_ets_toolkit(toolkit):
    try:
        traits.etsconfig.etsconfig.ETSConfig.toolkit = toolkit
    except ValueError:
        pass

# Get the backend from matplotlib
backend = plt.get_backend()
if ("WX" not in backend and
        "Qt" not in backend):
    if "inline" in backend:
        if current_toolkit in ("wx", "qt4"):
            try:
                ip = get_ipython()
                if ip is not None:
                    ip.enable_gui(current_toolkit)
                    set_ets_toolkit(current_toolkit)
            except:
                set_ets_toolkit("null")
    else:
        set_ets_toolkit("null")
elif "WX" in backend:
    set_ets_toolkit("wx")
elif "Qt" in backend:
    set_ets_toolkit("qt4")
else:
    if current_toolkit in ("wx", "qt4"):
        set_ets_toolkit(current_toolkit)
    else:
        set_ets_toolkit("null")
