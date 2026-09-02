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

"""IPython magic to switch the hyperspy plotting backend to anyplotlib.

Kept outside ``hyperspy.drawing`` so that ``hyperspy.api`` can register the
magic at import time without triggering the lazy import of the drawing
package (which registers the matplotlib backend).
"""


def _register_anyplotlib_magic(ip):
    """Register %anyplotlib with IPython."""
    from IPython.core.magic import register_line_magic

    @register_line_magic
    def anyplotlib(line):
        """Switch the hyperspy plotting backend to anyplotlib for this session.

        Usage
        -----
        %anyplotlib
        """
        from hyperspy.defaults_parser import preferences

        preferences.Plot.backend = "anyplotlib"
        print("hyperspy: switched plotting backend to anyplotlib")


def _register_if_active():
    """Register %anyplotlib if running inside an IPython session.

    Safe to call repeatedly and without IPython installed.
    """
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is not None:
            _register_anyplotlib_magic(ip)
    except ImportError:
        pass


def load_ipython_extension(ip):
    _register_anyplotlib_magic(ip)
