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

Deliberately kept outside the ``hyperspy.drawing`` package tree: importing
any ``hyperspy.drawing.*`` submodule runs ``hyperspy/drawing/__init__.py``,
which eagerly registers the (heavier) matplotlib backend — exactly the
lazy-import cost ``hyperspy.api`` avoids paying until a signal is actually
plotted.  This module has no such side effect, so ``hyperspy.api`` can
register the magic immediately on import and it's ready for ``%anyplotlib``
before the user has done anything else with hyperspy.
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
    """Register %anyplotlib now if running inside an active IPython session.

    Safe to call multiple times (re-registering a line magic just replaces
    it) and safe to call whether or not IPython is installed at all.
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
