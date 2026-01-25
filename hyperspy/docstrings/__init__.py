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

"""Common docstring snippets."""

START_HSPY = """When starting HyperSpy using the ``hyperspy`` script (e.g. by executing
``hyperspy`` in a console, using the context menu entries or using the links in
the ``Start Menu``, the :mod:`~hyperspy.api` package is imported in the user
namespace as ``hs``, i.e. by executing the following:

    >>> import hyperspy.api as hs


(Note that code snippets are indicated by three greater-than signs)

We recommend to import the HyperSpy API as above also when doing it manually.
The docstring examples assume that ``hyperspy.api`` has been imported as ``hs``,
``numpy`` as ``np`` and ``matplotlib.pyplot`` as ``plt``. """
