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

from hyperspy.misc.array_tools import are_aligned


def are_signals_aligned(*args):
    if len(args) < 2:
        raise ValueError(
            "This function requires at least two signal instances")
    args = list(args)
    am = args.pop().axes_manager

    while args:
        amo = args.pop().axes_manager
        if not (are_aligned(am.signal_shape[::-1], amo.signal_shape[::-1]) and
                are_aligned(am.navigation_shape[::-1],
                            amo.navigation_shape[::-1])):
            return False
    return True
