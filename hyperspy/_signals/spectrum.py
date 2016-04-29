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

from hyperspy._signals.signal1D import Signal1D
from hyperspy._signals.signal2D import Signal2DTools
from hyperspy.misc.hspy_warnings import VisibleDeprecationWarning


class Spectrum(Signal1D,
               Signal2DTools,):
    VisibleDeprecationWarning('The Spectrum class will be deprecated from\
                              version 1.0.0 and replaced with Signal1D')
