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

from hyperspy.misc.example_signals_loading import load_1D_EDS_SEM_spectrum as\
    EDS_SEM_Spectrum
from hyperspy.misc.example_signals_loading import load_1D_EDS_TEM_spectrum as\
    EDS_TEM_Spectrum
from hyperspy.misc.example_signals_loading import load_object_hologram as object_hologram
from hyperspy.misc.example_signals_loading import load_reference_hologram as reference_hologram


__all__ = [
    'EDS_SEM_Spectrum',
    'EDS_TEM_Spectrum',
    'object_hologram',
    'reference_hologram',
    ]


def __dir__():
    return sorted(__all__)
