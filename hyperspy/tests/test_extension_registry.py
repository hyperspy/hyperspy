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

from hyperspy.extensions import ALL_EXTENSIONS


signals = {key: value for key, value in ALL_EXTENSIONS["signals"].items()
           if not value["lazy"]}


hyperspy_signals = [
    'BaseSignal',
    'Signal1D',
    'Signal2D',
    'ComplexSignal',
    'ComplexSignal1D',
    'ComplexSignal2D',
    'HologramImage',
    'EELSSpectrum',
    'EDSTEMSpectrum',
    'EDSSEMSpectrum',
    'DielectricFunction',
    ]


for signal in hyperspy_signals:
    assert signal in signals.keys()

try:
    import lumipsy
    assert 'LumiSpectrum' in signals.keys()
    assert 'CLSEMSpectrum' in signals.keys()
except:
    pass


try:
    import pyxem
    assert 'Diffraction2D' in signals.keys()
    assert 'ElectronDiffraction' in signals.keys()
except:
    pass
