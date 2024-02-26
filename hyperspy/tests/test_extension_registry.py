# -*- coding: utf-8 -*-
# Copyright 2007-2024 The HyperSpy developers
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

import importlib

from hyperspy.extensions import ALL_EXTENSIONS


def test_signal_registry():
    signals = {
        key: value
        for key, value in ALL_EXTENSIONS["signals"].items()
        if not value["lazy"]
    }

    hyperspy_signals = [
        "BaseSignal",
        "Signal1D",
        "Signal2D",
        "ComplexSignal",
        "ComplexSignal1D",
        "ComplexSignal2D",
    ]

    for signal in hyperspy_signals:
        assert signal in signals.keys()

    exspy_spec = importlib.util.find_spec("exspy")
    if exspy_spec is not None:
        assert "EELSSpectrum" in signals.keys()
        assert "EDSTEMSpectrum" in signals.keys()
        assert "DielectricFunction" in signals.keys()

    holospy_spec = importlib.util.find_spec("holospy")
    if holospy_spec is not None:
        assert "HologramImage" in signals.keys()

    lumispy_spec = importlib.util.find_spec("lumispy")
    if lumispy_spec is not None:
        assert "LumiSpectrum" in signals.keys()
        assert "CLSEMSpectrum" in signals.keys()

    pyxem_spec = importlib.util.find_spec("pyxem")
    if pyxem_spec is not None:
        assert "Diffraction2D" in signals.keys()
        assert "ElectronDiffraction2D" in signals.keys()
