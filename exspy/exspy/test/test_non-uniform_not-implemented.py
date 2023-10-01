# -*- coding: utf-8 -*-
# Copyright 2007-2023 The HyperSpy developers
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

import pytest

import exspy

def test_eels():
    s = exspy.signals.EELSSpectrum(([0, 1]))
    s0 = s.deepcopy()
    s.axes_manager[0].convert_to_non_uniform_axis()
    with pytest.raises(NotImplementedError):
        s.align_zero_loss_peak()
    with pytest.raises(NotImplementedError):
        s.create_model(low_loss=s)
    with pytest.raises(NotImplementedError):
        s.fourier_log_deconvolution(0)
    with pytest.raises(NotImplementedError):
        s.fourier_ratio_deconvolution(s)
    with pytest.raises(NotImplementedError):
        s.fourier_ratio_deconvolution(s0)
    with pytest.raises(NotImplementedError):
        s0.fourier_ratio_deconvolution(s)
    with pytest.raises(NotImplementedError):
        s.richardson_lucy_deconvolution(s)
    with pytest.raises(NotImplementedError):
        s.kramers_kronig_analysis()
    m = s.create_model()
    g = exspy.components.EELSCLEdge('N_K')
    with pytest.raises(NotImplementedError):
        m.append(g)


def test_eds():
    s = exspy.signals.EDSTEMSpectrum(([0, 1]))
    s2 = exspy.signals.EDSSEMSpectrum(([0, 1]))
    s.axes_manager[0].convert_to_non_uniform_axis()
    s2.axes_manager[0].convert_to_non_uniform_axis()
    s.set_microscope_parameters(20)
    with pytest.raises(NotImplementedError):
        s.get_calibration_from(s)
    with pytest.raises(NotImplementedError):
        s2.get_calibration_from(s2)
    m = s.create_model()
    with pytest.raises(NotImplementedError):
        m.add_family_lines('Al_Ka')
    with pytest.raises(NotImplementedError):
        m._set_energy_scale('Al_Ka', [1.0])
    with pytest.raises(NotImplementedError):
        m._set_energy_offset('Al_Ka', [1.0])
