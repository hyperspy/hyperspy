# -*- coding: utf-8 -*-
# Copyright 2007-2020 The HyperSpy developers
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


import os
import numpy as np
from numpy.testing import assert_allclose
import pytest
import hyperspy.api as hs

dirpath = os.path.dirname(__file__)
test_file = os.path.join(dirpath, 'attolight_data', 'FolderCreatedBy_Attolight', 'HYPCard.bin')

ref_shape = (2, 2, 1024)


def test_file_reader():
    cl_sem = hs.load(test_file)
    #assert cl_sem._signal_type == 'CL_SEM_Spectrum', "The lumispy CLSEMSpectrum class was not loaded."
    assert cl_sem.axes_manager.signal_size == ref_shape[-1]
    assert cl_sem.axes_manager.navigation_shape == ref_shape[:-1]
    assert cl_sem.axes_manager.signal_dimension == 1
    assert cl_sem.axes_manager.navigation_dimension == 2


def test__save_background_metadata():
    cl_sem = hs.load(test_file)
    path = os.path.join(dirpath, 'attolight_data', 'FolderCreatedBy_Attolight', 'Background-bkg-700-300ms.txt')
    bkg = np.loadtxt(path)[1]
    assert (cl_sem.metadata.Signal.background[1] == bkg).all()


def test__create_navigation_axis():
    cl_sem = hs.load(test_file)
    x = cl_sem.axes_manager.navigation_axes[0]
    y = cl_sem.axes_manager.navigation_axes[1]
    assert x.units == '$nm$'
    assert y.units == '$nm$'
    assert_allclose(x.scale, 2483.2, 0.1)
    assert_allclose(y.scale, 2483.2, 0.1)


def test__create_signal_axis_in_wavelength():
    cl_sem = hs.load(test_file)
    s = cl_sem.axes_manager.signal_axes[0]
    assert s.name == 'Wavelength'
    assert s.units == '$nm$'
    assert_allclose(s.scale, 0.53, 0.01)
    assert_allclose(s.offset, 427, 1)


def test__store_metadata():
    cl_sem = hs.load(test_file)
    assert cl_sem.metadata.Acquisition_instrument is not None


def test__get_metadata():
    cl_sem = hs.load(test_file)
    assert cl_sem.metadata.Acquisition_instrument.Spectrometer.grating == 150.0
    assert_allclose(cl_sem.metadata.Acquisition_instrument.Spectrometer.central_wavelength_nm, 700, 1)
    assert cl_sem.metadata.Acquisition_instrument.SEM.resolution_x == 2
    assert cl_sem.metadata.Acquisition_instrument.SEM.resolution_y == 2
    assert_allclose(cl_sem.metadata.Acquisition_instrument.SEM.FOV, 26391, 1)
    assert cl_sem.metadata.Acquisition_instrument.CCD.binning == 1
    assert cl_sem.metadata.Acquisition_instrument.CCD.channels == 1024
    assert cl_sem.metadata.Acquisition_instrument.acquisition_system == 'cambridge_uk_attolight'
    assert cl_sem.metadata.Acquisition_instrument.CCD.amplification == 1
    assert cl_sem.metadata.Acquisition_instrument.CCD.readout_rate == 1
    assert cl_sem.metadata.Acquisition_instrument.CCD.exposure_time_s == 0.3
    assert_allclose(cl_sem.metadata.Acquisition_instrument.SEM.dwell_time_scan_s, 0.000293, atol=0.00001)
    assert_allclose(cl_sem.metadata.Acquisition_instrument.SEM.beam_acc_voltage_kv, 6, 0.1)
    assert cl_sem.metadata.Acquisition_instrument.SEM.gun_lens_amps == 1.2
    assert_allclose(cl_sem.metadata.Acquisition_instrument.SEM.obj_lens_amps, 0.325, 0.01)
    assert cl_sem.metadata.Acquisition_instrument.SEM.aperture_um == 100
    assert_allclose(cl_sem.metadata.Acquisition_instrument.SEM.chamber_pressure_torr, 1e-7, atol=1e-8)
    assert_allclose(cl_sem.metadata.Acquisition_instrument.SEM.real_magnification, 26400, atol=100)
