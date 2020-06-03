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
    assert cl_sem.axes_manager.signal_size == ref_shape[-1]
    assert cl_sem.axes_manager.navigation_shape == ref_shape[:-1]
    assert cl_sem.axes_manager.signal_dimension == 1
    assert cl_sem.axes_manager.navigation_dimension == 2


def test_warning():
    import warnings
    with pytest.warns(UserWarning):
        warnings.warn(
            "hyperspy.io:This file contains a signal provided by the lumispy Python package that is not currently installed. The signal will be loaded into a generic HyperSpy signal. Consider installing lumispy to load this dataset into its original signal class.",
            UserWarning)


def test__save_background_metadata():
    cl_sem = hs.load(test_file)
    path = os.path.join(dirpath, 'attolight_data', 'FolderCreatedBy_Attolight', 'Background-bkg-700-300ms.txt')
    bkg = np.loadtxt(path)[1]
    assert (cl_sem.metadata.Signal.background[1] == bkg).all()


def test__create_navigation_axis():
    cl_sem = hs.load(test_file)
    x = cl_sem.axes_manager.navigation_axes[0]
    y = cl_sem.axes_manager.navigation_axes[1]
    assert x.units == 'nm'
    assert y.units == 'nm'
    assert_allclose(x.scale, 2483.2, 0.1)
    assert_allclose(y.scale, 2483.2, 0.1)


def test__create_signal_axis_in_wavelength():
    cl_sem = hs.load(test_file)
    s = cl_sem.axes_manager.signal_axes[0]
    assert s.name == 'Wavelength'
    assert s.units == 'nm'
    assert_allclose(s.scale, 0.53, 0.01)
    assert_allclose(s.offset, 427, 1)


def test__store_metadata():
    cl_sem = hs.load(test_file)
    assert cl_sem.metadata.Acquisition_instrument is not None


def test_metadata():
    cl_sem = hs.load(test_file)

    assert_allclose(cl_sem.metadata.Acquisition_instrument.Spectrometer.Grating__Groove_Density.magnitude, 150.0, 1)
    assert_allclose(cl_sem.metadata.Acquisition_instrument.Spectrometer.Central_wavelength.magnitude, 700, 1)

    assert cl_sem.metadata.Acquisition_instrument.SEM.Resolution_X.magnitude == 2
    assert cl_sem.metadata.Acquisition_instrument.SEM.Resolution_Y.magnitude == 2
    assert_allclose(cl_sem.metadata.Acquisition_instrument.SEM.Real_Magnification.magnitude, 26391, 1)
    assert_allclose(cl_sem.metadata.Acquisition_instrument.SEM.Objective_Lens.magnitude, 0.325, 0.01)
    assert cl_sem.metadata.Acquisition_instrument.SEM.Aperture.magnitude == 100
    assert cl_sem.metadata.Acquisition_instrument.SEM.Aperture_Chamber_Pressure == '1.0426e-07 Torr'
    assert_allclose(cl_sem.metadata.Acquisition_instrument.SEM.HYP_Dwelltime.magnitude, 0.29, atol=0.1)
    assert_allclose(cl_sem.metadata.Acquisition_instrument.SEM.Beam_Energy.magnitude, 6000, 100)
    assert cl_sem.metadata.Acquisition_instrument.SEM.Gun_Lens.magnitude == 1.2

    assert cl_sem.metadata.Acquisition_instrument.CCD.Horizontal_Binning.magnitude == 1
    assert cl_sem.metadata.Acquisition_instrument.CCD.Channels.magnitude == 1024
    assert cl_sem.metadata.Acquisition_instrument.CCD.Signal_Amplification == 'x1'
    assert cl_sem.metadata.Acquisition_instrument.CCD.Readout_Rate_horizontal_pixel_shift == '1Mhz'
    assert cl_sem.metadata.Acquisition_instrument.CCD.Exposure_Time.magnitude == 0.3

    assert cl_sem.metadata.Acquisition_instrument.acquisition_system == 'cambridge_uk_attolight'

