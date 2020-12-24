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

import pytest
import numpy as np

import hyperspy.api as hs

my_path = os.path.dirname(__file__)

test_files = ['rawdata.ASW',
              'View000_0000000.img',
              'View000_0000001.map',
              'View000_0000002.map',
              'View000_0000003.map',
              'View000_0000004.map',
              'View000_0000005.map',
              'View000_0000006.pts'
              ]


def test_load_project():
    # test load all elements of the project rawdata.ASW
    filename = os.path.join(my_path, 'JEOL_files', test_files[0])
    s = hs.load(filename)
    # first file is always a 16bit image of the work area
    assert s[0].data.dtype == np.uint8
    assert s[0].data.shape == (512, 512)
    assert s[0].axes_manager.signal_dimension == 2
    assert s[0].axes_manager[0].units == 'µm'
    assert s[0].axes_manager[0].name == 'x'
    assert s[0].axes_manager[1].units == 'µm'
    assert s[0].axes_manager[1].name == 'y'
    # 1 to 16 files are a 16bit image of work area and elemental maps
    for map in s[:-1]:
        assert map.data.dtype == np.uint8
        assert map.data.shape == (512, 512)
        assert map.axes_manager.signal_dimension == 2
        assert map.axes_manager[0].units == 'µm'
        assert map.axes_manager[0].name == 'x'
        assert map.axes_manager[1].units == 'µm'
        assert map.axes_manager[1].name == 'y'
    # last file is the datacube
    assert s[-1].data.dtype == np.uint8
    assert s[-1].data.shape == (512, 512, 4096)
    assert s[-1].axes_manager.signal_dimension == 1
    assert s[-1].axes_manager.navigation_dimension == 2
    assert s[-1].axes_manager[0].units == 'µm'
    assert s[-1].axes_manager[0].name == 'x'
    assert s[-1].axes_manager[1].units == 'µm'
    assert s[-1].axes_manager[1].name == 'y'
    assert s[-1].axes_manager[2].units == 'keV'
    np.testing.assert_allclose(s[-1].axes_manager[2].offset, -0.000789965-0.00999866*96)
    np.testing.assert_allclose(s[-1].axes_manager[2].scale, 0.00999866)
    assert s[-1].axes_manager[2].name == 'Energy'


def test_load_image():
    # test load work area haadf image
    filename = os.path.join(my_path, 'JEOL_files', 'Sample', '00_View000', test_files[1])
    s = hs.load(filename)
    assert s.data.dtype == np.uint8
    assert s.data.shape == (512, 512)
    assert s.axes_manager.signal_dimension == 2
    assert s.axes_manager[0].units == 'px'
    assert s.axes_manager[0].scale == 1
    assert s.axes_manager[0].name == 'x'
    assert s.axes_manager[1].units == 'px'
    assert s.axes_manager[1].scale == 1
    assert s.axes_manager[1].name == 'y'


@pytest.mark.parametrize('SI_dtype', [np.int8, np.uint8])
def test_load_datacube(SI_dtype):
    # test load eds datacube
    filename = os.path.join(my_path, 'JEOL_files', 'Sample', '00_View000', test_files[-1])
    s = hs.load(filename, SI_dtype=SI_dtype)
    assert s.data.dtype == SI_dtype
    assert s.data.shape == (512, 512, 4096)
    assert s.axes_manager.signal_dimension == 1
    assert s.axes_manager.navigation_dimension == 2
    assert s.axes_manager[0].units == 'px'
    assert s.axes_manager[0].scale == 1
    assert s.axes_manager[0].name == 'x'
    assert s.axes_manager[1].units == 'px'
    assert s.axes_manager[1].scale == 1
    assert s.axes_manager[1].name == 'y'
    assert s.axes_manager[2].units == 'keV'
    np.testing.assert_allclose(s.axes_manager[2].offset, -0.000789965-0.00999866*96)
    np.testing.assert_allclose(s.axes_manager[2].scale, 0.00999866)
    assert s.axes_manager[2].name == 'Energy'


def test_load_datacube_rebin_energy():
    filename = os.path.join(my_path, 'JEOL_files', 'Sample', '00_View000', test_files[-1])
    s = hs.load(filename)
    s_sum = s.sum()

    ref_data = hs.signals.Signal1D(
        np.array([1032, 1229, 1409, 1336, 1239, 1169, 969, 850, 759, 782, 773,
                  779, 853, 810, 825, 927, 1110, 1271, 1656, 1948])
        )
    np.testing.assert_allclose(s_sum.isig[0.5:0.7].data, ref_data.data)

    rebin_energy = 2
    s2 = hs.load(filename, rebin_energy=rebin_energy)
    s2_sum = s2.sum()

    ref_data2 = ref_data.rebin(scale=(rebin_energy,))
    np.testing.assert_allclose(s2_sum.isig[0.5:0.7].data, ref_data2.data)

    with pytest.raises(ValueError, match='must be a multiple'):
        _ = hs.load(filename, rebin_energy=10)


def test_load_datacube_cutoff_at_kV():
    cutoff_at_kV = 10.
    filename = os.path.join(my_path, 'JEOL_files', 'Sample', '00_View000', test_files[-1])
    s = hs.load(filename, cutoff_at_kV=None)
    s2 = hs.load(filename, cutoff_at_kV=cutoff_at_kV)

    assert s2.axes_manager[-1].size == 1096
    np.testing.assert_allclose(s2.axes_manager[2].scale, 0.00999866)
    np.testing.assert_allclose(s2.axes_manager[2].offset, -0.9606613)

    np.testing.assert_allclose(s.sum().isig[:cutoff_at_kV].data, s2.sum().data)


def test_load_datacube_downsample():
    downsample = 8
    filename = os.path.join(my_path, 'JEOL_files', test_files[0])
    s = hs.load(filename, downsample=1)[-1]
    s2 = hs.load(filename, downsample=downsample)[-1]

    s_sum = s.sum(-1).rebin(scale=(downsample, downsample))
    s2_sum = s2.sum(-1)

    assert s2.axes_manager[-1].size == 4096
    np.testing.assert_allclose(s2.axes_manager[2].scale, 0.00999866)
    np.testing.assert_allclose(s2.axes_manager[2].offset, -0.9606613)

    for axis in s2.axes_manager.navigation_axes:
        assert axis.size == 64
        np.testing.assert_allclose(axis.scale, 0.069531247)
        np.testing.assert_allclose(axis.offset, 0.0)

    np.testing.assert_allclose(s_sum.data, s2_sum.data)

    with pytest.raises(ValueError, match='must be a multiple'):
        _ = hs.load(filename, downsample=10)[-1]

    downsample = [8, 16]
    s = hs.load(filename, downsample=downsample)[-1]
    assert s.axes_manager['x'].size * downsample[0] == 512
    assert s.axes_manager['y'].size * downsample[1] == 512

    with pytest.raises(ValueError, match='must be a multiple'):
        _ = hs.load(filename, downsample=[256, 100])[-1]

    with pytest.raises(ValueError, match='must be a multiple'):
        _ = hs.load(filename, downsample=[100, 256])[-1]
