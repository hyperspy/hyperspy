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
import pytest

from hyperspy.io import load

my_path = os.path.dirname(__file__)

test_files = ['rawdata.ASW',
              'View000_0000000.img',
              'View000_0000001.map',
              'View000_0000002.map',
              'View000_0000003.map',
              'View000_0000004.map',
              'View000_0000005.map',
              'View000_0000006.pts']

def test_load_project():
  # test load all elements of the project rawdata.ASW
  filename = os.path.join(my_path, 'JEOL_files', test_files[0])
  s = load(filename)
  # first file is always a 16bit image of the work area
  assert s[0].data.dtype == np.uint8
  assert s[0].data.shape == (512,512)
  assert s[0].axes_manager.signal_dimension == 2
  assert s[0].axes_manager[0].units == 'µm'
  assert s[0].axes_manager[0].name == 'width'
  assert s[0].axes_manager[1].units == 'µm'
  assert s[0].axes_manager[1].name == 'height'
  # 1 to 16 files are a 16bit image of work area and elemental maps
  for map in s[0:-1]:
    assert map.data.dtype == np.uint8
    assert map.data.shape == (512,512)
    assert map.axes_manager.signal_dimension == 2
    assert map.axes_manager[0].units == 'µm'
    assert map.axes_manager[0].name == 'width'
    assert map.axes_manager[1].units == 'µm'
    assert map.axes_manager[1].name == 'height'
  # last file is the datacube
  assert s[-1].data.dtype == np.uint8
  assert s[-1].data.shape == (512,512,4096)
  assert s[-1].axes_manager.signal_dimension == 1
  assert s[-1].axes_manager.navigation_dimension == 2
  assert s[-1].axes_manager[0].units == 'µm'
  assert s[-1].axes_manager[0].name == 'width'
  assert s[-1].axes_manager[1].units == 'µm'
  assert s[-1].axes_manager[1].name == 'height'
  assert s[-1].axes_manager[2].units == 'keV'
  np.testing.assert_allclose(s[-1].axes_manager[2].offset, -0.000789965-0.00999866*96)
  np.testing.assert_allclose(s[-1].axes_manager[2].scale, 0.00999866)
  assert s[-1].axes_manager[2].name == 'Energy'
  
def test_load_image():
  # test load work area haadf image
  filename = os.path.join(my_path, 'JEOL_files', 'Sample', '00_View000', test_files[1])
  s = load(filename)
  assert s.data.dtype == np.uint8
  assert s.data.shape == (512,512)
  assert s.axes_manager.signal_dimension == 2
  assert s.axes_manager[0].units == 'px'
  assert s.axes_manager[0].scale == 1
  assert s.axes_manager[0].name == 'width'
  assert s.axes_manager[1].units == 'px'
  assert s.axes_manager[1].scale == 1
  assert s.axes_manager[1].name == 'height'

def test_load_datacube():
  # test load eds datacube
  filename = os.path.join(my_path, 'JEOL_files', 'Sample', '00_View000', test_files[-1])
  s = load(filename)
  assert s.data.dtype == np.uint8
  assert s.data.shape == (512,512,4096)
  assert s.axes_manager.signal_dimension == 1
  assert s.axes_manager.navigation_dimension == 2
  assert s.axes_manager[0].units == 'px'
  assert s.axes_manager[0].scale == 1
  assert s.axes_manager[0].name == 'width'
  assert s.axes_manager[1].units == 'px'
  assert s.axes_manager[1].scale == 1
  assert s.axes_manager[1].name == 'height'
  assert s.axes_manager[2].units == 'keV'
  np.testing.assert_allclose(s.axes_manager[2].offset, -0.000789965-0.00999866*96)
  np.testing.assert_allclose(s.axes_manager[2].scale, 0.00999866)
  assert s.axes_manager[2].name == 'Energy'
