# -*- coding: utf-8 -*-
# Copyright 2007-2021 The HyperSpy developers
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
import traits.api as t

import hyperspy.api as hs
from hyperspy.io_plugins.empad import _parse_xml


DATA_DIR = os.path.join(os.path.dirname(__file__), "empad_data")
FILENAME_STACK_RAW = os.path.join(DATA_DIR, "series_x10.raw")
FILENAME_MAP_RAW = os.path.join(DATA_DIR, "scan_x4_y4.raw")


def _create_raw_data(filename, shape):
    size = np.prod(shape)
    data = np.arange(size).reshape(shape).astype("float32")
    data.tofile(filename)


def setup_module():
    _create_raw_data(FILENAME_STACK_RAW, (166400,))
    _create_raw_data(FILENAME_MAP_RAW, (4 * 4 * 130 * 128))


def teardown_module():
    fs = [f for f in [FILENAME_STACK_RAW, FILENAME_MAP_RAW] if os.path.exists(f)]

    for f in fs:
        try:
            os.remove(f)
        except Exception as e:
            print(e)


@pytest.mark.parametrize("lazy", (False, True))
def test_read_stack(lazy):
    # xml file version 0.51 211118
    s = hs.load(os.path.join(DATA_DIR, "stack_images.xml"), lazy=lazy)
    assert s.data.dtype == "float32"
    ref_data = np.arange(166400).reshape((10, 130, 128))[..., :128, :]
    np.testing.assert_allclose(s.data, ref_data.astype("float32"))
    signal_axes = s.axes_manager.signal_axes
    assert signal_axes[0].name == "width"
    assert signal_axes[1].name == "height"
    for axis in signal_axes:
        assert axis.units == t.Undefined
        assert axis.scale == 1.0
        assert axis.offset == -64
    navigation_axes = s.axes_manager.navigation_axes
    assert navigation_axes[0].name == "series_count"
    assert navigation_axes[0].units == "ms"
    assert navigation_axes[0].scale == 1.0
    assert navigation_axes[0].offset == 0.0

    assert s.metadata.General.date == "2019-06-07"
    assert s.metadata.General.time == "13:17:22.590279"
    assert s.metadata.Signal.signal_type == "electron_diffraction"


@pytest.mark.parametrize("lazy", (False, True))
def test_read_map(lazy):
    # xml file version 0.51 211118
    s = hs.load(os.path.join(DATA_DIR, "map4x4.xml"), lazy=lazy)
    assert s.data.dtype == "float32"
    ref_data = np.arange(266240).reshape((4, 4, 130, 128))[..., :128, :]
    np.testing.assert_allclose(s.data, ref_data.astype("float32"))
    signal_axes = s.axes_manager.signal_axes
    assert signal_axes[0].name == "width"
    assert signal_axes[1].name == "height"
    for axis in signal_axes:
        assert axis.units == "1/nm"
        np.testing.assert_allclose(axis.scale, 0.1826537)
        np.testing.assert_allclose(axis.offset, -11.689837)
    navigation_axes = s.axes_manager.navigation_axes
    assert navigation_axes[0].name == "scan_y"
    assert navigation_axes[1].name == "scan_x"
    for axis in navigation_axes:
        assert axis.units == "µm"
        np.testing.assert_allclose(axis.scale, 1.1415856)
        np.testing.assert_allclose(axis.offset, 0.0)

    assert s.metadata.General.date == "2019-06-06"
    assert s.metadata.General.time == "13:30:00.164675"
    assert s.metadata.Signal.signal_type == "electron_diffraction"


def test_parse_xml_1_2_0():
    # xml file version 1.2.0 (2020-10-29)
    filename = os.path.join(DATA_DIR, "map128x128_version1.2.0.xml")
    om, info = _parse_xml(filename)
    assert info['scan_x'] == 128
    assert info['scan_y'] == 128
    assert info['raw_filename'] == 'scan_x128_y128.raw'
