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
import traits.api as t
import pytest

import hyperspy.api as hs

FILES_PATH = os.path.join(os.path.dirname(__file__), 'emd_files')


def test_all_but_4D():
    filename = os.path.join(FILES_PATH, 'Si100_2D_3D_DPC_potential_2slices.emd')
    s = hs.load(filename)

    assert len(s) == 4
    # DPC: navigation dimension are DPC and depth
    assert s[0].data.shape == (2, 22, 22, 2)
    # annular detectors: navigation dimension is depth
    assert s[1].data.shape == (2, 22, 22)
    # virtual detectors: navigation dimension are depth and scattering angle
    assert s[2].data.shape == (16, 16, 4)
    # projected potential: navigation dimension is number of slices
    assert s[3].data.shape == (2, 22, 22, 18)

    def check_depth_axis(axis):
        assert "depth" in axis.name
        assert axis.units == t.Undefined
        np.testing.assert_allclose(axis.scale, 1)
        np.testing.assert_allclose(axis.offset, 0)
        np.testing.assert_allclose(axis.size, 2)

    def check_signal_axes(signal_axes):
        assert signal_axes[0].name == 'R_y'
        assert signal_axes[1].name == 'R_x'
        for axis in signal_axes:
            assert axis.units == "nm"
            np.testing.assert_allclose(axis.scale, 0.25)
            np.testing.assert_allclose(axis.offset, 0)
            np.testing.assert_allclose(axis.size, 22)

    for i in [0, 1, 3]:
        # signal axis
        check_signal_axes(s[i].axes_manager.signal_axes)
        # depth axis
        check_depth_axis(s[i].axes_manager.navigation_axes[-1])

    axis = s[0].axes_manager[1]
    assert axis.name == "DPC_CoM_depth"
    assert axis.units == t.Undefined
    np.testing.assert_allclose(axis.scale, 1, rtol=1E-6)
    np.testing.assert_allclose(axis.offset, 0)
    np.testing.assert_allclose(axis.size, 2)

    # DPC axis
    axis = s[0].axes_manager[0]
    assert axis.name == t.Undefined
    assert axis.units == t.Undefined
    np.testing.assert_allclose(axis.scale, 1)
    np.testing.assert_allclose(axis.offset, 0)
    np.testing.assert_allclose(axis.size, 2)

    axis = s[2].axes_manager[0]
    assert axis.name == "R_z"
    assert axis.units == "nm"
    np.testing.assert_allclose(axis.scale, 1.3575)
    np.testing.assert_allclose(axis.offset, 0)
    np.testing.assert_allclose(axis.size, 4)

    axis = s[1].axes_manager[0]
    assert axis.name == "annular_detector_depth"
    assert axis.units == t.Undefined

    # signal axes
    signal_axes = s[2].axes_manager.signal_axes
    assert signal_axes[0].name == 'R_y'
    assert signal_axes[1].name == 'R_x'
    for axis in signal_axes:
        assert axis.units == "nm"
        np.testing.assert_allclose(axis.scale, 0.339375)
        np.testing.assert_allclose(axis.offset, 0)
        np.testing.assert_allclose(axis.size, 16)


def test_all_but_4D_no_stack():
    filename = os.path.join(FILES_PATH, 'Si100_2D_3D_DPC_potential_2slices.emd')
    s = hs.load(filename, stack_group=False)
    assert len(s) == 7

    s2 = hs.load(filename, stack_group=True)
    np.testing.assert_allclose(np.stack([s[0].data, s[1].data]), s2[0].data)


@pytest.mark.parametrize("dataset_path",
                         ['4DSTEM_simulation/data/realslices/annular_detector_depth0000/realslice',
                          '4DSTEM_simulation/data/realslices/virtual_detector_depth0000/realslice',
                          '4DSTEM_simulation/data/realslices/DPC_CoM_depth0000/realslice',
                          '4DSTEM_simulation/data/realslices/ppotential/realslice'])
def test_load_single_dataset(dataset_path):
    filename = os.path.join(FILES_PATH, 'Si100_2D_3D_DPC_potential_2slices.emd')
    s = hs.load(filename, dataset_path=dataset_path)

    assert isinstance(s, hs.signals.Signal2D)


def test_load_specific_datasets():
    filename = os.path.join(FILES_PATH, 'Si100_2D_3D_DPC_potential_2slices.emd')
    dataset_path = ['4DSTEM_simulation/data/realslices/annular_detector_depth0000/realslice',
                    '4DSTEM_simulation/data/realslices/virtual_detector_depth0000/realslice']
    s = hs.load(filename, dataset_path=dataset_path)

    assert len(s) == 2


@pytest.mark.parametrize("lazy", (True, False))
def test_3D_only(lazy):
    filename = os.path.join(FILES_PATH, 'Si100_3D.emd')
    s = hs.load(filename, lazy=lazy)
    if lazy:
        s.compute(close_file=True)

    assert s.data.shape == (22, 22, 37)

    # scattering angle axis
    axis = s.axes_manager[0]
    assert axis.name == "bin_outer_angle"
    assert axis.units == "mrad"
    np.testing.assert_allclose(axis.scale, 0.001, rtol=1E-6)
    np.testing.assert_allclose(axis.offset, 0.0005)
    np.testing.assert_allclose(axis.size, 37)

    # signal axes
    signal_axes = s.axes_manager.signal_axes
    assert signal_axes[0].name == 'R_y'
    assert signal_axes[1].name == 'R_x'
    for axis in signal_axes:
        assert axis.units == "nm"
        np.testing.assert_allclose(axis.scale, 0.25)
        np.testing.assert_allclose(axis.offset, 0)
        np.testing.assert_allclose(axis.size, 22)


def test_non_square_3D():
    filename = os.path.join(FILES_PATH, 'Si100_2x1x1_3D.emd')
    s = hs.load(filename)

    assert s.data.shape == (44, 22)

    # signal axes
    signal_axes = s.axes_manager.signal_axes
    assert signal_axes[0].name == 'R_y'
    assert signal_axes[1].name == 'R_x'
    assert signal_axes[0].size == 22
    assert signal_axes[1].size == 44
    for axis in signal_axes:
        assert axis.units == "nm"
        np.testing.assert_allclose(axis.scale, 0.25)
        np.testing.assert_allclose(axis.offset, 0)


@pytest.mark.parametrize("lazy", (True, False))
def test_4D(lazy):
    filename = os.path.join(FILES_PATH, 'Si100_4D.emd')
    s = hs.load(filename, lazy=lazy)
    if lazy:
        s.compute(close_file=True)
    assert s.data.shape == (2, 11, 11, 8, 8)

    # navigation x, y axes
    navigation_axes = s.axes_manager.navigation_axes
    assert navigation_axes[0].name == 'R_y'
    assert navigation_axes[1].name == 'R_x'
    for axis in navigation_axes[:2]:
        assert axis.units == "nm"
        np.testing.assert_allclose(axis.scale, 0.5)
        np.testing.assert_allclose(axis.offset, 0)
        np.testing.assert_allclose(axis.size, 11)

    axis = s.axes_manager[2]
    assert axis.name == "CBED_array_depth"
    assert axis.units == t.Undefined
    # Needs metadata to know these values?
    np.testing.assert_allclose(axis.scale, 1)
    np.testing.assert_allclose(axis.offset, 0)
    np.testing.assert_allclose(axis.size, 2)

    # signal axes
    signal_axes = s.axes_manager.signal_axes
    assert signal_axes[0].name == 'Q_y'
    assert signal_axes[1].name == 'Q_x'
    assert signal_axes[0].size == 8
    assert signal_axes[1].size == 8
    for axis in signal_axes:
        assert axis.units == "1 / nm"
        np.testing.assert_allclose(axis.scale, 0.18416205)
        np.testing.assert_allclose(axis.offset, -0.73664826)
