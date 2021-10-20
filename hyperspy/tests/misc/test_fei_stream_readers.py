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

"""Test the FEI stream readers.

Because there is no official description of the format, these tests just tests
consistency between ``array_to_stream`` and ``stream_to*array``. In the
particular case of stream to sparse array, we use dask to compute the array
in order to mimic the usage in the FEI EMD reader.

"""
import numpy as np
import pytest

from hyperspy.misc.io.fei_stream_readers import (array_to_stream,
                                                 stream_to_array,
                                                 stream_to_sparse_COO_array)


@pytest.mark.parametrize("lazy", (True, False))
def test_dense_stream(lazy):
    arr = np.random.randint(0, 65535, size=(2, 3, 4, 5)).astype("uint16")
    stream = array_to_stream(arr)
    if lazy:
        arrs = stream_to_sparse_COO_array(
            stream, spatial_shape=(3, 4), sum_frames=False, channels=5,
            last_frame=2)
        arrs = arrs.compute()
        assert (arrs == arr).all()
    else:
        arrs = stream_to_array(
            stream, spatial_shape=(3, 4), sum_frames=False, channels=5,
            last_frame=2)
        assert (arrs == arr).all()


@pytest.mark.parametrize("lazy", (True, False))
def test_empty_stream(lazy):
    arr = np.zeros((2, 3, 4, 5), dtype="uint16")
    stream = array_to_stream(arr)
    if lazy:
        arrs = stream_to_sparse_COO_array(
            stream, spatial_shape=(3, 4), sum_frames=False, channels=5,
            last_frame=2)
        arrs = arrs.compute()
        assert not arrs.any()
    else:
        arrs = stream_to_array(
            stream, spatial_shape=(3, 4), sum_frames=False, channels=5,
            last_frame=2)
        assert not arrs.any()


@pytest.mark.parametrize("lazy", (True, False))
def test_sparse_stream(lazy):
    arr = np.zeros((2, 3, 4, 5), dtype="uint16")
    arr[0, 0, 0, 0] = 1
    arr[-1, -1, -1, -1] = 2
    arr[1, 1, 3, 3] = 3
    stream = array_to_stream(arr)
    if lazy:
        arrs = stream_to_sparse_COO_array(
            stream, spatial_shape=(3, 4), sum_frames=False, channels=5,
            last_frame=2)
        arrs = arrs.compute()
        assert (arrs == arr).all()
    else:
        arrs = stream_to_array(
            stream, spatial_shape=(3, 4), sum_frames=False, channels=5,
            last_frame=2)
        assert (arrs == arr).all()
