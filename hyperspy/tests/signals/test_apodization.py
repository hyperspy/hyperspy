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

import numpy as np
import pytest
from scipy.signal.windows import tukey

from hyperspy.misc.math_tools import hann_window_nth_order, outer_nd
from hyperspy.signals import BaseSignal, Signal1D, Signal2D


def test_hann_nth_order():
    np.testing.assert_allclose(
        np.hanning(1000),
        hann_window_nth_order(1000, order=1),
        rtol=1e-5,
    )
    with pytest.raises(ValueError):
        hann_window_nth_order(-1000, order=1)
    with pytest.raises(ValueError):
        hann_window_nth_order(1000.0, order=1)
    with pytest.raises(ValueError):
        hann_window_nth_order(1000, order=-1)
    with pytest.raises(ValueError):
        hann_window_nth_order(1000, order=1.0)


def _generate_parameters():
    parameters = []
    for lazy in [False, True]:
        for window_type in ["hann", "hamming", "tukey"]:
            for inplace in [False, True]:
                parameters.append([lazy, window_type, inplace])
    return parameters


@pytest.mark.parametrize("lazy, window_type, inplace", _generate_parameters())
def test_apodization(lazy, window_type, inplace):
    SIZE_NAV0 = 2
    SIZE_NAV1 = 3
    SIZE_NAV2 = 4
    SIZE_SIG0 = 50
    SIZE_SIG1 = 60
    SIZE_SIG2 = 70

    ax_dict0 = {"size": SIZE_NAV0, "navigate": True}
    ax_dict1 = {"size": SIZE_SIG0, "navigate": False}
    ax_dict2 = {"size": SIZE_SIG1, "navigate": False}
    ax_dict3 = {"size": SIZE_SIG2, "navigate": False}

    # 1. Test apodization for signal 1D, 2D, 3D:
    data = np.random.rand(SIZE_NAV0 * SIZE_NAV1 * SIZE_SIG0 * SIZE_NAV2).reshape(
        (SIZE_NAV0, SIZE_NAV1, SIZE_NAV2, SIZE_SIG0)
    )
    data2 = np.random.rand(SIZE_NAV0 * SIZE_NAV1 * SIZE_SIG0 * SIZE_SIG1).reshape(
        (SIZE_NAV0, SIZE_NAV1, SIZE_SIG0, SIZE_SIG1)
    )
    data3 = np.random.rand(SIZE_NAV0 * SIZE_SIG2 * SIZE_SIG0 * SIZE_SIG1).reshape(
        (SIZE_NAV0, SIZE_SIG0, SIZE_SIG1, SIZE_SIG2)
    )
    signal1d = Signal1D(data)
    signal2d = Signal2D(data2)
    signal3d = BaseSignal(data3, axes=[ax_dict0, ax_dict1, ax_dict2, ax_dict3])
    if lazy:
        signal1d = signal1d.as_lazy()
        signal2d = signal2d.as_lazy()
        signal3d = signal3d.as_lazy()
    if window_type == "hann":
        window = np.hanning(SIZE_SIG0)
        window1 = np.hanning(SIZE_SIG1)
        window2 = np.hanning(SIZE_SIG2)
        window2d = np.outer(window, window1)
        window3d = outer_nd(window, window1, window2)

        if inplace:
            signal1d_a = signal1d.deepcopy()
            signal1d_a.apply_apodization(window=window_type, inplace=inplace)
        else:
            signal1d_a = signal1d.apply_apodization(window=window_type)
        data_a = data * window[np.newaxis, np.newaxis, np.newaxis, :]

        if inplace:
            signal2d_a = signal2d.deepcopy()
            signal2d_a.apply_apodization(window=window_type, inplace=inplace)
        else:
            signal2d_a = signal2d.apply_apodization(window=window_type, inplace=inplace)
        data2_a = data2 * window2d[np.newaxis, np.newaxis, :, :]

        if inplace:
            signal3d_a = signal3d.deepcopy()
            signal3d_a.apply_apodization(window=window_type, inplace=inplace)
        else:
            signal3d_a = signal3d.apply_apodization(window=window_type, inplace=inplace)
        data3_a = data3 * window3d[np.newaxis, :, :, :]

        np.testing.assert_allclose(signal1d_a.data, data_a)
        np.testing.assert_allclose(signal2d_a.data, data2_a)
        np.testing.assert_allclose(signal3d_a.data, data3_a)

        for hann_order in 9 * (np.random.rand(5)) + 1:
            window = hann_window_nth_order(SIZE_SIG0, order=int(hann_order))
            signal1d_a = signal1d.apply_apodization(
                window=window_type, hann_order=int(hann_order)
            )
            data_a = data * window[np.newaxis, np.newaxis, np.newaxis, :]
            np.testing.assert_allclose(signal1d_a.data, data_a)
    elif window_type == "hamming":
        window = np.hamming(SIZE_SIG0)
        signal1d_a = signal1d.apply_apodization(window=window_type)
        data_a = data * window[np.newaxis, np.newaxis, np.newaxis, :]
        np.testing.assert_allclose(signal1d_a.data, data_a)
    elif window_type == "tukey":
        for tukey_alpha in np.random.rand(5):
            window = tukey(SIZE_SIG0, alpha=tukey_alpha)
            signal1d_a = signal1d.apply_apodization(
                window=window_type, tukey_alpha=tukey_alpha
            )
            data_a = data * window[np.newaxis, np.newaxis, np.newaxis, :]
            np.testing.assert_allclose(signal1d_a.data, data_a)

    # 2. Test raises:
    with pytest.raises(ValueError):
        signal1d.apply_apodization(window="hamm")
