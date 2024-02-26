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
import skimage

from hyperspy.misc.tv_denoise import tv_denoise


def test_tv_denoise_error():
    with pytest.raises(ValueError, match="may be denoised"):
        _ = tv_denoise(np.array([1, 2, 3]))


def test_2d_tv_denoise():
    rng = np.random.RandomState(123)
    data = skimage.data.camera().astype(float)
    data_noisy = data + data.std() * rng.randn(*data.shape)
    data_clean = tv_denoise(data, weight=60)

    norm_noisy = np.linalg.norm(data - data_noisy) / np.linalg.norm(data)
    norm_clean = np.linalg.norm(data - data_clean) / np.linalg.norm(data)

    np.testing.assert_allclose(norm_noisy, 0.49466990)
    np.testing.assert_allclose(norm_clean, 0.06453270)


def test_3d_tv_denoise():
    rng = np.random.RandomState(123)
    x, y, z = np.ogrid[0:40, 0:40, 0:40]
    data = (x - 22) ** 2 + (y - 20) ** 2 + (z - 17) ** 2 < 8**2
    data = 255 * data.astype(float)
    data_noisy = data + data.std() * rng.randn(*data.shape)
    data_clean = tv_denoise(data_noisy, weight=100)

    norm_noisy = np.linalg.norm(data - data_noisy) / np.linalg.norm(data)
    norm_clean = np.linalg.norm(data - data_clean) / np.linalg.norm(data)

    print(norm_clean)
    np.testing.assert_allclose(norm_noisy, 0.98151071)
    np.testing.assert_allclose(norm_clean, 0.12519535)
