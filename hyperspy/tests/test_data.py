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

import pytest

import hyperspy.api as hs


@pytest.mark.parametrize("navigation_dimension", (0, 1, 2, 3))
@pytest.mark.parametrize("uniform", (True, False))
@pytest.mark.parametrize("add_baseline", (True, False))
@pytest.mark.parametrize("add_noise", (True, False))
def test_get_luminescence_signal(
    navigation_dimension, uniform, add_baseline, add_noise
):
    # Creating signal
    s = hs.data.luminescence_signal(
        navigation_dimension, uniform, add_baseline, add_noise
    )
    # Checking that dimension initialisation works
    assert tuple([10 for i in range(navigation_dimension)] + [1024]) == s.data.shape
    # Verifying that both functional and uniform data axis work
    sax = s.axes_manager.signal_axes[0]
    assert sax.is_uniform == uniform
    # Verifying that baseline works
    if add_baseline:
        assert s.data.min() > 340
    # Verification that noise works
    # Case of baseline + energy axis is discarded because of
    # jacobian transformation
    if not (add_baseline and not uniform):
        # Verify that adding noise works
        noisedat = s.isig[:100].data
        assert (noisedat.std() > 0.1) == add_noise


@pytest.mark.parametrize("shape", ((128, 128), (256, 256)))
@pytest.mark.parametrize("add_noise", (True, False))
def test_get_wave_image(shape, add_noise):
    s = hs.data.wave_image(shape=shape, add_noise=add_noise)
    assert s.data.shape == shape
