# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
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

import dask.array as da
import numpy as np
import pytest

from hyperspy.misc import math_tools


def test_isfloat_float():
    assert math_tools.isfloat(3.0)


def test_isfloat_int():
    assert not math_tools.isfloat(3)


def test_isfloat_npfloat():
    assert math_tools.isfloat(np.float32(3.0))


def test_isfloat_npint():
    assert not math_tools.isfloat(np.int16(3))


@pytest.mark.parametrize("seed", [None, 123, np.random.default_rng(123)])
def test_random_state(seed):
    assert isinstance(math_tools.check_random_state(seed), np.random.Generator)


@pytest.mark.parametrize("seed", [None, 123, "dask_supported"])
def test_random_state_lazy(seed):
    if seed == "dask_supported":
        seed = da.random.default_rng(123)
    out = math_tools.check_random_state(seed, lazy=True)
    assert isinstance(out, da.random.Generator)


def test_random_state_error():
    with pytest.raises(TypeError, match="SeedSequence expects"):
        math_tools.check_random_state("string")
