# Copyright 2007-2022 The HyperSpy developers
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

import importlib

import pytest

from hyperspy.utils.parallel_pool import ParallelPool


def test_parallel_pool_multiprocessing():
    pool = ParallelPool(ipyparallel=False)
    assert pool.is_multiprocessing
    assert not pool.is_ipyparallel


def test_parallel_pool_ipyparallel_not_installed():
    pool = ParallelPool()
    ipyparallel_spec = importlib.util.find_spec("ipyparallel")
    if ipyparallel_spec is None:
        # ipyparallel is installed, use multiprocessing instead
        assert pool.is_multiprocessing
        assert not pool.is_ipyparallel

        with pytest.raises(ValueError):
            ParallelPool(ipyparallel=True)
