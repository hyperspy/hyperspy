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

from unittest.mock import patch

import pytest

from hyperspy.utils.parallel_pool import ParallelPool


def test_parallel_pool_multiprocessing():
    pool = ParallelPool(ipyparallel=False)
    assert pool.is_multiprocessing
    assert not pool.is_ipyparallel


def test_parallel_pool_ipyparallel_not_installed():
    """Test that multiprocessing fallback works when ipyparallel is not installed."""
    with patch("hyperspy.utils.parallel_pool._ipyparallel_installed", False):
        # When ipyparallel is not installed and ipyparallel=None (default),
        # should fall back to multiprocessing
        pool = ParallelPool()
        assert pool.is_multiprocessing
        assert not pool.is_ipyparallel

        # Explicitly requesting ipyparallel when not installed should raise
        with pytest.raises(ValueError, match="ipyparralel must be installed"):
            ParallelPool(ipyparallel=True)
