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

import dask.array as da
import numpy as np
import pytest

import hyperspy.api as hs


@pytest.mark.parametrize('ndim', [0, 1, 2, 3])
def test_plot_lazy(ndim):
    N = 10
    dim = ndim + 1
    s = hs.signals.Signal1D(da.arange(N**dim).reshape([N]*dim)).as_lazy()
    s.plot()

    if ndim == 0:
        assert s._plot.navigator_data_function == None
    elif ndim in [1, 2]:
        assert s.navigator.data.shape == tuple([N]*ndim)
        assert isinstance(s.navigator, hs.signals.BaseSignal)
    else:
        assert s._plot.navigator_data_function == 'slider'


def test_plot_lazy_chunks():
    N = 15
    dim = 3
    s = hs.signals.Signal1D(da.arange(N**dim).reshape([N]*dim)).as_lazy()
    s.data = s.data.rechunk(("auto", "auto", 5))
    s.plot()
    assert s.navigator.original_metadata.sum_from == [slice(5, 10, None)]


def test_compute_navigator():
    N = 15
    dim = 3
    s = hs.signals.Signal1D(da.arange(N**dim).reshape([N]*dim)).as_lazy()
    s.compute_navigator(chunks_number=3)
    assert s.navigator.original_metadata.sum_from == [slice(5, 10, None)]

    # change the navigator and check it is used when plotting
    s.navigator = s.navigator / s.navigator.mean()
    s.plot()
    np.testing.assert_allclose(s._plot.navigator_data_function(),
                               s.navigator.data)
