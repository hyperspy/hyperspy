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


@pytest.mark.parametrize('plot_kwargs', [{},
                                         {'navigator':'auto'} ,
                                         {'navigator':'spectrum'}])
def test_plot_lazy_chunks(plot_kwargs):
    N = 15
    dim = 3
    s = hs.signals.Signal1D(da.arange(N**dim).reshape([N]*dim)).as_lazy()
    s.data = s.data.rechunk(("auto", "auto", 5))
    s.plot(**plot_kwargs)
    assert s.navigator.data.shape == tuple([N]*(dim-1))
    assert s.navigator.original_metadata.sum_from == '[slice(5, 10, None)]'


def test_compute_navigator():
    N = 15
    dim = 3
    s = hs.signals.Signal1D(da.arange(N**dim).reshape([N]*dim)).as_lazy()
    s.compute_navigator(chunks_number=3)
    assert s.navigator.original_metadata.sum_from == '[slice(5, 10, None)]'

    # change the navigator and check it is used when plotting
    s.navigator = s.navigator / s.navigator.mean()
    s.plot()
    np.testing.assert_allclose(s._plot.navigator_data_function(),
                               s.navigator.data)


def test_navigator_deepcopy_with_new_data():
    shape = (15, 15, 30, 30)
    s = hs.signals.Signal2D(da.arange(np.prod(shape)).reshape(shape)).as_lazy()
    s.compute_navigator(chunks_number=3)

    s1 = s._deepcopy_with_new_data()
    # After transpose, the navigator should be removed
    assert not s1.metadata.has_item('_HyperSpy.navigator')
    assert s1.navigator is None

    s2 = s._deepcopy_with_new_data(copy_navigator=True)
    # After transpose, the navigator should be removed
    assert s2.metadata.has_item('_HyperSpy.navigator')
    assert s2.navigator == s.navigator


def test_remove_navigator_operation():
    shape = (15, 15, 30, 30)
    s = hs.signals.Signal2D(da.arange(np.prod(shape)).reshape(shape)).as_lazy()
    s.compute_navigator(chunks_number=3)
    s1 = s.T
    # After transpose, the navigator should be removed
    assert not s1.metadata.has_item('_HyperSpy.navigator')
    assert s1.navigator is None
    s1.plot()

    s2 = s1.sum(-1)
    # After transpose, the navigator should be removed
    assert not s2.metadata.has_item('_HyperSpy.navigator')
    assert s2.navigator is None
    s2.plot()


def test_compute_navigator_index():
    N = 15
    dim = 4
    s = hs.signals.Signal2D(da.arange(N**dim).reshape([N]*dim)).as_lazy()
    for ax in s.axes_manager.signal_axes:
        ax.scale = 0.1
        ax.offset = -0.75

    s.compute_navigator(index=0.0, chunks_number=3)
    assert s.navigator.original_metadata.sum_from ==  '[slice(5, 10, None), slice(5, 10, None)]'

    s.compute_navigator(index=0, chunks_number=3)
    assert s.navigator.original_metadata.sum_from == '[slice(0, 5, None), slice(0, 5, None)]'

    s.compute_navigator(index=-0.7, chunks_number=3)
    assert s.navigator.original_metadata.sum_from == '[slice(0, 5, None), slice(0, 5, None)]'

    s.compute_navigator(index=[-0.7, 0.0], chunks_number=3)
    assert s.navigator.original_metadata.sum_from ==  '[slice(0, 5, None), slice(5, 10, None)]'

    s.compute_navigator(index=0.0, chunks_number=[3, 5])
    assert s.navigator.original_metadata.sum_from ==  '[slice(5, 10, None), slice(6, 9, None)]'

    s.compute_navigator(index=[0.7, -0.7], chunks_number=[3, 5])
    assert s.navigator.original_metadata.sum_from ==  '[slice(10, 15, None), slice(0, 3, None)]'


def test_plot_navigator_signal():
    N = 15
    dim = 4
    s = hs.signals.Signal2D(da.arange(N**dim).reshape([N]*dim)).as_lazy()
    nav = s.inav[10, 10]
    nav.compute()
    nav *= -1
    s.plot(navigator=nav)
    np.testing.assert_allclose(s._plot.navigator_data_function(), nav)
