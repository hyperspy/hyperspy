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
# along with  HyperSpy.  If not, see <https://www.gnu.org/licenses/>.

from packaging.version import Version

import dask
import numpy as np
import pytest

import hyperspy.api as hs

try:
    import cupy as cp
    if Version(np.__version__) < Version('1.20'):
        raise ImportError('numpy 1.20 or newer is required.')
    if Version(dask.__version__) < Version('2021.3.0'):
        raise ImportError('dask 2021.3.0 or newer is required.')
except ImportError:
    pytest.skip("cupy is required", allow_module_level=True)


class TestCupy:
    def setup_method(self, method):
        N = 100
        ndim = 3
        data = cp.arange(N**ndim).reshape([N]*ndim)
        s = hs.signals.Signal1D(data)
        self.s = s

    @pytest.mark.parametrize('as_numpy', [True, False, None])
    def test_call_signal(self, as_numpy):
        s = self.s
        s2 = s(as_numpy=as_numpy)
        if not as_numpy:
            assert isinstance(s2, cp.ndarray)
            s2 = cp.asnumpy(s2)
        np.testing.assert_allclose(s2, np.arange(100))

    def test_roi(self):
        s = self.s
        roi = hs.roi.CircleROI(40, 60, 15)
        sr = roi(s)
        sr.plot()
        assert isinstance(sr.data, cp.ndarray)
        sr0 = cp.asnumpy(sr.inav[0, 0].data)
        np.testing.assert_allclose(np.nan_to_num(sr0), np.zeros_like(sr0))
        assert sr.isig[0].nansum() == 4.360798E08

    def test_as_signal(self):
        s = self.s
        _ = s.as_signal2D([0, 1])

    @pytest.mark.parametrize('parallel', [True, False, None])
    def test_map(self, parallel):
        s = self.s
        data_ref = s.data.copy()

        def dummy_function(data):
            return data * 10

        s.map(dummy_function, parallel, inplace=True,
              output_signal_size=s.axes_manager.signal_shape,
              output_dtype=s.data.dtype)

        assert (s.data == data_ref * 10).all()

    def test_map_lazy(self):
        s = self.s
        data_ref = s.data.copy()
        s = s.as_lazy()
        assert isinstance(s.data, dask.array.Array)
        assert isinstance(s.data[0, 0, 0].compute(), cp.ndarray)

        def dummy_function(data):
            return data * 10

        s.map(dummy_function, inplace=True,
              output_signal_size=s.axes_manager.signal_shape,
              output_dtype=s.data.dtype)

        assert isinstance(s.data, dask.array.Array)
        assert isinstance(s.data[0, 0, 0].compute(), cp.ndarray)

        s.compute()
        cp.testing.assert_allclose(s.data, data_ref * 10)

    def test_plot_images(self):
        s = self.s
        s2 = s.T.inav[:5]
        hs.plot.plot_images(s2, axes_decor=None)
        assert isinstance(s2.data, cp.ndarray)

        s_list = [_s for _s in s2]
        hs.plot.plot_images(s_list, axes_decor=None)
        assert isinstance(s_list[0].data, cp.ndarray)

        hs.plot.plot_images(s_list, axes_decor=None, colorbar='single')
        assert isinstance(s_list[0].data, cp.ndarray)

    @pytest.mark.parametrize('style', ['overlap', 'cascade', 'mosaic', 'heatmap'])
    def test_plot_spectra(self, style):
        s = self.s
        s2 = s.inav[:5, 0]
        hs.plot.plot_spectra(s2, style=style)
        assert isinstance(s2.data, cp.ndarray)

    def test_plot_images_spectra_lazy(self):
        # simply call the function to check if there is no issue
        # with dask and cupy
        s = self.s.as_lazy()
        hs.plot.plot_spectra(s.inav[:3, 0])
        hs.plot.plot_images(s.isig[:2].T)

    def test_plot_lazy(self):
        # simply call the function to check if there is no issue
        # with dask and cupy
        # 1D signal
        s = self.s.as_lazy()
        s.plot()
        isinstance(s._cache_dask_chunk, cp.ndarray)

        # 2D signal
        s2 = self.s.T.as_lazy()
        s2.plot()
        isinstance(s._cache_dask_chunk, cp.ndarray)

    def test_fit(self):
        s = self.s
        m = s.create_model()
        m.append(hs.model.components1D.Polynomial(legacy=False, order=1))
        m.fit()


@pytest.mark.parametrize('lazy', [False, True])
def test_to_device(lazy):
    data = np.arange(10)
    s = hs.signals.Signal1D(data)
    if lazy:
        s = s.as_lazy()
        assert isinstance(s, hs.hyperspy._signals.signal1d.LazySignal1D)
        with pytest.raises(BaseException):
            s.to_device()
    else:
        s.to_device()
        assert isinstance(s, hs.signals.Signal1D)
        assert isinstance(s.data, cp.ndarray)
        s.to_host()
        assert isinstance(s, hs.signals.Signal1D)
        assert isinstance(s.data, np.ndarray)
        np.testing.assert_allclose(s.data, data)
        # we have a different copy now
        assert s.data is not data


def test_decomposition():
    data = cp.random.random(size=(20, 10, 10))
    s = hs.signals.Signal1D(data)

    with pytest.raises(TypeError):
        s.decomposition(algorithm="NMF")
    with pytest.raises(ValueError):
        s.decomposition(algorithm="SVD", svd_solver='randomized')

    s.decomposition()
    s.plot_explained_variance_ratio()
    s.plot_decomposition_loadings(3)
    s.plot_decomposition_factors(3)

    s.blind_source_separation(2, algorithm='orthomax')
