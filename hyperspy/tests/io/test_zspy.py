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

import os

import dask.array as da
import numpy as np
import pytest
import zarr

from hyperspy._signals.signal1d import Signal1D
from hyperspy.io import load
from hyperspy.signal import BaseSignal

my_path = os.path.dirname(__file__)


class TestZspy:

    @pytest.fixture
    def signal(self):
        data = np.ones((10,10,10,10))
        s = Signal1D(data)
        return s

    def test_save_N5_type(self, signal, tmp_path):
        filename = tmp_path / 'testmodels.zspy'
        store = zarr.N5Store(path=filename)
        signal.save(store, write_to_storage=True)
        signal2 = load(filename)
        np.testing.assert_array_equal(signal2.data, signal.data)

    def test_save_N5_type_path(self, signal, tmp_path):
        filename = tmp_path / 'testmodels.zspy'
        store = zarr.N5Store(path=filename)
        signal.save(store.path, write_to_storage=True)
        signal2 = load(filename)
        np.testing.assert_array_equal(signal2.data, signal.data)

    @pytest.mark.parametrize("overwrite",[None, True, False])
    def test_overwrite(self,signal, overwrite, tmp_path):
        filename = tmp_path / 'testmodels.zspy'
        signal.save(filename=filename)
        signal2 = signal*2
        signal2.save(filename=filename, overwrite=overwrite)
        if overwrite is None:
            np.testing.assert_array_equal(signal.data,load(filename).data)
        elif overwrite:
            np.testing.assert_array_equal(signal2.data,load(filename).data)
        else:
            np.testing.assert_array_equal(signal.data,load(filename).data)

    def test_compression_opts(self, tmp_path):
        self.filename = tmp_path / 'testfile.zspy'
        from numcodecs import Blosc
        comp = Blosc(cname='zstd', clevel=1, shuffle=Blosc.SHUFFLE)
        BaseSignal([1, 2, 3]).save(self.filename, compressor=comp)
        f = zarr.open(self.filename.__str__(), mode='r+')
        d = f['Experiments/__unnamed__/data']
        assert (d.compressor == comp)\

    @pytest.mark.parametrize("compressor", (None, "default", "blosc"))
    def test_compression(self, compressor, tmp_path):
        if compressor == "blosc":
            from numcodecs import Blosc
            compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
        s = Signal1D(np.ones((3, 3)))
        s.save(tmp_path / 'test_compression.zspy',
               overwrite=True,
               compressor=compressor)
        load(tmp_path / 'test_compression.zspy')

    def test_overwrite(self, tmp_path):
        s = BaseSignal(np.ones((5, 5, 5)))

        fname = tmp_path / 'tmp.zspy'
        s.save(fname, overwrite=True)
        shape = (10, 10, 10)
        s2 = BaseSignal(np.ones(shape))
        s2.save(fname, overwrite=True)

        assert shape == s2.data.shape
