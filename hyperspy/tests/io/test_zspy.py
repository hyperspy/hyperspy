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

import gc
import os.path
import tempfile
from os import remove

import dask.array as da
import numpy as np
import pytest
import zarr

from hyperspy._signals.signal1d import Signal1D
from hyperspy.io import load
from hyperspy.signal import BaseSignal

my_path = os.path.dirname(__file__)


class TestLoadingOOMReadOnly:

    def setup_method(self, method):
        s = BaseSignal(np.empty((5, 5, 5)))
        s.save('tmp.zspy', overwrite=True)
        self.shape = (10000, 10000, 100)
        del s
        f = zarr.open('tmp.zspy', mode='r+')
        s = f['Experiments/__unnamed__']
        del s['data']
        s.create_dataset(
            'data',
            shape=self.shape,
            dtype='float64',
            chunks=True)

    def test_oom_loading(self):
        s = load('tmp.zspy', lazy=True)
        assert self.shape == s.data.shape
        assert isinstance(s.data, da.Array)
        assert s._lazy
        s.close_file()

    def teardown_method(self, method):
        gc.collect()  # Make sure any memmaps are closed first!
        try:
            remove('tmp.zspy')
        except BaseException:
            # Don't fail tests if we cannot remove
            pass


class TestZspy:
    @pytest.fixture
    def signal(self):
        data = np.ones((10,10,10,10))
        s = Signal1D(data)
        return s

    def test_save_N5_type(self,signal):
        with tempfile.TemporaryDirectory() as tmp:
            filename = tmp + '/testmodels.zspy'
        store = zarr.N5Store(path=filename)
        signal.save(store.path, write_to_storage=True)
        signal2 = load(filename)
        np.testing.assert_array_equal(signal2.data, signal.data)

    @pytest.mark.skip(reason="lmdb must be installed to test")
    def test_save_lmdb_type(self, signal):
        with tempfile.TemporaryDirectory() as tmp:
            os.mkdir(tmp+"/testmodels.zspy")
            filename = tmp + '/testmodels.zspy/'
            store = zarr.LMDBStore(path=filename)
            signal.save(store.path, write_to_storage=True)
            signal2 = load(store.path)
            np.testing.assert_array_equal(signal2.data, signal.data)

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
        if compressor is "blosc":
            from numcodecs import Blosc
            compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
        s = Signal1D(np.ones((3, 3)))
        s.save(tmp_path / 'test_compression.zspy',
               overwrite=True,
               compressor=compressor)
        load(tmp_path / 'test_compression.zspy')