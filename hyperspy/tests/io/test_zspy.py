# -*- coding: utf-8 -*-
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

import numpy as np
import logging
import os
import pytest

from hyperspy._signals.signal1d import Signal1D
from hyperspy.io import load
from hyperspy.signal import BaseSignal

# zarr (because of numcodecs) is only supported on x86_64 machines
zarr = pytest.importorskip("zarr", reason="zarr not installed")


class TestZspy:

    @pytest.fixture
    def signal(self):
        data = np.ones((10,10,10,10))
        s = Signal1D(data)
        return s

    @pytest.mark.parametrize('store_class', [zarr.N5Store, zarr.ZipStore])
    def test_save_store(self, signal, tmp_path, store_class):
        filename = tmp_path / 'testmodels.zspy'
        store = store_class(path=filename)
        signal.save(store)

        if store_class is zarr.ZipStore:
            assert os.path.isfile(filename)
        else:
            assert os.path.isdir(filename)

        store2 = store_class(path=filename)
        signal2 = load(store2)

        np.testing.assert_array_equal(signal2.data, signal.data)

    def test_save_ZipStore_close_file(self, signal, tmp_path):
        filename = tmp_path / 'testmodels.zspy'
        store = zarr.ZipStore(path=filename)
        signal.save(store, close_file=False)

        assert os.path.isfile(filename)

        store2 = zarr.ZipStore(path=filename)
        s2 = load(store2)

        np.testing.assert_array_equal(s2.data, signal.data)

    def test_save_wrong_store(self, signal, tmp_path, caplog):
        filename = tmp_path / 'testmodels.zspy'
        store = zarr.N5Store(path=filename)
        signal.save(store)

        store2 = zarr.N5Store(path=filename)
        s2 = load(store2)
        np.testing.assert_array_equal(s2.data, signal.data)

        store2 = zarr.NestedDirectoryStore(path=filename)
        with pytest.raises(BaseException):
            with caplog.at_level(logging.ERROR):
                _ = load(store2)

    @pytest.mark.parametrize("overwrite",[None, True, False])
    def test_overwrite(self, signal, overwrite, tmp_path):
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


def test_non_valid_zspy(tmp_path, caplog):
    filename = tmp_path / 'testfile.zspy'
    data = np.arange(10)

    f = zarr.group(filename)
    f.create_dataset('dataset', data=data)

    with pytest.raises(IOError):
        with caplog.at_level(logging.ERROR):
            _ = load(filename)
