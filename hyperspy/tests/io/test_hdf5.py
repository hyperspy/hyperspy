# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
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

import os.path
from os import remove
import gc
import time
import tempfile

import h5py
import numpy as np
import dask.array as da
import pytest

from hyperspy.io import load
from hyperspy.io_plugins.hdf5 import get_signal_chunks
from hyperspy.signal import BaseSignal
from hyperspy._signals.signal1d import Signal1D
from hyperspy.roi import Point2DROI
from hyperspy.datasets.example_signals import EDS_TEM_Spectrum

my_path = os.path.dirname(__file__)

data = np.array([4066., 3996., 3932., 3923., 5602., 5288., 7234., 7809.,
                 4710., 5015., 4366., 4524., 4832., 5474., 5718., 5034.,
                 4651., 4613., 4637., 4429., 4217.])
example1_original_metadata = {
    'BEAMDIAM -nm': 100.0,
    'BEAMKV   -kV': 120.0,
    'CHOFFSET': -168.0,
    'COLLANGLE-mR': 3.4,
    'CONVANGLE-mR': 1.5,
    'DATATYPE': 'XY',
    'DATE': '01-OCT-1991',
    'DWELLTIME-ms': 100.0,
    'ELSDET': 'SERIAL',
    'EMISSION -uA': 5.5,
    'FORMAT': 'EMSA/MAS Spectral Data File',
    'MAGCAM': 100.0,
    'NCOLUMNS': 1.0,
    'NPOINTS': 20.0,
    'OFFSET': 520.13,
    'OPERMODE': 'IMAG',
    'OWNER': 'EMSA/MAS TASK FORCE',
    'PROBECUR -nA': 12.345,
    'SIGNALTYPE': 'ELS',
    'THICKNESS-nm': 50.0,
    'TIME': '12:00',
    'TITLE': 'NIO EELS OK SHELL',
    'VERSION': '1.0',
    'XLABEL': 'Energy',
    'XPERCHAN': 3.1,
    'XUNITS': 'eV',
    'YLABEL': 'Counts',
    'YUNITS': 'Intensity'}


class Example1:

    def test_data(self):
        assert (
            [4066.0,
             3996.0,
             3932.0,
             3923.0,
             5602.0,
             5288.0,
             7234.0,
             7809.0,
             4710.0,
             5015.0,
             4366.0,
             4524.0,
             4832.0,
             5474.0,
             5718.0,
             5034.0,
             4651.0,
             4613.0,
             4637.0,
             4429.0,
             4217.0] == self.s.data.tolist())

    def test_original_metadata(self):
        assert (
            example1_original_metadata ==
            self.s.original_metadata.as_dictionary())


class TestExample1_12(Example1):

    def setup_method(self, method):
        self.s = load(os.path.join(
            my_path,
            "hdf5_files",
            "example1_v1.2.hdf5"))

    def test_date(self):
        assert (
            self.s.metadata.General.date == "1991-10-01")

    def test_time(self):
        assert self.s.metadata.General.time == "12:00:00"


class TestExample1_10(Example1):

    def setup_method(self, method):
        self.s = load(os.path.join(
            my_path,
            "hdf5_files",
            "example1_v1.0.hdf5"))


class TestExample1_11(Example1):

    def setup_method(self, method):
        self.s = load(os.path.join(
            my_path,
            "hdf5_files",
            "example1_v1.1.hdf5"))


class TestLoadingNewSavedMetadata:

    def setup_method(self, method):
        self.s = load(os.path.join(
            my_path,
            "hdf5_files",
            "with_lists_etc.hdf5"))

    def test_signal_inside(self):
        np.testing.assert_array_almost_equal(self.s.data,
                                             self.s.metadata.Signal.Noise_properties.variance.data)

    def test_empty_things(self):
        assert self.s.metadata.test.empty_list == []
        assert self.s.metadata.test.empty_tuple == ()

    def test_simple_things(self):
        assert self.s.metadata.test.list == [42]
        assert self.s.metadata.test.tuple == (1, 2)

    def test_inside_things(self):
        assert (
            self.s.metadata.test.list_inside_list == [
                42, 137, [
                    0, 1]])
        assert self.s.metadata.test.list_inside_tuple == (137, [42, 0])
        assert (
            self.s.metadata.test.tuple_inside_tuple == (137, (123, 44)))
        assert (
            self.s.metadata.test.tuple_inside_list == [
                137, (123, 44)])

    @pytest.mark.xfail(
        reason="dill is not guaranteed to load across Python versions")
    def test_binary_string(self):
        import dill
        # apparently pickle is not "full" and marshal is not
        # backwards-compatible
        f = dill.loads(self.s.metadata.test.binary_string)
        assert f(3.5) == 4.5


@pytest.fixture()
def tmpfilepath():
    with tempfile.TemporaryDirectory() as tmp:
        yield os.path.join(tmp, "test.hdf5")
        gc.collect()        # Make sure any memmaps are closed first!


class TestSavingMetadataContainers:

    def setup_method(self, method):
        self.s = BaseSignal([0.1])

    def test_save_unicode(self, tmpfilepath):
        s = self.s
        s.metadata.set_item('test', ['a', 'b', '\u6f22\u5b57'])
        s.save(tmpfilepath)
        l = load(tmpfilepath)
        assert isinstance(l.metadata.test[0], str)
        assert isinstance(l.metadata.test[1], str)
        assert isinstance(l.metadata.test[2], str)
        assert l.metadata.test[2] == '\u6f22\u5b57'

    def test_save_long_list(self, tmpfilepath):
        s = self.s
        s.metadata.set_item('long_list', list(range(10000)))
        start = time.time()
        s.save(tmpfilepath)
        end = time.time()
        assert end - start < 1.0  # It should finish in less that 1 s.

    def test_numpy_only_inner_lists(self, tmpfilepath):
        s = self.s
        s.metadata.set_item('test', [[1., 2], ('3', 4)])
        s.save(tmpfilepath)
        l = load(tmpfilepath)
        assert isinstance(l.metadata.test, list)
        assert isinstance(l.metadata.test[0], list)
        assert isinstance(l.metadata.test[1], tuple)

    def test_numpy_general_type(self, tmpfilepath):
        s = self.s
        s.metadata.set_item('test', [[1., 2], ['3', 4]])
        s.save(tmpfilepath)
        l = load(tmpfilepath)
        assert isinstance(l.metadata.test[0][0], float)
        assert isinstance(l.metadata.test[0][1], float)
        assert isinstance(l.metadata.test[1][0], str)
        assert isinstance(l.metadata.test[1][1], str)

    def test_general_type_not_working(self, tmpfilepath):
        s = self.s
        s.metadata.set_item('test', (BaseSignal([1]), 0.1, 'test_string'))
        s.save(tmpfilepath)
        l = load(tmpfilepath)
        assert isinstance(l.metadata.test, tuple)
        assert isinstance(l.metadata.test[0], Signal1D)
        assert isinstance(l.metadata.test[1], float)
        assert isinstance(l.metadata.test[2], str)

    def test_unsupported_type(self, tmpfilepath):
        s = self.s
        s.metadata.set_item('test', Point2DROI(1, 2))
        s.save(tmpfilepath)
        l = load(tmpfilepath)
        assert 'test' not in l.metadata

    def test_date_time(self, tmpfilepath):
        s = self.s
        date, time = "2016-08-05", "15:00:00.450"
        s.metadata.General.date = date
        s.metadata.General.time = time
        s.save(tmpfilepath)
        l = load(tmpfilepath)
        assert l.metadata.General.date == date
        assert l.metadata.General.time == time

    def test_general_metadata(self, tmpfilepath):
        s = self.s
        notes = "Dummy notes"
        authors = "Author 1, Author 2"
        doi = "doi"
        s.metadata.General.notes = notes
        s.metadata.General.authors = authors
        s.metadata.General.doi = doi
        s.save(tmpfilepath)
        l = load(tmpfilepath)
        assert l.metadata.General.notes == notes
        assert l.metadata.General.authors == authors
        assert l.metadata.General.doi == doi

    def test_quantity(self, tmpfilepath):
        s = self.s
        quantity = "Intensity (electron)"
        s.metadata.Signal.quantity = quantity
        s.save(tmpfilepath)
        l = load(tmpfilepath)
        assert l.metadata.Signal.quantity == quantity


def test_none_metadata():
    s = load(os.path.join(
        my_path,
        "hdf5_files",
        "none_metadata.hdf5"))
    assert s.metadata.should_be_None is None


def test_rgba16():
    s = load(os.path.join(
        my_path,
        "hdf5_files",
        "test_rgba16.hdf5"))
    data = np.load(os.path.join(
        my_path,
        "npy_files",
        "test_rgba16.npy"))
    assert (s.data == data).all()


class TestLoadingOOMReadOnly:

    def setup_method(self, method):
        s = BaseSignal(np.empty((5, 5, 5)))
        s.save('tmp.hdf5', overwrite=True)
        self.shape = (10000, 10000, 100)
        del s
        f = h5py.File('tmp.hdf5', model='r+')
        s = f['Experiments/__unnamed__']
        del s['data']
        s.create_dataset(
            'data',
            shape=self.shape,
            dtype='float64',
            chunks=True)
        f.close()

    def test_oom_loading(self):
        s = load('tmp.hdf5', lazy=True)
        assert self.shape == s.data.shape
        assert isinstance(s.data, da.Array)
        assert s._lazy

    def teardown_method(self, method):
        gc.collect()        # Make sure any memmaps are closed first!
        try:
            remove('tmp.hdf5')
        except:
            # Don't fail tests if we cannot remove
            pass


class TestPassingArgs:

    def setup_method(self, method):
        self.filename = 'testfile.hdf5'
        BaseSignal([1, 2, 3]).save(self.filename, compression_opts=8)

    def test_compression_opts(self):
        f = h5py.File(self.filename)
        d = f['Experiments/__unnamed__/data']
        assert d.compression_opts == 8
        assert d.compression == 'gzip'
        f.close()

    def teardown_method(self, method):
        remove(self.filename)


class TestAxesConfiguration:

    def setup_method(self, method):
        self.filename = 'testfile.hdf5'
        s = BaseSignal(np.zeros((2, 2, 2, 2, 2)))
        s.axes_manager.signal_axes[0].navigate = True
        s.axes_manager.signal_axes[0].navigate = True
        s.save(self.filename)

    def test_axes_configuration(self):
        s = load(self.filename)
        assert s.axes_manager.navigation_axes[0].index_in_array == 4
        assert s.axes_manager.navigation_axes[1].index_in_array == 3
        assert s.axes_manager.signal_dimension == 3

    def teardown_method(self, method):
        remove(self.filename)


def test_strings_from_py2():
    s = EDS_TEM_Spectrum()
    assert s.metadata.Sample.elements.dtype.char == "U"


def test_lazy_metadata_arrays(tmpfilepath):
    s = BaseSignal([1, 2, 3])
    s.metadata.array = np.arange(10.)
    s.save(tmpfilepath)
    l = load(tmpfilepath, lazy=True)
    # Can't deepcopy open hdf5 file handles
    with pytest.raises(TypeError):
        l.deepcopy()
    del l
