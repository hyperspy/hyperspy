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

import logging
from pathlib import Path
import sys
import time

import dask.array as da
import h5py
import numpy as np
import pytest

from hyperspy import __version__ as hs_version
from hyperspy.io import load
from hyperspy.axes import DataAxis, UniformDataAxis, FunctionalDataAxis, AxesManager
from hyperspy.signal import BaseSignal
from hyperspy._signals.signal1d import Signal1D
from hyperspy._signals.signal2d import Signal2D
from hyperspy.datasets.example_signals import EDS_TEM_Spectrum
from hyperspy.decorators import lazifyTestClass
from hyperspy.exceptions import VisibleDeprecationWarning
from hyperspy.misc.test_utils import assert_deep_almost_equal
from hyperspy.misc.test_utils import sanitize_dict as san_dict
from hyperspy.roi import Point2DROI
from hyperspy.utils import markers
from hyperspy.io_plugins.hspy import get_signal_chunks



my_path = Path(__file__).parent


try:
    # zarr (because of numcodecs) is only supported on x86_64 machines
    import zarr
    zspy_marker = pytest.mark.parametrize("file", ["test.hspy", "test.zspy"])
except ImportError:
    zspy_marker = pytest.mark.parametrize("file", ["test.hspy"])


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
    "Used as a base class for the TestExample classes below"
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
        self.s = load(my_path / "hdf5_files" / "example1_v1.2.hdf5")

    def test_date(self):
        assert (
            self.s.metadata.General.date == "1991-10-01")

    def test_time(self):
        assert self.s.metadata.General.time == "12:00:00"


class TestExample1_10(Example1):

    def setup_method(self, method):
        self.s = load(my_path / "hdf5_files" / "example1_v1.0.hdf5")


class TestExample1_11(Example1):

    def setup_method(self, method):
        self.s = load(my_path / "hdf5_files" / "example1_v1.1.hdf5")


class TestLoadingNewSavedMetadata:

    def setup_method(self, method):
        with pytest.warns(VisibleDeprecationWarning):
            self.s = load(my_path / "hdf5_files" / "with_lists_etc.hdf5")

    def test_signal_inside(self):
        np.testing.assert_array_almost_equal(
            self.s.data,
            self.s.metadata.Signal.Noise_properties.variance.data
        )

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



class TestSavingMetadataContainers:

    def setup_method(self, method):
        self.s = BaseSignal([0.1])

    @zspy_marker
    def test_save_unicode(self, tmp_path, file):
        s = self.s
        s.metadata.set_item('test', ['a', 'b', '\u6f22\u5b57'])
        fname = tmp_path / file
        s.save(fname)
        l = load(fname)
        assert isinstance(l.metadata.test[0], str)
        assert isinstance(l.metadata.test[1], str)
        assert isinstance(l.metadata.test[2], str)
        assert l.metadata.test[2] == '\u6f22\u5b57'

    @zspy_marker
    def test_save_long_list(self, tmp_path, file):
        s = self.s
        s.metadata.set_item('long_list', list(range(10000)))
        start = time.time()
        fname = tmp_path / file
        s.save(fname)
        end = time.time()
        # It should finish in less that 2 s on CI.
        assert end - start < 2.0  

    @zspy_marker
    def test_numpy_only_inner_lists(self, tmp_path, file):
        s = self.s
        s.metadata.set_item('test', [[1., 2], ('3', 4)])
        fname = tmp_path / file
        s.save(fname)
        l = load(fname)
        assert isinstance(l.metadata.test, list)
        assert isinstance(l.metadata.test[0], list)
        assert isinstance(l.metadata.test[1], tuple)

    @pytest.mark.xfail(sys.platform == 'win32',
                       reason="randomly fails in win32")
    @zspy_marker
    def test_numpy_general_type(self, tmp_path, file):
        s = self.s
        s.metadata.set_item('test', np.array([[1., 2], ['3', 4]]))
        fname = tmp_path / file
        s.save(fname)
        l = load(fname)
        np.testing.assert_array_equal(l.metadata.test, s.metadata.test)

    @pytest.mark.xfail(sys.platform == 'win32',
                       reason="randomly fails in win32")
    @zspy_marker
    def test_list_general_type(self, tmp_path, file):
        s = self.s
        s.metadata.set_item('test', [[1., 2], ['3', 4]])
        fname = tmp_path / file
        s.save(fname)
        l = load(fname)
        assert isinstance(l.metadata.test[0][0], float)
        assert isinstance(l.metadata.test[0][1], float)
        assert isinstance(l.metadata.test[1][0], str)
        assert isinstance(l.metadata.test[1][1], str)

    @pytest.mark.xfail(sys.platform == 'win32',
                       reason="randomly fails in win32")
    @zspy_marker
    def test_general_type_not_working(self, tmp_path, file):
        s = self.s
        s.metadata.set_item('test', (BaseSignal([1]), 0.1, 'test_string'))
        fname = tmp_path / file
        s.save(fname)
        l = load(fname)
        assert isinstance(l.metadata.test, tuple)
        assert isinstance(l.metadata.test[0], Signal1D)
        assert isinstance(l.metadata.test[1], float)
        assert isinstance(l.metadata.test[2], str)

    @zspy_marker
    def test_unsupported_type(self, tmp_path, file):
        s = self.s
        s.metadata.set_item('test', Point2DROI(1, 2))
        fname = tmp_path / file
        s.save(fname)
        l = load(fname)
        assert 'test' not in l.metadata

    @zspy_marker
    def test_date_time(self, tmp_path, file):
        s = self.s
        date, time = "2016-08-05", "15:00:00.450"
        s.metadata.General.date = date
        s.metadata.General.time = time
        fname = tmp_path / file
        s.save(fname)
        l = load(fname)
        assert l.metadata.General.date == date
        assert l.metadata.General.time == time

    @zspy_marker
    def test_general_metadata(self, tmp_path, file):
        s = self.s
        notes = "Dummy notes"
        authors = "Author 1, Author 2"
        doi = "doi"
        s.metadata.General.notes = notes
        s.metadata.General.authors = authors
        s.metadata.General.doi = doi
        fname = tmp_path / file
        s.save(fname)
        l = load(fname)
        assert l.metadata.General.notes == notes
        assert l.metadata.General.authors == authors
        assert l.metadata.General.doi == doi

    @zspy_marker
    def test_quantity(self, tmp_path, file):
        s = self.s
        quantity = "Intensity (electron)"
        s.metadata.Signal.quantity = quantity
        fname = tmp_path / file
        s.save(fname)
        l = load(fname)
        assert l.metadata.Signal.quantity == quantity

    @zspy_marker
    def test_save_axes_manager(self, tmp_path, file):
        s = self.s
        s.metadata.set_item('test', s.axes_manager)
        fname = tmp_path /  file
        s.save(fname)
        l = load(fname)
        #strange becuase you need the encoding...
        assert isinstance(l.metadata.test, AxesManager)

    @zspy_marker
    def test_title(self, tmp_path, file):
        s = self.s
        fname = tmp_path / file
        s.metadata.General.title = '__unnamed__'
        s.save(fname)
        l = load(fname)
        assert l.metadata.General.title == ""

    @zspy_marker
    def test_save_empty_tuple(self, tmp_path, file):
        s = self.s
        s.metadata.set_item('test', ())
        fname = tmp_path / file
        s.save(fname)
        l = load(fname)
        #strange becuase you need the encoding...
        assert l.metadata.test == s.metadata.test

    @zspy_marker
    def test_save_bytes(self, tmp_path,file):
        s = self.s
        byte_message = bytes("testing", 'utf-8')
        s.metadata.set_item('test', byte_message)
        fname = tmp_path / file
        s.save(fname)
        l = load(fname)
        assert l.metadata.test == s.metadata.test.decode()

    def test_metadata_binned_deprecate(self):
        with pytest.warns(UserWarning, match="Loading old file"):
            s = load(my_path / "hdf5_files" / 'example2_v2.2.hspy')
        assert s.metadata.has_item('Signal.binned') == False
        assert s.axes_manager[-1].is_binned == False


    def test_metadata_update_to_v3_1(self):
        md = {'Acquisition_instrument': {'SEM': {'Stage': {'tilt_alpha': 5.0}},
                                         'TEM': {'Detector': {'Camera': {'exposure': 0.20000000000000001}},
                                                 'Stage': {'tilt_alpha': 10.0},
                                                 'acquisition_mode': 'TEM',
                                                 'beam_current': 0.0,
                                                 'beam_energy': 200.0,
                                                 'camera_length': 320.00000000000006,
                                                 'microscope': 'FEI Tecnai'}},
              'General': {'date': '2014-07-09',
                          'original_filename': 'test_diffraction_pattern.dm3',
                          'time': '18:56:37',
                          'title': 'test_diffraction_pattern',
                          'FileIO': {
                              '0': {
                                  'operation': 'load',
                                  'hyperspy_version': hs_version,
                                  'io_plugin': 'hyperspy.io_plugins.hspy'
                              }
                          }
              },
              'Signal': {'Noise_properties': {'Variance_linear_model': {'gain_factor': 1.0,
                                                                        'gain_offset': 0.0}},
                         'quantity': 'Intensity',
                         'signal_type': ''},
              '_HyperSpy': {'Folding': {'original_axes_manager': None,
                                        'original_shape': None,
                                        'signal_unfolded': False,
                                        'unfolded': False}}}
        s = load(my_path / "hdf5_files" / 'example2_v3.1.hspy')
        # delete timestamp from metadata since it's runtime dependent
        del s.metadata.General.FileIO.Number_0.timestamp
        assert_deep_almost_equal(s.metadata.as_dictionary(), md)


def test_none_metadata():
    s = load(my_path / "hdf5_files" / "none_metadata.hdf5")
    assert s.metadata.should_be_None is None


def test_rgba16():
    with pytest.warns(VisibleDeprecationWarning):
        print(my_path)
        s = load(my_path / "hdf5_files" / "test_rgba16.hdf5")
    data = np.load(my_path / "npz_files" / "test_rgba16.npz")['a']
    assert (s.data == data).all()


def test_non_valid_hspy(tmp_path, caplog):
    filename = tmp_path / 'testfile.hspy'
    data = np.arange(10)

    with h5py.File(filename, mode='w') as f:
        f.create_dataset('dataset', data=data)

    with pytest.raises(IOError):
        with caplog.at_level(logging.ERROR):
            _ = load(filename)


@zspy_marker
@pytest.mark.parametrize('lazy', [True, False])
def test_nonuniformaxis(tmp_path, file, lazy):
    fname = tmp_path / file
    data = np.arange(10)
    axis = DataAxis(axis =1/np.arange(1, data.size+1), navigate=False)
    s = Signal1D(data, axes=(axis.get_axis_dictionary(), ))
    if lazy:
        s = s.as_lazy()
    s.save(fname, overwrite=True)
    s2 = load(fname)
    np.testing.assert_array_almost_equal(s.axes_manager[0].axis,
                                         s2.axes_manager[0].axis)
    assert(s2.axes_manager[0].is_uniform == False)
    assert(s2.axes_manager[0].navigate == False)
    assert(s2.axes_manager[0].size == data.size)


@zspy_marker
@pytest.mark.parametrize('lazy', [True, False])
def test_nonuniformFDA(tmp_path, file, lazy):
    fname = tmp_path / file
    data = np.arange(10)
    x0 = UniformDataAxis(size=data.size, offset=1)
    axis = FunctionalDataAxis(expression='1/x', x=x0, navigate=False)
    s = Signal1D(data, axes = (axis.get_axis_dictionary(), ))
    if lazy:
        s = s.as_lazy()
    print(axis.get_axis_dictionary())
    s.save(fname, overwrite=True)
    s2 = load(fname)
    np.testing.assert_array_almost_equal(s.axes_manager[0].axis,
                                         s2.axes_manager[0].axis)
    assert(s2.axes_manager[0].is_uniform == False)
    assert(s2.axes_manager[0].navigate == False)
    assert(s2.axes_manager[0].size == data.size)


def test_lazy_loading(tmp_path):
    s = BaseSignal(np.empty((5, 5, 5)))
    fname = tmp_path / 'tmp.hdf5'
    s.save(fname, overwrite=True)
    shape = (10000, 10000, 100)
    del s
    f = h5py.File(fname, mode='r+')
    s = f['Experiments/__unnamed__']
    del s['data']
    s.create_dataset(
        'data',
        shape=shape,
        dtype='float64',
        chunks=True)
    f.close()

    s = load(fname, lazy=True)
    assert shape == s.data.shape
    assert isinstance(s.data, da.Array)
    assert s._lazy
    s.close_file()


def test_passing_compression_opts_saving(tmp_path):
    filename = tmp_path / 'testfile.hdf5'
    BaseSignal([1, 2, 3]).save(filename, compression_opts=8)

    f = h5py.File(filename, mode='r+')
    d = f['Experiments/__unnamed__/data']
    assert d.compression_opts == 8
    assert d.compression == 'gzip'
    f.close()


@zspy_marker
def test_axes_configuration(tmp_path, file):
    fname = tmp_path / file
    s = BaseSignal(np.zeros((2, 2, 2, 2, 2)))
    s.axes_manager.signal_axes[0].navigate = True
    s.axes_manager.signal_axes[0].navigate = True

    s.save(fname, overwrite=True)
    s = load(fname)
    assert s.axes_manager.navigation_axes[0].index_in_array == 4
    assert s.axes_manager.navigation_axes[1].index_in_array == 3
    assert s.axes_manager.signal_dimension == 3


@zspy_marker
def test_axes_configuration_binning(tmp_path, file):
    fname = tmp_path / file
    s = BaseSignal(np.zeros((2, 2, 2)))
    s.axes_manager.signal_axes[-1].is_binned = True
    s.save(fname)

    s = load(fname)
    assert s.axes_manager.signal_axes[-1].is_binned


@lazifyTestClass
class Test_permanent_markers_io:

    def setup_method(self, method):
        s = Signal2D(np.arange(100).reshape(10, 10))
        self.s = s

    @zspy_marker
    def test_save_permanent_marker(self, tmp_path, file):
        filename = tmp_path / file
        s = self.s
        m = markers.point(x=5, y=5)
        s.add_marker(m, permanent=True)
        s.save(filename)

    @zspy_marker
    def test_save_load_empty_metadata_markers(self, tmp_path, file):
        filename = tmp_path / file
        s = self.s
        m = markers.point(x=5, y=5)
        m.name = "test"
        s.add_marker(m, permanent=True)
        del s.metadata.Markers.test
        s.save(filename)
        s1 = load(filename)
        assert len(s1.metadata.Markers) == 0

    @zspy_marker
    def test_save_load_permanent_marker(self, tmp_path, file):
        filename = tmp_path / file
        s = self.s
        x, y = 5, 2
        color = 'red'
        size = 10
        name = 'testname'
        m = markers.point(x=x, y=y, color=color, size=size)
        m.name = name
        s.add_marker(m, permanent=True)
        s.save(filename)
        s1 = load(filename)
        assert s1.metadata.Markers.has_item(name)
        m1 = s1.metadata.Markers.get_item(name)
        assert m1.get_data_position('x1') == x
        assert m1.get_data_position('y1') == y
        assert m1.get_data_position('size') == size
        assert m1.marker_properties['color'] == color
        assert m1.name == name

    @zspy_marker
    def test_save_load_permanent_marker_all_types(self, tmp_path, file):
        filename = tmp_path / file
        s = self.s
        x1, y1, x2, y2 = 5, 2, 1, 8
        m0_list = [
            markers.point(x=x1, y=y1),
            markers.horizontal_line(y=y1),
            markers.horizontal_line_segment(x1=x1, x2=x2, y=y1),
            markers.line_segment(x1=x1, x2=x2, y1=y1, y2=y2),
            markers.rectangle(x1=x1, x2=x2, y1=y1, y2=y2),
            markers.text(x=x1, y=y1, text="test"),
            markers.vertical_line(x=x1),
            markers.vertical_line_segment(x=x1, y1=y1, y2=y2),
        ]
        for m in m0_list:
            s.add_marker(m, permanent=True)
        s.save(filename)
        s1 = load(filename)
        markers_dict = s1.metadata.Markers
        m0_dict_list = []
        m1_dict_list = []
        for m in m0_list:
            m0_dict_list.append(san_dict(m._to_dictionary()))
            m1_dict_list.append(
                san_dict(markers_dict.get_item(m.name)._to_dictionary()))
        assert len(list(s1.metadata.Markers)) == 8
        for m0_dict, m1_dict in zip(m0_dict_list, m1_dict_list):
            assert m0_dict == m1_dict

    @zspy_marker
    def test_save_load_horizontal_line_marker(self,tmp_path,file):
        filename = tmp_path / file
        s = self.s
        y = 8
        color = 'blue'
        linewidth = 2.5
        name = "horizontal_line_test"
        m = markers.horizontal_line(y=y, color=color, linewidth=linewidth)
        m.name = name
        s.add_marker(m, permanent=True)
        s.save(filename)
        s1 = load(filename)
        m1 = s1.metadata.Markers.get_item(name)
        assert san_dict(m1._to_dictionary()) == san_dict(m._to_dictionary())

    @zspy_marker
    def test_save_load_horizontal_line_segment_marker(self, tmp_path, file):
        filename = tmp_path / file
        s = self.s
        x1, x2, y = 1, 5, 8
        color = 'red'
        linewidth = 1.2
        name = "horizontal_line_segment_test"
        m = markers.horizontal_line_segment(
            x1=x1, x2=x2, y=y, color=color, linewidth=linewidth)
        m.name = name
        s.add_marker(m, permanent=True)
        s.save(filename)
        s1 = load(filename)
        m1 = s1.metadata.Markers.get_item(name)
        assert san_dict(m1._to_dictionary()) == san_dict(m._to_dictionary())

    @zspy_marker
    def test_save_load_vertical_line_marker(self, tmp_path, file):
        filename = tmp_path / file
        s = self.s
        x = 9
        color = 'black'
        linewidth = 3.5
        name = "vertical_line_test"
        m = markers.vertical_line(x=x, color=color, linewidth=linewidth)
        m.name = name
        s.add_marker(m, permanent=True)
        s.save(filename)
        s1 = load(filename)
        m1 = s1.metadata.Markers.get_item(name)
        assert san_dict(m1._to_dictionary()) == san_dict(m._to_dictionary())

    @zspy_marker
    def test_save_load_vertical_line_segment_marker(self, tmp_path, file):
        filename = tmp_path / file
        s = self.s
        x, y1, y2 = 2, 1, 3
        color = 'white'
        linewidth = 4.2
        name = "vertical_line_segment_test"
        m = markers.vertical_line_segment(
            x=x, y1=y1, y2=y2, color=color, linewidth=linewidth)
        m.name = name
        s.add_marker(m, permanent=True)
        s.save(filename)
        s1 = load(filename)
        m1 = s1.metadata.Markers.get_item(name)
        assert san_dict(m1._to_dictionary()) == san_dict(m._to_dictionary())

    @zspy_marker
    def test_save_load_line_segment_marker(self, tmp_path, file):
        filename = tmp_path / file
        s = self.s
        x1, x2, y1, y2 = 1, 9, 4, 7
        color = 'cyan'
        linewidth = 0.7
        name = "line_segment_test"
        m = markers.line_segment(
            x1=x1, x2=x2, y1=y1, y2=y2, color=color, linewidth=linewidth)
        m.name = name
        s.add_marker(m, permanent=True)
        s.save(filename)
        s1 = load(filename)
        m1 = s1.metadata.Markers.get_item(name)
        assert san_dict(m1._to_dictionary()) == san_dict(m._to_dictionary())

    @zspy_marker
    def test_save_load_point_marker(self, tmp_path, file):
        filename = tmp_path / file
        s = self.s
        x, y = 9, 8
        color = 'purple'
        name = "point test"
        m = markers.point(
            x=x, y=y, color=color)
        m.name = name
        s.add_marker(m, permanent=True)
        s.save(filename)
        s1 = load(filename)
        m1 = s1.metadata.Markers.get_item(name)
        assert san_dict(m1._to_dictionary()) == san_dict(m._to_dictionary())

    @zspy_marker
    def test_save_load_rectangle_marker(self, tmp_path, file):
        filename = tmp_path / file
        s = self.s
        x1, x2, y1, y2 = 2, 4, 1, 3
        color = 'yellow'
        linewidth = 5
        name = "rectangle_test"
        m = markers.rectangle(
            x1=x1, x2=x2, y1=y1, y2=y2, color=color, linewidth=linewidth)
        m.name = name
        s.add_marker(m, permanent=True)
        s.save(filename)
        s1 = load(filename)
        m1 = s1.metadata.Markers.get_item(name)
        assert san_dict(m1._to_dictionary()) == san_dict(m._to_dictionary())

    @zspy_marker
    def test_save_load_text_marker(self, tmp_path, file):
        filename = tmp_path / file
        s = self.s
        x, y = 3, 9.5
        color = 'brown'
        name = "text_test"
        text = "a text"
        m = markers.text(
            x=x, y=y, text=text, color=color)
        m.name = name
        s.add_marker(m, permanent=True)
        s.save(filename)
        s1 = load(filename)
        m1 = s1.metadata.Markers.get_item(name)
        assert san_dict(m1._to_dictionary()) == san_dict(m._to_dictionary())

    @zspy_marker
    @pytest.mark.parametrize('lazy', [True, False])
    def test_save_load_multidim_navigation_marker(self, tmp_path, file, lazy):
        filename = tmp_path / file
        x, y = (1, 2, 3), (5, 6, 7)
        name = 'test point'
        s = Signal2D(np.arange(300).reshape(3, 10, 10))
        if lazy:
            s = s.as_lazy()
        m = markers.point(x=x, y=y)
        m.name = name
        s.add_marker(m, permanent=True)
        s.save(filename)
        s1 = load(filename)
        m1 = s1.metadata.Markers.get_item(name)
        assert san_dict(m1._to_dictionary()) == san_dict(m._to_dictionary())
        assert m1.get_data_position('x1') == x[0]
        assert m1.get_data_position('y1') == y[0]
        s1.axes_manager.navigation_axes[0].index = 1
        assert m1.get_data_position('x1') == x[1]
        assert m1.get_data_position('y1') == y[1]
        s1.axes_manager.navigation_axes[0].index = 2
        assert m1.get_data_position('x1') == x[2]
        assert m1.get_data_position('y1') == y[2]

    def test_load_unknown_marker_type(self):
        # test_marker_bad_marker_type.hdf5 has 5 markers,
        # where one of them has an unknown marker type
        with pytest.warns(VisibleDeprecationWarning):
            fname = my_path / "hdf5_files" / "test_marker_bad_marker_type.hdf5"
            s = load(fname)
        assert len(s.metadata.Markers) == 4

    def test_load_missing_y2_value(self):
        # test_marker_point_y2_data_deleted.hdf5 has 5 markers,
        # where one of them is missing the y2 value, however the
        # the point marker only needs the x1 and y1 value to work
        # so this should load
        fname = my_path / "hdf5_files" / "test_marker_point_y2_data_deleted.hdf5"
        with pytest.warns(VisibleDeprecationWarning):
            s = load(fname)
        assert len(s.metadata.Markers) == 5


@zspy_marker
@pytest.mark.parametrize('lazy', [True, False])
def test_save_load_model(tmp_path, file, lazy):
    from hyperspy._components.gaussian import Gaussian
    filename = tmp_path / file
    s = Signal1D(np.ones((10, 10, 10, 10)))
    if lazy:
        s = s.as_lazy()
    m = s.create_model()
    m.append(Gaussian())
    m.store("test")
    s.save(filename)
    signal2 = load(filename)
    m2 = signal2.models.restore("test")
    assert m.signal == m2.signal


@pytest.mark.parametrize("compression", (None, "gzip", "lzf"))
def test_compression(compression, tmp_path):
    s = Signal1D(np.ones((3,3)))
    s.save(tmp_path / 'test_compression.hspy', overwrite=True,
           compression=compression)
    load(tmp_path / 'test_compression.hspy')


def test_strings_from_py2():
    s = EDS_TEM_Spectrum()
    assert isinstance(s.metadata.Sample.elements, list)


@zspy_marker
def test_save_ragged_array(tmp_path, file):
    a = np.array([0, 1])
    b = np.array([0, 1, 2])
    s = BaseSignal(np.array([a, b], dtype=object), ragged=True)
    fname = tmp_path / file
    s.save(fname)
    s1 = load(fname)
    for i in range(len(s.data)):
        np.testing.assert_allclose(s.data[i], s1.data[i])
    assert s.__class__ == s1.__class__


@zspy_marker
def test_save_ragged_dim2(tmp_path, file):
    x = np.empty(5, dtype=object)
    for i in range(1, 6):
        x[i - 1] = np.array([list(range(i)), list(range(i))])

    s = BaseSignal(x, ragged=True)

    filename = tmp_path / file
    s.save(filename)
    s2 = load(filename)

    for i, j in zip(s.data,s2.data):
        np.testing.assert_array_equal(i,j)


def test_load_missing_extension(caplog):
    path = my_path / "hdf5_files" / "hspy_ext_missing.hspy"
    with pytest.warns(UserWarning):
        s = load(path)
    assert "This file contains a signal provided by the hspy_ext_missing" in caplog.text
    with pytest.raises(ImportError):
       _ = s.models.restore("a")


@zspy_marker
def test_save_chunks_signal_metadata(tmp_path, file):
    N = 10
    dim = 3
    s = Signal1D(np.arange(N**dim).reshape([N]*dim))
    s.navigator = s.sum(-1)
    s.change_dtype('float')
    s.decomposition()

    filename = tmp_path / file
    chunks = (5, 5, 10)
    s.save(filename, chunks=chunks)
    s2 = load(filename, lazy=True)
    assert tuple([c[0] for c in s2.data.chunks]) == chunks


@zspy_marker
def test_chunking_saving_lazy(tmp_path, file):
    s = Signal2D(da.zeros((50, 100, 100))).as_lazy()
    s.data = s.data.rechunk([50, 25, 25])

    filename = tmp_path / 'test_chunking_saving_lazy.hspy'
    filename2 = tmp_path / 'test_chunking_saving_lazy_chunks_True.hspy'
    filename3 = tmp_path / 'test_chunking_saving_lazy_chunks_specified.hspy'
    s.save(filename)
    s1 = load(filename, lazy=True)
    assert s.data.chunks == s1.data.chunks

    # with chunks=True, use h5py chunking
    s.save(filename2, chunks=True)
    s2 = load(filename2, lazy=True)
    assert tuple([c[0] for c in s2.data.chunks]) == (7, 25, 25)

    # specify chunks
    chunks = (50, 10, 10)
    s.save(filename3, chunks=chunks)
    s3 = load(filename3, lazy=True)
    assert tuple([c[0] for c in s3.data.chunks]) == chunks


def test_saving_close_file(tmp_path):
    # Setup that we will reopen
    s = Signal1D(da.zeros((10, 100))).as_lazy()
    fname = tmp_path / 'test.hspy'
    s.save(fname, close_file=True)

    s2 = load(fname, lazy=True, mode='a')
    assert s2._get_file_handle() is not None
    s2.save(fname, close_file=True, overwrite=True)
    assert s2._get_file_handle() is None

    s3 = load(fname, lazy=True, mode='a')
    assert s3._get_file_handle() is not None
    s3.save(fname, close_file=False, overwrite=True)
    assert s3._get_file_handle() is not None
    s3.close_file()
    assert s3._get_file_handle() is None

    s4 = load(fname, lazy=True)
    assert s4._get_file_handle() is not None
    with pytest.raises(OSError):
        s4.save(fname, overwrite=True)


@zspy_marker
def test_saving_overwrite_data(tmp_path, file):
    s = Signal1D(da.zeros((10, 100))).as_lazy()
    kwds = dict(close_file=True) if file == 'test.hspy' else {}
    fname = tmp_path / file
    s.save(fname, **kwds)

    s2 = load(fname, lazy=True, mode='a')
    s2.axes_manager[0].units = 'nm'
    s2.data = da.ones((10, 100))
    s2.save(fname, overwrite=True, write_dataset=False)

    s3 = load(fname)
    assert s3.axes_manager[0].units == 'nm'
    # Still old data
    np.testing.assert_allclose(s3.data, np.zeros((10, 100)))

    s4 = load(fname)
    s4.data = da.ones((10, 100))
    if file == 'test.hspy':
        # try with opening non-lazily to check opening mode
        # only relevant for hdf5 file
        with pytest.raises(ValueError):
            s4.save(fname, overwrite=True, write_dataset=False, mode='w')

    s4.save(fname, overwrite=True, write_dataset=True)
    # make sure we can open it after, file haven't been corrupted
    _ = load(fname)
    # now new data
    @zspy_marker
    def test_chunking_saving_lazy_specify(self, tmp_path, file):
        filename = tmp_path / file
        s = Signal2D(da.zeros((50, 100, 100))).as_lazy()
        # specify chunks
        chunks = (50, 10, 10)
        s.data = s.data.rechunk([50, 25, 25])
        s.save(filename, chunks=chunks)
        s1 = load(filename, lazy=True)
        assert tuple([c[0] for c in s1.data.chunks]) == chunks


@pytest.mark.parametrize("target_size", (1e6,1e7))
def test_get_signal_chunks(target_size):
    shape = (2, 150, 3, 200, 1, 600, 1)
    chunks = get_signal_chunks(shape=shape,
                               dtype=np.int64,
                               signal_axes=(2, 3),
                               target_size=target_size)
    assert (np.prod(chunks)*8 < target_size)
    # The chunks must be smaller or equal that the corresponding sizes
    assert (np.array(chunks) <= np.array(shape)).all()


def test_get_signal_chunks_big_signal():
    # One signal exceeds the target size
    shape = (10, 1000, 5, 1000)
    chunks = get_signal_chunks(shape=shape,
                               dtype=np.int32,
                               signal_axes=(1, 3),
                               target_size=1e6)
    # The chunks must be smaller or equal that the corresponding sizes
    assert chunks == (1, 1000, 1, 1000)

def test_get_signal_chunks_small_dataset():
    # Whole dataset fits in one chunk
    shape = (10, 10, 2, 2)
    chunks = get_signal_chunks(shape=shape,
                               dtype=np.int32,
                               signal_axes=(2, 3),
                               target_size=1e6)
    # The chunks must be smaller or equal that the corresponding sizes
    assert chunks == shape

@zspy_marker
def test_error_saving(tmp_path, file):
    filename = tmp_path / file
    s = Signal1D(np.arange(10))

    with pytest.raises(ValueError):
        s.save(filename, write_dataset='unsupported_type')
        assert not s.metadata.Signal.has_item('record_by')


def test_more_recent_version_warning(tmp_path):
    filename = tmp_path / 'test.hspy'
    s = Signal1D(np.arange(10))
    s.save(filename)

    with h5py.File(filename, mode='a') as f:
        f.attrs["file_format_version"] = '99999999'

    with pytest.warns(UserWarning):
        s2 = load(filename)
    np.testing.assert_allclose(s.data, s2.data)


def test_load_future():
    """Code for creating future.hspy with hyperspy version 2.0.0

    import hyperspy.api as hs
    import numpy as np
    s = hs.signals.Signal2D(np.ones((10,10,10,10)))
    pos = np.array([[np.random.randint(0,10, (np.random.randint(1,10),2))  for i in range(10)]for j in range(10)], dtype=object)
    p = hs.plot.markers.Points(offsets = pos)
    s.add_marker(p, permanent=True)
    s.save("future.hspy")
    """
    with pytest.warns(UserWarning):
        future = load(filenames=my_path / "hdf5_files" / "future.hspy")
