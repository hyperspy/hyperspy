import os.path
from os import remove
import datetime
import h5py
import gc

import nose.tools as nt
import numpy as np

from hyperspy.io import load
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
        nt.assert_equal(
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
             4217.0], self.s.data.tolist())

    def test_original_metadata(self):
        nt.assert_equal(
            example1_original_metadata,
            self.s.original_metadata.as_dictionary())


class TestExample1_10(Example1):

    def setUp(self):
        self.s = load(os.path.join(
            my_path,
            "hdf5_files",
            "example1_v1.0.hdf5"))


class TestExample1_11(Example1):

    def setUp(self):
        self.s = load(os.path.join(
            my_path,
            "hdf5_files",
            "example1_v1.1.hdf5"))

# The following is commented out because
# the feature was removed in HyperSpy 1.0
# to fix a security flaw.
# class TestExample1_12(Example1):
#
#     def setUp(self):
#         self.s = load(os.path.join(
#             my_path,
#             "hdf5_files",
#             "example1_v1.2.hdf5"))
#
#     def test_date(self):
#         nt.assert_equal(
#             self.s.metadata.General.date,
#             datetime.date(
#                 1991,
#                 10,
#                 1))
#
#     def test_time(self):
#         nt.assert_equal(self.s.metadata.General.time, datetime.time(12, 0))


class TestLoadingNewSavedMetadata:

    def setUp(self):
        self.s = load(os.path.join(
            my_path,
            "hdf5_files",
            "with_lists_etc.hdf5"))

    def test_signal_inside(self):
        np.testing.assert_array_almost_equal(self.s.data,
                                             self.s.metadata.Signal.Noise_properties.variance.data)

    def test_empty_things(self):
        nt.assert_equal(self.s.metadata.test.empty_list, [])
        nt.assert_equal(self.s.metadata.test.empty_tuple, ())

    def test_simple_things(self):
        nt.assert_equal(self.s.metadata.test.list, [42])
        nt.assert_equal(self.s.metadata.test.tuple, (1, 2))

    def test_inside_things(self):
        nt.assert_equal(
            self.s.metadata.test.list_inside_list, [
                42, 137, [
                    0, 1]])
        nt.assert_equal(self.s.metadata.test.list_inside_tuple, (137, [42, 0]))
        nt.assert_equal(
            self.s.metadata.test.tuple_inside_tuple, (137, (123, 44)))
        nt.assert_equal(
            self.s.metadata.test.tuple_inside_list, [
                137, (123, 44)])

    def test_binary_string(self):
        import dill
        # apparently pickle is not "full" and marshal is not
        # backwards-compatible
        f = dill.loads(self.s.metadata.test.binary_string)
        nt.assert_equal(f(3.5), 4.5)


class TestSavingMetadataContainers:

    def setUp(self):
        self.s = BaseSignal([0.1])

    def test_save_unicode(self):
        s = self.s
        s.metadata.set_item('test', ['a', 'b', '\u6f22\u5b57'])
        s.save('tmp.hdf5', overwrite=True)
        l = load('tmp.hdf5')
        nt.assert_is_instance(l.metadata.test[0], str)
        nt.assert_is_instance(l.metadata.test[1], str)
        nt.assert_is_instance(l.metadata.test[2], str)
        nt.assert_equal(l.metadata.test[2], '\u6f22\u5b57')

    @nt.timed(1.0)
    def test_save_long_list(self):
        s = self.s
        s.metadata.set_item('long_list', list(range(10000)))
        s.save('tmp.hdf5', overwrite=True)

    def test_numpy_only_inner_lists(self):
        s = self.s
        s.metadata.set_item('test', [[1., 2], ('3', 4)])
        s.save('tmp.hdf5', overwrite=True)
        l = load('tmp.hdf5')
        nt.assert_is_instance(l.metadata.test, list)
        nt.assert_is_instance(l.metadata.test[0], list)
        nt.assert_is_instance(l.metadata.test[1], tuple)

    def test_numpy_general_type(self):
        s = self.s
        s.metadata.set_item('test', [[1., 2], ['3', 4]])
        s.save('tmp.hdf5', overwrite=True)
        l = load('tmp.hdf5')
        nt.assert_is_instance(l.metadata.test[0][0], float)
        nt.assert_is_instance(l.metadata.test[0][1], float)
        nt.assert_is_instance(l.metadata.test[1][0], str)
        nt.assert_is_instance(l.metadata.test[1][1], str)

    def test_general_type_not_working(self):
        s = self.s
        s.metadata.set_item('test', (BaseSignal([1]), 0.1, 'test_string'))
        s.save('tmp.hdf5', overwrite=True)
        l = load('tmp.hdf5')
        nt.assert_is_instance(l.metadata.test, tuple)
        nt.assert_is_instance(l.metadata.test[0], Signal1D)
        nt.assert_is_instance(l.metadata.test[1], float)
        nt.assert_is_instance(l.metadata.test[2], str)

    def test_unsupported_type(self):
        s = self.s
        s.metadata.set_item('test', Point2DROI(1, 2))
        s.save('tmp.hdf5', overwrite=True)
        l = load('tmp.hdf5')
        nt.assert_not_in('test', l.metadata)

    def tearDown(self):
        gc.collect()        # Make sure any memmaps are closed first!
        remove('tmp.hdf5')


def test_none_metadata():
    s = load(os.path.join(
        my_path,
        "hdf5_files",
        "none_metadata.hdf5"))
    nt.assert_is(s.metadata.should_be_None, None)


def test_rgba16():
    s = load(os.path.join(
        my_path,
        "hdf5_files",
        "test_rgba16.hdf5"))
    data = np.load(os.path.join(
        my_path,
        "npy_files",
        "test_rgba16.npy"))
    nt.assert_true((s.data == data).all())


class TestLoadingOOMReadOnly:

    def setUp(self):
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
        s = load('tmp.hdf5', load_to_memory=False)
        nt.assert_equal(self.shape, s.data.shape)
        nt.assert_is_instance(s.data, h5py.Dataset)

    def tearDown(self):
        gc.collect()        # Make sure any memmaps are closed first!
        try:
            remove('tmp.hdf5')
        except:
            # Don't fail tests if we cannot remove
            pass


class TestPassingArgs:

    def setUp(self):
        self.filename = 'testfile.hdf5'
        BaseSignal([1, 2, 3]).save(self.filename, compression_opts=8)

    def test_compression_opts(self):
        f = h5py.File(self.filename)
        d = f['Experiments/__unnamed__/data']
        nt.assert_equal(d.compression_opts, 8)
        nt.assert_equal(d.compression, 'gzip')
        f.close()

    def tearDown(self):
        remove(self.filename)


class TestAxesConfiguration:

    def setUp(self):
        self.filename = 'testfile.hdf5'
        s = BaseSignal(np.zeros((2, 2, 2, 2, 2)))
        s.axes_manager.signal_axes[0].navigate = True
        s.axes_manager.signal_axes[0].navigate = True
        s.save(self.filename)

    def test_axes_configuration(self):
        s = load(self.filename)
        nt.assert_equal(s.axes_manager.navigation_axes[0].index_in_array, 4)
        nt.assert_equal(s.axes_manager.navigation_axes[1].index_in_array, 3)
        nt.assert_equal(s.axes_manager.signal_dimension, 3)

    def tearDown(self):
        remove(self.filename)


def test_strings_from_py2():
    s = EDS_TEM_Spectrum()
    nt.assert_equal(s.metadata.Sample.elements.dtype.char, "U")
