import os.path
from os import remove
import datetime

import nose.tools as nt
import numpy as np

from hyperspy.io import load
from hyperspy.signal import Signal

my_path = os.path.dirname(__file__)

data = np.array([4066., 3996., 3932., 3923., 5602., 5288., 7234., 7809.,
                 4710., 5015., 4366., 4524., 4832., 5474., 5718., 5034.,
                 4651., 4613., 4637., 4429., 4217.])
example1_original_metadata = {
    u'BEAMDIAM -nm': 100.0,
    u'BEAMKV   -kV': 120.0,
    u'CHOFFSET': -168.0,
    u'COLLANGLE-mR': 3.4,
    u'CONVANGLE-mR': 1.5,
    u'DATATYPE': u'XY',
    u'DATE': u'01-OCT-1991',
    u'DWELLTIME-ms': 100.0,
    u'ELSDET': u'SERIAL',
    u'EMISSION -uA': 5.5,
    u'FORMAT': u'EMSA/MAS Spectral Data File',
    u'MAGCAM': 100.0,
    u'NCOLUMNS': 1.0,
    u'NPOINTS': 20.0,
    u'OFFSET': 520.13,
    u'OPERMODE': u'IMAG',
    u'OWNER': u'EMSA/MAS TASK FORCE',
    u'PROBECUR -nA': 12.345,
    u'SIGNALTYPE': u'ELS',
    u'THICKNESS-nm': 50.0,
    u'TIME': u'12:00',
    u'TITLE': u'NIO EELS OK SHELL',
    u'VERSION': u'1.0',
    u'XLABEL': u'Energy',
    u'XPERCHAN': 3.1,
    u'XUNITS': u'eV',
    u'YLABEL': u'Counts',
    u'YUNITS': u'Intensity'}


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


class TestExample1_12(Example1):

    def setUp(self):
        self.s = load(os.path.join(
            my_path,
            "hdf5_files",
            "example1_v1.2.hdf5"))

    def test_date(self):
        nt.assert_equal(
            self.s.metadata.General.date,
            datetime.date(
                1991,
                10,
                1))

    def test_time(self):
        nt.assert_equal(self.s.metadata.General.time, datetime.time(12, 0))


class TestLoadingNewSavedMetadata:

    def setUp(self):
        self.s = load(os.path.join(
            my_path,
            "hdf5_files",
            "with_lists_etc.hdf5"))

    def test_signal_inside(self):
        nt.assert_true(
            np.all(
                self.s.data == self.s.metadata.Signal.Noise_properties.variance.data))

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
        import marshal
        import types
        f = types.FunctionType(
            marshal.loads(
                self.s.metadata.test.binary_string),
            globals())
        nt.assert_equal(f(3.5), 4.5)


class TestSavingMetadataContainers:

    def setUp(self):
        self.s = Signal([0.1])

    def test_save_unicode(self):
        s = self.s
        s.metadata.set_item('test', [u'a', u'b', u'\u6f22\u5b57'])
        s.save('tmp.hdf5', overwrite=True)
        l = load('tmp.hdf5')
        nt.assert_is_instance(l.metadata.test[0], unicode)
        nt.assert_is_instance(l.metadata.test[1], unicode)
        nt.assert_is_instance(l.metadata.test[2], unicode)
        nt.assert_equal(l.metadata.test[2], u'\u6f22\u5b57')

    @nt.timed(0.1)
    def test_save_long_list(self):
        s = self.s
        s.metadata.set_item('long_list', range(10000))
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
        nt.assert_is_instance(l.metadata.test[1][0], basestring)
        nt.assert_is_instance(l.metadata.test[1][1], basestring)

    def test_general_type_not_working(self):
        s = self.s
        s.metadata.set_item('test', (Signal([1]), 0.1, 'test_string'))
        s.save('tmp.hdf5', overwrite=True)
        l = load('tmp.hdf5')
        nt.assert_is_instance(l.metadata.test, tuple)
        nt.assert_is_instance(l.metadata.test[0], Signal)
        nt.assert_is_instance(l.metadata.test[1], float)
        nt.assert_is_instance(l.metadata.test[2], unicode)

    def tearDown(self):
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
