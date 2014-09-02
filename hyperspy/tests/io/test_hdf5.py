import os.path
import datetime

from nose.tools import (assert_equal,
                        assert_true,
                        assert_is)
import numpy as np

from hyperspy.io import load

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


class Example1():

    def test_data(self):
        assert_equal(
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
        assert_equal(
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
        assert_equal(self.s.metadata.General.date, datetime.date(1991, 10, 1))

    def test_time(self):
        assert_equal(self.s.metadata.General.time, datetime.time(12, 0))


def test_none_metadata():
    s = load(os.path.join(
        my_path,
        "hdf5_files",
        "none_metadata.hdf5"))
    assert_is(s.metadata.should_be_None, None)


def test_rgba16():
    s = load(os.path.join(
        my_path,
        "hdf5_files",
        "test_rgba16.hdf5"))
    data = np.load(os.path.join(
        my_path,
        "npy_files",
        "test_rgba16.npy"))
    assert_true((s.data == data).all())
