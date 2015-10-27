# -*- coding: utf-8 -*-
# Copyright 2015 by Forschungszentrum Juelich GmbH
# Author: Jan Caron
#

import os.path
from os import remove
import datetime
import h5py

import nose.tools as nt
import numpy as np
from numpy.testing import assert_equal

from hyperspy.io import load
from hyperspy.signals import Signal, Image, Spectrum

import unittest

my_path = os.path.dirname(__file__)


data_signal = np.arange(27, dtype=np.float32).reshape((3, 3, 3)) / 2.
data_image = np.arange(16, dtype=np.float32).reshape((4, 4)) / 2.
data_spectrum = np.arange(10, dtype=np.float32) / 2.
data_image_int = np.arange(16, dtype=np.int32).reshape((4, 4))
data_image_complex = (data_image_int + 1j*data_image).astype(np.complex64)
title = 'This is a test!'


class TestCaseSignal3D(unittest.TestCase):

    def setUp(self):
        self.signal = load(os.path.join(my_path, 'unf_files', 'example_signal_3d.unf'))

    def test_signal_3d_loading(self):
        assert_equal(self.signal.data, data_signal)
        assert_equal(self.signal.original_metadata.IFORM, 2)  # float
        assert isinstance(self.signal, Signal)


class TestCaseImage2D(unittest.TestCase):

    def setUp(self):
        self.signal = load(os.path.join(my_path, 'unf_files', 'example_image_2d.unf'))

    def test_image_2d_loading(self):
        assert_equal(self.signal.data, data_image)
        assert_equal(self.signal.original_metadata.IFORM, 2)  # float
        assert isinstance(self.signal, Image)


class TestCaseSpectrum1D(unittest.TestCase):

    def setUp(self):
        self.signal = load(os.path.join(my_path, 'unf_files', 'example_spectrum_1d.unf'))

    def test_spectrum_1d_loading(self):
        assert_equal(self.signal.data, data_spectrum)
        assert_equal(self.signal.original_metadata.IFORM, 2)  # float
        assert isinstance(self.signal, Spectrum)


class TestCaseImageInt(unittest.TestCase):

    def setUp(self):
        self.signal = load(os.path.join(my_path, 'unf_files', 'example_image_int.unf'))

    def test_image_int_loading(self):
        assert_equal(self.signal.data, data_image_int)
        assert_equal(self.signal.original_metadata.IFORM, 1)  # int
        assert isinstance(self.signal, Image)


class TestCaseImageComplex(unittest.TestCase):

    def setUp(self):
        self.signal = load(os.path.join(my_path, 'unf_files', 'example_image_complex.unf'))

    def test_image_complex_loading(self):
        assert_equal(self.signal.data, data_image_complex)
        assert_equal(self.signal.original_metadata.IFORM, 3)  # complex
        assert isinstance(self.signal, Image)


class TestCaseWithTitle(unittest.TestCase):

    def setUp(self):
        self.signal = load(os.path.join(my_path, 'unf_files', 'example_with_title.unf'))

    def test_with_title_loading(self):
        assert_equal(self.signal.data, data_image)
        assert_equal(self.signal.original_metadata.IFORM, 2)  # float
        assert_equal(self.signal.metadata.General.title, 'This is a test!')
        assert isinstance(self.signal, Image)


class TestCaseNoLabel(unittest.TestCase):

    def setUp(self):
        self.signal = load(os.path.join(my_path, 'unf_files', 'example_no_label.unf'))

    def test_no_label_loading(self):
        assert_equal(self.signal.data, data_image)
        assert_equal(self.signal.original_metadata.ILABEL, 0)
        assert isinstance(self.signal, Image)


class TestCaseSaveAndRead(unittest.TestCase):

    def test_save_and_read(self):
        signal_ref = Image(data_image)
        signal_ref.metadata.General.title = 'This is a test!'
        signal_ref.save(os.path.join(my_path, 'unf_files', 'example_temp.unf'), overwrite=True)
        signal = load(os.path.join(my_path, 'unf_files', 'example_temp.unf'))
        assert_equal(signal.data, signal_ref.data)
        assert_equal(signal.metadata.General.title, 'This is a test!')
        assert isinstance(signal, Image)

    def tearDown(self):
        remove(os.path.join(my_path, 'unf_files', 'example_temp.unf'))


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCaseSignal3D)
    unittest.TextTestRunner(verbosity=2).run(suite)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCaseImage2D)
    unittest.TextTestRunner(verbosity=2).run(suite)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCaseSpectrum1D)
    unittest.TextTestRunner(verbosity=2).run(suite)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCaseImageInt)
    unittest.TextTestRunner(verbosity=2).run(suite)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCaseImageComplex)
    unittest.TextTestRunner(verbosity=2).run(suite)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCaseWithTitle)
    unittest.TextTestRunner(verbosity=2).run(suite)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCaseNoLabel)
    unittest.TextTestRunner(verbosity=2).run(suite)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCaseSaveAndRead)
    unittest.TextTestRunner(verbosity=2).run(suite)
