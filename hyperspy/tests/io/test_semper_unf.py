# -*- coding: utf-8 -*-
# Copyright 2015 by Forschungszentrum Juelich GmbH
# Author: Jan Caron
#

import sys
import os.path
from os import remove

import nose
import nose.tools as nt
import numpy as np

from hyperspy.io import load
from hyperspy.signals import Signal, Image, Spectrum


my_path = os.path.dirname(__file__)

# Reference data:
data_signal = np.arange(27, dtype=np.float32).reshape((3, 3, 3)) / 2.
data_image = np.arange(16, dtype=np.float32).reshape((4, 4)) / 2.
data_spectrum = np.arange(10, dtype=np.float32) / 2.
data_image_int = np.arange(16, dtype=np.byte).reshape((4, 4))
data_image_int16 = np.arange(16, dtype=np.int16).reshape((4, 4))
data_image_int32 = np.arange(16, dtype=np.int32).reshape((4, 4))
data_image_complex = (data_image_int + 1j * data_image).astype(np.complex64)
test_title = 'This is a test!'


def test_signal_3d_loading():
    signal = load(os.path.join(my_path, 'unf_files', 'example_signal_3d.unf'))
    np.testing.assert_equal(signal.data, data_signal)
    np.testing.assert_equal(signal.original_metadata.IFORM, 2)  # float
    nt.assert_is_instance(signal, Signal)


def test_image_2d_loading():
    signal = load(os.path.join(my_path, 'unf_files', 'example_image_2d.unf'))
    np.testing.assert_equal(signal.data, data_image)
    np.testing.assert_equal(signal.original_metadata.IFORM, 2)  # float
    nt.assert_is_instance(signal, Image)


def test_spectrum_1d_loading():
    signal = load(
        os.path.join(
            my_path,
            'unf_files',
            'example_spectrum_1d.unf'))
    np.testing.assert_equal(signal.data, data_spectrum)
    np.testing.assert_equal(signal.original_metadata.IFORM, 2)  # float
    nt.assert_is_instance(signal, Spectrum)


def test_image_int16_loading():
    signal = load(
        os.path.join(
            my_path,
            'unf_files',
            'example_image_int16.unf'))
    np.testing.assert_equal(signal.data, data_image_int16)
    np.testing.assert_equal(signal.original_metadata.IFORM, 1)  # int16
    nt.assert_is_instance(signal, Image)


def test_image_int32_loading():
    signal = load(
        os.path.join(
            my_path,
            'unf_files',
            'example_image_int32.unf'))
    np.testing.assert_equal(signal.data, data_image_int32)
    np.testing.assert_equal(signal.original_metadata.IFORM, 4)  # int32
    nt.assert_is_instance(signal, Image)


def test_image_complex_loading():
    signal = load(
        os.path.join(
            my_path,
            'unf_files',
            'example_image_complex.unf'))
    np.testing.assert_equal(signal.data, data_image_complex)
    np.testing.assert_equal(signal.original_metadata.IFORM, 3)  # complex
    nt.assert_is_instance(signal, Image)


def test_with_title_loading():
    signal = load(os.path.join(my_path, 'unf_files', 'example_with_title.unf'))
    np.testing.assert_equal(signal.data, data_image)
    np.testing.assert_equal(signal.original_metadata.IFORM, 2)  # float
    np.testing.assert_equal(signal.metadata.General.title, 'This is a test!')
    nt.assert_is_instance(signal, Image)


def test_no_label_loading():
    signal = load(os.path.join(my_path, 'unf_files', 'example_no_label.unf'))
    np.testing.assert_equal(signal.data, data_image)
    np.testing.assert_equal(signal.original_metadata.ILABEL, 0)
    nt.assert_is_instance(signal, Image)


class TestCaseSaveAndRead():

    def test_save_and_read(self):
        signal_ref = Image(data_image)
        signal_ref.metadata.General.title = 'This is a test!'
        signal_ref.save(
            os.path.join(
                my_path,
                'unf_files',
                'example_temp.unf'),
            overwrite=True)
        signal = load(os.path.join(my_path, 'unf_files', 'example_temp.unf'))
        np.testing.assert_equal(signal.data, signal_ref.data)
        np.testing.assert_equal(
            signal.metadata.General.title,
            'This is a test!')
        nt.assert_is_instance(signal, Image)

    def tearDown(self):
        remove(os.path.join(my_path, 'unf_files', 'example_temp.unf'))


if __name__ == '__main__':
    nose.run(argv=[sys.argv[0], sys.modules[__name__].__file__, '-v'])
