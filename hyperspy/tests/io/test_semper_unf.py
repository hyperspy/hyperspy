# -*- coding: utf-8 -*-
# Copyright 2016 by Forschungszentrum Juelich GmbH
# Author: Jan Caron
#


import os.path
from os import remove

import nose.tools as nt
import numpy as np

from hyperspy.io import load
from hyperspy.signals import BaseSignal, Signal2D, Signal1D, ComplexSignal


my_path = os.path.dirname(__file__)

# Reference data:
data_signal = np.arange(27, dtype=np.float32).reshape((3, 3, 3)) / 2.
data_image = np.arange(16, dtype=np.float32).reshape((4, 4)) / 2.
data_spectrum = np.arange(10, dtype=np.float32) / 2.
data_image_byte = np.arange(
    25, dtype=np.byte).reshape(
    (5, 5))  # Odd dim. tests strange read/write
data_image_int16 = np.arange(16, dtype=np.int16).reshape((4, 4))
data_image_int32 = np.arange(16, dtype=np.int32).reshape((4, 4))
data_image_complex = (data_image_int32 + 1j * data_image).astype(np.complex64)
test_title = 'This is a test!'


def test_signal_3d_loading():
    signal = load(os.path.join(my_path, 'unf_files', 'example_signal_3d.unf'))
    np.testing.assert_equal(signal.data, data_signal)
    np.testing.assert_equal(signal.original_metadata.IFORM, 2)  # float
    nt.assert_is_instance(signal, BaseSignal)


def test_image_2d_loading():
    signal = load(os.path.join(my_path, 'unf_files', 'example_image_2d.unf'))
    np.testing.assert_equal(signal.data, data_image)
    np.testing.assert_equal(signal.original_metadata.IFORM, 2)  # float
    nt.assert_is_instance(signal, Signal2D)


def test_spectrum_1d_loading():
    signal = load(
        os.path.join(
            my_path,
            'unf_files',
            'example_spectrum_1d.unf'))
    np.testing.assert_equal(signal.data, data_spectrum)
    np.testing.assert_equal(signal.original_metadata.IFORM, 2)  # float
    nt.assert_is_instance(signal, Signal1D)


def test_image_byte_loading():
    signal = load(os.path.join(my_path, 'unf_files', 'example_image_byte.unf'))
    np.testing.assert_equal(signal.data, data_image_byte)
    np.testing.assert_equal(signal.original_metadata.IFORM, 0)  # byte
    nt.assert_is_instance(signal, Signal2D)


def test_image_int16_loading():
    signal = load(
        os.path.join(
            my_path,
            'unf_files',
            'example_image_int16.unf'))
    np.testing.assert_equal(signal.data, data_image_int16)
    np.testing.assert_equal(signal.original_metadata.IFORM, 1)  # int16
    nt.assert_is_instance(signal, Signal2D)


def test_image_int32_loading():
    signal = load(
        os.path.join(
            my_path,
            'unf_files',
            'example_image_int32.unf'))
    np.testing.assert_equal(signal.data, data_image_int32)
    np.testing.assert_equal(signal.original_metadata.IFORM, 4)  # int32
    nt.assert_is_instance(signal, Signal2D)


def test_image_complex_loading():
    signal = load(
        os.path.join(
            my_path,
            'unf_files',
            'example_image_complex.unf'))
    np.testing.assert_equal(signal.data, data_image_complex)
    np.testing.assert_equal(signal.original_metadata.IFORM, 3)  # complex
    nt.assert_is_instance(signal, ComplexSignal)


def test_with_title_loading():
    signal = load(os.path.join(my_path, 'unf_files', 'example_with_title.unf'))
    np.testing.assert_equal(signal.data, data_image)
    np.testing.assert_equal(signal.original_metadata.IFORM, 2)  # float
    np.testing.assert_equal(signal.metadata.General.title, test_title)
    nt.assert_is_instance(signal, Signal2D)


def test_no_label_loading():
    signal = load(os.path.join(my_path, 'unf_files', 'example_no_label.unf'))
    np.testing.assert_equal(signal.data, data_image)
    np.testing.assert_equal(signal.original_metadata.ILABEL, 0)
    nt.assert_is_instance(signal, Signal2D)


class TestCaseSaveAndReadImage():

    def test_save_and_read(self):
        signal_ref = Signal2D(data_image)
        signal_ref.metadata.General.title = test_title
        signal_ref.save(
            os.path.join(
                my_path,
                'unf_files',
                'example_temp.unf'),
            overwrite=True)
        signal = load(os.path.join(my_path, 'unf_files', 'example_temp.unf'))
        np.testing.assert_equal(signal.data, signal_ref.data)
        np.testing.assert_equal(signal.metadata.General.title, test_title)
        nt.assert_is_instance(signal, Signal2D)

    def tearDown(self):
        remove(os.path.join(my_path, 'unf_files', 'example_temp.unf'))


class TestCaseSaveAndReadByte():

    def test_save_and_read(self):
        signal_ref = Signal2D(data_image_byte)
        signal_ref.metadata.General.title = test_title
        signal_ref.save(os.path.join(my_path, 'unf_files', 'example_temp.unf'),
                        overwrite=True)
        signal = load(os.path.join(my_path, 'unf_files', 'example_temp.unf'))
        np.testing.assert_equal(signal.data, signal_ref.data)
        np.testing.assert_equal(signal.metadata.General.title, test_title)
        nt.assert_is_instance(signal, Signal2D)

    def tearDown(self):
        remove(os.path.join(my_path, 'unf_files', 'example_temp.unf'))


if __name__ == '__main__':
    import nose
    nose.run(defaultTest=__name__)
