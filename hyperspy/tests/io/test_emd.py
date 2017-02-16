# -*- coding: utf-8 -*-
# Copyright 2015 by Forschungszentrum Juelich GmbH
# Author: Jan Caron
#


import os.path
from os import remove
import tempfile

import numpy as np
import h5py

from hyperspy.io import load
from hyperspy.signals import BaseSignal, Signal2D, Signal1D


my_path = os.path.dirname(__file__)

# Reference data:
data_signal = np.arange(27).reshape((3, 3, 3))
data_image = np.arange(9).reshape((3, 3))
data_spectrum = np.arange(3)
data_save = np.arange(24).reshape((2, 3, 4))
sig_metadata = {'a': 1, 'b': 2}
user = {'name': 'John Doe', 'institution': 'TestUniversity',
        'department': 'Microscopy', 'email': 'johndoe@web.de'}
microscope = {'name': 'Titan', 'voltage': '300kV'}
sample = {'material': 'TiO2', 'preparation': 'FIB'}
comments = {'comment': 'Test'}
test_title = 'This is a test!'


def test_signal_3d_loading():
    signal = load(os.path.join(my_path, 'emd_files', 'example_signal.emd'))
    np.testing.assert_equal(signal.data, data_signal)
    assert isinstance(signal, BaseSignal)


def test_image_2d_loading():
    signal = load(os.path.join(my_path, 'emd_files', 'example_image.emd'))
    np.testing.assert_equal(signal.data, data_image)
    assert isinstance(signal, Signal2D)


def test_spectrum_1d_loading():
    signal = load(os.path.join(my_path, 'emd_files', 'example_spectrum.emd'))
    np.testing.assert_equal(signal.data, data_spectrum)
    assert isinstance(signal, Signal1D)


def test_metadata():
    signal = load(os.path.join(my_path, 'emd_files', 'example_metadata.emd'))
    np.testing.assert_equal(signal.data, data_image)
    np.testing.assert_equal(signal.metadata.General.title, test_title)
    np.testing.assert_equal(signal.metadata.General.user.as_dictionary(), user)
    np.testing.assert_equal(
        signal.metadata.General.microscope.as_dictionary(),
        microscope)
    np.testing.assert_equal(
        signal.metadata.General.sample.as_dictionary(), sample)
    np.testing.assert_equal(
        signal.metadata.General.comments.as_dictionary(),
        comments)
    for key, ref_value in sig_metadata.items():
        np.testing.assert_equal(
            signal.metadata.Signal.as_dictionary().get(key), ref_value)
    assert isinstance(signal, Signal2D)


def test_metadata_with_bytes_string():
    filename = os.path.join(
            my_path, 'emd_files', 'example_bytes_string_metadata.emd')
    f = h5py.File(filename, 'r')
    dim1 = f['test_group']['data_group']['dim1']
    dim1_name = dim1.attrs['name']
    dim1_units = dim1.attrs['units']
    f.close()
    assert type(dim1_name) is np.bytes_
    assert type(dim1_units) is np.bytes_
    signal = load(os.path.join(my_path, 'emd_files', filename))


def test_data_numpy_object_dtype():
    filename = os.path.join(
            my_path, 'emd_files', 'example_object_dtype_data.emd')
    signal = load(filename)
    assert len(signal) == 0


def test_data_axis_length_1():
    filename = os.path.join(
            my_path, 'emd_files', 'example_axis_len_1.emd')
    signal = load(filename)
    assert signal.data.shape == (5, 1, 5)


class TestMinimalSave():

    def setup_method(self, method):
        with tempfile.TemporaryDirectory() as tmp:
            self.filename = tmp + '/testfile.emd'
        self.signal = Signal1D([0, 1])

    def test_minimal_save(self):
        self.signal.save(self.filename)


class TestCaseSaveAndRead():

    def test_save_and_read(self):
        signal_ref = BaseSignal(data_save)
        signal_ref.metadata.General.title = test_title
        signal_ref.axes_manager[0].name = 'x'
        signal_ref.axes_manager[1].name = 'y'
        signal_ref.axes_manager[2].name = 'z'
        signal_ref.axes_manager[0].scale = 2
        signal_ref.axes_manager[1].scale = 3
        signal_ref.axes_manager[2].scale = 4
        signal_ref.axes_manager[0].offset = 10
        signal_ref.axes_manager[1].offset = 20
        signal_ref.axes_manager[2].offset = 30
        signal_ref.axes_manager[0].units = 'nmx'
        signal_ref.axes_manager[1].units = 'nmy'
        signal_ref.axes_manager[2].units = 'nmz'
        signal_ref.save(os.path.join(my_path, 'emd_files', 'example_temp.emd'), overwrite=True,
                        signal_metadata=sig_metadata, user=user, microscope=microscope,
                        sample=sample, comments=comments)
        signal = load(os.path.join(my_path, 'emd_files', 'example_temp.emd'))
        np.testing.assert_equal(signal.data, signal_ref.data)
        np.testing.assert_equal(signal.axes_manager[0].name, 'x')
        np.testing.assert_equal(signal.axes_manager[1].name, 'y')
        np.testing.assert_equal(signal.axes_manager[2].name, 'z')
        np.testing.assert_equal(signal.axes_manager[0].scale, 2)
        np.testing.assert_equal(signal.axes_manager[1].scale, 3)
        np.testing.assert_equal(signal.axes_manager[2].scale, 4)
        np.testing.assert_equal(signal.axes_manager[0].offset, 10)
        np.testing.assert_equal(signal.axes_manager[1].offset, 20)
        np.testing.assert_equal(signal.axes_manager[2].offset, 30)
        np.testing.assert_equal(signal.axes_manager[0].units, 'nmx')
        np.testing.assert_equal(signal.axes_manager[1].units, 'nmy')
        np.testing.assert_equal(signal.axes_manager[2].units, 'nmz')
        np.testing.assert_equal(signal.metadata.General.title, test_title)
        np.testing.assert_equal(
            signal.metadata.General.user.as_dictionary(), user)
        np.testing.assert_equal(
            signal.metadata.General.microscope.as_dictionary(),
            microscope)
        np.testing.assert_equal(
            signal.metadata.General.sample.as_dictionary(), sample)
        np.testing.assert_equal(
            signal.metadata.General.comments.as_dictionary(), comments)
        for key, ref_value in sig_metadata.items():
            np.testing.assert_equal(
                signal.metadata.Signal.as_dictionary().get(key), ref_value)
        assert isinstance(signal, BaseSignal)

    def teardown_method(self, method):
        remove(os.path.join(my_path, 'emd_files', 'example_temp.emd'))


if __name__ == '__main__':

    import pytest
    pytest.main(__name__)
