# -*- coding: utf-8 -*-
# Copyright 2015 by Forschungszentrum Juelich GmbH
# Author: Jan Caron
#


import os.path
from os import remove

import nose.tools as nt
import numpy as np

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
    nt.assert_is_instance(signal, BaseSignal)


def test_image_2d_loading():
    signal = load(os.path.join(my_path, 'emd_files', 'example_image.emd'))
    np.testing.assert_equal(signal.data, data_image)
    nt.assert_is_instance(signal, Signal2D)


def test_spectrum_1d_loading():
    signal = load(os.path.join(my_path, 'emd_files', 'example_spectrum.emd'))
    np.testing.assert_equal(signal.data, data_spectrum)
    nt.assert_is_instance(signal, Signal1D)


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
    nt.assert_is_instance(signal, Signal2D)


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
        nt.assert_is_instance(signal, BaseSignal)

    def tearDown(self):
        remove(os.path.join(my_path, 'emd_files', 'example_temp.emd'))


if __name__ == '__main__':
    import nose
    nose.run(defaultTest=__name__)
