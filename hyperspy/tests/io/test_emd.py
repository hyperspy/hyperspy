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
data_signal = np.arange(27).reshape((3, 3, 3))
data_image = np.arange(9).reshape((3, 3))
data_spectrum = np.arange(3)




#data_image_byte = np.arange(25, dtype=np.byte).reshape((5, 5))  # Odd dim. tests strange read/write
#data_image_int16 = np.arange(16, dtype=np.int16).reshape((4, 4))
#data_image_int32 = np.arange(16, dtype=np.int32).reshape((4, 4))
#data_image_complex = (data_image_int32 + 1j*data_image).astype(np.complex64)


#filename, signal, signal_metadata=None, user=None,
#                microscope=None, sample=None, comments=None, **kwds

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
    nt.assert_is_instance(signal, Signal)


def test_image_2d_loading():
    signal = load(os.path.join(my_path, 'emd_files', 'example_image.emd'))
    np.testing.assert_equal(signal.data, data_image)
    nt.assert_is_instance(signal, Image)


def test_spectrum_1d_loading():
    signal = load(os.path.join(my_path, 'emd_files', 'example_spectrum.emd'))
    np.testing.assert_equal(signal.data, data_spectrum)
    nt.assert_is_instance(signal, Spectrum)


def test_metadata():
    signal = load(os.path.join(my_path, 'emd_files', 'example_metadata.emd'))
    np.testing.assert_equal(signal.data, data_image)
    np.testing.assert_equal(signal.metadata.General.title, test_title)
    np.testing.assert_equal(signal.metadata.General.user.as_dictionary(), user)
    np.testing.assert_equal(signal.metadata.General.microscope.as_dictionary(), microscope)
    np.testing.assert_equal(signal.metadata.General.sample.as_dictionary(), sample)
    np.testing.assert_equal(signal.metadata.General.comments.as_dictionary(), comments)
    for key, ref_value in sig_metadata.iteritems():
        np.testing.assert_equal(signal.metadata.Signal.as_dictionary().get(key), ref_value)
    nt.assert_is_instance(signal, Image)


class TestCaseSaveAndRead():

    def test_save_and_read(self):
        signal_ref = Image(data_image)
        signal_ref.metadata.General.title = test_title
        signal_ref.save(os.path.join(my_path, 'emd_files', 'example_temp.emd'), overwrite=True,
                        signal_metadata=sig_metadata, user=user, microscope=microscope,
                        sample=sample, comments=comments)
        signal = load(os.path.join(my_path, 'emd_files', 'example_temp.emd'))
        np.testing.assert_equal(signal.data, signal_ref.data)
        np.testing.assert_equal(signal.metadata.General.title, test_title)
        np.testing.assert_equal(signal.metadata.General.user.as_dictionary(), user)
        np.testing.assert_equal(signal.metadata.General.microscope.as_dictionary(), microscope)
        np.testing.assert_equal(signal.metadata.General.sample.as_dictionary(), sample)
        np.testing.assert_equal(signal.metadata.General.comments.as_dictionary(), comments)
        for key, ref_value in sig_metadata.iteritems():
            np.testing.assert_equal(signal.metadata.Signal.as_dictionary().get(key), ref_value)
        nt.assert_is_instance(signal, Image)

    def tearDown(self):
        remove(os.path.join(my_path, 'emd_files', 'example_temp.emd'))


if __name__ == '__main__':
    nose.run(argv=[sys.argv[0], sys.modules[__name__].__file__, '-v'])
