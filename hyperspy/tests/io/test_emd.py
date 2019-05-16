# -*- coding: utf-8 -*-
# Copyright 2007-2015 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.

# The EMD format is a hdf5 standard proposed at Lawrence Berkeley
# National Lab (see http://emdatasets.com/ for more information).
# NOT to be confused with the FEI EMD format which was developed later.


import os.path
from os import remove
import shutil
import tempfile
import gc

from numpy.testing import assert_allclose
import numpy as np
import h5py
from dateutil import tz
from datetime import datetime
import pytest

from hyperspy.io import load
from hyperspy.signals import BaseSignal, Signal2D, Signal1D, EDSTEMSpectrum
from hyperspy.misc.test_utils import assert_deep_almost_equal


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
test_title = '/signals/This is a test!'


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
    pytest.importorskip("natsort", minversion="5.1.0")
    filename = os.path.join(
        my_path, 'emd_files', 'example_bytes_string_metadata.emd')
    f = h5py.File(filename, 'r')
    dim1 = f['test_group']['data_group']['dim1']
    dim1_name = dim1.attrs['name']
    dim1_units = dim1.attrs['units']
    f.close()
    assert isinstance(dim1_name, np.bytes_)
    assert isinstance(dim1_units, np.bytes_)
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


class TestDatasetName:

    def setup_method(self):
        tmpdir = tempfile.TemporaryDirectory()
        hdf5_dataset_path = os.path.join(tmpdir.name, "test_dataset.emd")
        f = h5py.File(hdf5_dataset_path, mode="w")
        f.attrs.create('version_major', 0)
        f.attrs.create('version_minor', 2)

        dataset_name_list = [
            '/experimental/science_data_0',
            '/experimental/science_data_1',
            '/processed/science_data_0']
        data_size_list = [(50, 50), (20, 10), (16, 32)]

        for dataset_name, data_size in zip(dataset_name_list, data_size_list):
            group = f.create_group(dataset_name)
            group.attrs.create('emd_group_type', 1)
            group.create_dataset(name='data', data=np.random.random(data_size))
            group.create_dataset(name='dim1', data=range(data_size[0]))
            group.create_dataset(name='dim2', data=range(data_size[1]))

        f.close()

        self.hdf5_dataset_path = hdf5_dataset_path
        self.tmpdir = tmpdir
        self.dataset_name_list = dataset_name_list
        self.data_size_list = data_size_list

    def teardown_method(self):
        self.tmpdir.cleanup()

    def test_load_with_dataset_name(self):
        s = load(self.hdf5_dataset_path)
        assert len(s) == len(self.dataset_name_list)
        for dataset_name, data_size in zip(
                self.dataset_name_list, self.data_size_list):
            s = load(self.hdf5_dataset_path, dataset_name=dataset_name)
            assert s.metadata.General.title == dataset_name
            assert s.data.shape == data_size

    def test_load_with_dataset_name_several(self):
        dataset_name = self.dataset_name_list[0:2]
        s = load(self.hdf5_dataset_path, dataset_name=dataset_name)
        assert len(s) == len(dataset_name)
        assert s[0].metadata.General.title in dataset_name
        assert s[1].metadata.General.title in dataset_name

    def test_wrong_dataset_name(self):
        with pytest.raises(IOError):
            load(self.hdf5_dataset_path, dataset_name='a_wrong_name')
        with pytest.raises(IOError):
            load(self.hdf5_dataset_path,
                 dataset_name=[self.dataset_name_list[0], 'a_wrong_name'])


class TestMinimalSave():

    def test_minimal_save(self):
        self.signal = Signal1D([0, 1])
        with tempfile.TemporaryDirectory() as tmp:
            self.signal.save(os.path.join(tmp, 'testfile.emd'))


class TestReadSeveralDatasets:

    def setup_method(self):
        tmpdir = tempfile.TemporaryDirectory()
        hdf5_dataset_path = os.path.join(tmpdir.name, "test_dataset.emd")
        f = h5py.File(hdf5_dataset_path, mode="w")
        f.attrs.create('version_major', 0)
        f.attrs.create('version_minor', 2)

        group_path_list = ['/exp/data_0', '/exp/data_1', '/calc/data_0']

        for group_path in group_path_list:
            group = f.create_group(group_path)
            group.attrs.create('emd_group_type', 1)
            data = np.random.random((128, 128))
            group.create_dataset(name='data', data=data)
            group.create_dataset(name='dim1', data=range(128))
            group.create_dataset(name='dim2', data=range(128))

        f.close()

        self.group_path_list = group_path_list
        self.hdf5_dataset_path = hdf5_dataset_path
        self.tmpdir = tmpdir

    def teardown_method(self):
        self.tmpdir.cleanup()

    def test_load_file(self):
        s = load(self.hdf5_dataset_path)
        assert len(s) == len(self.group_path_list)
        title_list = [s_temp.metadata.General.title for s_temp in s]
        assert sorted(self.group_path_list) == sorted(title_list)


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
        signal_ref.axes_manager[0].units = 'nm'
        signal_ref.axes_manager[1].units = 'µm'
        signal_ref.axes_manager[2].units = 'mm'
        signal_ref.save(os.path.join(my_path, 'emd_files', 'example_temp.emd'),
                        overwrite=True, signal_metadata=sig_metadata,
                        user=user, microscope=microscope, sample=sample,
                        comments=comments)
        signal = load(os.path.join(my_path, 'emd_files', 'example_temp.emd'))
        np.testing.assert_equal(signal.data, signal_ref.data)
        np.testing.assert_equal(signal.axes_manager[0].name, 'x')
        np.testing.assert_equal(signal.axes_manager[1].name, 'y')
        np.testing.assert_equal(signal.axes_manager[2].name, 'z')
        np.testing.assert_equal(signal.axes_manager[0].scale, 2)
        np.testing.assert_almost_equal(signal.axes_manager[1].scale, 3.0)
        np.testing.assert_almost_equal(signal.axes_manager[2].scale, 4.0)
        np.testing.assert_equal(signal.axes_manager[0].offset, 10)
        np.testing.assert_almost_equal(signal.axes_manager[1].offset, 20.0)
        np.testing.assert_almost_equal(signal.axes_manager[2].offset, 30.0)
        np.testing.assert_equal(signal.axes_manager[0].units, 'nm')
        np.testing.assert_equal(signal.axes_manager[1].units, 'µm')
        np.testing.assert_equal(signal.axes_manager[2].units, 'mm')
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


def _generate_parameters():
    parameters = []
    for lazy in [True, False]:
        for sum_EDS_detectors in [True, False]:
            parameters.append([lazy, sum_EDS_detectors])
    return parameters


class TestFeiEMD():

    fei_files_path = os.path.join(my_path, "emd_files", "fei_emd_files")

    @classmethod
    def setup_class(cls):
        import zipfile
        zipf = os.path.join(my_path, "emd_files", "fei_emd_files.zip")
        with zipfile.ZipFile(zipf, 'r') as zipped:
            zipped.extractall(cls.fei_files_path)

    @classmethod
    def teardown_class(cls):
        gc.collect()
        shutil.rmtree(cls.fei_files_path)

    @pytest.mark.parametrize("lazy", (True, False))
    def test_fei_emd_image(self, lazy):
        stage = {'tilt_alpha': 0.006,
                 'tilt_beta': 0.000,
                 'x': -0.000009,
                 'y': 0.000144,
                 'z': 0.000029}
        md = {'Acquisition_instrument': {'TEM': {'beam_energy': 200.0,
                                                 'camera_length': 98.0,
                                                 'magnification': 40000.0,
                                                 'microscope': 'Talos',
                                                 'Stage': stage}},
              'General': {'original_filename': 'fei_emd_image.emd',
                          'date': '2017-03-06',
                          'time': '09:56:41',
                          'time_zone': 'BST',
                          'title': 'HAADF'},
              'Signal': {'binned': False, 'signal_type': 'image'},
              '_HyperSpy': {'Folding': {'original_axes_manager': None,
                                        'original_shape': None,
                                        'signal_unfolded': False,
                                        'unfolded': False}}}

        # Update time and time_zone to local ones
        md['General']['time_zone'] = tz.tzlocal().tzname(datetime.today())
        dt = datetime.fromtimestamp(1488794201, tz=tz.tzutc())
        date, time = dt.astimezone(
            tz.tzlocal()).isoformat().split('+')[0].split('T')
        md['General']['date'] = date
        md['General']['time'] = time

        signal = load(os.path.join(self.fei_files_path, 'fei_emd_image.emd'),
                      lazy=lazy)
        if lazy:
            assert signal._lazy
            signal.compute(close_file=True)
        fei_image = np.load(os.path.join(self.fei_files_path,
                                         'fei_emd_image.npy'))
        assert signal.axes_manager[0].name == 'x'
        assert signal.axes_manager[0].units == 'um'
        assert_allclose(signal.axes_manager[0].scale, 0.00530241, rtol=1E-5)
        assert signal.axes_manager[1].name == 'y'
        assert signal.axes_manager[1].units == 'um'
        assert_allclose(signal.axes_manager[1].scale, 0.00530241, rtol=1E-5)
        assert_allclose(signal.data, fei_image)
        assert_deep_almost_equal(signal.metadata.as_dictionary(), md)
        assert isinstance(signal, Signal2D)

    @pytest.mark.parametrize("lazy", (True, False))
    def test_fei_emd_spectrum(self, lazy):
        signal = load(os.path.join(
            self.fei_files_path, 'fei_emd_spectrum.emd'), lazy=lazy)
        if lazy:
            assert signal._lazy
            signal.compute(close_file=True)
        fei_spectrum = np.load(os.path.join(self.fei_files_path,
                                            'fei_emd_spectrum.npy'))
        np.testing.assert_equal(signal.data, fei_spectrum)
        assert isinstance(signal, Signal1D)

    @pytest.mark.parametrize("lazy", (True, False))
    def test_fei_emd_si(self, lazy):
        signal = load(os.path.join(self.fei_files_path, 'fei_emd_si.emd'),
                      lazy=lazy)
        if lazy:
            assert signal[1]._lazy
            signal[1].compute(close_file=True)
        fei_si = np.load(os.path.join(self.fei_files_path, 'fei_emd_si.npy'))
        np.testing.assert_equal(signal[1].data, fei_si)
        assert isinstance(signal[1], Signal1D)

    @pytest.mark.parametrize("lazy", (True, False))
    def test_fei_emd_si_non_square_10frames(self, lazy):
        s = load(os.path.join(
            self.fei_files_path, 'fei_SI_SuperX-HAADF_10frames_10x50.emd'),
            lazy=lazy)
        signal = s[1]
        if lazy:
            assert signal._lazy
            signal.compute(close_file=True)
        assert isinstance(signal, EDSTEMSpectrum)
        assert signal.axes_manager[0].name == 'x'
        assert signal.axes_manager[0].size == 10
        assert signal.axes_manager[0].units == 'nm'
        assert_allclose(signal.axes_manager[0].scale, 1.234009, atol=1E-5)
        assert signal.axes_manager[1].name == 'y'
        assert signal.axes_manager[1].size == 50
        assert signal.axes_manager[1].units == 'nm'
        assert_allclose(signal.axes_manager[1].scale, 1.234009, atol=1E-5)
        assert signal.axes_manager[2].name == 'X-ray energy'
        assert signal.axes_manager[2].size == 4096
        assert signal.axes_manager[2].units == 'keV'
        assert_allclose(signal.axes_manager[2].scale, 0.005, atol=1E-5)

        signal0 = s[0]
        if lazy:
            assert signal0._lazy
            signal0.compute(close_file=True)
        assert isinstance(signal0, Signal2D)
        assert signal0.axes_manager[0].name == 'x'
        assert signal0.axes_manager[0].size == 10
        assert signal0.axes_manager[0].units == 'nm'
        assert_allclose(signal0.axes_manager[0].scale, 1.234009, atol=1E-5)
        assert signal0.axes_manager[1].name == 'y'
        assert signal0.axes_manager[1].size == 50
        assert signal0.axes_manager[1].units == 'nm'

        s = load(os.path.join(self.fei_files_path,
                              'fei_SI_SuperX-HAADF_10frames_10x50.emd'),
                 sum_frames=False,
                 SI_dtype=np.uint8,
                 rebin_energy=256,
                 lazy=lazy)
        signal = s[1]
        if lazy:
            assert signal._lazy
            signal.compute(close_file=True)
        assert isinstance(signal, EDSTEMSpectrum)
        assert signal.axes_manager.navigation_shape == (10, 50, 10)
        assert signal.axes_manager[0].name == 'x'
        assert signal.axes_manager[0].size == 10
        assert signal.axes_manager[0].units == 'nm'
        assert_allclose(signal.axes_manager[0].scale, 1.234009, atol=1E-5)
        assert signal.axes_manager[1].name == 'y'
        assert signal.axes_manager[1].size == 50
        assert signal.axes_manager[1].units == 'nm'
        assert_allclose(signal.axes_manager[1].scale, 1.234009, atol=1E-5)
        assert signal.axes_manager[2].name == 'Time'
        assert signal.axes_manager[2].size == 10
        assert signal.axes_manager[2].units == 's'
        assert_allclose(signal.axes_manager[2].scale, 0.76800, atol=1E-5)
        assert signal.axes_manager[3].name == 'X-ray energy'
        assert signal.axes_manager[3].size == 16
        assert signal.axes_manager[3].units == 'keV'
        assert_allclose(signal.axes_manager[3].scale, 1.28, atol=1E-5)

        s = load(os.path.join(self.fei_files_path,
                              'fei_SI_SuperX-HAADF_10frames_10x50.emd'),
                 sum_frames=False,
                 last_frame=5,
                 SI_dtype=np.uint8,
                 rebin_energy=256,
                 lazy=lazy)
        signal = s[1]
        if lazy:
            assert signal._lazy
            signal.compute(close_file=True)
        assert isinstance(signal, EDSTEMSpectrum)
        assert signal.axes_manager.navigation_shape == (10, 50, 5)
        assert signal.axes_manager[0].name == 'x'
        assert signal.axes_manager[0].size == 10
        assert signal.axes_manager[0].units == 'nm'
        assert_allclose(signal.axes_manager[0].scale, 1.234009, atol=1E-5)
        assert signal.axes_manager[1].name == 'y'
        assert signal.axes_manager[1].size == 50
        assert signal.axes_manager[1].units == 'nm'
        assert_allclose(signal.axes_manager[1].scale, 1.234009, atol=1E-5)
        assert signal.axes_manager[2].name == 'Time'
        assert signal.axes_manager[2].size == 5
        assert signal.axes_manager[2].units == 's'
        assert_allclose(signal.axes_manager[2].scale, 0.76800, atol=1E-5)
        assert signal.axes_manager[3].name == 'X-ray energy'
        assert signal.axes_manager[3].size == 16
        assert signal.axes_manager[3].units == 'keV'
        assert_allclose(signal.axes_manager[3].scale, 1.28, atol=1E-5)

        s = load(os.path.join(self.fei_files_path,
                              'fei_SI_SuperX-HAADF_10frames_10x50.emd'),
                 sum_frames=False,
                 first_frame=4,
                 SI_dtype=np.uint8,
                 rebin_energy=256,
                 lazy=lazy)
        signal = s[1]
        if lazy:
            assert signal._lazy
            signal.compute(close_file=True)
        assert isinstance(signal, EDSTEMSpectrum)
        assert signal.axes_manager.navigation_shape == (10, 50, 6)
        assert signal.axes_manager[0].name == 'x'
        assert signal.axes_manager[0].size == 10
        assert signal.axes_manager[0].units == 'nm'
        assert_allclose(signal.axes_manager[0].scale, 1.234009, atol=1E-5)
        assert signal.axes_manager[1].name == 'y'
        assert signal.axes_manager[1].size == 50
        assert signal.axes_manager[1].units == 'nm'
        assert_allclose(signal.axes_manager[1].scale, 1.234009, atol=1E-5)
        assert signal.axes_manager[2].name == 'Time'
        assert signal.axes_manager[2].size == 6
        assert signal.axes_manager[2].units == 's'
        assert_allclose(signal.axes_manager[2].scale, 0.76800, atol=1E-5)
        assert signal.axes_manager[3].name == 'X-ray energy'
        assert signal.axes_manager[3].size == 16
        assert signal.axes_manager[3].units == 'keV'
        assert_allclose(signal.axes_manager[3].scale, 1.28, atol=1E-5)

    @pytest.mark.parametrize("lazy", (True, False))
    def test_fei_emd_si_non_square_20frames(self, lazy):
        s = load(os.path.join(
            self.fei_files_path,
            'fei_SI_SuperX-HAADF_20frames_10x50.emd'),
            lazy=lazy)
        signal = s[1]
        if lazy:
            assert signal._lazy
            signal.compute(close_file=True)
        assert isinstance(signal, EDSTEMSpectrum)
        assert signal.axes_manager[0].name == 'x'
        assert signal.axes_manager[0].size == 10
        assert signal.axes_manager[0].units == 'nm'
        assert_allclose(signal.axes_manager[0].scale, 1.234009, atol=1E-5)
        assert signal.axes_manager[1].name == 'y'
        assert signal.axes_manager[1].size == 50
        assert signal.axes_manager[1].units == 'nm'
        assert_allclose(signal.axes_manager[1].scale, 1.234009, atol=1E-5)
        assert signal.axes_manager[2].name == 'X-ray energy'
        assert signal.axes_manager[2].size == 4096
        assert signal.axes_manager[2].units == 'keV'
        assert_allclose(signal.axes_manager[2].scale, 0.005, atol=1E-5)

    @pytest.mark.parametrize("lazy", (True, False))
    def test_fei_emd_si_non_square_20frames_2eV(self, lazy):
        s = load(os.path.join(
            self.fei_files_path,
            'fei_SI_SuperX-HAADF_20frames_10x50_2ev.emd'),
            lazy=lazy)
        signal = s[1]
        if lazy:
            assert signal._lazy
            signal.compute(close_file=True)
        assert isinstance(signal, EDSTEMSpectrum)
        assert signal.axes_manager[0].name == 'x'
        assert signal.axes_manager[0].size == 10
        assert signal.axes_manager[0].units == 'nm'
        assert_allclose(signal.axes_manager[0].scale, 1.234009, atol=1E-5)
        assert signal.axes_manager[1].name == 'y'
        assert signal.axes_manager[1].size == 50
        assert signal.axes_manager[1].units == 'nm'
        assert_allclose(signal.axes_manager[1].scale, 1.234009, atol=1E-5)
        assert signal.axes_manager[2].name == 'X-ray energy'
        assert signal.axes_manager[2].size == 4096
        assert signal.axes_manager[2].units == 'keV'
        assert_allclose(signal.axes_manager[2].scale, 0.002, atol=1E-5)

    @pytest.mark.parametrize("lazy", (True, False))
    def test_fei_emd_si_frame_range(self, lazy):
        signal = load(os.path.join(self.fei_files_path, 'fei_emd_si.emd'),
                      first_frame=2, last_frame=4, lazy=lazy)
        fei_si = np.load(os.path.join(self.fei_files_path,
                                      'fei_emd_si_frame.npy'))
        if lazy:
            assert signal[1]._lazy
            signal[1].compute(close_file=True)
        np.testing.assert_equal(signal[1].data, fei_si)
        assert isinstance(signal[1], Signal1D)

    @pytest.mark.parametrize(["lazy", "sum_EDS_detectors"],
                             _generate_parameters())
    def test_fei_si_4detectors(self, lazy, sum_EDS_detectors):
        fname = os.path.join(self.fei_files_path,
                             'fei_SI_EDS-HAADF-4detectors_2frames.emd')
        signal = load(fname, sum_EDS_detectors=sum_EDS_detectors, lazy=lazy)
        if lazy:
            assert signal[1]._lazy
            signal[1].compute(close_file=True)
        length = 6
        if not sum_EDS_detectors:
            length += 3
        assert len(signal) == length
        # TODO: add parsing azimuth_angle

    def test_fei_emd_ceta_camera(self):
        signal = load(
            os.path.join(
                self.fei_files_path,
                '1532 Camera Ceta.emd'))
        assert_allclose(signal.data, np.zeros((64, 64)))
        assert isinstance(signal, Signal2D)
        date, time = self._convert_datetime(1512055942.914275).split('T')
        assert signal.metadata.General.date == date
        assert signal.metadata.General.time == time
        assert signal.metadata.General.time_zone == self._get_local_time_zone()

        signal = load(
            os.path.join(
                self.fei_files_path,
                '1854 Camera Ceta.emd'))
        assert_allclose(signal.data, np.zeros((64, 64)))
        assert isinstance(signal, Signal2D)

    def _convert_datetime(self, unix_time):
        # Since we don't know the actual time zone of where the data have been
        # acquired, we convert the datetime to the local time for convenience
        dt = datetime.fromtimestamp(float(unix_time), tz=tz.tzutc())
        return dt.astimezone(tz.tzlocal()).isoformat().split('+')[0]

    def _get_local_time_zone(self):
        return tz.tzlocal().tzname(datetime.today())

    def time_loading_frame(self):
        # Run this function to check the loading time when loading EDS data
        import time
        frame_number = 100
        point_measurement = 15
        frame_offsets = np.arange(0, point_measurement * frame_number,
                                  frame_number)
        time_data = np.zeros_like(frame_offsets)
        path = 'path to large dataset'
        for i, frame_offset in enumerate(frame_offsets):
            print(frame_offset + frame_number)
            t0 = time.time()
            load(os.path.join(path, 'large dataset.emd'),
                 first_frame=frame_offset, last_frame=frame_offset + frame_number)
            t1 = time.time()
            time_data[i] = t1 - t0
        import matplotlib.pyplot as plt
        plt.plot(frame_offsets, time_data)
        plt.xlabel('Frame offset')
        plt.xlabel('Loading time (s)')
