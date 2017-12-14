import gzip
import hashlib
import os.path
import os
import shutil
import tempfile
import gc

import numpy as np
from numpy.testing import assert_allclose
import pytest

from hyperspy.io import load
from hyperspy import signals

TEST_FILES = ('Live Map 2_Img.ipr',
              'single_spect.spc',
              'spd_map.spc',
              'spd_map.spd')
MY_PATH = os.path.dirname(__file__)
TMP_DIR = tempfile.TemporaryDirectory()


def setup_module():
    import zipfile
    zipf = os.path.join(MY_PATH, "edax_files.zip")
    with zipfile.ZipFile(zipf, 'r') as zipped:
        zipped.extractall(TMP_DIR.name)
        # print(TMP_DIR.name)
        # spd_fname = os.path.join(tmp, TEST_FILES[3])
        # print(os.listdir(TMP_DIR.name))
        # print(spd_fname)


def teardown_module():
    TMP_DIR.cleanup()


class TestSpcSpectrum:

    @classmethod
    def setup_class(cls):
        cls.spc = load(os.path.join(TMP_DIR.name, "single_spect.spc"))

    @classmethod
    def teardown_class(cls):
        del cls.spc
        gc.collect()

    def test_data(self):
        assert np.uint32 == TestSpcSpectrum.spc.data.dtype     # test datatype
        assert (4096,) == TestSpcSpectrum.spc.data.shape       # test data shape
        assert (
            [0, 0, 0, 0, 0, 0, 1, 2, 3, 3, 10, 4, 10, 10, 45, 87, 146, 236,
             312, 342] == TestSpcSpectrum.spc.data[:20].tolist()) # test 1st 20 datapoints

    def test_parameters(self):
        elements = TestSpcSpectrum.spc.metadata.as_dictionary()['Sample']['elements']
        sem_dict = TestSpcSpectrum.spc.metadata.as_dictionary()[
            'Acquisition_instrument']['SEM']
        eds_dict = sem_dict['Detector']['EDS']
        signal_dict = TestSpcSpectrum.spc.metadata.as_dictionary()['Signal']

        # Testing SEM parameters
        assert_allclose(22, sem_dict['beam_energy'])
        assert_allclose(0, sem_dict['Stage']['tilt_alpha'])

        # Testing EDS parameters
        assert_allclose(0, eds_dict['azimuth_angle'])
        assert_allclose(34, eds_dict['elevation_angle'])
        assert_allclose(129.31299, eds_dict['energy_resolution_MnKa'],
                        atol=1E-5)
        assert_allclose(50.000004, eds_dict['live_time'], atol=1E-6)

        # Testing elements
        assert ({'Al', 'C', 'Ce', 'Cu', 'F', 'Ho', 'Mg', 'O'} ==
                set(elements))

        # Testing HyperSpy parameters
        assert True == signal_dict['binned']
        assert 'EDS_SEM' == signal_dict['signal_type']
        assert isinstance(TestSpcSpectrum.spc, signals.EDSSEMSpectrum)

    def test_axes(self):
        spc_ax_manager = {'axis-0': {'name': 'Energy',
                                     'navigate': False,
                                     'offset': 0.0,
                                     'scale': 0.01,
                                     'size': 4096,
                                     'units': 'keV'}}
        assert (spc_ax_manager ==
                TestSpcSpectrum.spc.axes_manager.as_dictionary())


class TestSpdMap:

    @classmethod
    def setup_class(cls):
        cls.spd = load(os.path.join(TMP_DIR.name, "spd_map.spd"))

    @classmethod
    def teardown_class(cls):
        del cls.spd
        gc.collect()

    def test_data(self):
        assert np.uint16 == TestSpdMap.spd.data.dtype     # test d_type
        assert (200, 256, 2500) == TestSpdMap.spd.data.shape  # test d_shape
        assert ([[[0, 0, 0, 0, 0],              # test random data
                  [0, 0, 1, 0, 1],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]],
                 [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]],
                 [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1],
                  [0, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0]],
                 [[0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0]],
                 [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 1],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0]]] ==
                TestSpdMap.spd.data[15:20, 15:20, 15:20].tolist())

    def test_parameters(self):
        elements = TestSpdMap.spd.metadata.as_dictionary()['Sample']['elements']
        sem_dict = TestSpdMap.spd.metadata.as_dictionary()[
            'Acquisition_instrument']['SEM']
        eds_dict = sem_dict['Detector']['EDS']
        signal_dict = TestSpdMap.spd.metadata.as_dictionary()['Signal']

        # Testing SEM parameters
        assert_allclose(22, sem_dict['beam_energy'])
        assert_allclose(0, sem_dict['Stage']['tilt_alpha'])

        # Testing EDS parameters
        assert_allclose(0, eds_dict['azimuth_angle'])
        assert_allclose(34, eds_dict['elevation_angle'])
        assert_allclose(126.60252, eds_dict['energy_resolution_MnKa'],
                        atol=1E-5)
        assert_allclose(2621.4399, eds_dict['live_time'], atol=1E-4)

        # Testing elements
        assert {'Ce', 'Co', 'Cr', 'Fe', 'Gd', 'La', 'Mg', 'O',
                'Sr'} == set(elements)

        # Testing HyperSpy parameters
        assert True == signal_dict['binned']
        assert 'EDS_SEM' == signal_dict['signal_type']
        assert isinstance(TestSpdMap.spd, signals.EDSSEMSpectrum)

    def test_axes(self):
        spd_ax_manager = {'axis-0': {'name': 'y',
                                     'navigate': True,
                                     'offset': 0.0,
                                     'scale': 0.014227345585823059,
                                     'size': 200,
                                     'units': '$\\mu m$'},
                          'axis-1': {'name': 'x',
                                     'navigate': True,
                                     'offset': 0.0,
                                     'scale': 0.014235896058380604,
                                     'size': 256,
                                     'units': '$\\mu m$'},
                          'axis-2': {'name': 'Energy',
                                     'navigate': False,
                                     'offset': 0.0,
                                     'scale': 0.0050000000000000001,
                                     'size': 2500,
                                     'units': 'keV'}}
        assert (spd_ax_manager ==
                TestSpdMap.spd.axes_manager.as_dictionary())

    def test_ipr_reading(self):
        ipr_header = TestSpdMap.spd.original_metadata['ipr_header']
        assert_allclose(0.014235896, ipr_header['mppX'])
        assert_allclose(0.014227346, ipr_header['mppY'])

    def test_spc_reading(self):
        # Test to make sure that spc metadata matches spd metadata
        spc_header = TestSpdMap.spd.original_metadata['spc_header']

        elements = TestSpdMap.spd.metadata.as_dictionary()['Sample']['elements']
        sem_dict = TestSpdMap.spd.metadata.as_dictionary()[
            'Acquisition_instrument']['SEM']
        eds_dict = sem_dict['Detector']['EDS']

        assert_allclose(spc_header.azimuth,
                        eds_dict['azimuth_angle'])
        assert_allclose(spc_header.detReso,
                        eds_dict['energy_resolution_MnKa'])
        assert_allclose(spc_header.elevation,
                        eds_dict['elevation_angle'])
        assert_allclose(spc_header.liveTime,
                        eds_dict['live_time'])
        assert_allclose(spc_header.evPerChan,
                        TestSpdMap.spd.axes_manager[2].scale * 1000)
        assert_allclose(spc_header.kV,
                        sem_dict['beam_energy'])
        assert_allclose(spc_header.numElem,
                        len(elements))
