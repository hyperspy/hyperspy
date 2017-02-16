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


@pytest.fixture(scope="module")
def tmpdir():
    import zipfile
    zipf = os.path.join(MY_PATH, "edax_files.zip")
    with zipfile.ZipFile(zipf, 'r') as zipped:
        with tempfile.TemporaryDirectory() as tmp:
            zipped.extractall(tmp)
            # spd_fname = os.path.join(tmp, TEST_FILES[3])
            # print(os.listdir(tmpdir))
            # print(spd_fname)
            yield tmp
            # Force files release (required in Windows)
            gc.collect()


@pytest.fixture(scope="module")
def spd(tmpdir):
    signal = load(os.path.join(tmpdir, 'spd_map.spd'))
    yield signal
    signal.data._mmap.close()


@pytest.fixture(scope="module")
def spc(tmpdir):
    os.listdir(tmpdir)
    return load(os.path.join(tmpdir, "single_spect.spc"))


class TestSpcSpectrum:

    def test_data(self, spc):
        assert np.uint32 == spc.data.dtype     # test datatype
        assert (4096,) == spc.data.shape       # test data shape
        assert (
            [0, 0, 0, 0, 0, 0, 1, 2, 3, 3, 10, 4, 10, 10, 45, 87, 146, 236,
             312, 342] == spc.data[:20].tolist())   # test 1st 20 datapoints

    def test_parameters(self, spc):
        elements = spc.metadata.as_dictionary()['Sample']['elements']
        sem_dict = spc.metadata.as_dictionary()[
            'Acquisition_instrument']['SEM']
        eds_dict = sem_dict['Detector']['EDS']
        signal_dict = spc.metadata.as_dictionary()['Signal']

        # Testing SEM parameters
        assert_allclose(22, sem_dict['beam_energy'])
        assert_allclose(0, sem_dict['tilt_stage'])

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
        assert isinstance(spc, signals.EDSSEMSpectrum)

    def test_axes(self, spc):
        spc_ax_manager = {'axis-0': {'name': 'Energy',
                                     'navigate': False,
                                     'offset': 0.0,
                                     'scale': 0.01,
                                     'size': 4096,
                                     'units': 'keV'}}
        assert (spc_ax_manager ==
                spc.axes_manager.as_dictionary())


class TestSpdMap:

    def test_data(self, spd):
        assert np.uint16 == spd.data.dtype     # test d_type
        assert (200, 256, 2500) == spd.data.shape  # test d_shape
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
                spd.data[15:20, 15:20, 15:20].tolist())

    def test_parameters(self, spd):
        elements = spd.metadata.as_dictionary()['Sample']['elements']
        sem_dict = spd.metadata.as_dictionary()[
            'Acquisition_instrument']['SEM']
        eds_dict = sem_dict['Detector']['EDS']
        signal_dict = spd.metadata.as_dictionary()['Signal']

        # Testing SEM parameters
        assert_allclose(22, sem_dict['beam_energy'])
        assert_allclose(0, sem_dict['tilt_stage'])

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
        assert isinstance(spd, signals.EDSSEMSpectrum)

    def test_axes(self, spd):
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
                spd.axes_manager.as_dictionary())

    def test_ipr_reading(self, spd):
        ipr_header = spd.original_metadata['ipr_header']
        assert_allclose(0.014235896, ipr_header['mppX'])
        assert_allclose(0.014227346, ipr_header['mppY'])

    def test_spc_reading(self, spd):
        # Test to make sure that spc metadata matches spd metadata
        spc_header = spd.original_metadata['spc_header']

        elements = spd.metadata.as_dictionary()['Sample']['elements']
        sem_dict = spd.metadata.as_dictionary()[
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
                        spd.axes_manager[2].scale * 1000)
        assert_allclose(spc_header.kV,
                        sem_dict['beam_energy'])
        assert_allclose(spc_header.numElem,
                        len(elements))
