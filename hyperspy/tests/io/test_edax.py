import os.path
import os
import tempfile
import gc
import urllib.request
import zipfile
import hashlib

import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest
import requests

from hyperspy.io import load
from hyperspy import signals
from hyperspy.misc.test_utils import assert_deep_almost_equal


MY_PATH = os.path.dirname(__file__)
ZIPF = os.path.join(MY_PATH, "edax_files.zip")
TMP_DIR = tempfile.TemporaryDirectory()
TEST_FILES_OK = os.path.isfile(ZIPF)
REASON = ""
SHA256SUM = "e217c71efbd208da4b52e9cf483443f9da2175f2924a96447ed393086fe32008"


# The test files are not included in HyperSpy v1.4 because their file size is 36.5MB
# taking the HyperSpy source distribution file size above PyPI's 60MB limit.
# As a temporary solution, we attempt to download the test files from GitHub
# and skip the tests if the download fails.
if not TEST_FILES_OK:
    try:
        r = requests.get(
            "https://github.com/hyperspy/hyperspy/blob/e7a323a3bb9b237c24bd9267d2cc4fcb31bb99f3/hyperspy/tests/io/edax_files.zip?raw=true")

        SHA256SUM_GOT = hashlib.sha256(r.content).hexdigest()
        if SHA256SUM_GOT == SHA256SUM:
            ZIPF = os.path.join(TMP_DIR.name, "edax_files.zip")
            with open(ZIPF, 'wb') as f:
                f.write(r.content)
            TEST_FILES_OK = True
        else:
            REASON = "wrong sha256sum of downloaded file. Expected: %s, got: %s" % SHA256SUM, SHA256SUM_GOT
    except BaseException as e:
        REASON = "download of EDAX test files failed: %s" % e


def setup_module():
    if TEST_FILES_OK:
        with zipfile.ZipFile(ZIPF, 'r') as zipped:
            zipped.extractall(TMP_DIR.name)


pytestmark = pytest.mark.skipif(not TEST_FILES_OK,
                                reason=REASON)


def teardown_module():
    TMP_DIR.cleanup()


class TestSpcSpectrum_v061_xrf:

    @classmethod
    def setup_class(cls):
        cls.spc = load(os.path.join(TMP_DIR.name, "spc0_61-ipr333_xrf.spc"))
        cls.spc_loadAll = load(os.path.join(TMP_DIR.name,
                                            "spc0_61-ipr333_xrf.spc"),
                               load_all_spc=True)

    @classmethod
    def teardown_class(cls):
        del cls.spc, cls.spc_loadAll
        gc.collect()

    def test_data(self):
        # test datatype
        assert np.uint32 == TestSpcSpectrum_v061_xrf.spc.data.dtype
        # test data shape
        assert (4000,) == TestSpcSpectrum_v061_xrf.spc.data.shape
        # test 40 datapoints
        assert (
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 319, 504, 639, 924,
             1081, 1326, 1470, 1727, 1983, 2123, 2278, 2509, 2586, 2639,
             2681, 2833, 2696, 2704, 2812, 2745, 2709, 2647, 2608, 2620,
             2571, 2669] == TestSpcSpectrum_v061_xrf.spc.data[:40].tolist())

    def test_parameters(self):
        elements = TestSpcSpectrum_v061_xrf.spc.metadata.as_dictionary()[
            'Sample']['elements']
        sem_dict = TestSpcSpectrum_v061_xrf.spc.metadata.as_dictionary()[
            'Acquisition_instrument']['SEM']  # this will eventually need to
        #  be changed when XRF-specific
        #  features are added
        eds_dict = sem_dict['Detector']['EDS']
        signal_dict = TestSpcSpectrum_v061_xrf.spc.metadata.as_dictionary()[
            'Signal']

        # Testing SEM parameters
        assert_allclose(30, sem_dict['beam_energy'])
        assert_allclose(0, sem_dict['Stage']['tilt_alpha'])

        # Testing EDS parameters
        assert_allclose(45, eds_dict['azimuth_angle'])
        assert_allclose(35, eds_dict['elevation_angle'])
        assert_allclose(137.92946, eds_dict['energy_resolution_MnKa'],
                        atol=1E-5)
        assert_allclose(2561.0, eds_dict['live_time'], atol=1E-6)

        # Testing elements
        assert ({'Al', 'Ca', 'Cl', 'Cr', 'Fe', 'K', 'Mg', 'Mn', 'Si', 'Y'} ==
                set(elements))

        # Testing HyperSpy parameters
        assert True == signal_dict['binned']
        assert 'EDS_SEM' == signal_dict['signal_type']
        assert isinstance(TestSpcSpectrum_v061_xrf.spc, signals.EDSSEMSpectrum)

    def test_axes(self):
        spc_ax_manager = {'axis-0': {'name': 'Energy',
                                     'navigate': False,
                                     'offset': 0.0,
                                     'scale': 0.01,
                                     'size': 4000,
                                     'units': 'keV'}}
        assert (spc_ax_manager ==
                TestSpcSpectrum_v061_xrf.spc.axes_manager.as_dictionary())

    def test_load_all_spc(self):
        spc_header = TestSpcSpectrum_v061_xrf.spc_loadAll.original_metadata[
            'spc_header']

        assert_allclose(4, spc_header['analysisType'])
        assert_allclose(4, spc_header['analyzerType'])
        assert_allclose(2013, spc_header['collectDateYear'])
        assert_allclose(9, spc_header['collectDateMon'])
        assert_allclose(26, spc_header['collectDateDay'])
        assert_equal(b'Garnet1.', spc_header['fileName'].view('|S8')[0])
        assert_allclose(45, spc_header['xRayTubeZ'])


class TestSpcSpectrum_v070_eds:

    @classmethod
    def setup_class(cls):
        cls.spc = load(os.path.join(TMP_DIR.name, "single_spect.spc"))
        cls.spc_loadAll = load(os.path.join(TMP_DIR.name,
                                            "single_spect.spc"),
                               load_all_spc=True)

    @classmethod
    def teardown_class(cls):
        del cls.spc, cls.spc_loadAll
        gc.collect()

    def test_data(self):
        # test datatype
        assert np.uint32 == TestSpcSpectrum_v070_eds.spc.data.dtype
        # test data shape
        assert (4096,) == TestSpcSpectrum_v070_eds.spc.data.shape
        # test 1st 20 datapoints
        assert (
            [0, 0, 0, 0, 0, 0, 1, 2, 3, 3, 10, 4, 10, 10, 45, 87, 146, 236,
             312, 342] == TestSpcSpectrum_v070_eds.spc.data[:20].tolist())

    def test_parameters(self):
        elements = TestSpcSpectrum_v070_eds.spc.metadata.as_dictionary()[
            'Sample']['elements']
        sem_dict = TestSpcSpectrum_v070_eds.spc.metadata.as_dictionary()[
            'Acquisition_instrument']['SEM']
        eds_dict = sem_dict['Detector']['EDS']
        signal_dict = TestSpcSpectrum_v070_eds.spc.metadata.as_dictionary()[
            'Signal']

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
        assert isinstance(TestSpcSpectrum_v070_eds.spc, signals.EDSSEMSpectrum)

    def test_axes(self):
        spc_ax_manager = {'axis-0': {'name': 'Energy',
                                     'navigate': False,
                                     'offset': 0.0,
                                     'scale': 0.01,
                                     'size': 4096,
                                     'units': 'keV'}}
        assert (spc_ax_manager ==
                TestSpcSpectrum_v070_eds.spc.axes_manager.as_dictionary())

    def test_load_all_spc(self):
        spc_header = TestSpcSpectrum_v070_eds.spc_loadAll.original_metadata[
            'spc_header']

        assert_allclose(4, spc_header['analysisType'])
        assert_allclose(5, spc_header['analyzerType'])
        assert_allclose(2016, spc_header['collectDateYear'])
        assert_allclose(4, spc_header['collectDateMon'])
        assert_allclose(19, spc_header['collectDateDay'])
        assert_equal(b'C:\\ProgramData\\EDAX\\jtaillon\\Cole\\Mapping\\Lsm\\'
                     b'GFdCr\\950\\Area 1\\spectrum20160419153851427_0.spc',
                     spc_header['longFileName'].view('|S256')[0])
        assert_allclose(0, spc_header['xRayTubeZ'])


class TestSpdMap_070_eds:

    @classmethod
    def setup_class(cls):
        cls.spd = load(os.path.join(TMP_DIR.name, "spd_map.spd"),
                       convert_units=True)

    @classmethod
    def teardown_class(cls):
        del cls.spd
        gc.collect()

    def test_data(self):
        # test d_type
        assert np.uint16 == TestSpdMap_070_eds.spd.data.dtype
        # test d_shape
        assert (200, 256, 2500) == TestSpdMap_070_eds.spd.data.shape
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
                TestSpdMap_070_eds.spd.data[15:20, 15:20, 15:20].tolist())

    def test_parameters(self):
        elements = TestSpdMap_070_eds.spd.metadata.as_dictionary()[
            'Sample']['elements']
        sem_dict = TestSpdMap_070_eds.spd.metadata.as_dictionary()[
            'Acquisition_instrument']['SEM']
        eds_dict = sem_dict['Detector']['EDS']
        signal_dict = TestSpdMap_070_eds.spd.metadata.as_dictionary()['Signal']

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
        assert isinstance(TestSpdMap_070_eds.spd, signals.EDSSEMSpectrum)

    def test_axes(self):
        spd_ax_manager = {'axis-0': {'name': 'y',
                                     'navigate': True,
                                     'offset': 0.0,
                                     'scale': 14.227345585823057,
                                     'size': 200,
                                     'units': 'nm'},
                          'axis-1': {'name': 'x',
                                     'navigate': True,
                                     'offset': 0.0,
                                     'scale': 14.235896058380602,
                                     'size': 256,
                                     'units': 'nm'},
                          'axis-2': {'name': 'Energy',
                                     'navigate': False,
                                     'offset': 0.0,
                                     'scale': 0.0050000000000000001,
                                     'size': 2500,
                                     'units': 'keV'}}
        assert (spd_ax_manager ==
                TestSpdMap_070_eds.spd.axes_manager.as_dictionary())

    def test_ipr_reading(self):
        ipr_header = TestSpdMap_070_eds.spd.original_metadata['ipr_header']
        assert_allclose(0.014235896, ipr_header['mppX'])
        assert_allclose(0.014227346, ipr_header['mppY'])

    def test_spc_reading(self):
        # Test to make sure that spc metadata matches spd metadata
        spc_header = TestSpdMap_070_eds.spd.original_metadata['spc_header']

        elements = TestSpdMap_070_eds.spd.metadata.as_dictionary()[
            'Sample']['elements']
        sem_dict = TestSpdMap_070_eds.spd.metadata.as_dictionary()[
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
                        TestSpdMap_070_eds.spd.axes_manager[2].scale * 1000)
        assert_allclose(spc_header.kV,
                        sem_dict['beam_energy'])
        assert_allclose(spc_header.numElem,
                        len(elements))


class TestSpdMap_061_xrf:

    @classmethod
    def setup_class(cls):
        cls.spd = load(os.path.join(TMP_DIR.name, "spc0_61-ipr333_xrf.spd"),
                       convert_units=True)

    @classmethod
    def teardown_class(cls):
        del cls.spd
        gc.collect()

    def test_data(self):
        # test d_type
        assert np.uint16 == TestSpdMap_061_xrf.spd.data.dtype
        # test d_shape
        assert (200, 256, 2000) == TestSpdMap_061_xrf.spd.data.shape
        assert ([[[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0]],
                 [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]],
                 [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1]],
                 [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1]],
                 [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]]] ==
                TestSpdMap_061_xrf.spd.data[15:20, 15:20, 15:20].tolist())

    def test_parameters(self):
        elements = TestSpdMap_061_xrf.spd.metadata.as_dictionary()['Sample'][
            'elements']
        sem_dict = TestSpdMap_061_xrf.spd.metadata.as_dictionary()[
            'Acquisition_instrument']['SEM']
        eds_dict = sem_dict['Detector']['EDS']
        signal_dict = TestSpdMap_061_xrf.spd.metadata.as_dictionary()['Signal']

        # Testing SEM parameters
        assert_allclose(30, sem_dict['beam_energy'])
        assert_allclose(0, sem_dict['Stage']['tilt_alpha'])

        # Testing EDS parameters
        assert_allclose(45, eds_dict['azimuth_angle'])
        assert_allclose(35, eds_dict['elevation_angle'])
        assert_allclose(137.92946, eds_dict['energy_resolution_MnKa'],
                        atol=1E-5)
        assert_allclose(2561.0, eds_dict['live_time'], atol=1E-4)

        # Testing elements
        assert {'Al', 'Ca', 'Cl', 'Cr', 'Fe', 'K', 'Mg', 'Mn', 'Si',
                'Y'} == set(elements)

        # Testing HyperSpy parameters
        assert True == signal_dict['binned']
        assert 'EDS_SEM' == signal_dict['signal_type']
        assert isinstance(TestSpdMap_061_xrf.spd, signals.EDSSEMSpectrum)

    def test_axes(self):
        spd_ax_manager = {'axis-0': {'name': 'y',
                                     'navigate': True,
                                     'offset': 0.0,
                                     'scale': 0.5651920166015625,
                                     'size': 200,
                                     'units': 'mm'},
                          'axis-1': {'name': 'x',
                                     'navigate': True,
                                     'offset': 0.0,
                                     'scale': 0.5651920166015625,
                                     'size': 256,
                                     'units': 'mm'},
                          'axis-2': {'name': 'Energy',
                                     'navigate': False,
                                     'offset': 0.0,
                                     'scale': 0.01,
                                     'size': 2000,
                                     'units': 'keV'}}
        assert (spd_ax_manager ==
                TestSpdMap_061_xrf.spd.axes_manager.as_dictionary())

    def test_ipr_reading(self):
        ipr_header = TestSpdMap_061_xrf.spd.original_metadata['ipr_header']
        assert_allclose(565.1920166015625, ipr_header['mppX'])
        assert_allclose(565.1920166015625, ipr_header['mppY'])

    def test_spc_reading(self):
        # Test to make sure that spc metadata matches spd_061_xrf metadata
        spc_header = TestSpdMap_061_xrf.spd.original_metadata['spc_header']

        elements = TestSpdMap_061_xrf.spd.metadata.as_dictionary()['Sample'][
            'elements']
        sem_dict = TestSpdMap_061_xrf.spd.metadata.as_dictionary()[
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
                        TestSpdMap_061_xrf.spd.axes_manager[2].scale * 1000)
        assert_allclose(spc_header.kV,
                        sem_dict['beam_energy'])
        assert_allclose(spc_header.numElem,
                        len(elements))
