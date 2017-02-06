import os

import json

import numpy as np
import pytest
from numpy.testing import assert_allclose

from hyperspy.io import load
from hyperspy import signals
from hyperspy.misc.test_utils import assert_deep_almost_equal

test_files = ['P45_instructively_packed_16bit_compressed.bcf',
              'P45_12bit_packed_8bit.bcf',
              'P45_the_default_job.bcf',
              'test_TEM.bcf',
              'Hitachi_TM3030Plus.bcf']
np_file = ['P45_16bit.npy', 'P45_16bit_ds.npy']

my_path = os.path.dirname(__file__)


def test_load_16bit():
    lxml = pytest.importorskip("lxml")
    # test bcf from hyperspy load function level
    # some of functions can be not covered
    # it cant use cython parsing implementation, as it is not compiled
    filename = os.path.join(my_path, 'bcf_data', test_files[0])
    print('testing bcf instructively packed 16bit...')
    s = load(filename)
    bse, sei, hype = s
    # Bruker saves all images in true 16bit:
    assert bse.data.dtype == np.uint16
    assert sei.data.dtype == np.uint16
    assert bse.data.shape == (75, 100)
    np_filename = os.path.join(my_path, 'bcf_data', np_file[0])
    np.testing.assert_array_equal(hype.data[:22, :22, 222],
                                  np.load(np_filename))
    assert hype.data.shape == (75, 100, 2048)


def test_load_16bit_reduced():
    lxml = pytest.importorskip("lxml")
    filename = os.path.join(my_path, 'bcf_data', test_files[0])
    print('testing downsampled 16bit bcf...')
    s = load(filename, downsample=4, cutoff_at_kV=10)
    bse, sei, hype = s
    # sem images never are downsampled
    assert bse.data.shape == (75, 100)
    np_filename = os.path.join(my_path, 'bcf_data', np_file[1])
    np.testing.assert_array_equal(hype.data[:2, :2, 222],
                                  np.load(np_filename))
    assert hype.data.shape == (19, 25, 1047)
    # Bruker saves all images in true 16bit:
    assert bse.data.dtype == np.uint16
    assert sei.data.dtype == np.uint16
    # hypermaps should always return unsigned integers:
    assert str(hype.data.dtype)[0] == 'u'


def test_load_8bit():
    lxml = pytest.importorskip("lxml")
    for bcffile in test_files[1:3]:
        filename = os.path.join(my_path, 'bcf_data', bcffile)
        print('testing simple 8bit bcf...')
        s = load(filename)
        bse, sei, hype = s
        # Bruker saves all images in true 16bit:
        assert bse.data.dtype == np.uint16
        assert sei.data.dtype == np.uint16
        # hypermaps should always return unsigned integers:
        assert str(hype.data.dtype)[0] == 'u'


def test_hyperspy_wrap():
    lxml = pytest.importorskip("lxml")
    filename = os.path.join(my_path, 'bcf_data', test_files[0])
    print('testing bcf wrap to hyperspy signal...')
    hype = load(filename, select_type='spectrum')
    assert_allclose(
        hype.axes_manager[0].scale,
        8.7367850619778,
        atol=1E-12)
    assert_allclose(
        hype.axes_manager[1].scale,
        8.7367850619778,
        atol=1E-12)
    assert hype.axes_manager[1].units == 'µm'
    assert_allclose(hype.axes_manager[2].scale, 0.010001)
    assert_allclose(hype.axes_manager[2].offset, -0.472397235)
    assert hype.axes_manager[2].units == 'keV'

    md_ref = {'_HyperSpy': {'Folding': {'original_shape': None,
                                        'unfolded': False,
                                        'original_axes_manager': None,
                                        'signal_unfolded': False}},
              'Sample': {'xray_lines': ['Al_Ka', 'Ca_Ka', 'Cl_Ka', 'Fe_Ka', 'K_Ka', 'Mg_Ka', 'Na_Ka', 'O_Ka', 'P_Ka', 'Si_Ka', 'Ti_Ka'],
                         'elements': ['Al', 'Ca', 'Cl', 'Fe', 'K', 'Mg', 'Na', 'O', 'P', 'Si', 'Ti'],
                         'name': 'Map data 232'},
              'Acquisition_instrument': {'SEM': {'beam_energy': 20.0,
                                                 'Detector': {'EDS': {'detector_type': 'XFlash 6|10',
                                                                      'energy_resolution_MnKa': 130.0,
                                                                      'elevation_angle': 35.0,
                                                                      'azimuth_angle': 90.0,
                                                                      'real_time': 328.8}},
                                                 'magnification': 131.1433,
                                                 'tilt_stage': 0.5}},
              'General': {'title': 'EDX',
                          'time': '17:05:03',
                          'original_filename': 'P45_instructively_packed_16bit_compressed.bcf',
                          'date': '2016-04-01'},
              'Signal': {'binned': True,
                         'quantity': 'X-rays (Counts)',
                         'signal_type': 'EDS_SEM'}}

    md_ref['General']['original_filename'] = hype.metadata.General.original_filename
    filename_omd = os.path.join(my_path,
                                'bcf_data',
                                'test_original_metadata.json')
    with open(filename_omd) as fn:
        # original_metadata:
        omd_ref = json.load(fn)
    assert_deep_almost_equal(hype.metadata.as_dictionary(), md_ref)
    assert_deep_almost_equal(hype.original_metadata.as_dictionary(), omd_ref)
    assert hype.metadata.General.date == "2016-04-01"
    assert hype.metadata.General.time == "17:05:03"
    assert hype.metadata.Signal.quantity == "X-rays (Counts)"


def test_hyperspy_wrap_downsampled():
    lxml = pytest.importorskip("lxml")
    filename = os.path.join(my_path, 'bcf_data', test_files[0])
    print('testing bcf wrap to hyperspy signal...')
    hype = load(filename, select_type='spectrum', downsample=5)
    assert_allclose(
        hype.axes_manager[0].scale,
        43.683925309889,
        atol=1E-12)
    assert_allclose(
        hype.axes_manager[1].scale,
        43.683925309889,
        atol=1E-12)
    assert hype.axes_manager[1].units == 'µm'


def test_fast_bcf():
    lxml = pytest.importorskip("lxml")
    from hyperspy.io_plugins import bcf

    for bcffile in test_files:
        filename = os.path.join(my_path, 'bcf_data', bcffile)
        thingy = bcf.BCF_reader(filename)
        for j in range(2, 5, 1):
            print('downsampling:', j)
            bcf.fast_unbcf = True              # manually enabling fast parsing
            hmap1 = thingy.parse_hypermap(downsample=j)    # using cython
            bcf.fast_unbcf = False            # manually disabling fast parsing
            hmap2 = thingy.parse_hypermap(downsample=j)    # py implementation
            np.testing.assert_array_equal(hmap1, hmap2)


def test_get_mode():
    lxml = pytest.importorskip("lxml")
    filename = os.path.join(my_path, 'bcf_data', test_files[0])
    s = load(filename, select_type='spectrum', instrument='SEM')
    assert s.metadata.Signal.signal_type == "EDS_SEM"
    assert isinstance(s, signals.EDSSEMSpectrum)

    filename = os.path.join(my_path, 'bcf_data', test_files[0])
    s = load(filename, select_type='spectrum', instrument='TEM')
    assert s.metadata.Signal.signal_type == "EDS_TEM"
    assert isinstance(s, signals.EDSTEMSpectrum)

    filename = os.path.join(my_path, 'bcf_data', test_files[0])
    s = load(filename, select_type='spectrum')
    assert s.metadata.Signal.signal_type == "EDS_SEM"
    assert isinstance(s, signals.EDSSEMSpectrum)

    filename = os.path.join(my_path, 'bcf_data', test_files[3])
    s = load(filename, select_type='spectrum')
    assert s.metadata.Signal.signal_type == "EDS_TEM"
    assert isinstance(s, signals.EDSTEMSpectrum)


def test_wrong_file():
    lxml = pytest.importorskip("lxml")
    filename = os.path.join(my_path, 'bcf_data', 'Nope.bcf')
    with pytest.raises(TypeError):
        load(filename)
