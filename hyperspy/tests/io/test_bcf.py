import os

import numpy as np
import json
from nose.plugins.skip import SkipTest

import nose.tools as nt
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

try:
    import lxml
    skip_test = False
except ImportError:
    skip_test = True


def test_load_16bit():
    if skip_test:
        raise SkipTest
    # test bcf from hyperspy load function level
    # some of functions can be not covered
    # it cant use cython parsing implementation, as it is not compiled
    filename = os.path.join(my_path, 'bcf_data', test_files[0])
    print('testing bcf instructively packed 16bit...')
    s = load(filename)
    bse, sei, hype = s
    # Bruker saves all images in true 16bit:
    nt.assert_true(bse.data.dtype == np.uint16)
    nt.assert_true(sei.data.dtype == np.uint16)
    nt.assert_true(bse.data.shape == (75, 100))
    np_filename = os.path.join(my_path, 'bcf_data', np_file[0])
    np.testing.assert_array_equal(hype.data[:22, :22, 222],
                                  np.load(np_filename))
    nt.assert_true(hype.data.shape == (75, 100, 2048))


def test_load_16bit_reduced():
    if skip_test:
        raise SkipTest
    filename = os.path.join(my_path, 'bcf_data', test_files[0])
    print('testing downsampled 16bit bcf...')
    s = load(filename, downsample=4, cutoff_at_kV=10)
    bse, sei, hype = s
    # sem images never are downsampled
    nt.assert_true(bse.data.shape == (75, 100))
    np_filename = os.path.join(my_path, 'bcf_data', np_file[1])
    np.testing.assert_array_equal(hype.data[:2, :2, 222],
                                  np.load(np_filename))
    nt.assert_true(hype.data.shape == (19, 25, 1047))
    # Bruker saves all images in true 16bit:
    nt.assert_true(bse.data.dtype == np.uint16)
    nt.assert_true(sei.data.dtype == np.uint16)
    # hypermaps should always return unsigned integers:
    nt.assert_true(str(hype.data.dtype)[0] == 'u')


def test_load_8bit():
    if skip_test:
        raise SkipTest
    for bcffile in test_files[1:3]:
        filename = os.path.join(my_path, 'bcf_data', bcffile)
        print('testing simple 8bit bcf...')
        s = load(filename)
        bse, sei, hype = s
        # Bruker saves all images in true 16bit:
        nt.assert_true(bse.data.dtype == np.uint16)
        nt.assert_true(sei.data.dtype == np.uint16)
        # hypermaps should always return unsigned integers:
        nt.assert_true(str(hype.data.dtype)[0] == 'u')


def test_hyperspy_wrap():
    if skip_test:
        raise SkipTest
    filename = os.path.join(my_path, 'bcf_data', test_files[0])
    print('testing bcf wrap to hyperspy signal...')
    hype = load(filename, select_type='spectrum')
    nt.assert_almost_equal(
        hype.axes_manager[0].scale,
        8.7367850619778,
        places=12)
    nt.assert_almost_equal(
        hype.axes_manager[1].scale,
        8.7367850619778,
        places=12)
    nt.assert_equal(hype.axes_manager[1].units, 'µm')
    nt.assert_almost_equal(hype.axes_manager[2].scale, 0.010001)
    nt.assert_almost_equal(hype.axes_manager[2].offset, -0.472397235)
    nt.assert_equal(hype.axes_manager[2].units, 'keV')

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
        #original_metadata:
        omd_ref = json.load(fn)
    assert_deep_almost_equal(hype.metadata.as_dictionary(), md_ref)
    assert_deep_almost_equal(hype.original_metadata.as_dictionary(), omd_ref)
    nt.assert_equal(hype.metadata.General.date, "2016-04-01")
    nt.assert_equal(hype.metadata.General.time, "17:05:03")
    nt.assert_equal(hype.metadata.Signal.quantity, "X-rays (Counts)")


def test_hyperspy_wrap_downsampled():
    if skip_test:
        raise SkipTest
    filename = os.path.join(my_path, 'bcf_data', test_files[0])
    print('testing bcf wrap to hyperspy signal...')
    hype = load(filename, select_type='spectrum', downsample=5)
    nt.assert_almost_equal(
        hype.axes_manager[0].scale,
        43.683925309889,
        places=12)
    nt.assert_almost_equal(
        hype.axes_manager[1].scale,
        43.683925309889,
        places=12)
    nt.assert_true(hype.axes_manager[1].units == 'µm')


def test_fast_bcf():
    if skip_test:
        raise SkipTest
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
    if skip_test:
        raise SkipTest
    filename = os.path.join(my_path, 'bcf_data', test_files[0])
    s = load(filename, select_type='spectrum', instrument='SEM')
    nt.assert_equal(s.metadata.Signal.signal_type, "EDS_SEM")
    nt.assert_true(isinstance(s, signals.EDSSEMSpectrum))

    filename = os.path.join(my_path, 'bcf_data', test_files[0])
    s = load(filename, select_type='spectrum', instrument='TEM')
    nt.assert_equal(s.metadata.Signal.signal_type, "EDS_TEM")
    nt.assert_true(isinstance(s, signals.EDSTEMSpectrum))

    filename = os.path.join(my_path, 'bcf_data', test_files[0])
    s = load(filename, select_type='spectrum')
    nt.assert_equal(s.metadata.Signal.signal_type, "EDS_SEM")
    nt.assert_true(isinstance(s, signals.EDSSEMSpectrum))

    filename = os.path.join(my_path, 'bcf_data', test_files[3])
    s = load(filename, select_type='spectrum')
    nt.assert_equal(s.metadata.Signal.signal_type, "EDS_TEM")
    nt.assert_true(isinstance(s, signals.EDSTEMSpectrum))
    

def test_wrong_file():
    if skip_test:
        raise SkipTest
    filename = os.path.join(my_path, 'bcf_data', 'Nope.bcf')
    with nt.assert_raises(TypeError):
        load(filename)
