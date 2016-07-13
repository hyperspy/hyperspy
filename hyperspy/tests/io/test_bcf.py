import os

import numpy as np
from nose.plugins.skip import SkipTest

from nose.tools import assert_true, assert_almost_equal
from hyperspy.io import load

test_files = ['P45_instructively_packed_16bit_compressed.bcf',
              'P45_12bit_packed_8bit.bcf',
              'P45_the_default_job.bcf']
np_file = ['P45_16bit.npy', 'P45_16bit_ds.npy']

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
    my_path = os.path.dirname(__file__)
    filename = os.path.join(my_path, 'bcf_data', test_files[0])
    print('testing bcf instructively packed 16bit...')
    s = load(filename)
    bse, sei, hype = s
    #Bruker saves all images in true 16bit:
    assert_true(bse.data.dtype == np.uint16)
    assert_true(sei.data.dtype == np.uint16)
    assert_true(bse.data.shape == (75, 100))
    np_filename = os.path.join(my_path, 'bcf_data', np_file[0])
    np.testing.assert_array_equal(hype.data[:22, :22, 222],
                                  np.load(np_filename))
    assert_true(hype.data.shape == (75, 100, 2048))


def test_load_16bit_reduced():
    if skip_test:
        raise SkipTest
    my_path = os.path.dirname(__file__)
    filename = os.path.join(my_path, 'bcf_data', test_files[0])
    print('testing downsampled 16bit bcf...')
    s = load(filename, downsample=4, cutoff_at_kV=10)
    bse, sei, hype = s
    assert_true(bse.data.shape == (75, 100))  # sem images never are downsampled
    np_filename = os.path.join(my_path, 'bcf_data', np_file[1])
    np.testing.assert_array_equal(hype.data[:2, :2, 222],
                                  np.load(np_filename))
    assert_true(hype.data.shape == (19, 25, 1047))
    #Bruker saves all images in true 16bit:
    assert_true(bse.data.dtype == np.uint16)
    assert_true(sei.data.dtype == np.uint16)
    #hypermaps should always return unsigned integers:
    assert_true(str(hype.data.dtype)[0] == 'u')


def test_load_8bit():
    if skip_test:
        raise SkipTest
    for bcffile in test_files[1:]:
        my_path = os.path.dirname(__file__)
        filename = os.path.join(my_path, 'bcf_data', bcffile)
        print('testing simple 8bit bcf...')
        s = load(filename)
        bse, sei, hype = s
        #Bruker saves all images in true 16bit:
        assert_true(bse.data.dtype == np.uint16)
        assert_true(sei.data.dtype == np.uint16)
        #hypermaps should always return unsigned integers:
        assert_true(str(hype.data.dtype)[0] == 'u')


def test_hyperspy_wrap():
    if skip_test:
        raise SkipTest
    my_path = os.path.dirname(__file__)
    filename = os.path.join(my_path, 'bcf_data', test_files[0])
    print('testing bcf wrap to hyperspy signal...')
    hype = load(filename, select_type='spectrum')
    assert_almost_equal(hype.axes_manager[0].scale, 8.7367850619778, places=12)
    assert_almost_equal(hype.axes_manager[1].scale, 8.7367850619778, places=12)
    assert_true(hype.axes_manager[1].units == 'µm')
    assert_almost_equal(hype.axes_manager[2].scale, 0.010001)
    assert_almost_equal(hype.axes_manager[2].offset, -0.472397235)
    assert_true(hype.axes_manager[2].units == 'keV')


def test_hyperspy_wrap_downsampled():
    if skip_test:
        raise SkipTest
    my_path = os.path.dirname(__file__)
    filename = os.path.join(my_path, 'bcf_data', test_files[0])
    print('testing bcf wrap to hyperspy signal...')
    hype = load(filename, select_type='spectrum', downsample=5)
    assert_almost_equal(hype.axes_manager[0].scale, 43.683925309889, places=12)
    assert_almost_equal(hype.axes_manager[1].scale, 43.683925309889, places=12)
    assert_true(hype.axes_manager[1].units == 'µm')


def test_fast_bcf():
    if skip_test:
        raise SkipTest
    from hyperspy.io_plugins import bcf

    for bcffile in test_files:
        my_path = os.path.dirname(__file__)
        filename = os.path.join(my_path, 'bcf_data', bcffile)
        thingy = bcf.BCF_reader(filename)
        for j in range(2, 5, 1):
            print('downsampling:', j)
            bcf.fast_unbcf = True              # manually enabling fast parsing
            hmap1 = thingy.parse_hypermap(downsample=j)    # using cython
            bcf.fast_unbcf = False            # manually disabling fast parsing
            hmap2 = thingy.parse_hypermap(downsample=j)    # py implementation
            np.testing.assert_array_equal(hmap1, hmap2)
