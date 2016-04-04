import os

import numpy as np

from nose.tools import assert_true
from hyperspy.io import load

test_files = ['P45_instructively_packed_16bit_compressed.bcf',
              'P45_12bit_packed_8bit.bcf',
              'P45_the_default_job.bcf']
np_file = ['P45_16bit.npy', 'P45_16bit_ds.npy']


def test_load_16bit():
    # test bcf from hyperspy load function level
    # some of functions can be not covered
    # it cant use cython parsing implementation, as it is not compiled
    my_path = os.path.dirname(__file__)
    filename = os.path.join(my_path, 'bcf_data', test_files[0])
    print('testing bcf instructivele packed16bit...')
    s = load(filename)
    bse, sei, hype = s
    assert_true(bse.data.dtype == np.uint16)
    np_filename = os.path.join(my_path, 'bcf_data', np_file[0])
    np.testing.assert_array_equal(hype.data[:22, :22, 222],
                                  np.load(np_filename))
    assert_true(hype.data.shape == (75, 100, 2048))


def test_load_16bit_reduced():
    my_path = os.path.dirname(__file__)
    filename = os.path.join(my_path, 'bcf_data', test_files[0])
    print('testing downsampled 16bit bcf...')
    s = load(filename, downsample=4, cutoff_at_kV=10)
    bse, sei, hype = s
    np_filename = os.path.join(my_path, 'bcf_data', np_file[1])
    np.testing.assert_array_equal(hype.data[:2, :2, 222],
                                  np.load(np_filename))
    assert_true(hype.data.shape == (18, 25, 1047))


def test_load_8bit():
    for bcffile in test_files[1:]:
        my_path = os.path.dirname(__file__)
        filename = os.path.join(my_path, 'bcf_data', bcffile)
        print('testing simple 8bit bcf...')
        s = load(filename)
        bse, sei, hype = s


#def test_fast_bcf():
#    from hyperspy.io_plugins import unbcf_fast
#    unbcf_fast
