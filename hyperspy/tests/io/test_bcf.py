import os

import numpy as np

from nose.tools import assert_true
from hyperspy.io import load
from hyperspy.io_plugins import bcf

test_files = ['P45_instructively_packed_16bit_compressed.bcf',
              'P45_12bit_packed_8bit.bcf']  # for not compressed testing


def test_load():
    for thingy in test_files:
        my_path = os.path.dirname(__file__)
        filename = os.path.join(my_path, 'bcf_data', thingy)
        print('loading bcf test file...')
        s = load(filename, downsample=2, cutoff_at_kV=10)
        bse, sei, hype = s
        assert_true(bse.data.dtype == np.uint16)
        assert_true(sei.data.dtype == np.uint16)

def test_py_parsing():
    bcf.fast_unbcf = False  #force to ignore cython code
    for thingy in test_files:
        my_path = os.path.dirname(__file__)
        filename = os.path.join(my_path, 'bcf_data', thingy)
        print('loading bcf test file...')
        s = load(filename, downsample=2, cutoff_at_kV=10)