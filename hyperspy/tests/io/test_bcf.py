import os

import numpy as np

from nose.tools import assert_true
from hyperspy.io import load

test_files = ['P45_instructively_packed_16bit_compressed.bcf',
              'P45_12bit_packed_8bit.bcf',
              'P45_the_default_job.bcf']  # for not compressed testing


def test_load():
    # test bcf from hyperspy load function level
    # some of functions can be not covered
    # it cant use cython parsing implementation, as it is not compiled
    for thingy in test_files:
        my_path = os.path.dirname(__file__)
        filename = os.path.join(my_path, 'bcf_data', thingy)
        print('testing bcf loading from hyperspy level...')
        s = load(filename, downsample=2, cutoff_at_kV=10)
        s = load(filename)
        bse, sei, hype = s
        assert_true(bse.data.dtype == np.uint16)


def test_fast_bcf():
    from hyperspy.io_plugins import unbcf_fast
    unbcf_fast
