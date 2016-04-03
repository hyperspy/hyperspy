import os

import numpy as np

from nose.tools import assert_true
from hyperspy.io import load

def test_load():
    my_path = os.path.dirname(__file__)
    filename = os.path.join(my_path, 'bcf_data',
                        'P45_instructively_packed_16bit_compressed.bcf')
    print('loading bcf test file...')
    s = load(filename)
    bse, sei, hype = s
    assert_true(bse.data.dtype == np.uint16)
    assert_true(sei.data.dtype == np.uint16)
    assert_true(hype.data.dtype == np.uint16)