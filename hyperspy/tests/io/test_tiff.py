import os

import numpy as np
from nose.tools import assert_true

import hyperspy.api as hs

my_path = os.path.dirname(__file__)


def test_rgba16():
    s = hs.load(os.path.join(
        my_path,
        "tiff_files",
        "test_rgba16.tif"))
    data = np.load(os.path.join(
        my_path,
        "npy_files",
        "test_rgba16.npy"))
    assert_true((s.data == data).all())
