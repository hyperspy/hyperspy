import hyperspy.misc.utils as utils
from hyperspy import roi


def test_signal_range_from_roi():
    sr = roi.SpanROI(20, 50)
    left, right = utils.signal_range_from_roi(sr)
    assert left == 20
    assert right == 50
    left, right = utils.signal_range_from_roi((20, 50))
    assert left == 20
    assert right == 50


def test_slugify():
    assert utils.slugify('a') == 'a'
    assert utils.slugify('1a') == '1a'
    assert utils.slugify('1') == '1'
    assert utils.slugify('a a') == 'a_a'

    assert utils.slugify('a', valid_variable_name=True) == 'a'
    assert utils.slugify('1a', valid_variable_name=True) == 'Number_1a'
    assert utils.slugify('1', valid_variable_name=True) == 'Number_1'

    assert utils.slugify('a', valid_variable_name=False) == 'a'
    assert utils.slugify('1a', valid_variable_name=False) == '1a'
    assert utils.slugify('1', valid_variable_name=False) == '1'
