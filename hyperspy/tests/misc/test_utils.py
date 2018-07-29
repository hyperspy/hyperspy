import pytest
import numpy as np
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


@pytest.mark.parametrize("value, index", [(4, 4), (7, 7), (-10, 0), (90, 9)])
def test_find_nearest_index_simple(value, index):
    array = np.arange(10)
    assert utils.find_nearest_index(array, value) == index


@pytest.mark.parametrize("value, index", [
    (510, 100), (510.5, 105), (550., 499), (100., 0), (999.9, 499)])
def test_find_nearest_index_float(value, index):
    array = np.arange(500, 550, 0.1)
    assert utils.find_nearest_index(array, value) == index


@pytest.mark.parametrize("value, index", [
    (4, 4), (7, 7), (-10, 0), (90, 90), (200, 99)])
def test_find_nearest_index_from_right_simple_ramp(value, index):
    array = np.arange(100)
    assert utils.find_nearest_index_from_right(array, value) == index


@pytest.mark.parametrize("value, index", [
    (20, 70), (0, 50), (200, 0), (90, 4), (80, 9)])
def test_find_nearest_index_from_right_double_ramp(value, index):
    array0 = np.arange(0, 50)[::-1] * 2
    array1 = np.arange(0, 50)
    array = np.append(array0, array1)
    assert utils.find_nearest_index_from_right(array, value) == index


@pytest.mark.parametrize("threshold, index", [
    (1, 81), (2, 82), (10, 90), (15, 95)])
def test_find_nearest_index_from_right_threshold(threshold, index):
    array = np.arange(100)
    assert utils.find_nearest_index_from_right(array, 81, threshold) == index


def test_find_nearest_index_from_right_wrong_threshold_input():
    with pytest.raises(ValueError):
        utils.find_nearest_index_from_right(np.arange(10), 5, -2.1)
