# Copyright 2007-2016 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <http://www.gnu.org/licenses/>.


import numpy as np

from numpy.testing import assert_allclose
from hyperspy.samfire_utils.strategy import (LocalStrategy,
                                             GlobalStrategy)
from hyperspy.misc.utils import DictionaryTreeBrowser
from hyperspy.signals import Signal1D
from hyperspy.components1d import Gaussian


class someweight(object):

    def __init__(self):
        self.model = None

    def map(self, calc_pixels, slices=slice(None, None)):
        # return of slices shape thing, with np.nan where mask is False
        thing = self.model.chisq.data[slices].copy()
        thing[np.logical_not(calc_pixels)] = np.nan
        return thing

    def function(self, ind):
        return self.model.chisq.data[ind]


def create_artificial_samfire(shape):
    artificial_samfire = DictionaryTreeBrowser()
    artificial_samfire.add_node('running_pixels')
    artificial_samfire.running_pixels = []
    artificial_samfire.add_node('model')
    artificial_samfire.add_node('metadata')
    artificial_samfire.metadata.add_node('marker')
    artificial_samfire.metadata.marker = np.zeros(shape)
    artificial_samfire.add_node('_scale')
    artificial_samfire._scale = 1.0
    return artificial_samfire


def compare_two_value_dicts(ans_r, ans):
    test = True
    for k, v in ans_r.items():
        test = test and k in ans
        if test:
            for p, pv in v.items():
                test = test and p in ans[k]
                if test:
                    assert np.allclose(
                        np.array(pv),
                        np.array(
                            ans[k][p]))
    return test


class TestLocalSimple:

    def setup_method(self, method):
        self.shape = (5, 7)
        self.s = LocalStrategy('test diffusion strategy')
        self.samf = create_artificial_samfire(self.shape)

        m = DictionaryTreeBrowser()
        m.set_item('chisq.data', np.ones(self.shape) * 5.)

        self.samf.model = m

    def test_default_init(self):
        s = self.s
        assert s.samf is None
        assert s.radii is None
        assert s.weight is None
        assert s._untruncated is None
        assert s._mask_all is None
        assert s._radii_changed

    def test_samf_weight_setters(self):
        s = self.s
        samf = self.samf
        w = someweight()
        s.weight = w
        assert s._weight is w
        assert w.model is None
        s.samf = samf
        assert w.model is samf.model
        w2 = someweight()
        s.weight = w2
        assert w.model is None

    def test_radii(self):
        s = self.s
        s.samf = self.samf
        assert s.radii is None
        assert s._radii_changed
        s._radii_changed = False
        s.radii = 1.
        assert s.radii == (1.0, 1.0)
        assert s._radii_changed

    def test_clean(self):
        s = self.s
        s._untruncated = 12
        s._mask_all = 2.
        s._radii_changed = False
        s.clean()
        assert s._untruncated is None
        assert s._mask_all is None
        assert s._radii_changed

    def test_refresh_overwrite(self):
        s = self.s
        s.radii = (1.9, 2.1)
        samf = self.samf
        s.samf = samf
        s.decay_function = lambda x: np.exp(-x)
        s.weight = someweight()
        s.samf.metadata.marker[1, 1] = -2
        s.samf.metadata.marker[0, 1] = -1
        s.samf.model.chisq.data[0, 1] = 50

        ans1 = np.array([[1.63810767e-03, -1.00000000e+00, 1.63810767e-03, 2.61027907e-23],
                         [2.47875218e-03, -
                          1.00000000e+00, 2.47875218e-03, 9.11881966e-04],
                         [1.63810767e-03,
                          2.47875218e-03,
                          1.63810767e-03,
                          0.00000000e+00],
                         [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])

        s.refresh(True, given_pixels=None)
        assert np.allclose(ans1, s.samf.metadata.marker[:4, :4])

        given = np.ones(self.shape, dtype=bool)
        given[0, 1] = False
        s.refresh(True, given_pixels=given)
        assert_allclose(
            s.samf.metadata.marker[
                ~given][0],
            0.011624353837970535)
        assert np.all(s.samf.metadata.marker[given] == -1)

    def test_refresh_nooverwrite(self):
        s = self.s
        s.radii = (1.9, 2.1)
        samf = self.samf
        s.samf = samf
        s.decay_function = lambda x: np.exp(-x)
        s.weight = someweight()
        s.samf.metadata.marker[1, 1] = -2
        s.samf.metadata.marker[0, 1] = -1
        s.samf.model.chisq.data[0, 1] = 50

        ans1 = np.array([[7.09547416e-23, -1.00000000e+00, 7.09547416e-23, 2.61027907e-23],
                         [4.68911365e-23, -
                          2.00000000e+00, 4.68911365e-23, 0.00000000e+00],
                         [0.00000000e+00,
                          0.00000000e+00,
                          0.00000000e+00,
                          0.00000000e+00],
                         [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])

        s.refresh(False, given_pixels=None)
        assert np.allclose(ans1, s.samf.metadata.marker[:4, :4])

        s.samf.metadata.marker[0, 0] = -1
        given = np.ones(self.shape, dtype=bool)
        given[0, 0] = False
        # should stay the same, as the new point [0,0] is not in "given"
        s.refresh(False, given_pixels=given)
        assert np.allclose(ans1, s.samf.metadata.marker[:4, :4])

    def test_get_distance_array(self):
        s = self.s
        s.samf = self.samf
        # use floats intentionally, since it was broken at one point
        s.radii = (2.1, 1.9)
        ind = (0, 0)
        distances, slices, centre, mask = s._get_distance_array(
            self.shape, ind)
        assert centre == ind
        assert slices == (slice(0, 4, None), slice(0, 3, None))
        tmp = np.array([[0., 1., np.nan],
                        [1., 1.41421356, np.nan],
                        [2., np.nan, np.nan],
                        [np.nan, np.nan, np.nan]])
        tmp_m = np.array([[False, True, False],
                          [True, True, False],
                          [True, False, False],
                          [False, False, False]], dtype=bool)
        assert np.all(tmp_m == mask)
        assert np.allclose(tmp[mask], distances[mask])
        assert not s._radii_changed
        tmp_ma = np.array([[3.14884957, 2.31782464, 2.04081633, 2.31782464, 3.14884957],
                           [2.01506272,
                            1.18403779,
                            0.90702948,
                            1.18403779,
                            2.01506272],
                           [1.33479061,
                            0.50376568,
                            0.22675737,
                            0.50376568,
                            1.33479061],
                           [1.10803324,
                            0.27700831,
                            0.,
                            0.27700831,
                            1.10803324],
                           [1.33479061,
                            0.50376568,
                            0.22675737,
                            0.50376568,
                            1.33479061],
                           [2.01506272,
                            1.18403779,
                            0.90702948,
                            1.18403779,
                            2.01506272],
                           [3.14884957, 2.31782464, 2.04081633, 2.31782464, 3.14884957]])
        assert np.allclose(s._mask_all, tmp_ma)
        tmp_un = np.array([[3.60555128, 3.16227766, 3., 3.16227766, 3.60555128],
                           [2.82842712,
                            2.23606798,
                            2.,
                            2.23606798,
                            2.82842712],
                           [2.23606798,
                            1.41421356,
                            1.,
                            1.41421356,
                            2.23606798],
                           [2., 1., 0., 1., 2.],
                           [2.23606798,
                            1.41421356,
                            1.,
                            1.41421356,
                            2.23606798],
                           [2.82842712,
                            2.23606798,
                            2.,
                            2.23606798,
                            2.82842712],
                           [3.60555128, 3.16227766, 3., 3.16227766, 3.60555128]])
        assert np.allclose(s._untruncated, tmp_un)

        # now check that the stored values are used
        # mask:
        s._mask_all[0, 0] = 1.0
        ind = (4, 6)
        distances, slices, centre, mask = s._get_distance_array(
            self.shape, ind)
        assert centre == (3.0, 2.0)
        assert slices == (slice(1, 5, None), slice(4, 7, None))
        tmp_m = np.array([[True, False, False],
                          [False, False, True],
                          [False, True, True],
                          [False, True, False]], dtype=bool)
        tmp = np.array([[3.60555128, np.nan, np.nan],
                        [np.nan, np.nan, 2.],
                        [np.nan, 1.41421356, 1.],
                        [np.nan, 1., 0.]])
        assert np.all(tmp_m == mask)
        assert np.allclose(tmp[mask], distances[mask])
        assert not s._radii_changed

        # now mask radii changed and check that the correct result is
        # calculated again
        s._radii_changed = True

        distances, slices, centre, mask = s._get_distance_array(
            self.shape, ind)
        assert centre == (3.0, 2.0)
        assert slices == (slice(1, 5, None), slice(4, 7, None))
        tmp_m = np.array([[False, False, False],
                          [False, False, True],
                          [False, True, True],
                          [False, True, False]], dtype=bool)
        tmp = np.array([[np.nan, np.nan, np.nan],
                        [np.nan, np.nan, 2.],
                        [np.nan, 1.41421356, 1.],
                        [np.nan, 1., 0.]])
        assert np.all(tmp_m == mask)
        assert np.allclose(tmp[mask], distances[mask])
        assert not s._radii_changed

    def test_update_marker(self):
        s = self.s
        # use floats intentionally, since it was broken at one point
        s.radii = (1.9, 2.1)
        samf = self.samf
        samf.metadata.marker[0, 1] = -0.33
        samf.metadata.marker[1, 2] = 100
        s.samf = samf
        s.decay_function = lambda x: np.exp(-x)
        s.weight = someweight()

        ind = (0, 0)
        s._update_marker(ind)
        tmp_m1 = np.array([[-1.00000000e+00, -3.30000000e-01, 9.11881966e-04, 0.00000000e+00],
                           [2.47875218e-03,
                            1.63810767e-03,
                            1.00000000e+02,
                            0.00000000e+00],
                           [0.00000000e+00,
                            0.00000000e+00,
                            0.00000000e+00,
                            0.00000000e+00],
                           [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])
        assert np.allclose(tmp_m1, s.samf.metadata.marker[:4, :4])

        ind = (1, 1)
        s.samf.running_pixels.append((1, 2))
        s.samf._scale = 13
        s._update_marker(ind)
        tmp_m2 = np.array([[-1.00000000e+00, -3.30000000e-01, 2.54998964e-03, 0.00000000e+00],
                           [4.95750435e-03, -
                            1.30000000e+01, 0.00000000e+00, 9.11881966e-04],
                           [1.63810767e-03,
                            2.47875218e-03,
                            1.63810767e-03,
                            0.00000000e+00],
                           [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])
        assert np.allclose(tmp_m2, s.samf.metadata.marker[:4, :4])


class TestLocalWithModel:

    def setup_method(self, method):
        self.shape = (5, 7)
        self.s = LocalStrategy('test diffusion strategy')
        self.samf = create_artificial_samfire(self.shape)

        m = Signal1D(np.empty(self.shape + (100,))).create_model()
        m.extend([Gaussian() for _ in range(3)])
        m.chisq.data.fill(5.)

        self.samf.model = m

    def test_values(self):
        s = self.s
        # use floats intentionally, since it was broken at one point
        s.radii = (1.9, 2.1)
        samf = self.samf
        s.samf = samf
        s.decay_function = lambda x: np.exp(-x)
        s.weight = someweight()

        samf.model[0].active = False

        samf.model[1].active_is_multidimensional = True
        samf.model[1]._active_array[0, 0] = False
        samf.model[1].A.map['values'].fill(10.)
        samf.model[1].centre.map['values'].fill(3.)
        samf.model[1].centre.map['values'][0, 1] = 15.

        samf.model[2].A.free = False

        d1 = s.values((0, 0))
        assert d1 == {}

        samf.metadata.marker[0, 0] = -1
        assert (
            s.values(
                (1, 0)) == {
                'Gaussian_1': {
                    'centre': 0.0, 'sigma': 0.0}})

        samf.metadata.marker[1, 1] = -1

        ans_r = {'Gaussian_0': {'A': 10.0, 'centre': 2.9999999999999996, 'sigma': 0.0},
                 'Gaussian_1': {'centre': 0.0, 'sigma': 0.0}}

        ans = s.values((1, 0))

        test = compare_two_value_dicts(ans_r, ans)
        assert test

        samf.metadata.marker[0, 1] = -1

        ans_r2 = {'Gaussian_0': {'A': 10.0, 'centre': 7.7748266350745405, 'sigma': 0.0},
                  'Gaussian_1': {'centre': 0.0, 'sigma': 0.0}}

        ans2 = s.values((1, 0))

        test2 = compare_two_value_dicts(ans_r2, ans2)
        assert test2


class TestGlobalStrategy:

    def setup_method(self, method):
        # TODO: actually finish setup+ tests
        self.shape = (5, 7)
        self.s = GlobalStrategy('test segmenter strategy')
        self.samf = create_artificial_samfire(self.shape)

        m = Signal1D(np.empty(self.shape + (100,))).create_model()
        m.extend([Gaussian() for _ in range(3)])
        m.chisq.data.fill(5.)

        self.samf.model = m

    def test_refresh_nooverwrite_nogiven(self):
        samf = self.samf
        samf.set_item('update_every', np.nan)
        s = self.s
        s.samf = samf

        samf.metadata.marker[0, 0] = -1
        samf.metadata.marker[0, 1] = -2

        s.refresh(False)
        assert samf.metadata.marker[0, 0] == -1
        assert samf.metadata.marker[0, 1] == -2
        assert np.all(samf.metadata.marker.ravel()[2:] == 1)

    def test_refresh_nooverwrite_given(self):
        samf = self.samf
        samf.set_item('update_every', np.nan)
        s = self.s
        s.samf = samf

        samf.metadata.marker.fill(3.)
        samf.metadata.marker[0, 0] = -1
        samf.metadata.marker[0, 1] = -2

        given = np.zeros(self.shape, dtype=bool)
        given[0, 1] = True
        s.refresh(False, given_pixels=given)
        assert samf.metadata.marker[0, 0] == 1
        assert samf.metadata.marker[0, 1] == -2
        assert np.all(samf.metadata.marker.ravel()[2:] == 1)

    def test_refresh_overwrite_nogiven(self):
        samf = self.samf
        samf.set_item('update_every', np.nan)
        s = self.s
        s.samf = samf

        samf.metadata.marker[0, 0] = -1
        samf.metadata.marker[0, 1] = -2

        s.refresh(True)
        assert samf.metadata.marker[0, 0] == -1
        assert samf.metadata.marker[0, 1] == -1
        assert np.all(samf.metadata.marker.ravel()[2:] == 1)

    def test_refresh_overwrite_given(self):
        samf = self.samf
        samf.set_item('update_every', np.nan)
        s = self.s
        s.samf = samf

        samf.metadata.marker.fill(3.)
        samf.metadata.marker[0, 0] = -1
        samf.metadata.marker[0, 1] = -2

        given = np.zeros(self.shape, dtype=bool)
        given[0, 1] = True
        s.refresh(True, given_pixels=given)
        assert samf.metadata.marker[0, 0] == 1
        assert samf.metadata.marker[0, 1] == -1
        assert np.all(samf.metadata.marker.ravel()[2:] == 1)

    def test_update_marker(self):
        s = self.s
        samf = self.samf
        s.samf = samf
        ind = (1, 1)
        s._update_marker(ind)
        assert samf.metadata.marker[ind] == -samf._scale
        mask = np.ones(self.shape, dtype=bool)
        mask[ind] = False
        assert np.all(samf.metadata.marker[mask] == 0)

    def test_package_values(self):

        s = self.s
        samf = self.samf
        s.samf = samf

        samf.model[0].active = False
        samf.model[1].active_is_multidimensional = True
        samf.model[1]._active_array[0, 0] = False
        samf.model[1].A.map['values'].fill(10.)
        samf.model[1].centre.map['values'].fill(3.)
        samf.model[1].centre.map['values'][0, 1] = 15.
        samf.model[2].sigma.map['values'][0, 2] = 9.

        s.samf.metadata.marker[0, 0] = -100

        ans_r1 = {'Gaussian_0': {'A': np.array([], dtype=float),
                                 'centre': np.array([], dtype=float),
                                 'sigma': np.array([], dtype=float)},
                  'Gaussian_1': {'centre': np.array([0.]), 'sigma': np.array([0.])}}

        ans = s._package_values()
        t = compare_two_value_dicts(ans_r1, ans)
        assert t

        s.samf.metadata.marker[0, 2] = -100

        ans_r2 = {'Gaussian_0': {'A': np.array([10.]),
                                 'centre': np.array([3.]),
                                 'sigma': np.array([0.])},
                  'Gaussian_1': {'centre': np.array([0., 0.]), 'sigma': np.array([0., 9.])}}

        ans2 = s._package_values()
        t2 = compare_two_value_dicts(ans_r2, ans2)
        assert t2
