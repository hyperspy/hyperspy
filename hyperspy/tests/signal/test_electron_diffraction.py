# Copyright 2007-2016 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np
import nose.tools as nt

from hyperspy.signals import ElectronDiffraction, Signal1D, Signal2D
from hyperspy.defaults_parser import preferences


class Test_metadata:

    def setUp(self):
        # Create an empty diffraction pattern
        dp = ElectronDiffraction(np.ones((2, 2, 2, 2)))
        dp.axes_manager.signal_axes[0].scale = 1e-3
        dp.metadata.Acquisition_instrument.TEM.accelerating_voltage = 200
        dp.metadata.Acquisition_instrument.TEM.convergence_angle = 15.0
        dp.metadata.Acquisition_instrument.TEM.rocking_angle = 18.0
        dp.metadata.Acquisition_instrument.TEM.rocking_frequency = 63
        dp.metadata.Acquisition_instrument.TEM.Detector.Diffraction.exposure_time = 35
        self.signal = dp

    def test_default_param(self):
        dp = self.signal
        md = dp.metadata
        nt.assert_equal(md.Acquisition_instrument.TEM.rocking_angle,
                        preferences.ElectronDiffraction.ed_precession_angle)

class Test_direct_beam_methods:

    def setUp(self):
        dp = ElectronDiffraction(np.zeros((4, 8, 8)))
        dp.data[0]= np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 1., 0., 0., 0., 0.],
                              [0., 0., 1., 2., 1., 0., 0., 0.],
                              [0., 0., 0., 1., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.]])

        dp.data[1]= np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 1., 0., 0., 0.],
                              [0., 0., 0., 1., 2., 1., 0., 0.],
                              [0., 0., 0., 0., 1., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.]])

        dp.data[2]= np.array([[0., 0., 0., 0., 0., 0., 0., 2.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 1., 0., 0., 0., 0.],
                              [0., 0., 1., 2., 1., 0., 0., 0.],
                              [0., 0., 0., 1., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.]])

        dp.data[3]= np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 2., 0., 0., 0.],
                              [0., 0., 0., 2., 2., 2., 0., 0.],
                              [0., 0., 0., 0., 2., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.]])
        self.signal = dp

    def test_get_direct_beam_position(self):
        dp = self.signal
        c = dp.get_direct_beam_position(radius=2)
        np.testing.assert_equal(c, np.array([[3, 3], [4, 4], [3, 3], [4, 4]]))

    def test_get_direct_beam_shifts_no_centers(self):
        dp = self.signal
        s = dp.get_direct_beam_shifts(radius=2)
        np.testing.assert_equal(s, np.array([[-0.5, -0.5],
                                             [ 0.5,  0.5],
                                             [-0.5, -0.5],
                                             [ 0.5,  0.5]]))

    def test_get_direct_beam_shifts_with_centers(self):
        dp = self.signal
        c = np.array([[3, 3], [4, 4], [3, 3], [4, 4]])
        s = dp.get_direct_beam_shifts(radius=2)
        np.testing.assert_equal(s, np.array([[-0.5, -0.5],
                                             [ 0.5,  0.5],
                                             [-0.5, -0.5],
                                             [ 0.5,  0.5]]))

    def test_get_direct_beam_shifts_with_wrong_centers(self):
        dp = self.signal
        c = np.array([[3, 3], [4, 4], [3, 3]])
        nt.assert_raises(ValueError, dp.get_direct_beam_shifts, centers=c)

    def test_get_direct_beam_mask_signal_type(self):
        dp = self.signal
        mask = dp.get_direct_beam_mask(2)
        nt.assert_true(isinstance(mask, Signal2D))

    def test_get_direct_beam_mask(self):
        dp = self.signal
        a = np.array([[False, False, False, False, False, False, False, False],
                      [False, False, False, False, False, False, False, False],
                      [False, False, False,  True,  True, False, False, False],
                      [False, False,  True,  True,  True,  True, False, False],
                      [False, False,  True,  True,  True,  True, False, False],
                      [False, False, False,  True,  True, False, False, False],
                      [False, False, False, False, False, False, False, False],
                      [False, False, False, False, False, False, False, False]])
        mask = dp.get_direct_beam_mask(2)
        np.testing.assert_equal(mask, a)

    def test_get_direct_beam_mask_with_center(self):
        dp = self.signal
        a = np.array([[False, False, False, False, False, False, False, False],
                      [False, False, False, False, False, False, False, False],
                      [False, False, False, False,  True,  True, False, False],
                      [False, False, False,  True,  True,  True,  True, False],
                      [False, False, False,  True,  True,  True,  True, False],
                      [False, False, False, False,  True,  True, False, False],
                      [False, False, False, False, False, False, False, False],
                      [False, False, False, False, False, False, False, False]])
        mask = dp.get_direct_beam_mask(2, center=(4.5, 3.5))
        np.testing.assert_equal(mask, a)

    def test_get_vacuum_mask(self):
        dp = self.signal
        vm = dp.get_vacuum_mask(radius=3, threshold=1,
                                closing=False, opening=False)
        np.testing.assert_equal(vm, np.array([True, True, False, True]))

    def test_vacuum_mask_with_closing(self):
        dp = self.signal
        vm = dp.get_vacuum_mask(radius=3, threshold=1,
                                closing=True, opening=False)
        np.testing.assert_equal(vm, np.array([True, True, True, True]))

    def test_vacuum_mask_with_opening(self):
        dp = self.signal
        vm = dp.get_vacuum_mask(radius=3, threshold=1,
                                closing=False, opening=True)
        np.testing.assert_equal(vm, np.array([True, True, False, False]))


class Test_radial_profile:

    def setUp(self):
        dp = ElectronDiffraction(np.zeros((2, 8, 8)))
        dp.data[0]= np.array([[0., 0., 1., 2., 2., 1., 0., 0.],
                              [0., 1., 2., 3., 3., 2., 1., 0.],
                              [1., 2., 3., 4., 4., 3., 2., 1.],
                              [2., 3., 4., 5., 5., 4., 3., 2.],
                              [2., 3., 4., 5., 5., 4., 3., 2.],
                              [1., 2., 3., 4., 4., 3., 2., 1.],
                              [0., 1., 2., 3., 3., 2., 1., 0.],
                              [0., 0., 1., 2., 2., 1., 0., 0.]])

        dp.data[1]= np.array([[0., 1., 2., 3., 3., 3., 2., 1.],
                              [1., 2., 3., 4., 4., 4., 3., 2.],
                              [2., 3., 4., 5., 5., 5., 4., 3.],
                              [2., 3., 4., 5., 6., 5., 4., 3.],
                              [2., 3., 4., 5., 5., 5., 4., 3.],
                              [1., 2., 3., 4., 4., 4., 3., 2.],
                              [0., 1., 2., 3., 3., 3., 2., 1.],
                              [0., 0., 1., 2., 2., 2., 1., 0.]])
        self.signal = dp

    def test_radial_profile_signal_type(self):
        dp=self.signal
        rp = dp.get_radial_profile()
        nt.assert_true(isinstance(rp, Signal1D))

    def test_radial_profile_no_centers(self):
        dp = self.signal
        rp = dp.get_radial_profile()
        np.testing.assert_allclose(rp.data, np.array([[5., 4.25, 2.875,
                                                       1.7, 0.92857143, 0.],
                                                      [5., 4.75, 3.625,
                                                       2.5, 1.71428571,0.6]]),
                                                      atol=1e-3)

    def test_radial_profile_with_centers(self):
        dp = self.signal
        rp = dp.get_radial_profile(centers=np.array([[4, 3], [4, 3]]))
        np.testing.assert_allclose(rp.data, np.array([[5., 4.25, 2.875,
                                                       1.7, 0.92857143, 0.],
                                                      [5., 4.375, 3.5,
                                                       2.4, 2.07142857, 1.]]),
                                                      atol=1e-3)


class Test_correct_geometric_distortion:

    def setUp(self):
        dp = ElectronDiffraction(np.zeros((2, 8, 8)))
        dp.data[0]= np.array([[0., 0., 1., 2., 2., 1., 0., 0.],
                              [0., 1., 2., 3., 3., 2., 1., 0.],
                              [1., 2., 3., 4., 4., 3., 2., 1.],
                              [2., 3., 4., 5., 5., 4., 3., 2.],
                              [2., 3., 4., 5., 5., 4., 3., 2.],
                              [1., 2., 3., 4., 4., 3., 2., 1.],
                              [0., 1., 2., 3., 3., 2., 1., 0.],
                              [0., 0., 1., 2., 2., 1., 0., 0.]])

        dp.data[1]= np.array([[0., 1., 2., 3., 3., 3., 2., 1.],
                              [1., 2., 3., 4., 4., 4., 3., 2.],
                              [2., 3., 4., 5., 5., 5., 4., 3.],
                              [2., 3., 4., 5., 6., 5., 4., 3.],
                              [2., 3., 4., 5., 5., 5., 4., 3.],
                              [1., 2., 3., 4., 4., 4., 3., 2.],
                              [0., 1., 2., 3., 3., 3., 2., 1.],
                              [0., 0., 1., 2., 2., 2., 1., 0.]])
        self.signal = dp

    def test_correct_geometric_distortion_signal_type(self):
        dp=self.signal
        dp.correct_geometric_distortion(D=np.array([[1., 0., 0.],
                                                    [0., 1., 0.],
                                                    [0., 0., 1.]]))
        nt.assert_true(isinstance(dp, ElectronDiffraction))

#    def test_geometric_distortion_rotation_origin(self):
#        dp = self.signal
#        dp.correct_geometric_distortion()
#        np.testing.assert_allclose(rp.data, np.array([[5., 4.25, 2.875,
#                                                       1.7, 0.92857143, 0.],
#                                                      [5., 4.75, 3.625,
#                                                       2.5, 1.71428571,0.6]]),
#                                                      atol=1e-3)

#    def test_geometric_distortion(self):
#        dp = self.signal
#        rp = dp.get_radial_profile(centers=np.array([[4, 3], [4, 3]]))
#        np.testing.assert_allclose(rp.data, np.array([[5., 4.25, 2.875,
#                                                       1.7, 0.92857143, 0.],
#                                                      [5., 4.375, 3.5,
#                                                       2.4, 2.07142857, 1.]]),
#                                                      atol=1e-3)
