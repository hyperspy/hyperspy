# -*- coding: utf-8 -*-
# Copyright 2007-2024 The HyperSpy developers
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
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.


import numpy as np
import pytest
from scipy.optimize import OptimizeResult

import hyperspy.api as hs
from hyperspy.decorators import lazifyTestClass

TOL = 1e-5


@lazifyTestClass
class TestModel2D:
    def setup_method(self, method):
        g = hs.model.components2D.Gaussian2D(
            centre_x=-5.0, centre_y=-5.0, sigma_x=1.0, sigma_y=2.0
        )
        scale = 0.1
        x = np.arange(-10, 10, scale)
        y = np.arange(-10, 10, scale)
        X, Y = np.meshgrid(x, y)
        im = hs.signals.Signal2D(g.function(X, Y))
        im.axes_manager[0].scale = scale
        im.axes_manager[0].offset = -10
        im.axes_manager[1].scale = scale
        im.axes_manager[1].offset = -10

        self.m = im.create_model()
        gt = hs.model.components2D.Gaussian2D(
            centre_x=-4.5, centre_y=-4.5, sigma_x=0.5, sigma_y=1.5
        )
        self.m.append(gt)

    def _check_model_values(self, model, expected, **kwargs):
        print(
            f"({self.m.p0[0]:.7f}, {self.m.p0[1]:.7f}, {self.m.p0[2]:.7f}, {self.m.p0[3]:.7f}, {self.m.p0[4]:.7f})"
        )
        np.testing.assert_allclose(model.A.value, expected[0], **kwargs)
        np.testing.assert_allclose(model.centre_x.value, expected[1], **kwargs)
        np.testing.assert_allclose(model.centre_y.value, expected[2], **kwargs)
        np.testing.assert_allclose(model.sigma_x.value, expected[3], **kwargs)
        np.testing.assert_allclose(model.sigma_y.value, expected[4], **kwargs)

    @pytest.mark.parametrize(
        "bounded, expected",
        [
            (False, (1.0, -5.0, -5.0, 1.0, 2.0)),
            (True, (1.0002339, -4.75, -5.0, 1.0317225, 2.0)),
        ],
    )
    def test_fit_lm(self, bounded, expected):
        if bounded:
            self.m[0].centre_x.bmin = -4.75

        self.m.fit(optimizer="lm", bounded=bounded)
        self._check_model_values(self.m[0], expected, rtol=TOL)

        assert isinstance(self.m.fit_output, OptimizeResult)
        assert self.m.p_std is not None
        assert len(self.m.p_std) == 5
        assert np.all(~np.isnan(self.m.p_std))

    @pytest.mark.parametrize(
        "grad, expected",
        [
            (None, (0.9999993, -5.0000023, -4.9999982, 0.9999994, 1.9999986)),
            ("fd", (0.9999993, -5.0000023, -4.9999982, 0.9999994, 1.9999986)),
        ],
    )
    def test_fit_scipy_minimize(self, grad, expected):
        self.m.fit(optimizer="L-BFGS-B", grad=grad)
        self._check_model_values(self.m[0], expected, rtol=TOL)

        assert isinstance(self.m.fit_output, OptimizeResult)
        assert self.m.p_std is None

    def test_fit_no_analytical_gradient_error(self):
        with pytest.raises(
            ValueError, match="Analytical gradients not implemented for Model2D"
        ):
            self.m.fit(optimizer="L-BFGS-B", grad="analytical")

    def test_fit_no_odr_error(self):
        with pytest.raises(NotImplementedError, match="is not implemented for Model2D"):
            self.m.fit(optimizer="odr")

    def test_call(self):
        with pytest.raises(ValueError):
            self.m._get_current_data(component_list=0)


def test_Model2D_NotImplementedError_fitting():
    im = hs.signals.Signal2D(np.ones((128, 128)))
    m = im.create_model()
    gt = hs.model.components2D.Gaussian2D(
        centre_x=-4.5, centre_y=-4.5, sigma_x=0.5, sigma_y=1.5
    )
    m.append(gt)

    for member_f in [
        "_jacobian",
        "_function4odr",
        "_jacobian4odr",
        "_poisson_likelihood_function",
        "_gradient_ml",
        "_gradient_ls",
        "_huber_loss_function",
        "_gradient_huber",
    ]:
        with pytest.raises(NotImplementedError):
            _ = getattr(m, member_f)(None, None)


def test_Model2D_NotImplementedError_plot():
    im = hs.signals.Signal2D(np.ones((128, 128)))
    m = im.create_model()
    gt = hs.model.components2D.Gaussian2D(
        centre_x=-4.5, centre_y=-4.5, sigma_x=0.5, sigma_y=1.5
    )
    m.append(gt)

    for member_f in ["plot", "enable_adjust_position", "disable_adjust_position"]:
        with pytest.raises(NotImplementedError):
            _ = getattr(m, member_f)()

    for member_f in [
        "_plot_component",
        "_connect_component_line",
        "_disconnect_component_line",
    ]:
        with pytest.raises(NotImplementedError):
            _ = getattr(m, member_f)(None)


def test_channelswitches_mask():
    g = hs.model.components2D.Gaussian2D(
        A=1, centre_x=-5.0, centre_y=-5.0, sigma_x=1.0, sigma_y=2.0
    )

    scale = 0.1
    x = np.arange(-10, 10, scale)
    y = np.arange(-10, 10, scale)
    X, Y = np.meshgrid(x, y)

    im = hs.signals.Signal2D(g.function(X, Y))
    im.axes_manager[0].scale = scale
    im.axes_manager[0].offset = -10
    im.axes_manager[1].scale = scale
    im.axes_manager[1].offset = -10

    mask = im < 0.01

    # Add another Gaussian to get different results if the mask is ignored
    g2 = hs.model.components2D.Gaussian2D(
        A=1, centre_x=5.0, centre_y=5.0, sigma_x=1.0, sigma_y=2.0
    )
    im += hs.signals.Signal2D(g2.function(X, Y))

    m = im.create_model()
    gt = hs.model.components2D.Gaussian2D(
        centre_x=0.0, centre_y=0.0, sigma_x=50, sigma_y=50
    )
    m.append(gt)
    m._channel_switches = ~mask.data
    m.fit()

    assert not m._channel_switches[0, 0]
    assert m._channel_switches[50, 50]

    np.testing.assert_allclose(gt.centre_x.value, -5.0)
    np.testing.assert_allclose(gt.centre_y.value, -5.0)
    np.testing.assert_allclose(gt.sigma_x.value, 1.0)
    np.testing.assert_allclose(gt.sigma_y.value, 2.0)


class TestModel2DSetSignalRange:
    def setup_method(self, method):
        s = hs.signals.Signal2D(np.random.rand(10, 10, 20))
        s.axes_manager[-2].scale = 0.5
        s.axes_manager[-1].scale = 0.5
        m = s.create_model()
        self.s = s
        self.m = m

    def test_set_signal_range_from_mask(self):
        m = self.m
        mask = np.ones((10, 20), dtype=bool)
        mask[slice(1, 6), slice(15, 19)] = False
        m.set_signal_range_from_mask(mask)
        np.testing.assert_allclose(m._channel_switches, mask)

    def test_set_signal_range_from_mask_error(self):
        m = self.m
        shape = m.signal.axes_manager.signal_shape
        mask = np.ones(shape, dtype=bool)
        with pytest.raises(ValueError):
            m.set_signal_range_from_mask(mask)

        mask = np.ones(shape)
        with pytest.raises(ValueError):
            m.set_signal_range_from_mask(mask)

    def test_set_signal_range(self):
        m = self.m
        signal_shape = self.s.axes_manager._signal_shape_in_array
        ch = m._channel_switches

        m._set_signal_range_in_pixels(17, 19, 1, 3)
        mask1 = np.zeros(signal_shape, dtype=bool)
        mask1[slice(17, 20), slice(1, 4)] = True
        np.testing.assert_allclose(ch, mask1)

        m.reset_signal_range()
        mask2 = np.ones(signal_shape, dtype=bool)
        np.testing.assert_allclose(ch, mask2)

        m.set_signal_range(8.5, 9.5, 0.5, 1.5)
        np.testing.assert_allclose(ch, mask1)

    def test_add_signal_range(self):
        m = self.m
        signal_shape = self.s.axes_manager.signal_shape[::-1]
        ch = m._channel_switches

        # Set all channel to zeros
        zeros = np.zeros(signal_shape, dtype=bool)
        m.set_signal_range_from_mask(zeros)
        np.testing.assert_allclose(m._channel_switches, zeros)

        m._add_signal_range_in_pixels(17, 19, 1, 3)
        mask = np.zeros(signal_shape, dtype=bool)
        mask[slice(17, 20), slice(1, 4)] = True
        np.testing.assert_allclose(ch, mask)

        m._add_signal_range_in_pixels(5, 6, 1, 3)
        mask[slice(5, 7), slice(1, 4)] = True
        np.testing.assert_allclose(ch, mask)

        # Set all channel to zeros
        zeros = np.zeros(signal_shape, dtype=bool)
        m.set_signal_range_from_mask(zeros)
        np.testing.assert_allclose(m._channel_switches, zeros)

        m.add_signal_range(8.5, 9.5, 0.5, 1.5)
        mask = np.zeros(signal_shape, dtype=bool)
        mask[slice(17, 20), slice(1, 4)] = True
        np.testing.assert_allclose(ch, mask)

        m.add_signal_range(2.5, 3, 0.5, 1.5)
        mask[slice(5, 7), slice(1, 4)] = True
        np.testing.assert_allclose(ch, mask)

    def test_remove_signal_range(self):
        m = self.m
        signal_shape = self.s.axes_manager.signal_shape[::-1]
        ch = m._channel_switches

        m._remove_signal_range_in_pixels(17, 19, 1, 3)
        mask = np.ones(signal_shape, dtype=bool)
        mask[slice(17, 20), slice(1, 4)] = False
        np.testing.assert_allclose(ch, mask)

        m._remove_signal_range_in_pixels(5, 6, 1, 3)
        mask[slice(5, 7), slice(1, 4)] = False
        np.testing.assert_allclose(ch, mask)

        # Set all channel to ones
        ones = np.ones(signal_shape, dtype=bool)
        m.set_signal_range_from_mask(ones)
        np.testing.assert_allclose(m._channel_switches, ones)

        m.remove_signal_range(8.5, 9.5, 0.5, 1.5)
        mask = np.ones(signal_shape, dtype=bool)
        mask[slice(17, 20), slice(1, 4)] = False
        np.testing.assert_allclose(ch, mask)

        m.remove_signal_range(2.5, 3, 0.5, 1.5)
        mask[slice(5, 7), slice(1, 4)] = False
        np.testing.assert_allclose(ch, mask)

    def test_initial_mask(self):
        m = self.m
        assert m._channel_switches.shape == (10, 20)
