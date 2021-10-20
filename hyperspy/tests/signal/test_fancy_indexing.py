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
from numpy.testing import assert_array_equal
import pytest

from hyperspy import signals
from hyperspy import roi


class Test1D:

    def setup_method(self, method):
        self.signal = signals.Signal1D(np.arange(10))
        self.data = self.signal.data.copy()

    def test_slice_None(self):
        s = self.signal.isig[:]
        d = self.data
        np.testing.assert_array_equal(s.data, d)
        assert (s.axes_manager._axes[0].offset ==
                self.signal.axes_manager._axes[0].offset)
        assert (s.axes_manager._axes[0].scale ==
                self.signal.axes_manager._axes[0].scale)

    def test_slice_out_of_range_interval_not_in_axis(self):
        with pytest.raises(IndexError):
            self.signal.isig[20.:30.]

    def test_slice_out_of_range_interval_in_axis(self):
        s = self.signal.isig[-20.:100.]
        assert (s.axes_manager[0].low_value ==
                self.signal.axes_manager[0].low_value)
        assert (s.axes_manager[0].high_value ==
                self.signal.axes_manager[0].high_value)
        np.testing.assert_array_equal(s.data, self.signal.data)

    def test_reverse_slice(self):
        s = self.signal.isig[-1:1:-1]
        d = self.data[-1:1:-1]
        np.testing.assert_array_equal(s.data, d)
        assert s.axes_manager._axes[0].offset == 9
        assert (s.axes_manager._axes[0].scale ==
                self.signal.axes_manager._axes[0].scale * -1)

    def test_slice_out_of_axis(self):
        np.testing.assert_array_equal(
            self.signal.isig[-1.:].data, self.signal.data)
        np.testing.assert_array_equal(
            self.signal.isig[
                :11.].data, self.signal.data)

    def test_step0_slice(self):
        with pytest.raises(ValueError):
            self.signal.isig[::0]

    def test_index(self):
        s = self.signal.isig[3]
        assert s.data == 3
        assert len(s.axes_manager._axes) == 1
        assert s.data.shape == (1,)

    def test_float_index(self):
        s = self.signal.isig[3.4]
        assert s.data == 3
        assert len(s.axes_manager._axes) == 1
        assert s.data.shape == (1,)

    def test_signal_indexer_slice(self):
        s = self.signal.isig[1:-1]
        d = self.data[1:-1]
        np.testing.assert_array_equal(s.data, d)
        assert s.axes_manager._axes[0].offset == 1
        assert (s.axes_manager._axes[0].scale ==
                self.signal.axes_manager._axes[0].scale)

    def test_signal_indexer_reverse_slice(self):
        s = self.signal.isig[-1:1:-1]
        d = self.data[-1:1:-1]
        np.testing.assert_array_equal(s.data, d)
        assert s.axes_manager._axes[0].offset == 9
        assert (s.axes_manager._axes[0].scale ==
                self.signal.axes_manager._axes[0].scale * -1)

    def test_signal_indexer_step2_slice(self):
        s = self.signal.isig[1:-1:2]
        d = self.data[1:-1:2]
        np.testing.assert_array_equal(s.data, d)
        assert s.axes_manager._axes[0].offset == 1
        assert (np.sign(s.axes_manager._axes[0].scale) ==
                np.sign(self.signal.axes_manager._axes[0].scale))
        assert (s.axes_manager._axes[0].scale ==
                self.signal.axes_manager._axes[0].scale * 2.)

    def test_signal_indexer_index(self):
        s = self.signal.isig[3]
        assert s.data == 3
        assert len(s.axes_manager._axes) == 1
        assert s.data.shape == (1,)

    def test_navigation_indexer_navdim0(self):
        with pytest.raises(IndexError):
            self.signal.inav[3]

    def test_minus_one_index(self):
        s = self.signal.isig[-1]
        assert s.data == self.data[-1]

    def test_units(self):
        self.signal.axes_manager[0].scale = 0.5
        self.signal.axes_manager[0].units = 'µm'
        s = self.signal.isig[:'4000.0 nm']
        assert_array_equal(s.data, self.data[:8])
        s = self.signal.isig[:'4 µm']
        assert_array_equal(s.data, self.data[:8])

    def test_units_error(self):
        self.signal.axes_manager[0].scale = 0.5
        self.signal.axes_manager[0].units = 'µm'
        with pytest.raises(ValueError, message='should contains an units'):
            s = self.signal.isig[:'4000.0']


class Test2D:

    def setup_method(self, method):
        self.signal = signals.Signal2D(np.arange(24).reshape(6, 4))
        self.data = self.signal.data.copy()

    def test_index(self):
        s = self.signal.isig[3, 2]
        assert s.data[0] == 11
        assert len(s.axes_manager._axes) == 1
        assert s.data.shape == (1,)

    def test_partial(self):
        s = self.signal.isig[3, 2:5]
        np.testing.assert_array_equal(s.data, [11, 15, 19])
        assert len(s.axes_manager._axes) == 1
        assert s.data.shape == (3,)


class Test3D_SignalDim0:

    def setup_method(self, method):
        self.signal = signals.BaseSignal(np.arange(24).reshape((2, 3, 4)))
        self.data = self.signal.data.copy()
        self.signal.axes_manager.set_signal_dimension(0)

    def test_signal_indexer_signal_dim0_idx_error1(self):
        s = self.signal
        with pytest.raises(IndexError):
            s.isig[:].data

    def test_signal_indexer_signal_dim0_idx_error2(self):
        s = self.signal
        with pytest.raises(IndexError):
            s.isig[:, :].data

    def test_signal_indexer_signal_dim0_idx_error3(self):
        s = self.signal
        with pytest.raises(IndexError):
            s.isig[0]

    def test_navigation_indexer_signal_dim0(self):
        s = self.signal
        np.testing.assert_array_equal(s.data, s.inav[:].data)


class Test3D_Navigate_0_and_1:

    def setup_method(self, method):
        self.signal = signals.Signal1D(np.arange(24).reshape((2, 3, 4)))
        self.data = self.signal.data.copy()

    def test_1px_navigation_indexer_slice(self):
        s = self.signal.inav[1:2]
        d = self.data[:, 1:2]
        np.testing.assert_array_equal(s.data, d)
        assert s.axes_manager._axes[1].offset == 1
        assert s.axes_manager._axes[1].size == 1
        assert (s.axes_manager._axes[1].scale ==
                self.signal.axes_manager._axes[1].scale)

    def test_1px_signal_indexer_slice(self):
        s = self.signal.isig[1:2]
        d = self.data[:, :, 1:2]
        np.testing.assert_array_equal(s.data, d)
        assert s.axes_manager.signal_axes[0].offset == 1
        assert s.axes_manager.signal_axes[0].size == 1
        assert (s.axes_manager.signal_axes[0].scale ==
                self.signal.axes_manager.signal_axes[0].scale)

    def test_signal_indexer_slice_variance_signal(self):
        s1 = self.signal
        s1.estimate_poissonian_noise_variance()
        s1_1 = s1.isig[1:2]
        np.testing.assert_array_equal(
            s1.metadata.Signal.Noise_properties.variance.data[:, :, 1:2],
            s1_1.metadata.Signal.Noise_properties.variance.data)

    def test_navigation_indexer_slice_variance_signal(self):
        s1 = self.signal
        s1.estimate_poissonian_noise_variance()
        s1_1 = s1.inav[1:2]
        np.testing.assert_array_equal(
            s1.metadata.Signal.Noise_properties.variance.data[:, 1:2],
            s1_1.metadata.Signal.Noise_properties.variance.data)

    def test_signal_indexer_slice_variance_float(self):
        s1 = self.signal
        s1.metadata.set_item("Signal.Noise_properties.variance", 1.2)
        s1_1 = s1.isig[1:2]
        assert (
            s1.metadata.Signal.Noise_properties.variance ==
            s1_1.metadata.Signal.Noise_properties.variance)

    def test_navigation_indexer_slice_variance_float(self):
        s1 = self.signal
        s1.metadata.set_item("Signal.Noise_properties.variance", 1.2)
        s1_1 = s1.inav[1:2]
        assert (
            s1.metadata.Signal.Noise_properties.variance ==
            s1_1.metadata.Signal.Noise_properties.variance)

    def test_dimension_when_indexing(self):
        s = self.signal.inav[0]
        assert s.data.shape == self.data[:, 0, :].shape

    def test_dimension_when_slicing(self):
        s = self.signal.inav[0:1]
        assert s.data.shape == self.data[:, 0:1, :].shape


class Test3D_Navigate_1:

    def setup_method(self, method):
        self.signal = signals.BaseSignal(np.arange(24).reshape((2, 3, 4)))
        self.data = self.signal.data.copy()
        self.signal.axes_manager._axes[0].navigate = False
        self.signal.axes_manager._axes[1].navigate = True
        self.signal.axes_manager._axes[2].navigate = False

    def test_1px_navigation_indexer_slice(self):
        s = self.signal.inav[1:2]
        d = self.data[:, 1:2]
        np.testing.assert_array_equal(s.data, d)
        assert s.axes_manager._axes[1].offset == 1
        assert s.axes_manager._axes[1].size == 1
        assert (s.axes_manager._axes[1].scale ==
                self.signal.axes_manager._axes[1].scale)

    def test_1px_signal_indexer_slice(self):
        s = self.signal.isig[1:2]
        d = self.data[:, :, 1:2]
        np.testing.assert_array_equal(s.data, d)
        assert s.axes_manager.signal_axes[0].offset == 1
        assert s.axes_manager.signal_axes[0].size == 1
        assert (s.axes_manager.signal_axes[0].scale ==
                self.signal.axes_manager.signal_axes[0].scale)

    def test_subclass_assignment(self):
        im = self.signal.as_signal2D((-2, -1))
        assert isinstance(im.isig[0], signals.Signal1D)


class TestFloatArguments:

    def setup_method(self, method):
        self.signal = signals.BaseSignal(np.arange(10))
        self.signal.axes_manager.set_signal_dimension(1)
        self.signal.axes_manager[0].scale = 0.5
        self.signal.axes_manager[0].offset = 0.25
        self.data = self.signal.data.copy()

    def test_float_start(self):
        s = self.signal.isig[0.75:-1]
        d = self.data[1:-1]
        np.testing.assert_array_equal(s.data, d)
        assert s.axes_manager._axes[0].offset == 0.75
        assert (s.axes_manager._axes[0].scale ==
                self.signal.axes_manager._axes[0].scale)

    def test_float_end(self):
        s = self.signal.isig[1:4.75]
        d = self.data[1:-1]
        np.testing.assert_array_equal(s.data, d)
        assert s.axes_manager._axes[0].offset == 0.75
        assert (s.axes_manager._axes[0].scale ==
                self.signal.axes_manager._axes[0].scale)

    def test_float_both(self):
        s = self.signal.isig[0.75:4.75]
        d = self.data[1:-1]
        np.testing.assert_array_equal(s.data, d)
        assert s.axes_manager._axes[0].offset == 0.75
        assert (s.axes_manager._axes[0].scale ==
                self.signal.axes_manager._axes[0].scale)

    def test_float_step(self):
        s = self.signal.isig[::1.1]
        d = self.data[::2]
        np.testing.assert_array_equal(s.data, d)
        assert s.axes_manager._axes[0].offset == 0.25
        assert (s.axes_manager._axes[0].scale ==
                self.signal.axes_manager._axes[0].scale * 2)

    def test_negative_float_step(self):
        s = self.signal.isig[::-1.1]
        d = self.data[::-2]
        np.testing.assert_array_equal(s.data, d)
        assert s.axes_manager._axes[0].offset == 4.75
        assert (s.axes_manager._axes[0].scale ==
                self.signal.axes_manager._axes[0].scale * -2)


class TestEllipsis:

    def setup_method(self, method):
        self.signal = signals.BaseSignal(np.arange(2 ** 5).reshape(
            (2, 2, 2, 2, 2)))
        self.signal.axes_manager.set_signal_dimension(1)
        self.data = self.signal.data.copy()

    def test_in_between(self):
        s = self.signal.inav[0, ..., 0]
        np.testing.assert_array_equal(s.data, self.data[0, ..., 0, :])

    def test_ellipsis_navigation(self):
        s = self.signal.inav[..., 0]
        np.testing.assert_array_equal(s.data, self.data[0, ...])

    def test_ellipsis_navigation2(self):
        self.signal.axes_manager._axes[-2].navigate = False
        self.signal.axes_manager._axes[-3].navigate = False
        s = self.signal.isig[..., 0]
        np.testing.assert_array_equal(s.data, self.data[:, :, 0, ...])


class TestROISlicing:

    def setup_method(self, method):
        s = signals.Signal1D(np.random.random((10, 20, 1)))
        s.axes_manager[0].scale = 0.5
        s.axes_manager[1].scale = 2
        self.s = s

    def test_span_roi(self):
        s = self.s
        srx = roi.SpanROI(left=1.5, right=10)
        sry = roi.SpanROI(left=-1000, right=2)
        assert_array_equal(s.inav[srx, :].data, s.inav[1.5:10., ].data)
        assert_array_equal(
            s.inav[
                srx, sry].data, s.inav[
                1.5:10., -1000.:2.].data)

    def test_rectangular_roi(self):
        s = self.s
        sr = roi.RectangularROI(left=1.5, right=10, top=-1000, bottom=2)
        assert_array_equal(s.inav[sr].data, s.inav[1.5:10., -1000.:2.].data)

    def test_point2D_roi(self):
        s = self.s
        sr = roi.Point2DROI(x=1.5, y=10)
        assert_array_equal(s.inav[sr].data, s.inav[1.5, 10.].data)
