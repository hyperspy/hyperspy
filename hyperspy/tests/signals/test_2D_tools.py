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

from unittest import mock

import numpy as np
import numpy.testing as npt
import pytest

try:
    # scipy >=1.10
    from scipy.datasets import ascent, face
except ImportError:
    # scipy <1.10
    from scipy.misc import ascent, face
from scipy.ndimage import fourier_shift

import hyperspy.api as hs
from hyperspy.decorators import lazifyTestClass
from hyperspy.exceptions import SignalDimensionError
from hyperspy.signal_tools import LineInSignal2D, Signal2DCalibration


def _generate_parameters():
    parameters = []
    for normalize_corr in [False, True]:
        for reference in ["current", "cascade", "stat"]:
            parameters.append([normalize_corr, reference])
    return parameters


@lazifyTestClass
class TestSubPixelAlign:
    def setup_method(self, method):
        ref_image = ascent()
        center = np.array((256, 256))
        shifts = np.array(
            [
                (0.0, 0.0),
                (4.3, 2.13),
                (1.65, 3.58),
                (-2.3, 2.9),
                (5.2, -2.1),
                (2.7, 2.9),
                (5.0, 6.8),
                (-9.1, -9.5),
                (-9.0, -9.9),
                (-6.3, -9.2),
            ]
        )
        s = hs.signals.Signal2D(np.zeros((10, 100, 100)))
        for i in range(10):
            # Apply each sup-pixel shift using FFT and InverseFFT
            offset_image = fourier_shift(np.fft.fftn(ref_image), shifts[i])
            offset_image = np.fft.ifftn(offset_image).real

            # Crop central regions of shifted images to avoid wrap around
            s.data[i, ...] = offset_image[
                center[0] : center[0] + 100, center[1] : center[1] + 100
            ]

        self.signal = s
        self.shifts = shifts

    def test_align_subpix(self):
        # Align signal
        s = self.signal
        shifts = self.shifts
        s.align2D(shifts=shifts)
        # Compare by broadcasting
        np.testing.assert_allclose(s.data[4], s.data[0], rtol=0.5)
        s.estimate_shift2D(reference="cascade", sub_pixel_factor=10)

    @pytest.mark.parametrize(("normalize_corr", "reference"), _generate_parameters())
    def test_estimate_subpix(self, normalize_corr, reference):
        s = self.signal
        shifts = s.estimate_shift2D(
            sub_pixel_factor=200, normalize_corr=normalize_corr, reference=reference
        )
        np.testing.assert_allclose(shifts, self.shifts, rtol=2, atol=0.2, verbose=True)

    @pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
    @pytest.mark.parametrize(("plot"), [True, "reuse"])
    def test_estimate_subpix_plot(self, plot):
        # To avoid this function plotting many figures and holding the test, we
        # make sure the backend is set to `agg` in case it is set to something
        # else in the testing environment
        import matplotlib.pyplot as plt

        plt.switch_backend("agg")
        s = self.signal
        s.estimate_shift2D(sub_pixel_factor=200, plot=plot)

    def test_align_crop_error(self):
        s = self.signal
        shifts = self.shifts
        s_size = np.array(s.axes_manager.signal_shape)
        shifts[0] = s_size + 1
        with pytest.raises(ValueError, match="Cannot crop signal"):
            s.align2D(shifts=shifts, crop=True)


@lazifyTestClass
class TestAlignTools:
    def setup_method(self, method):
        im = face(gray=True)
        self.ascent_offset = np.array((256, 256))
        s = hs.signals.Signal2D(np.zeros((10, 100, 100)))
        self.scales = np.array((0.1, 0.3))
        self.offsets = np.array((-2, -3))
        izlp = []
        for ax, offset, scale in zip(
            s.axes_manager.signal_axes, self.offsets, self.scales
        ):
            ax.scale = scale
            ax.offset = offset
            izlp.append(ax.value2index(0))
        self.izlp = izlp
        self.ishifts = np.array(
            [
                (0, 0),
                (4, 2),
                (1, 3),
                (-2, 2),
                (5, -2),
                (2, 2),
                (5, 6),
                (-9, -9),
                (-9, -9),
                (-6, -9),
            ]
        )
        self.new_offsets = self.offsets - self.ishifts.min(0) * self.scales
        zlp_pos = self.ishifts + self.izlp
        for i in range(10):
            slices = self.ascent_offset - zlp_pos[i, ...]
            s.data[i, ...] = im[
                slices[0] : slices[0] + 100, slices[1] : slices[1] + 100
            ]
        self.signal = s

        # How image should be after successfull alignment
        smin = self.ishifts.min(0)
        smax = self.ishifts.max(0)
        offsets = self.ascent_offset + self.offsets / self.scales - smin
        size = np.array((100, 100)) - (smax - smin)
        self.aligned = im[
            int(offsets[0]) : int(offsets[0] + size[0]),
            int(offsets[1]) : int(offsets[1] + size[1]),
        ]

    def test_estimate_shift(self):
        s = self.signal
        shifts = s.estimate_shift2D()
        np.testing.assert_allclose(shifts, self.ishifts)

    def test_align_no_shift(self):
        s = self.signal
        shifts = s.estimate_shift2D()
        shifts.fill(0)
        with pytest.warns(UserWarning, match="provided shifts are all zero"):
            shifts = s.align2D(shifts=shifts)
            assert shifts is None

    def test_align_twice(self):
        s = self.signal
        s.align2D()
        with pytest.warns(UserWarning, match="the images are already aligned"):
            shifts = s.align2D()
            assert shifts.sum() == 0

    def test_align(self):
        # Align signal
        m = mock.Mock()
        s = self.signal
        s.events.data_changed.connect(m.data_changed)
        s.align2D()
        # Compare by broadcasting
        assert np.all(s.data == self.aligned)
        assert m.data_changed.called

    def test_align_expand(self):
        s = self.signal
        s.align2D(expand=True)
        # Check the numbers of NaNs to make sure expansion happened properly
        ds = self.ishifts.max(0) - self.ishifts.min(0)
        Nnan = np.sum(ds) * 100 + np.prod(ds)
        Nnan_data = np.sum(1 * np.isnan(s.data), axis=(1, 2))
        # Due to interpolation, the number of NaNs in the data might
        # be 2 higher (left and right side) than expected
        assert np.all(Nnan_data - Nnan <= 2)

        # Check alignment is correct
        d_al = s.data[:, ds[0] : -ds[0], ds[1] : -ds[1]]
        assert np.all(d_al == self.aligned)


@lazifyTestClass
class TestGetSignal2DScale:
    def setup_method(self, method):
        self.s0 = hs.signals.Signal2D(np.ones((100, 50)))
        self.s1 = hs.signals.Signal2D(np.ones((100, 200)))

    def test_default_scale(self):
        s = self.s0
        x0, y0, x1, y1, length = 10.0, 10.0, 30.0, 10.0, 80.0
        scale0 = s._get_signal2d_scale(x0=x0, y0=y0, x1=x1, y1=y1, length=length)
        x0, y0, x1, y1 = 10, 10, 30, 10
        scale1 = s._get_signal2d_scale(x0=x0, y0=y0, x1=x1, y1=y1, length=length)
        # With the default scale (1), scale0 and scale1 have the same value
        assert scale0 == scale1 == 4.0

    def test_non_one_scale(self):
        s = self.s0
        sa = s.axes_manager.signal_axes
        x0, y0, x1, y1, length = 4.0, 2.0, 4.0, 6.0, 16.0
        sa[0].scale, sa[1].scale = 0.1, 0.1
        scale0 = s._get_signal2d_scale(x0=x0, y0=y0, x1=x1, y1=y1, length=length)
        x0, y0, x1, y1 = 4, 2, 4, 6
        scale1 = s._get_signal2d_scale(x0=x0, y0=y0, x1=x1, y1=y1, length=length)
        assert scale0 == 0.4
        assert scale1 == 4

    def test_diagonal(self):
        length = (100**2 + 50**2) ** 0.5
        s = self.s1
        sa = s.axes_manager.signal_axes
        sa[0].scale, sa[1].scale = 0.5, 1.0
        scale0 = s._get_signal2d_scale(
            x0=0.0, y0=0.0, x1=50.0, y1=50.0, length=length / 2
        )
        scale1 = s._get_signal2d_scale(x0=0, y0=0, x1=100, y1=50, length=length)
        assert scale0 == 0.5
        assert scale1 == 1.0

    def test_string_input(self):
        s = self.s0
        with pytest.raises(TypeError):
            s._get_signal2d_scale(x0="2.", y0="1", x1="3", y1="5", length=2)


@lazifyTestClass
class TestCalibrate2D:
    def setup_method(self, method):
        self.s0 = hs.signals.Signal2D(np.ones((100, 100)))
        self.s1 = hs.signals.Signal2D(np.ones((100, 200)))
        self.s2 = hs.signals.Signal2D(np.ones((3, 30, 40)))
        self.s3 = hs.signals.Signal2D(np.ones((2, 3, 30, 440)))

    def test_cli_default_scale(self):
        s = self.s0
        x0, y0, x1, y1, new_length = 10.0, 10.0, 30.0, 10.0, 5.0
        units = "test"
        s._calibrate(x0=x0, y0=y0, x1=x1, y1=y1, new_length=new_length, units=units)
        sa = s.axes_manager.signal_axes
        assert sa[0].units == sa[1].units == units
        assert sa[0].scale == sa[1].scale == 0.25

    def test_cli_non_one_scale(self):
        s = self.s0
        sa = s.axes_manager.signal_axes
        sa[0].scale, sa[1].scale = 0.25, 0.25
        x0, y0, x1, y1, new_length = 10.0, 10.0, 10.0, 20.0, 40.0
        s._calibrate(x0=x0, y0=y0, x1=x1, y1=y1, new_length=new_length)
        assert sa[0].scale == sa[1].scale == 1.0
        x0, y0, x1, y1 = 10, 10, 10, 20
        s._calibrate(x0=x0, y0=y0, x1=x1, y1=y1, new_length=new_length)
        assert sa[0].scale == sa[1].scale == 4.0

    def test_non_interactive_missing_parameters(self):
        s = self.s1
        with pytest.raises(ValueError, match="With interactive=False x0, y0,*"):
            s.calibrate(x0=20, y0=30, interactive=False)

    def test_non_interactive_not_same_input_scale(self, caplog):
        s = self.s1
        s.axes_manager[0].scale = 0.1
        s.axes_manager[1].scale = 0.5
        s.calibrate(x0=5, y0=2, x1=9, y1=5, new_length=2.5, interactive=False)
        assert "The previous scaling is not the same for both axes" in caplog.text

    def test_non_interactive_not_same_input_units(self, caplog):
        s = self.s1
        s.axes_manager[0].units = "nm"
        s.axes_manager[1].units = "mm"
        s.calibrate(x0=5, y0=2, x1=9, y1=5, new_length=2.5, interactive=False)
        assert "The signal axes does not have the same units" in caplog.text

    def test_non_interactive_default_scale(self):
        s = self.s1
        x0, y0, x1, y1, new_length = 10.0, 10.0, 10.0, 20.0, 40.0
        units = "nm"
        s.calibrate(
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            new_length=new_length,
            interactive=False,
            units=units,
        )
        sa = s.axes_manager.signal_axes
        assert sa[0].units == sa[1].units == units
        assert sa[0].scale == sa[1].scale == 4.0
        x0, y0, x1, y1 = 10, 10, 10, 20
        s.calibrate(
            x0=x0, y0=y0, x1=x1, y1=y1, new_length=new_length, interactive=False
        )
        assert sa[0].scale == sa[1].scale == 4.0
        assert sa[0].units == sa[1].units == units

    def test_3d_signal(self):
        s = self.s2
        x0, y0, x1, y1, new_length = 10.0, 10.0, 30.0, 10.0, 5.0
        units = "test"
        s._calibrate(x0=x0, y0=y0, x1=x1, y1=y1, new_length=new_length, units=units)
        sa = s.axes_manager.signal_axes
        assert sa[0].units == sa[1].units == units
        assert sa[0].scale == sa[1].scale == 0.25

    def test_4d_signal(self):
        s = self.s3
        x0, y0, x1, y1, new_length = 10.0, 10.0, 30.0, 10.0, 5.0
        units = "test"
        s._calibrate(x0=x0, y0=y0, x1=x1, y1=y1, new_length=new_length, units=units)
        sa = s.axes_manager.signal_axes
        assert sa[0].units == sa[1].units == units
        assert sa[0].scale == sa[1].scale == 0.25

    def test_non_interactive_non_one_scale(self):
        s = self.s1
        x0, y0, x1, y1, new_length = 10.0, 10.0, 10.0, 20.0, 40.0
        sa = s.axes_manager.signal_axes
        sa[0].scale, sa[1].scale = 5.0, 5.0
        s.calibrate(
            x0=x0, y0=y0, x1=x1, y1=y1, new_length=new_length, interactive=False
        )
        assert sa[0].scale == sa[1].scale == 20.0
        x0, y0, x1, y1 = 10, 10, 10, 20
        s.calibrate(
            x0=x0, y0=y0, x1=x1, y1=y1, new_length=new_length, interactive=False
        )
        assert sa[0].scale == sa[1].scale == 4.0


def test_add_ramp():
    s = hs.signals.Signal2D(np.indices((3, 3)).sum(axis=0) + 4)
    s.add_ramp(-1, -1, -4)
    npt.assert_allclose(s.data, 0)


def test_add_ramp_lazy():
    s = hs.signals.Signal2D(np.indices((3, 3)).sum(axis=0) + 4).as_lazy()
    s.add_ramp(-1, -1, -4)
    npt.assert_almost_equal(s.data.compute(), 0)


def test_line_in_signal2d_wrong_signal_dimension():
    s = hs.signals.Signal1D(np.zeros(100))
    with pytest.raises(SignalDimensionError):
        _ = LineInSignal2D(s)


def test_line_in_signal2d():
    s = hs.signals.Signal2D(np.zeros((100, 100)))
    _ = LineInSignal2D(s)


def test_signal_2d_calibration_wrong_signal_dimension():
    s = hs.signals.Signal1D(np.zeros(100))
    with pytest.raises(SignalDimensionError):
        _ = Signal2DCalibration(s)


def test_signal_2d_calibration():
    s = hs.signals.Signal2D(np.zeros((100, 100)))
    s2dc = Signal2DCalibration(s)
    s2dc.x0, s2dc.x1 = 1, 11
    s2dc.y0, s2dc.y1 = 20, 20
    s2dc.new_length, s2dc.units = 50, "nm"
    s2dc.apply()
    assert s.axes_manager[0].scale == 5
    assert s.axes_manager[1].scale == 5
    assert s.axes_manager[0].units == "nm"
    assert s.axes_manager[1].units == "nm"


def test_signal_2d_calibration_no_new_length(caplog):
    s = hs.signals.Signal2D(np.zeros((100, 100)))
    s2dc = Signal2DCalibration(s)
    s2dc.x0, s2dc.x1 = 1, 11
    s2dc.y0, s2dc.y1 = 20, 20
    s2dc.apply()
    assert "Input a new length before pressing apply." in caplog.text


def test_signal_2d_calibration_value_nan(caplog):
    s = hs.signals.Signal2D(np.zeros((100, 100)))
    s2dc = Signal2DCalibration(s)
    s2dc.x0, s2dc.x1, s2dc.y0, s2dc.y1 = 1, np.nan, 20, 20
    s2dc.new_length = 50
    s2dc.apply()
    assert "Line position is not valid" in caplog.text
