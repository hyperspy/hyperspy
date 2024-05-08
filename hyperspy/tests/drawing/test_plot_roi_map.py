# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
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

import matplotlib.pyplot as plt
import numpy as np
import pytest

import hyperspy.api as hs
from hyperspy.utils.plot import plot_roi_map

BASELINE_DIR = "plot_roi_map"
DEFAULT_TOL = 2.0
STYLE_PYTEST_MPL = "default"


# params different shapes of data, sig, nav dims
@pytest.fixture(
    params=[(1, 1), (1, 2), (2, 1), (2, 2)], ids=lambda sn: f"s{sn[0]}n{sn[1]}"
)
def test_signal(request):
    sig_dims, nav_dims = request.param

    sig_size = 13
    nav_size = 3

    sig_shape = (sig_size,) * sig_dims
    nav_shape = (nav_size,) * nav_dims
    shape = (*nav_shape, *sig_shape)
    test_data = np.zeros(shape)

    axes = []

    for _, name in zip(range(nav_dims), "xyz"):
        axes.append(
            {"name": name, "size": nav_size, "offset": 0, "scale": 1, "units": "um"}
        )

    for _, name in zip(range(sig_dims), ["Ix", "Iy", "Iz"]):
        axes.append(
            {
                "name": name,
                "size": sig_size,
                "offset": 0,
                "scale": 1,
                "units": "nm",
            }
        )
    sig = hs.signals.BaseSignal(
        test_data,
        axes=axes,
    )

    sig = sig.transpose(sig_dims)

    sig.inav[0, ...].isig[0, ...] = 1
    sig.inav[1, ...].isig[2, ...] = 2
    sig.inav[2, ...].isig[4, ...] = 3

    return sig


def test_args_wrong_shape():
    rng = np.random.default_rng()

    sig2 = hs.signals.BaseSignal(rng.random(size=(2, 2)))

    no_sig = sig2.transpose(0)
    no_nav = sig2.transpose(2)

    sig5 = hs.signals.BaseSignal(rng.random(size=(2, 2, 2, 2, 2)))
    three_sigs = sig5.transpose(3)
    three_navs = sig5.transpose(2)

    with pytest.raises(ValueError):
        plot_roi_map(no_sig, 1)

    # navigation needed
    with pytest.raises(ValueError):
        plot_roi_map(no_nav, [hs.roi.Point1DROI(0)])

    # unsupported signal
    for sig in [no_nav, three_sigs]:
        with pytest.raises(ValueError):
            plot_roi_map(sig, [hs.roi.Point1DROI(0)])

    # value error also raised because 1D ROI not right shape
    with pytest.raises(ValueError):
        plot_roi_map(sig, [hs.roi.Point1DROI(0)])

    # 3 navigation works fine
    plot_roi_map(three_navs, [hs.roi.CircleROI()])

    # 3 navigation works fine
    plot_roi_map(three_navs)


def test_passing_rois():
    s = hs.signals.Signal1D(np.arange(100).reshape(10, 10))
    int_rois, int_roi_sums = plot_roi_map(s, 3)

    rois, roi_sums = plot_roi_map(s, int_rois)

    assert rois is int_rois

    # passing the rois rather than generating own should yield same results
    assert int_roi_sums is not roi_sums
    assert int_roi_sums == roi_sums


def test_roi_positioning():
    s = hs.signals.Signal1D(np.arange(100).reshape(10, 10))
    rois, _ = plot_roi_map(s, 1)

    assert len(rois) == 1

    assert rois[0].left == 0
    assert rois[0].right == 4.5

    rois, _ = plot_roi_map(s, 2)

    assert len(rois) == 2
    assert rois[0].left == 0
    assert rois[0].right == 2.25
    assert rois[1].left == 2.25
    assert rois[1].right == 4.5

    # no overlap
    assert rois[0].right <= rois[1].left

    rois, _ = plot_roi_map(s, 3)

    assert len(rois) == 3
    assert rois[0].left == 0
    assert rois[0].right == 1.5
    assert rois[1].left == 1.5
    assert rois[1].right == 3
    assert rois[2].left == 3
    assert rois[2].right == 4.5

    # no overlap
    assert rois[0].right <= rois[1].left and rois[1].right <= rois[2].left


@pytest.mark.parametrize("nrois", [1, 2, 3])
def test_navigator(test_signal, nrois):
    rois, roi_sums = plot_roi_map(test_signal, nrois)
    assert len(rois) == nrois


def test_roi_sums():
    # check that the sum is correct
    s = hs.signals.Signal1D(np.arange(100).reshape(10, 10))

    rois, roi_sums = hs.plot.plot_roi_map(s, 2)

    for roi, roi_sum in zip(rois, roi_sums):
        np.testing.assert_allclose(roi(s).sum(-1), roi_sum.data)


def test_circle_roi():
    data = np.zeros((2, 2, 7, 7))
    data[-1, -1, 0, 0] = 10000
    s = hs.signals.Signal2D(data)
    roi = hs.roi.CircleROI(cx=3, cy=3, r=4, r_inner=0)

    rois, roi_sums = hs.plot.plot_roi_map(s, rois=[roi])
    roi_sum = roi_sums[0]

    assert not np.any(np.isin(10000, roi_sum))
    assert np.allclose(roi_sum, np.zeros((2, 2)))

    roi.cx = 4  # force update
    roi.cx = 3

    # no change expected
    assert not np.any(np.isin(10000, roi_sum))
    assert np.allclose(roi_sum, np.zeros((2, 2)))

    roi.cx = 0
    roi.cy = 0

    # check can actually find 10000
    assert np.any(np.isin(10000, roi_sum))


def test_pass_ROI():
    rng = np.random.default_rng(0)
    data = rng.random(size=(10, 10, 50))
    s = hs.signals.Signal2D(data)

    roi = hs.roi.CircleROI()
    hs.plot.plot_roi_map(s, rois=roi)


def test_color():
    rng = np.random.default_rng(0)
    data = rng.random(size=(10, 10, 50))
    s = hs.signals.Signal2D(data)

    # same number
    hs.plot.plot_roi_map(s, rois=3, color=["C0", "C1", "C2"])

    with pytest.raises(ValueError):
        hs.plot.plot_roi_map(s, rois=3, color=["C0", "C1"])

    with pytest.raises(ValueError):
        hs.plot.plot_roi_map(s, rois=3, color=["C0", "C1", "C2", "C3"])

    with pytest.raises(ValueError):
        hs.plot.plot_roi_map(s, rois=1, color=["unvalid_cmap"])


@pytest.mark.mpl_image_compare(
    baseline_dir=BASELINE_DIR, tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL
)
@pytest.mark.parametrize("cmap", (None, "gray"))
def test_cmap_image(cmap):
    rng = np.random.default_rng(0)
    data = rng.random(size=(10, 10, 50))
    s = hs.signals.Signal1D(data)

    rois, roi_sums = hs.plot.plot_roi_map(s, rois=2, cmap=cmap)

    return roi_sums[0]._plot.signal_plot.figure


@pytest.mark.mpl_image_compare(
    baseline_dir=BASELINE_DIR, tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL
)
@pytest.mark.parametrize("color", (None, ["r", "b"]))
def test_color_image(color):
    rng = np.random.default_rng(0)
    data = rng.random(size=(10, 10, 50))
    s = hs.signals.Signal1D(data)

    rois, roi_sums = hs.plot.plot_roi_map(s, rois=2, color=color)

    return roi_sums[0]._plot.signal_plot.figure


def test_close():
    # We can't test with `single_figure=True`, because `plt.close()`
    # doesn't call on close callback.
    # https://github.com/matplotlib/matplotlib/issues/18609
    rng = np.random.default_rng(0)
    data = rng.random(size=(10, 10, 50))
    s = hs.signals.Signal1D(data)

    rois, roi_sums = hs.plot.plot_roi_map(s, rois=3)
    # check that it closes and remove the roi from the figure
    for roi, roi_sum in zip(rois, roi_sums):
        assert len(roi.signal_map) == 1
        roi_sum._plot.close()
        assert len(roi.signal_map) == 0


def test_cmap_error():
    rng = np.random.default_rng(0)
    data = rng.random(size=(10, 10, 50))
    s = hs.signals.Signal2D(data)

    # same number
    hs.plot.plot_roi_map(s, rois=3, cmap=["C0", "C1", "C2"])

    with pytest.raises(ValueError):
        hs.plot.plot_roi_map(s, rois=3, cmap=["C0", "C1"])

    with pytest.raises(ValueError):
        hs.plot.plot_roi_map(s, rois=3, cmap=["C0", "C1", "C2", "C3"])


@pytest.mark.mpl_image_compare(
    baseline_dir=BASELINE_DIR, tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL
)
@pytest.mark.parametrize("cmap", (None, "gray"))
def test_single_figure_image(cmap):
    rng = np.random.default_rng(0)
    data = rng.random(size=(10, 10, 50))
    s = hs.signals.Signal1D(data)

    hs.plot.plot_roi_map(s, rois=3, cmap=cmap, single_figure=True, scalebar=False)

    return plt.gcf()


@pytest.mark.parametrize("color", (None, ["C0", "C1"], ["r", "b"]))
def test_single_figure_spectra(color):
    rng = np.random.default_rng(0)
    data = rng.random(size=(50, 10, 10))
    s = hs.signals.Signal2D(data)

    hs.plot.plot_roi_map(s, rois=2, color=color, single_figure=True)
    if color is None:
        color = ["b", "g"]

    ax = plt.gca()
    for line, color_ in zip(ax.lines, color):
        assert line.get_color() == color_
