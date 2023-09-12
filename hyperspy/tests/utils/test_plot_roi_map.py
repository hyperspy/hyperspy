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

from functools import wraps

import pytest
import numpy as np

import hyperspy.api as hs

from hyperspy.utils.plot import plot_roi_map


@pytest.fixture
def test_signal():
    test_data = np.zeros((64, 64, 1024))
    test_data[32:, :, 300:] = 1
    test_data[48:, :, 500:] = 2
    sig = hs.signals.Signal1D(
        test_data,
        axes=[
            {"name": "x", "size": 64, "offset": 0, "scale": 1, "units": "um"},
            {"name": "y", "size": 64, "offset": 0, "scale": 1, "units": "um"},
            {
                "name": "Wavelength",
                "size": 1024,
                "offset": 0,
                "scale": 1,
                "units": "nm",
            },
        ],
    )

    sig.set_signal_type("CLSEM")
    return sig


def sig_mpl_compare(type: set(["sig", "nav"])):
    def wrapper(f):
        @pytest.mark.mpl_image_compare(baseline_dir="plot_span_map")
        @wraps(f)
        def wrapped(*args, **kwargs):
            sig = f(*args, **kwargs)
            if type == "sig":
                fig = sig._plot.signal_plot.figure
            elif type == "nav":
                fig = sig._plot.navigation_plot.figure

            return fig

        return wrapped

    return wrapper


def test_plot_span_map_args(test_signal):
    with pytest.raises(ValueError):
        plot_roi_map(test_signal, 4)

    with pytest.raises(ValueError):
        plot_roi_map(test_signal, [hs.roi.SpanROI(0, 1),
                                    hs.roi.SpanROI(1, 2),
                                    hs.roi.SpanROI(2, 3),
                                    hs.roi.SpanROI(3, 4)])

    line_spectra = test_signal.inav[0, :]

    with pytest.raises(
        ValueError,
        match=("This method is designed for data with 1 signal and 2 "
               "navigation dimensions, not 1 and 1 respectively"),
    ):
        plot_roi_map(line_spectra)

    single_spectra = line_spectra.inav[0]

    with pytest.raises(
        ValueError,
        match=("This method is designed for data with 1 signal and 2 "
               "navigation dimensions, not 1 and 0 respectively"),
    ):
        plot_roi_map(single_spectra)


def test_passing_spans(test_signal):
    _, int_spans, int_span_sigs, int_span_sums = plot_roi_map(test_signal, 3)

    _, spans, span_sigs, span_sums = plot_roi_map(test_signal, int_spans)

    assert spans is int_spans

    # passing the spans rather than generating own should yield same results
    assert int_span_sigs is not span_sigs
    assert int_span_sigs == span_sigs

    assert int_span_sums is not span_sums
    assert int_span_sums == span_sums


def test_span_positioning(test_signal):
    _, spans, *_ = plot_roi_map(test_signal, 1)

    assert len(spans) == 1

    assert spans[0].left == pytest.approx(0)
    assert spans[0].right == pytest.approx(1023 / 2)

    _, spans, *_ = plot_roi_map(test_signal, 2)

    assert len(spans) == 2
    assert spans[0].left == pytest.approx(0)
    assert spans[0].right == pytest.approx(1023 / 4)
    assert spans[1].left == pytest.approx(1023 / 4)
    assert spans[1].right == pytest.approx(1023 / 2)

    # no overlap
    assert spans[0].right <= spans[1].left

    _, spans, *_ = plot_roi_map(test_signal, 3)

    assert len(spans) == 3
    assert spans[0].left == pytest.approx(0)
    assert spans[0].right == pytest.approx(1023 / 6)
    assert spans[1].left == pytest.approx(1023 / 6)
    assert spans[1].right == pytest.approx(1023 / 3)
    assert spans[2].left == pytest.approx(1023 / 3)
    assert spans[2].right == pytest.approx(1023 / 2)

    # no overlap
    assert spans[0].right <= spans[1].left and spans[1].right <= spans[2].left


@sig_mpl_compare("sig")
@pytest.mark.parametrize("nspans", [1, 2, 3])
def test_navigator(test_signal, nspans):
    all_sums, spans, span_sigs, span_sums = plot_roi_map(test_signal, nspans)

    assert np.all(all_sums.data == test_signal.sum().data)

    return all_sums


@sig_mpl_compare("sig")
@pytest.mark.parametrize("nspans", [1, 2, 3])
@pytest.mark.parametrize("span_out", [1, 2, 3])
def test_span_sums(test_signal, nspans, span_out):
    all_sums, spans, span_sigs, span_sums = plot_roi_map(test_signal, nspans)

    if span_out > nspans:
        span_out = nspans

    return span_sums[span_out - 1]


@sig_mpl_compare("sig")
@pytest.mark.parametrize("which_plot", ["all_sums", "span_sums"])
def test_interaction(test_signal, which_plot):
    all_sums, spans, span_sigs, span_sums = plot_roi_map(test_signal, 1)

    spans[0].left = 200
    spans[0].right = 1000

    if which_plot == "all_sums":
        return all_sums
    elif which_plot == "span_sums":
        return span_sums[0]
