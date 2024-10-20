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
import traits.api as t

import hyperspy
from hyperspy.decorators import lazifyTestClass
from hyperspy.misc.array_tools import round_half_towards_zero
from hyperspy.roi import (
    CircleROI,
    Line2DROI,
    Point1DROI,
    Point2DROI,
    PolygonROI,
    RectangularROI,
    SpanROI,
    _get_central_half_limits_of_axis,
    combine_rois,
    mask_from_rois,
)
from hyperspy.signals import Signal1D, Signal2D


@lazifyTestClass
class TestROIs:
    def setup_method(self, method):
        np.random.seed(0)  # Same random every time, Line2DROi test requires it
        s_s = Signal1D(np.random.rand(50, 60, 4))
        s_s.axes_manager[0].scale = 5
        s_s.axes_manager[0].units = "nm"
        s_s.axes_manager[1].scale = 5
        s_s.axes_manager[1].units = "nm"

        # 4D dataset
        s_i = Signal2D(np.random.rand(100, 100, 4, 4))

        # Generate ROI for test of angle measurements
        r = []
        t = np.tan(30.0 / 180.0 * np.pi)
        for x in [-1.0, -t, t, 1]:
            for y in [-1.0, -t, t, 1]:
                r.append(Line2DROI(x1=0.0, x2=x, y1=0.0, y2=y))

        self.s_s = s_s
        self.s_i = s_i
        self.r = r

    def test_point1d_spectrum(self):
        s = self.s_s
        r = Point1DROI(35)
        sr = r(s)
        scale = s.axes_manager[0].scale
        assert sr.axes_manager.navigation_shape == s.axes_manager.navigation_shape[1:]
        if s._lazy:
            s.compute()
            sr.compute()
        np.testing.assert_equal(sr.data, s.data[:, int(35 / scale), ...])

    def test_point1d_spectrum_ronded_coord(self):
        s = self.s_s
        r = Point1DROI(37.0)
        sr = r(s)
        scale = s.axes_manager[0].scale
        if s._lazy:
            s.compute()
            sr.compute()
        np.testing.assert_equal(sr.data, s.data[:, int(round(37 / scale)), ...])
        r = Point1DROI(39.0)
        sr = r(s)
        np.testing.assert_equal(sr.data, s.data[:, int(round(39 / scale)), ...])

    def test_point1d_image(self):
        s = self.s_i
        r = Point1DROI(35)
        sr = r(s)
        scale = s.axes_manager[0].scale
        assert sr.axes_manager.navigation_shape == s.axes_manager.navigation_shape[1:]
        if s._lazy:
            s.compute()
            sr.compute()
        np.testing.assert_equal(sr.data, s.data[:, int(35 / scale), ...])

    def test_point1d_getitem(self):
        r = Point1DROI(35)
        assert (35,) == tuple(r)

    def test_point2d_image(self):
        s = self.s_i
        r = Point2DROI(35, 40)
        sr = r(s)
        scale = s.axes_manager[0].scale
        assert sr.axes_manager.navigation_shape == s.axes_manager.navigation_shape[2:]
        if s._lazy:
            s.compute()
            sr.compute()
        np.testing.assert_equal(sr.data, s.data[int(40 / scale), int(35 / scale), ...])

    def test_point2d_image_sig(self):
        s = self.s_i
        r = Point2DROI(1, 2)
        sr = r(s, axes=s.axes_manager.signal_axes)
        scale = s.axes_manager.signal_axes[0].scale
        assert sr.axes_manager.signal_shape == s.axes_manager.signal_shape[2:]
        if s._lazy:
            s.compute()
            sr.compute()
        np.testing.assert_equal(sr.data, s.data[..., int(2 / scale), int(1 / scale)])

    def test_point2d_getitem(self):
        r = Point2DROI(1, 2)
        assert tuple(r) == (1, 2)

    def test_span_roi_init(self):
        with pytest.raises(ValueError):
            SpanROI(30, 15)
        with pytest.raises(ValueError):
            SpanROI(15, 15)

    def test_span_spectrum_nav(self):
        s = self.s_s
        r = SpanROI(15, 30)
        sr = r(s)
        scale = s.axes_manager[0].scale
        n = (30 - 15) / scale
        assert (
            sr.axes_manager.navigation_shape
            == (n,) + s.axes_manager.navigation_shape[1:]
        )
        if s._lazy:
            s.compute()
            sr.compute()
        np.testing.assert_equal(
            sr.data, s.data[:, int(15 / scale) : int(30 // scale), ...]
        )

    def test_roi_add_widget(self):
        s = Signal1D(np.random.rand(60, 4))
        s.axes_manager[0].name = "nav axis"
        # Test adding roi to plot
        s.plot(navigator="spectrum")

        # Try using different argument types
        for axes in [0, s.axes_manager[0], "nav axis", [0], ["nav axis"]]:
            r = SpanROI(0, 60)
            r.add_widget(s, axes=axes)
            np.testing.assert_equal(r(s).data, s.data)

        # invalid arguments
        for axes in ["not a DataAxis name", ["not a DataAxis name"], [0, 1]]:
            r2 = SpanROI(0, 60)
            with pytest.raises(ValueError):
                r2.add_widget(s, axes=axes)

        for axes in [2, [2]]:
            r3 = SpanROI(0, 60)
            with pytest.raises(IndexError):
                r3.add_widget(s, axes=axes)

    def test_roi_add_widget_plot_missing(self):
        s = Signal1D(np.random.rand(60, 4))
        r = SpanROI(0, 60)
        with pytest.raises(RuntimeError):
            r.interactive(s)

    def test_span_spectrum_nav_boundary_roi(self):
        s = Signal1D(np.random.rand(60, 4))
        r = SpanROI(0, 60)
        # Test adding roi to plot
        s.plot(navigator="spectrum")
        r.add_widget(s)
        np.testing.assert_equal(r(s).data, s.data)

        s.axes_manager[0].scale = 0.2
        r2 = SpanROI(0, 12)
        # Test adding roi to plot
        s.plot(navigator="spectrum")
        w2 = r2.add_widget(s)
        np.testing.assert_equal(r2(s).data, s.data)

        w2.set_bounds(x=-10)  # below min x
        assert w2._pos[0] == -0.1
        w2.set_bounds(width=0.1)  # below min width
        assert w2._size[0] == 0.2
        w2.set_bounds(width=30.0)  # above max width
        assert w2._size[0] == 11.8

        w2.set_bounds(x=10, width=20)
        assert w2._pos[0] == 9.9
        np.testing.assert_allclose(w2._size[0], 1.8)

        w2.set_bounds(x=10)
        w2.set_bounds(width=20)
        assert w2._pos[0] == 9.9
        np.testing.assert_allclose(w2._size[0], 1.8)

    def test_spanroi_getitem(self):
        r = SpanROI(15, 30)
        assert tuple(r) == (15, 30)

    @pytest.mark.parametrize(
        "roi", [Point1DROI, Point2DROI, RectangularROI, SpanROI, Line2DROI, CircleROI]
    )
    @pytest.mark.parametrize("axes", [None, "signal"])
    def test_add_widget_ROI_undefined(self, axes, roi):
        s = self.s_i
        s.plot()
        r = roi()
        if axes == "signal":
            axes = s.axes_manager.signal_axes[: r.ndim]
        r.add_widget(s, axes=axes)
        if axes is None:
            expected_axes = s.axes_manager.navigation_axes
        else:
            expected_axes = axes
        r.signal_map[s][1][0] in expected_axes

    def test_span_spectrum_sig(self):
        s = self.s_s
        r = SpanROI(1, 3)
        sr = r(s, axes=s.axes_manager.signal_axes)
        scale = s.axes_manager.signal_axes[0].scale
        n = (3 - 1) / scale
        assert sr.axes_manager.signal_shape == (n,)
        if s._lazy:
            s.compute()
            sr.compute()
        np.testing.assert_equal(sr.data, s.data[..., int(1 / scale) : int(3 / scale)])

    def test_rect_image(self):
        s = self.s_i
        s.axes_manager[0].scale = 0.2
        s.axes_manager[1].scale = 0.8
        r = RectangularROI(left=2.3, top=5.6, right=3.5, bottom=12.2)
        sr = r(s)
        scale0 = s.axes_manager[0].scale
        scale1 = s.axes_manager[1].scale
        n = (
            (
                int(round_half_towards_zero(2.3 / scale0)),
                int(round_half_towards_zero(3.5 / scale0)),
            ),
            (
                int(round_half_towards_zero(5.6 / scale1)),
                int(round_half_towards_zero(12.2 / scale1)),
            ),
        )
        assert sr.axes_manager.navigation_shape == (
            n[0][1] - n[0][0],
            n[1][1] - n[1][0],
        )
        if s._lazy:
            s.compute()
            sr.compute()
        np.testing.assert_equal(
            sr.data, s.data[n[1][0] : n[1][1], n[0][0] : n[0][1], ...]
        )

    def test_rectroi_getitem(self):
        r = RectangularROI(left=2.3, top=5.6, right=3.5, bottom=12.2)
        assert tuple(r) == (2.3, 3.5, 5.6, 12.2)

    def test_rect_image_boundary_roi(self):
        s = self.s_i
        r = RectangularROI(0, 0, 100, 100)
        # Test adding roi to plot
        s.plot()
        w = r.add_widget(s)
        sr = r(s)
        if s._lazy:
            s.compute()
            sr.compute()
        np.testing.assert_equal(sr.data, s.data)

        # width and height should range between 1 and axes shape
        with pytest.raises(ValueError):
            w.width = 101
        with pytest.raises(ValueError):
            w.height = 101

        s.axes_manager[0].scale = 0.2
        s.axes_manager[1].scale = 0.8
        r2 = RectangularROI(0, 0, 20, 80)
        # Test adding roi to plot
        s.plot()
        w2 = r2.add_widget(s)
        np.testing.assert_equal(r2(s).data, s.data)

        w2.set_bounds(x=-10)  # below min x
        assert w2._pos[0] == 0
        w2.set_bounds(width=0.1)  # below min width
        assert w2._size[0] == 0.2
        w2.set_bounds(width=30.0)  # above max width
        assert w2._size[0] == 20

        w2.set_bounds(y=0)  # min y
        w2.set_bounds(height=0.7)  # below min height
        assert w2._size[1] == 0.8
        w2.set_bounds(height=90.0)  # about max height
        assert w2._size[1] == 80.0

        # by indices
        # width and height should range between 1 and axes shape
        with pytest.raises(ValueError):
            w2.width = 0
        with pytest.raises(ValueError):
            w2.height = 0
        with pytest.raises(ValueError):
            w2.width = 101
        with pytest.raises(ValueError):
            w2.height = 101

        # the combination of the two is not valid
        w2.set_bounds(x=10, width=20)
        assert w2._pos[0] == 0.0
        assert w2._size[0] == 20.0

        # the combination of the two is not valid
        w2.set_bounds(y=40, height=60)
        assert w2._pos[1] == 0
        assert w2._size[1] == 80

        w2.set_bounds(x=10)
        w2.set_bounds(width=20)
        assert w2._pos[0] == 0
        assert w2._size[0] == 20
        w2.set_bounds(y=10)
        w2.set_bounds(height=79.2)
        assert w2._pos[1] == 0.0
        assert w2._size[1] == 79.2

    def test_circle_spec(self):
        s = self.s_s
        s.data = np.ones_like(s.data)
        r = CircleROI(20, 25, 20)
        r_ann = CircleROI(20, 25, 20, 15)
        sr = r(s)
        sr_ann = r_ann(s)
        scale = s.axes_manager[0].scale
        n = int(round(40 / scale))
        assert sr.axes_manager.navigation_shape == (n, n)
        assert sr_ann.axes_manager.navigation_shape == (n, n)
        # Check that mask is same for all images:
        for i in range(n):
            for j in range(n):
                assert np.all(sr.data[j, i, :] == np.nan) or np.all(
                    sr.data[j, i, :] != np.nan
                )
                assert np.all(sr_ann.data[j, i, :] == np.nan) or np.all(
                    sr_ann.data[j, i, :] != np.nan
                )
        # Check that the correct elements has been masked out:
        mask = sr.data[:, :, 0]
        print(mask)  # To help debugging, this shows the shape of the mask
        np.testing.assert_array_equal(
            np.where(np.isnan(mask.flatten()))[0],
            [0, 1, 6, 7, 8, 15, 48, 55, 56, 57, 62, 63],
        )
        mask_ann = sr_ann.data[:, :, 0]
        print(mask_ann)  # To help debugging, this shows the shape of the mask
        np.testing.assert_array_equal(
            np.where(np.isnan(mask_ann.flatten()))[0],
            [
                0,
                1,
                6,
                7,
                8,
                10,
                11,
                12,
                13,
                15,
                17,
                18,
                19,
                20,
                21,
                22,
                25,
                26,
                27,
                28,
                29,
                30,
                33,
                34,
                35,
                36,
                37,
                38,
                41,
                42,
                43,
                44,
                45,
                46,
                48,
                50,
                51,
                52,
                53,
                55,
                56,
                57,
                62,
                63,
            ],
        )
        # Check that mask works for sum
        assert np.nansum(sr.data) == (n**2 - 3 * 4) * 4
        assert np.nansum(sr_ann.data) == 4 * 5 * 4

        s.plot()
        r_signal = r.interactive(signal=s)
        r_ann_signal = r_ann.interactive(signal=s)

        assert np.sum(r_signal.nansum().data) == (n**2 - 3 * 4) * 4
        assert np.sum(r_ann_signal.nansum().data) == 4 * 5 * 4

    def test_circle_getitem(self):
        r = CircleROI(20, 25, 20)
        assert tuple(r) == (20, 25, 20, 0)

    def test_annulus_getitem(self):
        r_ann = CircleROI(20, 25, 20, 15)
        assert tuple(r_ann) == (20, 25, 20, 15)

    def test_polygon_spec(self):
        s = self.s_s

        s.data = np.ones_like(s.data)
        r = PolygonROI(
            [(20, 5), (35, 15), (55, 0), (50, 50), (45, 45), (15, 40), (20, 35)]
        )
        sr = r(s)

        n_x = int(40 // s.axes_manager[0].scale) + 1
        n_y = int(50 // s.axes_manager[1].scale) + 1
        assert sr.axes_manager.navigation_shape == (n_x, n_y)
        # Check that mask is same for all images:
        for i in range(n_x):
            for j in range(n_y):
                assert np.all(np.isnan(sr.data[j, i, :])) or np.all(
                    ~np.isnan(sr.data[j, i, :])
                )

        # Check that the correct elements have been masked out:
        desired_mask = """
            X X X X X X X X O
            X O X X X X X O O
            X O O O X O O O O
            X O O O O O O O O
            X O O O O O O O O
            X O O O O O O O X
            X O O O O O O O X
            X O O O O O O O X
            O O O O O O O O X
            X X X X X X O O X
            X X X X X X X O X
        """.strip().replace(" ", "")
        desired_mask = [
            [c == "O" for c in line.strip()] for line in desired_mask.splitlines()
        ]
        desired_mask = np.array(desired_mask)

        mask = sr.data[:, :, 0]
        np.testing.assert_array_equal(~np.isnan(mask), desired_mask)

        # Test signal axes
        s_t = s.T
        sr = r(s_t)
        mask = sr.data[0, :, :]
        np.testing.assert_array_equal(~np.isnan(mask), desired_mask)

        # Test inverted mask

        sr = r(s, inverted=True)
        assert sr.axes_manager.navigation_shape == s.axes_manager.navigation_shape
        # Check that mask is same for all images:
        for i in range(sr.axes_manager.navigation_shape[0]):
            for j in range(sr.axes_manager.navigation_shape[1]):
                assert np.all(np.isnan(sr.data[j, i, :])) or np.all(
                    ~np.isnan(sr.data[j, i, :])
                )

        mask = sr.data[:, :, 0]
        # Pad desired_mask to same shape
        pad_left = int(15 // s.axes_manager[1].scale)
        pad_right = mask.shape[1] - desired_mask.shape[1] - pad_left
        pad_bottom = mask.shape[0] - desired_mask.shape[0]
        desired_mask = np.pad(
            desired_mask,
            ((0, pad_bottom), (pad_left, pad_right)),
            constant_values=False,
        )
        np.testing.assert_array_equal(np.isnan(mask), desired_mask)

        # Test square ROI. Useful for testing edge cases.
        r_square = PolygonROI([(10, 10), (20, 10), (20, 40), (10, 40)])
        sr_square = r_square(s, inverted=True)
        mask = sr_square.data[:, :, 0]
        desired_mask = np.ones_like(mask, dtype=bool)
        desired_mask[2:9, 2:5] = False
        np.testing.assert_array_equal(~np.isnan(mask), desired_mask)

        # Test empty ROI
        r_empty = PolygonROI()
        r_empty.vertices = []
        s_empty = r_empty(s)
        assert np.all(np.isnan(s_empty))
        assert s_empty.axes_manager.navigation_shape == s.axes_manager.navigation_shape

        s_empty_inv = r_empty(s, inverted=True)
        np.testing.assert_array_equal(s_empty_inv.data, s.data)

        # Test multiple polygons

        r1 = PolygonROI(
            vertices=[(0.61, 0.52), (0.5, 30.51), (2.99, 30.51), (2.87, 0.64)]
        )
        r2 = PolygonROI(
            vertices=[
                (5.95, 0.52),
                (5.71, 30.75),
                (17.92, 30.87),
                (17.92, 28.26),
                (9.27, 28.14),
                (9.15, 0.52),
            ]
        )
        r3 = PolygonROI(
            vertices=[
                (22.21, 29.45),
                (25.05, 29.45),
                (26.24, 21.15),
                (32.05, 21.38),
                (33.71, 29.33),
                (37.03, 29.45),
                (30.62, 0.16),
                (28.61, 0.04),
                (24.58, 13.09),
                (27.54, 13.2),
                (29.2, 7.75),
                (30.86, 15.22),
                (27.3, 15.22),
                (27.66, 13.2),
                (24.46, 13.2),
                (20.55, 29.68),
            ]
        )
        r4 = PolygonROI(
            vertices=[
                (39.99, 30.25),
                (39.99, 0.57),
                (47.07, 0.22),
                (54.49, 3.85),
                (55.01, 10.41),
                (51.04, 13.68),
                (45.0, 13.51),
                (45.0, 9.72),
                (48.79, 9.03),
                (50.35, 7.13),
                (50.17, 5.75),
                (48.62, 4.02),
                (44.82, 4.02),
                (45.0, 9.54),
                (45.34, 9.54),
                (45.0, 13.68),
                (55.18, 29.73),
                (49.31, 29.73),
                (44.82, 18.86),
                (45.17, 30.25),
            ]
        )
        r5 = PolygonROI(
            vertices=[(57.61, 0.52), (57.5, 30.51), (59.99, 30.51), (59.87, 0.64)]
        )
        r6 = PolygonROI(
            vertices=[
                (64.21, 29.45),
                (67.05, 29.45),
                (68.24, 21.15),
                (74.05, 21.38),
                (75.71, 29.33),
                (79.03, 29.45),
                (72.62, 0.16),
                (70.61, 0.04),
                (66.58, 13.09),
                (69.54, 13.2),
                (71.2, 7.75),
                (72.86, 15.22),
                (69.3, 15.22),
                (69.66, 13.2),
                (66.46, 13.2),
                (62.55, 29.68),
            ]
        )
        r7 = PolygonROI(
            vertices=[
                (17.73, 4.54),
                (13.59, 0.74),
                (10.83, 3.68),
                (11.69, 8.33),
                (17.39, 13.86),
                (23.25, 8.68),
                (24.63, 3.85),
                (22.56, 1.26),
            ]
        )

        s = hyperspy.signals.Signal2D(np.ones((33, 80)))

        all_polygons = [
            polygon._vertices
            for polygon in [r1, r2, r3, r4, r5, r6, r7]
            if polygon.is_valid()
        ]
        sr = r1._combine(s, additional_polygons=all_polygons)

        n_x = int(79 // s.axes_manager[0].scale) + 1
        n_y = int(31 // s.axes_manager[1].scale) + 1
        assert sr.axes_manager.signal_shape == (n_x, n_y)

        # Check more complex mask
        desired_mask = " ".join(
            "0000000000000000000000000000000000000000000000000000000000000000000000000000000001110011110001100000000000001111000000001111111111000000001110000000001111000000011100111100111100000111000011110000000011111111111100000011100000000011110000000111001111011111100011111000111100000000111111111111110000111000000000111100000001110011110111111101111111011111000000001111111111111111001110000000011111000000011100111101111111111111100111111000000011111100011111110011100000000111111000000111001111011111111111111001111110000000111111000011111100111000000001111110000001110011110111111111111110111111100000001111110000111111001110000000111111100000011100111100111111111111001111111000000011111100001111110011100000001111111000000111001111001111111111110011111111000000111111000111111100111000000011111111000001110011110001111111111000111111110000001111111111111111001110000000111111110000011100111100001111111100011110111100000011111111111111100011100000011110111100000111001111000001111100000111101111000000111111111111110000111000000111101111000001110011110000000110000001111011110000001111111111111000001110000001111011110000011100111100000000000000111110011110000011111100000000000011100000111110011110000111001111000000000000001111000111100000111111100000000000111000001111000111100001110011110000000000000011111111111000001111111000000000001110000011111111111000011100111100000000000000111111111110000011111111000000000011100000111111111110000111001111000000000000011111111111110000111111111000000000111000011111111111110001110011110000000000000111111111111100001111111110000000001110000111111111111100011100111100000000000001111111111111000011111111110000000011100001111111111111000111001111000000000000011111111111110000111111111110000000111000011111111111110001110011110000000000001111100000111100001111111111100000001110001111100000111100011100111100000000000011111000001111100011111101111100000011100011111000001111100111001111000000000000111110000001111000111111011111100000111000111110000001111001110011110000000000001111100000011110001111110111111000001110001111100000011110011100111100000000000111111000000111100011111100111111000011100111111000000111100111001111000000000001111100000001111000111111001111110000111001111100000001111001110011110000000000011111000000011111001111110001111110001110011111000000011111011100111111111111100111110000000011110011111100011111110011100111110000000011110111001111111111111000000000000000000000111111000000000000111000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
        )
        desired_mask = np.fromstring(desired_mask, "int8", sep=" ")
        desired_mask = desired_mask.astype(bool).reshape((n_y, n_x))

        np.testing.assert_array_equal(~np.isnan(sr.data), desired_mask)

        # Test combined boolean mask
        sigaxes = s.axes_manager.signal_axes
        combined_mask = r1._boolean_mask(
            x_scale=sigaxes[0].scale,
            y_scale=sigaxes[1].scale,
            additional_polygons=all_polygons[1:],
        )
        np.testing.assert_array_equal(combined_mask, desired_mask)

    def test_set_polygon_errors(self):
        r = PolygonROI()
        r.vertices = []
        r.vertices = [
            (20, 5),
            (35, 15),
            (55, 0),
            (50, 50),
            (45, 45),
            (15, 40),
            (20, 35),
        ]

        with pytest.raises(ValueError):
            r.vertices = [(10, 2)]
        with pytest.raises(ValueError):
            r.vertices = [
                (20, 5),
                (35, 15),
                (55, 0),
                (50, 50),
                (45, 45),
                (15, 40),
                (20, None),
            ]
        with pytest.raises(ValueError):
            r.vertices = [
                (20, 5),
                (35, 15),
                (55, 0),
                (50, 50),
                (45, 45),
                (15, 40),
                (20, 20, 4),
            ]

    def test_polygon_getitem(self):
        r = PolygonROI([(2, 5), (5, 6), (5, 3)])
        assert tuple(r) == ((2, 5), (5, 6), (5, 3))

    def test_2d_line_getitem(self):
        r = Line2DROI(10, 10, 150, 50, 5)
        assert tuple(r) == (10, 10, 150, 50, 5)

    def test_2d_line_spec_plot(self):
        r = Line2DROI(10, 10, 150, 50, 5)
        s = self.s_s
        s2 = r(s)
        np.testing.assert_allclose(
            s2.data,
            np.array(
                [
                    [0.96779467, 0.5468849, 0.27482357, 0.59223042],
                    [0.89676116, 0.40673335, 0.55207828, 0.27165277],
                    [0.27734027, 0.52437981, 0.11738029, 0.15984529],
                    [0.04680635, 0.97073144, 0.00386035, 0.17857997],
                    [0.61286675, 0.0813696, 0.8818965, 0.71962016],
                    [0.96638997, 0.50763555, 0.30040368, 0.54950057],
                    [0.22956744, 0.50686296, 0.73685316, 0.09767637],
                    [0.5149222, 0.93841202, 0.22864655, 0.67714114],
                    [0.5149222, 0.93841202, 0.22864655, 0.67714114],
                    [0.59288027, 0.0100637, 0.4758262, 0.70877039],
                    [0.80546244, 0.58610794, 0.56928692, 0.51208072],
                    [0.97176308, 0.36384478, 0.78791575, 0.55529411],
                    [0.39563367, 0.95546593, 0.59831597, 0.11891694],
                    [0.4175392, 0.78158173, 0.69374702, 0.91634033],
                    [0.44679332, 0.83699037, 0.22182403, 0.49394526],
                    [0.92961874, 0.66721471, 0.79807902, 0.55099397],
                    [0.98046646, 0.58866215, 0.04551071, 0.1979828],
                    [0.70340703, 0.35307496, 0.15442542, 0.31268984],
                    [0.88432423, 0.95853234, 0.20751273, 0.78846839],
                    [0.27334874, 0.88713154, 0.16554561, 0.66595992],
                    [0.08421126, 0.97389332, 0.70063334, 0.84181574],
                    [0.15946909, 0.41702974, 0.42681952, 0.26810926],
                    [0.13159685, 0.03921054, 0.02523183, 0.27155029],
                    [0.13159685, 0.03921054, 0.02523183, 0.27155029],
                    [0.46185344, 0.72624328, 0.4748717, 0.90405082],
                    [0.52917427, 0.54280647, 0.71405379, 0.51655594],
                    [0.13307599, 0.77345467, 0.4062725, 0.96309389],
                    [0.28351378, 0.26307878, 0.3335074, 0.57231702],
                    [0.89486974, 0.17628164, 0.2796788, 0.58167984],
                    [0.64937273, 0.5006921, 0.28355772, 0.2861476],
                    [0.31342052, 0.19085, 0.90192363, 0.85839813],
                ]
            ),
            rtol=0.05,
        )
        r.linewidth = 50
        s3 = r(s)
        np.testing.assert_allclose(
            s3.data,
            np.array(
                [
                    [0.40999384, 0.27111487, 0.3345655, 0.47553854],
                    [0.44475117, 0.40330205, 0.48113292, 0.26780132],
                    [0.57911599, 0.38999298, 0.38509116, 0.37418655],
                    [0.29175157, 0.37856367, 0.34420691, 0.48316543],
                    [0.55975912, 0.57155145, 0.57640677, 0.39718605],
                    [0.41300845, 0.45929259, 0.27489573, 0.40120352],
                    [0.46271229, 0.60908378, 0.25796662, 0.46526239],
                    [0.37843991, 0.54919334, 0.40469436, 0.48612034],
                    [0.44717148, 0.44934708, 0.29064827, 0.51334849],
                    [0.3966089, 0.59853786, 0.50392157, 0.39123649],
                    [0.50281456, 0.62863149, 0.43051921, 0.32015553],
                    [0.40527468, 0.44258442, 0.55694228, 0.41142292],
                    [0.47856163, 0.49720026, 0.62012372, 0.47537808],
                    [0.46695064, 0.5159018, 0.53532036, 0.4691573],
                    [0.44267241, 0.46886762, 0.37363574, 0.54369291],
                    [0.76138395, 0.54406653, 0.47305104, 0.45083095],
                    [0.74812744, 0.53414434, 0.38487816, 0.44611049],
                    [0.59011489, 0.5456799, 0.41782293, 0.5948403],
                    [0.47546595, 0.52536805, 0.39267032, 0.58787463],
                    [0.39387115, 0.4784124, 0.36765754, 0.46951847],
                    [0.54076839, 0.69257203, 0.44540576, 0.39236971],
                    [0.41195904, 0.5148879, 0.51199686, 0.63694563],
                    [0.44885787, 0.46886977, 0.42150512, 0.52556669],
                    [0.60826081, 0.3987657, 0.55875628, 0.5293137],
                    [0.44151911, 0.4188617, 0.37734811, 0.51166705],
                    [0.52878209, 0.41050467, 0.57149806, 0.52577575],
                    [0.50474464, 0.3294767, 0.63519013, 0.56126315],
                    [0.37607782, 0.58086952, 0.45089019, 0.62929377],
                    [0.59956085, 0.5173887, 0.64790597, 0.49865165],
                    [0.57646846, 0.46468029, 0.45267259, 0.44889072],
                    [0.4382186, 0.49576157, 0.6192481, 0.45031413],
                ]
            ),
        )

    def test_2d_line_img_plot(self):
        s = self.s_i
        r = Line2DROI(0, 0, 4, 4, 1)
        s2 = r(s)
        np.testing.assert_allclose(
            s2.data,
            np.array(
                [
                    [
                        [0.5646904, 0.83974605, 0.37688365, 0.499676],
                        [0.08130241, 0.3241552, 0.91565131, 0.85345237],
                        [0.5941565, 0.90536555, 0.42692772, 0.93761072],
                        [0.9458708, 0.56996783, 0.05020319, 0.88466194],
                    ],
                    [
                        [0.55342858, 0.71776076, 0.9698018, 0.84684608],
                        [0.77676046, 0.32998726, 0.49284904, 0.63849364],
                        [0.94969472, 0.99393561, 0.79184028, 0.60493951],
                        [0.99584095, 0.83632682, 0.51592399, 0.53049253],
                    ],
                    [
                        [0.55342858, 0.71776076, 0.9698018, 0.84684608],
                        [0.77676046, 0.32998726, 0.49284904, 0.63849364],
                        [0.94969472, 0.99393561, 0.79184028, 0.60493951],
                        [0.99584095, 0.83632682, 0.51592399, 0.53049253],
                    ],
                    [
                        [0.32270396, 0.28878038, 0.64165074, 0.92820531],
                        [0.24836647, 0.37477366, 0.18406007, 0.11019336],
                        [0.38678734, 0.9174347, 0.47658793, 0.45095935],
                        [0.95232706, 0.96468026, 0.5158903, 0.69112322],
                    ],
                    [
                        [0.72414297, 0.64417135, 0.17938658, 0.12279276],
                        [0.90632348, 0.90345183, 0.21473533, 0.34087282],
                        [0.2579504, 0.65663038, 0.27606922, 0.33695786],
                        [0.46466925, 0.34991125, 0.73593611, 0.32203574],
                    ],
                    [
                        [0.72414297, 0.64417135, 0.17938658, 0.12279276],
                        [0.90632348, 0.90345183, 0.21473533, 0.34087282],
                        [0.2579504, 0.65663038, 0.27606922, 0.33695786],
                        [0.46466925, 0.34991125, 0.73593611, 0.32203574],
                    ],
                    [
                        [0.97259866, 0.13527587, 0.48531393, 0.31607768],
                        [0.13656701, 0.40578067, 0.64221493, 0.46036815],
                        [0.30466093, 0.88706533, 0.30914269, 0.01833664],
                        [0.56143007, 0.09026307, 0.81898535, 0.4518825],
                    ],
                ]
            ),
        )
        r.linewidth = 10
        s3 = r(s)
        np.testing.assert_allclose(
            s3.data,
            np.array(
                [
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.12385935, 0.17534623, 0.08266437, 0.08533342],
                        [0.06072978, 0.18213069, 0.13162582, 0.14526251],
                        [0.11950599, 0.09530544, 0.05814531, 0.10613925],
                        [0.13243216, 0.13388253, 0.15641767, 0.07678893],
                    ],
                    [
                        [0.10387718, 0.18591981, 0.21704829, 0.16594489],
                        [0.26554947, 0.27280648, 0.23534874, 0.15751378],
                        [0.11329239, 0.16440693, 0.19378236, 0.23418843],
                        [0.20414672, 0.24669051, 0.08809065, 0.21252996],
                    ],
                    [
                        [0.32737802, 0.24354627, 0.25713232, 0.42447693],
                        [0.22132115, 0.34440789, 0.1769873, 0.18348862],
                        [0.32205928, 0.29038094, 0.22570116, 0.20305065],
                        [0.45399669, 0.29687212, 0.313637, 0.27469796],
                    ],
                    [
                        [0.38104394, 0.2654458, 0.51666151, 0.47973295],
                        [0.34333797, 0.36907303, 0.34349318, 0.25681538],
                        [0.32849871, 0.27963978, 0.47319042, 0.37358476],
                        [0.48767599, 0.23022751, 0.32004745, 0.37714935],
                    ],
                    [
                        [0.59093609, 0.54976286, 0.54934114, 0.54753303],
                        [0.48284716, 0.35797562, 0.49739056, 0.46934957],
                        [0.29954848, 0.45448276, 0.50639968, 0.56140708],
                        [0.55790493, 0.55105139, 0.40859302, 0.47408336],
                    ],
                    [
                        [0.63293155, 0.38872956, 0.55044015, 0.37731745],
                        [0.49091568, 0.54173188, 0.51292652, 0.53813843],
                        [0.56463766, 0.73848284, 0.41183566, 0.37515417],
                        [0.48426503, 0.23582684, 0.45947953, 0.49322732],
                    ],
                ]
            ),
        )

    def test_line2droi_angle(self):
        # 1. Testing quantitative measurement for different quadrants:
        r = self.r
        r_angles = np.array([rr.angle() for rr in r])
        angles_h = np.array(
            [
                -135.0,
                -150.0,
                150.0,
                135.0,
                -120.0,
                -135.0,
                135.0,
                120.0,
                -60.0,
                -45.0,
                45.0,
                60.0,
                -45.0,
                -30,
                30.0,
                45.0,
            ]
        )
        angles_v = np.array(
            [
                -135.0,
                -120.0,
                -60.0,
                -45.0,
                -150.0,
                -135.0,
                -45.0,
                -30.0,
                150.0,
                135.0,
                45.0,
                30.0,
                135.0,
                120.0,
                60.0,
                45.0,
            ]
        )
        np.testing.assert_allclose(r_angles, angles_h)
        r_angles = np.array([rr.angle(axis="vertical") for rr in r])
        np.testing.assert_allclose(r_angles, angles_v)

        # 2. Testing unit conversation
        r = Line2DROI(
            np.random.rand(), np.random.rand(), np.random.rand(), np.random.rand()
        )
        assert r.angle(units="degrees") == (r.angle(units="radians") / np.pi * 180.0)

        # 3. Testing raises:
        with pytest.raises(ValueError):
            r.angle(units="meters")
        with pytest.raises(ValueError):
            r.angle(axis="z")

    def test_repr_None(self):
        # Setting the args=None sets them as traits.Undefined, which didn't
        # have a string representation in the old %s style.
        for roi in [Point1DROI, Point2DROI, RectangularROI, SpanROI]:
            r = roi()
            for value in tuple(r):
                assert value == t.Undefined
            repr(r)
        for roi in [CircleROI, Line2DROI]:
            r = roi()
            for value in tuple(r)[:-1]:
                assert value == t.Undefined
            assert tuple(r)[-1] == 0
            repr(r)
        for roi in [PolygonROI]:
            r = roi()
            assert tuple(r) == tuple()
            repr(r)

    def test_repr_vals(self):
        repr(Point1DROI(1.1))
        repr(Point2DROI(1.1, 2.1))
        repr(Line2DROI(0, 0, 1, 1, 0.1))
        repr(RectangularROI(0, 0, 1, 1))
        repr(SpanROI(3.0, 5.0))
        repr(CircleROI(5, 5, 3))
        repr(CircleROI(5, 5, 3, 1))
        repr(PolygonROI([(0, 0), (0, 6.0), (2.1, 3)]))
        repr(PolygonROI([(0, 0), (0, 6.0), (2.1, 3), (1, -1)]))

    def test_undefined_call(self):
        for roi in [
            Point1DROI,
            Point2DROI,
            RectangularROI,
            SpanROI,
            Line2DROI,
            CircleROI,
        ]:
            r = roi()
            with pytest.raises(ValueError, match="not yet been set"):
                r(self.s_s)

    def test_default_values_call(self):
        for roi in [
            Point1DROI,
            Point2DROI,
            RectangularROI,
            SpanROI,
            Line2DROI,
            CircleROI,
        ]:
            r = roi()
            r._set_default_values(self.s_s)
            r(self.s_s)

    def test_default_values_call_specify_signal_axes(self):
        s = self.s_i
        for roi in [
            Point1DROI,
            Point2DROI,
            RectangularROI,
            SpanROI,
            Line2DROI,
            CircleROI,
        ]:
            r = roi()
            r._set_default_values(s, axes=s.axes_manager.signal_axes)
            r(s)

    @pytest.mark.parametrize("axes", ("sig", (2, 3)))
    def test_call_signal_axes(self, axes):
        s = self.s_i
        r = RectangularROI(1, 1, 3, 3)
        s_roi = r(s, axes=axes)
        assert s_roi.data.shape == (100, 100, 2, 2)

    @pytest.mark.parametrize("axes", ("nav", (0, 1)))
    def test_call_navigation_axes(self, axes):
        s = self.s_i
        r = RectangularROI(25, 25, 75, 75)
        s_roi = r(s, axes=axes)
        assert s_roi.data.shape == (50, 50, 4, 4)

    def test_get_central_half_limits(self):
        ax = self.s_s.axes_manager[0]
        assert _get_central_half_limits_of_axis(ax) == (73.75, 221.25)

    def test_line2droi_length(self):
        line = Line2DROI(x1=0.0, x2=2, y1=0.0, y2=2)
        np.testing.assert_allclose(line.length, np.sqrt(8))

    def test_combined_rois_polygon(self):
        # Test only for `PolygonROI`

        s = self.s_s

        s.data = np.ones_like(s.data)

        r1 = PolygonROI(
            vertices=[(0.61, 0.52), (0.5, 30.51), (2.99, 30.51), (2.87, 0.64)]
        )
        r2 = PolygonROI(
            vertices=[
                (5.95, 0.52),
                (5.71, 30.75),
                (17.92, 30.87),
                (17.92, 28.26),
                (9.27, 28.14),
                (9.15, 0.52),
            ]
        )
        r3 = PolygonROI(
            vertices=[
                (22.21, 29.45),
                (25.05, 29.45),
                (26.24, 21.15),
                (32.05, 21.38),
                (33.71, 29.33),
                (37.03, 29.45),
                (30.62, 0.16),
                (28.61, 0.04),
                (24.58, 13.09),
                (27.54, 13.2),
                (29.2, 7.75),
                (30.86, 15.22),
                (27.3, 15.22),
                (27.66, 13.2),
                (24.46, 13.2),
                (20.55, 29.68),
            ]
        )
        r4 = PolygonROI(
            vertices=[
                (39.99, 30.25),
                (39.99, 0.57),
                (47.07, 0.22),
                (54.49, 3.85),
                (55.01, 10.41),
                (51.04, 13.68),
                (45.0, 13.51),
                (45.0, 9.72),
                (48.79, 9.03),
                (50.35, 7.13),
                (50.17, 5.75),
                (48.62, 4.02),
                (44.82, 4.02),
                (45.0, 9.54),
                (45.34, 9.54),
                (45.0, 13.68),
                (55.18, 29.73),
                (49.31, 29.73),
                (44.82, 18.86),
                (45.17, 30.25),
            ]
        )
        r5 = PolygonROI(
            vertices=[(57.61, 0.52), (57.5, 30.51), (59.99, 30.51), (59.87, 0.64)]
        )
        r6 = PolygonROI(
            vertices=[
                (64.21, 29.45),
                (67.05, 29.45),
                (68.24, 21.15),
                (74.05, 21.38),
                (75.71, 29.33),
                (79.03, 29.45),
                (72.62, 0.16),
                (70.61, 0.04),
                (66.58, 13.09),
                (69.54, 13.2),
                (71.2, 7.75),
                (72.86, 15.22),
                (69.3, 15.22),
                (69.66, 13.2),
                (66.46, 13.2),
                (62.55, 29.68),
            ]
        )
        r7 = PolygonROI(
            vertices=[
                (17.73, 4.54),
                (13.59, 0.74),
                (10.83, 3.68),
                (11.69, 8.33),
                (17.39, 13.86),
                (23.25, 8.68),
                (24.63, 3.85),
                (22.56, 1.26),
            ]
        )

        rois = [r1, r2, r3, r4, r5, r6, r7]
        other_polygons = [
            polygon._vertices for polygon in rois[1:] if polygon.is_valid()
        ]

        combined_slice = combine_rois(s, rois=rois)

        # Check same result as method in `PolygonROI`
        np.testing.assert_array_equal(
            combined_slice, rois[0]._combine(s, additional_polygons=other_polygons)
        )

        combined_mask = mask_from_rois(rois=rois, axes_manager=s.axes_manager)
        desired_mask = rois[0]._boolean_mask(
            axes_manager=s.axes_manager, additional_polygons=other_polygons
        )

        np.testing.assert_array_equal(combined_mask, desired_mask)


@lazifyTestClass
class TestInteractive:
    def setup_method(self, method):
        s = Signal1D(np.arange(2000).reshape((20, 10, 10)))
        self.s = s

    def test_out(self):
        s = self.s
        r = RectangularROI(left=3, right=7, top=2, bottom=5)
        sr = r(s)
        sr_ref = r(s)

        sr.data = sr.data + 2

        sr2 = r(s)
        r(s, out=sr)

        if s._lazy:
            s.compute()
            sr.compute()
            sr2.compute()
            sr_ref.compute()

        np.testing.assert_array_equal(sr2.data, sr.data)
        np.testing.assert_array_equal(sr2.data, sr_ref.data)

    def test_out_special_case(self):
        s = self.s.inav[0]
        r = CircleROI(3, 5, 2)
        sr = r(s)
        np.testing.assert_array_equal(
            np.where(np.isnan(sr.data.flatten()))[0], [0, 3, 12, 15]
        )
        r.r_inner = 1
        r.cy = 16
        sr2 = r(s)
        r(s, out=sr)
        np.testing.assert_array_equal(
            np.where(np.isnan(sr.data.flatten()))[0], [0, 3, 5, 6, 9, 10, 12, 15]
        )
        np.testing.assert_array_equal(sr2.data, sr.data)

    def test_interactive_special_case(self):
        s = self.s.inav[0]
        r = CircleROI(3, 5, 2)
        sr = r.interactive(s, None, color="blue")
        np.testing.assert_array_equal(
            np.where(np.isnan(sr.data.flatten()))[0], [0, 3, 12, 15]
        )
        r.r_inner = 1
        r.cy = 16
        sr2 = r(s)
        np.testing.assert_array_equal(
            np.where(np.isnan(sr.data.flatten()))[0], [0, 3, 5, 6, 9, 10, 12, 15]
        )
        np.testing.assert_array_equal(sr2.data, sr.data)

    def test_interactive(self):
        s = self.s
        r = RectangularROI(left=3, right=7, top=2, bottom=5)
        sr = r.interactive(s, None)
        r.x += 5
        sr2 = r(s)
        np.testing.assert_array_equal(sr.data, sr2.data)

    def test_interactive_default_values(self):
        rois = [Point1DROI, Point2DROI, RectangularROI, SpanROI, Line2DROI, CircleROI]
        values = [
            (4.5,),
            (4.5, 9.5),
            (2.25, 6.75, 4.75, 14.25),
            (2.25, 6.75),
            (2.25, 4.75, 6.75, 14.25, 0.0),
            (4.5, 9.5, 4.5, 0.0),
        ]
        self.s.plot()
        for roi, vals in zip(rois, values):
            r = roi()
            r.interactive(signal=self.s)
            assert tuple(r) == vals

    @pytest.mark.parametrize("snap", [True, False, "default"])
    def test_interactive_snap(self, snap):
        kwargs = {}
        if snap != "default":
            kwargs["snap"] = snap
        else:
            # default is True
            snap = True
        s = self.s
        r = RectangularROI(left=3, right=7, top=2, bottom=5)
        s.plot()
        _ = r.interactive(s, **kwargs)
        for w in r.widgets:
            old_position = w.position
            new_position = (3.25, 2.2)
            w.position = new_position
            assert w.position == old_position if snap else new_position
            assert w.snap_all == snap
            assert w.snap_position == snap
            assert w.snap_size == snap

        p1 = Point1DROI(4)
        _ = p1.interactive(s, **kwargs)
        for w in p1.widgets:
            old_position = w.position
            new_position = (4.2,)
            w.position = new_position
            assert w.position == old_position if snap else new_position
            assert w.snap_position == snap

        p2 = Point2DROI(4, 5)
        _ = p2.interactive(s, **kwargs)
        for w in p2.widgets:
            old_position = w.position
            new_position = (4.3, 5.3)
            w.position = new_position
            assert w.position == old_position if snap else new_position
            assert w.snap_position == snap

        span = SpanROI(4.01, 5)
        _ = span.interactive(s, **kwargs)
        for w in span.widgets:
            old_position = w.position
            new_position = (4.2,)
            w.position = new_position
            assert w.position == old_position if snap else new_position
            assert w.snap_all == snap
            assert w.snap_position == snap
            assert w.snap_size == snap

            # check that changing snap is working fine
            new_snap = not snap
            w.snap_all = new_snap
            old_position = w.position
            new_position = (4.2,)
            w.position = new_position
            assert w.position == old_position if new_snap else new_position

        line2d = Line2DROI(4, 5, 6, 6, 1)
        _ = line2d.interactive(s, **kwargs)
        for w in line2d.widgets:
            old_position = w.position
            new_position = ([4.3, 5.3], [6.0, 6.0])
            w.position = new_position
            assert w.position == old_position if snap else new_position
            assert w.snap_all == snap
            assert w.snap_position == snap
            assert w.snap_size == snap


@pytest.mark.parametrize(
    "axis_class", ["DataAxis", "FunctionalDataAxis", "UniformDataAxis"]
)
def test_roi_non_uniform_axes(axis_class):
    nav_length = 10
    sig_length = 20

    nav_axes = [hyperspy.axes.UniformDataAxis(size=nav_length)]
    if axis_class == "UniformDataAxis":
        sig_axes = [hyperspy.axes.UniformDataAxis(size=sig_length)]
    elif axis_class == "FunctionalDataAxis":
        sig_axes = [
            hyperspy.axes.FunctionalDataAxis(
                size=sig_length,
                expression="x",
            )
        ]
    else:
        sig_axes = [hyperspy.axes.DataAxis(axis=np.arange(sig_length))]
    s = Signal1D(
        np.arange(nav_length * nav_length * sig_length).reshape(
            nav_length, nav_length, sig_length
        ),
        axes=2 * nav_axes + sig_axes,
    )

    roi = RectangularROI()
    s.plot()
    s_roi = roi.interactive(s, recompute_out_event=None)
    assert roi.x == 2.25
    assert s_roi.data.sum() == 899500

    # change position to trigger events
    roi.x = 4
    assert roi.x == 4
    assert s_roi.data.sum() == 959600
