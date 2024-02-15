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

import math

import numpy as np
from pytest import approx

from hyperspy.components2d import Gaussian2D

sigma2fwhm = 2 * np.sqrt(2 * np.log(2))


def test_function():
    g = Gaussian2D()
    g.A.value = 14
    g.sigma_x.value = 1.0
    g.sigma_y.value = 2.0
    g.centre_x.value = -5.0
    g.centre_y.value = -5.0
    np.testing.assert_allclose(g.function(-5, -5), 1.1140846)
    np.testing.assert_allclose(g.function(-2, -3), 0.007506643)
    assert g._is2D
    assert g._position_x == g.centre_x
    assert g._position_y == g.centre_y


def test_util_fwhm_set():
    g1 = Gaussian2D()
    g1.fwhm_x = 0.33
    g1.fwhm_y = 0.33
    g1.A.value = 1.0
    np.testing.assert_allclose(g1.fwhm_x, g1.sigma_x.value * sigma2fwhm)
    np.testing.assert_allclose(g1.fwhm_y, g1.sigma_y.value * sigma2fwhm)


def test_util_fwhm_get():
    g1 = Gaussian2D(sigma_x=0.33, sigma_y=0.33)
    g1.A.value = 1.0
    np.testing.assert_allclose(g1.fwhm_x, g1.sigma_x.value * sigma2fwhm)
    np.testing.assert_allclose(g1.fwhm_y, g1.sigma_y.value * sigma2fwhm)


def test_util_fwhm_getset():
    g1 = Gaussian2D()
    g1.fwhm_x = 0.33
    g1.fwhm_y = 0.33
    assert g1.fwhm_x == 0.33
    assert g1.fwhm_y == 0.33


def test_height_value():
    g = Gaussian2D()
    g.sigma_x.value = 0.1
    g.sigma_y.value = 0.5
    g.A.value = 99
    x = np.arange(-2, 2, 0.01)
    y = np.arange(-2, 2, 0.01)
    xx, yy = np.meshgrid(x, y)
    g_image = g.function(xx, yy)
    assert approx(g_image.max()) == g.height


def test_util_height_set():
    g = Gaussian2D()
    g.height = 0.33
    np.testing.assert_allclose(
        g.height, g.A.value / (2 * np.pi * g.sigma_x.value * g.sigma_y.value)
    )


def test_util_height_get():
    g = Gaussian2D(A=55)
    np.testing.assert_allclose(
        g.height, g.A.value / (2 * np.pi * g.sigma_x.value * g.sigma_y.value)
    )


def test_util_height_getset():
    g = Gaussian2D()
    g.height = 0.165
    assert g.height == 0.165


def test_properties():
    g = Gaussian2D(add_rotation=True)
    angle = np.radians(20)
    g.rotation_angle.value = angle
    np.testing.assert_allclose(g.rotation_angle_wrapped, angle)

    angle = np.radians(380)
    g.rotation_angle.value = angle
    np.testing.assert_allclose(g.rotation_angle_wrapped, math.fmod(angle, 2 * np.pi))

    g = Gaussian2D(add_rotation=True)
    g.sigma_x.value = 0.5
    g.sigma_y.value = 0.1
    assert g.ellipticity == 5.0
    assert g.rotation_angle.value == 0
    assert g.sigma_major == 0.5
    assert g.sigma_minor == 0.1
    angle = np.radians(20)
    g.rotation_angle.value = angle
    np.testing.assert_allclose(g.rotation_angle_wrapped, angle)
    np.testing.assert_allclose(g.rotation_major_axis, angle)

    g = Gaussian2D(add_rotation=True)
    g.sigma_x.value = 0.1
    g.sigma_y.value = 0.5
    assert g.ellipticity == 5.0
    assert g.rotation_angle.value == 0
    assert g.sigma_major == 0.5
    assert g.sigma_minor == 0.1
    angle = np.radians(20)
    g.rotation_angle.value = angle
    np.testing.assert_allclose(g.rotation_angle_wrapped, angle)
    np.testing.assert_allclose(g.rotation_major_axis, angle - np.pi / 2)
