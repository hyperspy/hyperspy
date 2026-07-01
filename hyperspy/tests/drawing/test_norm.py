# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
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

"""Unit tests for the backend-agnostic norm descriptors in hyperspy.drawing.norm."""

from hyperspy.drawing.norm import LinearNorm, LogNorm, PowerNorm, SymLogNorm


def test_linear_norm_stores_parameters():
    norm = LinearNorm(vmin=1.0, vmax=2.0, clip=True)
    assert norm.vmin == 1.0
    assert norm.vmax == 2.0
    assert norm.clip is True


def test_linear_norm_defaults():
    norm = LinearNorm()
    assert norm.vmin is None
    assert norm.vmax is None
    assert norm.clip is False


def test_hypernorm_repr_lists_public_attributes():
    norm = LinearNorm(vmin=1.0, vmax=2.0)
    r = repr(norm)
    assert r.startswith("LinearNorm(")
    assert "vmin=1.0" in r
    assert "vmax=2.0" in r
    assert "clip=False" in r


def test_hypernorm_eq_same_type_and_params():
    assert LinearNorm(vmin=1.0, vmax=2.0) == LinearNorm(vmin=1.0, vmax=2.0)
    assert LinearNorm(vmin=1.0) != LinearNorm(vmin=2.0)


def test_hypernorm_eq_different_type_not_equal():
    assert LinearNorm(vmin=1.0, vmax=2.0) != LogNorm(vmin=1.0, vmax=2.0)
    assert LinearNorm() != object()


def test_log_norm_stores_parameters():
    norm = LogNorm(vmin=0.1, vmax=10.0, clip=True)
    assert norm.vmin == 0.1
    assert norm.vmax == 10.0
    assert norm.clip is True


def test_power_norm_stores_parameters():
    norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=1.0)
    assert norm.gamma == 0.5
    assert norm.vmin == 0.0
    assert norm.vmax == 1.0
    assert norm.clip is False


def test_symlog_norm_stores_parameters():
    norm = SymLogNorm(linthresh=0.1, linscale=2.0, vmin=-10.0, vmax=10.0, base=2.0)
    assert norm.linthresh == 0.1
    assert norm.linscale == 2.0
    assert norm.vmin == -10.0
    assert norm.vmax == 10.0
    assert norm.base == 2.0
