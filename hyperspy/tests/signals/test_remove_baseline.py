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


def test_remove_baseline():
    import hyperspy.api as hs

    s = hs.data.two_gaussians().inav[:5, :5]

    assert s.data.mean() > 100
    s2 = s.remove_baseline(algorithm="aspls", lam=1e7, inplace=False)
    assert s.data.mean() > 100
    assert s2.data.mean() < 10

    s.remove_baseline(algorithm="aspls", lam=1e7)
    assert s.data.mean() < 10
