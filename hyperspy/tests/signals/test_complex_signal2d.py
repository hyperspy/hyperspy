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


import numpy as np

import hyperspy.api as hs


def test_add_phase_ramp():
    s = hs.signals.ComplexSignal2D(
        np.exp(1j * (np.indices((3, 3)).sum(axis=0) + 4)))
    s.add_phase_ramp(-1, -1, -4)
    np.testing.assert_allclose(s.phase.data, np.zeros_like(s.phase.data),
                    atol=np.finfo(float).eps * 1.5)


def test_lazy_add_phase_ramp():
    s = hs.signals.ComplexSignal2D(
        np.exp(1j * (np.indices((3, 3)).sum(axis=0) + 4))).as_lazy()
    s.add_phase_ramp(-1, -1, -4)
    np.testing.assert_almost_equal(s.phase.data.compute(), 0)
