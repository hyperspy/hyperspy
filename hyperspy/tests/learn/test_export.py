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

import numpy as np

from hyperspy import signals


class TestMVAExport:
    def setup_method(self, method):
        s = signals.Signal1D(np.random.random((2, 3, 4, 5)))
        sa = s.axes_manager[-1]
        na = s.axes_manager[0]
        sa.offset = 100
        sa.scale = 0.1
        s.learning_results.components = np.arange(5 * 5).reshape((5, 5))
        s.learning_results.scores = np.arange(24 * 5).reshape((24, 5))
        s.learning_results.bss_components = np.arange(5 * 2).reshape((5, 2))
        s.learning_results.bss_scores = np.arange(24 * 2).reshape((24, 2))
        self.s = s
        self.na = na
        self.sa = sa

    def test_get_bss_components(self):
        bss_components = self.s.get_bss_components()
        assert bss_components.axes_manager[-1].scale == self.sa.scale
        assert bss_components.axes_manager[-1].offset == self.sa.offset
        assert (
            bss_components.axes_manager.signal_shape == self.s.axes_manager.signal_shape
        )

    def test_get_bss_scores(self):
        bss_scores = self.s.get_bss_scores()
        assert bss_scores.axes_manager.navigation_dimension == 1
        assert (
            bss_scores.axes_manager.signal_shape == self.s.axes_manager.navigation_shape
        )
