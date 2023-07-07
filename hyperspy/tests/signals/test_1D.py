# -*- coding: utf-8 -*-
# Copyright 2007-2023 The HyperSpy developers
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

from hyperspy.components1d import Gaussian
from hyperspy.decorators import lazifyTestClass
from hyperspy.signals import Signal1D
from hyperspy.drawing.marker_collection import MarkerCollection
from matplotlib.collections import LineCollection


@lazifyTestClass
class Test1D:
    def setup_method(self, method):
        gaussian = Gaussian()
        gaussian.A.value = 20
        gaussian.sigma.value = 10
        gaussian.centre.value = 50
        self.signal = Signal1D(gaussian.function(np.arange(0, 100, 0.01)))
        self.signal.axes_manager[0].scale = 0.01

    @pytest.fixture
    def zero_d_navigate(self):
        return Signal1D(np.arange(0, 100, 0.01))

    @pytest.fixture
    def one_d_navigate(self):
        return Signal1D(np.repeat(np.arange(0, 100, 1)[np.newaxis,:], 3, axis=0))

    def test_integrate1D(self):
        integrated_signal = self.signal.integrate1D(axis=0)
        np.testing.assert_allclose(integrated_signal.data, 20, rtol=1e-6)

    @pytest.mark.parametrize("signal", ("zero_d_navigate", "one_d_navigate"))
    @pytest.mark.parametrize("start", (0.0, None))
    def test_get_line_intensity(self, request, signal, start):
        s = request.getfixturevalue(signal)
        intensities = s.get_line_intensity(indexes=[3, 5],
                                           start=start)
        if start is None:
            vector_shape = (2, 2)
        else:
            vector_shape = (2, 2, 2)
        if signal == "zero_d_navigate":
            assert intensities.data.dtype == float
            assert intensities.data.shape == vector_shape
        else:
            assert intensities.data.dtype == object
            assert intensities.data.shape == (3,)

        if start is None:
            markers = MarkerCollection.from_signal(intensities)
            s.add_marker(markers)
        else:
            markers = MarkerCollection.from_signal(intensities,
                                                   key="segments",
                                                   collection_class=LineCollection,
                                                   )
            s.add_marker(markers)


