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
import pytest

import numpy as np
from matplotlib.patches import Ellipse, Rectangle
import matplotlib.pyplot as plt
import dask.array as da

from hyperspy.drawing._markers.iter_patch_collection import IterPatchCollection
from hyperspy._signals.signal2d import Signal2D, BaseSignal, Signal1D

BASELINE_DIR = "marker_collection"
DEFAULT_TOL = 2.0
STYLE_PYTEST_MPL = "default"
plt.style.use(STYLE_PYTEST_MPL)


class TestIterCollections:

    def test_iter_patch_collection(self):
        s = Signal2D(np.ones((3, 5, 6)))
        xy = np.empty(s.axes_manager.navigation_shape, dtype=object)
        angles = np.empty(s.axes_manager.navigation_shape, dtype=object)
        widths = np.empty(s.axes_manager.navigation_shape, dtype=object)

        for ind in np.ndindex(xy.shape):
            xy[ind] = np.ones((10, 2)) * 100
            angles[ind] = np.ones((10,)) * 180
            widths[ind] = np.ones((10,)) * 4
        markers = IterPatchCollection(patch=Ellipse, xy=xy, angle=angles, width=widths, height=(1,))
        s.plot()
        s.add_marker(markers)
        s.axes_manager.navigation_axes[0].index = 1
