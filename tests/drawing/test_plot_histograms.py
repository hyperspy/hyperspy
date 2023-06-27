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

import hyperspy.api as hs
import numpy as np


def test_plot_histograms():
    img = hs.signals.Signal2D(np.random.chisquare(1,[10,10,100]))
    img2 = hs.signals.Signal2D(np.random.chisquare(2,[10,10,100]))
    ax = hs.plot.plot_histograms([img, img2], legend=['hist1', 'hist2'])
    assert len(ax.lines) == 2
    l1 = ax.lines[0]
    assert l1.get_drawstyle() == 'steps-mid'
