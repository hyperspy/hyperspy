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

from hyperspy.drawing import widget
from hyperspy import signals


def test_get_step():
    s = signals.Signal1D(np.zeros((4, 4)))
    axis = s.axes_manager.navigation_axes[0]
    step = widget.ResizableDraggableWidgetBase._get_step(s,s.axes_manager.navigation_axes[0])
    assert(step == 1)
    axis.index = 3
    step = widget.ResizableDraggableWidgetBase._get_step(s,s.axes_manager.navigation_axes[0])
    assert(step == 1)
    
    
def test_scalebar_remove():
    im = signals.Signal2D(-np.arange(10000).reshape([100, 100]))
    for ax in im.axes_manager.signal_axes:
        ax.scale = 1.2
        ax.units = 'nm'
    im.plot()
    assert im._plot.signal_plot.ax.scalebar is not None
    im._plot.signal_plot.ax.scalebar.remove()
