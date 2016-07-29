# Copyright 2007-2016 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import traits.api as t
from matplotlib.testing.decorators import image_comparison

import hyperspy.api as hs
from hyperspy.misc.test_utils import get_matplotlib_version_label

mplv = get_matplotlib_version_label()
scalebar_color = 'blue'


def _set_navigation_axes(axes_manager, name=t.Undefined, units=t.Undefined,
                         scale=1.0, offset=0.0):
    for nav_axis in axes_manager.navigation_axes:
        nav_axis.units = units
        nav_axis.scale = scale
        nav_axis.offset = offset
    return axes_manager


def _set_signal_axes(axes_manager, name=t.Undefined, units=t.Undefined,
                     scale=1.0, offset=0.0):
    for sig_axis in axes_manager.signal_axes:
        sig_axis.name = name
        sig_axis.units = units
        sig_axis.scale = scale
        sig_axis.offset = offset
    return axes_manager

    
""" Navigation 0, Signal 1 """


@image_comparison(baseline_images=['%s_nav0_signal1_sig' % mplv],
                  extensions=['png'])
def test_plot_nav0_sig1():
    s = hs.signals.Signal1D(np.arange(20))
    s.axes_manager = _set_signal_axes(s.axes_manager, name='Energy',
                                      units='eV', scale=500.0, offset=300.0)
    s.metadata.General.title = '1: Nav 0, Sig 1'
    s.plot()


""" Navigation 1, Signal 1 """


def _setup_nav1_sig1():
    data = np.arange(10 * 20).reshape((10, 20))
    s = hs.signals.Signal1D(data)
    s.axes_manager = _set_signal_axes(s.axes_manager, name='Energy',
                                      units='eV', scale=500.0, offset=300.0)
    s.axes_manager = _set_navigation_axes(s.axes_manager, name='',
                                          units='m', scale=1E-6, offset=5E-6)
    return s

    
@image_comparison(baseline_images=['%s_nav1_signal1_1nav' % mplv,
                                   '%s_nav1_signal1_1sig' % mplv],
                  extensions=['png'])
def test_plot_nav1_sig1():
    s = _setup_nav1_sig1()
    s.metadata.General.title = '1: Nav 1, Sig 1'
    s.plot()


""" Navigation 2, Signal 1 """
    

def _setup_nav2_sig1():
    data = np.arange(5 * 10 * 20).reshape((5, 10, 20))
    s = hs.signals.Signal1D(data)
    s.axes_manager = _set_signal_axes(s.axes_manager, name='Energy',
                                      units='eV', scale=500.0, offset=300.0)
    s.axes_manager = _set_navigation_axes(s.axes_manager, name='',
                                          units='m', scale=1E-6, offset=5E-6)
    return s

@image_comparison(baseline_images=['%s_nav2_signal1_1nav' % mplv,
                                   '%s_nav2_signal1_1sig' % mplv],
                  extensions=['png'])
def test_plot_nav2_sig1():
    s = _setup_nav2_sig1()
    s.metadata.General.title = '1: Nav 2, Sig 1'
    s.plot()
