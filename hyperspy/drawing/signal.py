# -*- coding: utf-8 -*-
# Copyright 2007-2020 The HyperSpy developers
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

# This file contains plotting code generic to the BaseSignal class.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from traits.api import Undefined

from hyperspy.drawing.utils import set_axes_decor


def _plot_1D_component(factors, idx, axes_manager, ax=None,
                       calibrate=True, comp_label=None,
                       same_window=False):
    if ax is None:
        ax = plt.gca()
    axis = axes_manager.signal_axes[0]
    if calibrate:
        x = axis.axis
        plt.xlabel(axis.units)
    else:
        x = np.arange(axis.size)
        plt.xlabel('Channel index')
    ax.plot(x, factors[:, idx], label='%i' % idx)
    if comp_label and not same_window:
        plt.title('%s' % comp_label)
    return ax


def _plot_2D_component(factors, idx, axes_manager,
                       calibrate=True, ax=None,
                       comp_label=None, cmap=plt.cm.gray,
                       axes_decor='all'
                       ):
    if ax is None:
        ax = plt.gca()
    axes = axes_manager.signal_axes[::-1]
    shape = axes_manager._signal_shape_in_array
    extent = None
    if calibrate:
        extent = (axes[1].low_value,
                  axes[1].high_value,
                  axes[0].high_value,
                  axes[0].low_value)
    if comp_label:
        plt.title('%s' % idx)
    im = ax.imshow(factors[:, idx].reshape(shape),
                   cmap=cmap, interpolation='nearest',
                   extent=extent)

    # Set axes decorations based on user input
    set_axes_decor(ax, axes_decor)

    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    return ax


def _plot_loading(loadings, idx, axes_manager, ax=None,
                  comp_label=None, no_nans=True,
                  calibrate=True, cmap=plt.cm.gray,
                  same_window=False, axes_decor='all'):
    if ax is None:
        ax = plt.gca()
    if no_nans:
        loadings = np.nan_to_num(loadings)
    axes = axes_manager.navigation_axes
    if axes_manager.navigation_dimension == 2:
        extent = None
        # get calibration from a passed axes_manager
        shape = axes_manager._navigation_shape_in_array
        if calibrate:
            extent = (axes[0].low_value,
                      axes[0].high_value,
                      axes[1].high_value,
                      axes[1].low_value)
        im = ax.imshow(loadings[idx].reshape(shape),
                       cmap=cmap, extent=extent,
                       interpolation='nearest')
        if calibrate:
            plt.xlabel(axes[0].units)
            plt.ylabel(axes[1].units)
        else:
            plt.xlabel('pixels')
            plt.ylabel('pixels')
        if comp_label:
            if same_window:
                plt.title('%s' % idx)
            else:
                plt.title('%s #%s' % (comp_label, idx))

        # Set axes decorations based on user input
        set_axes_decor(ax, axes_decor)

        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    elif axes_manager.navigation_dimension == 1:
        if calibrate:
            x = axes[0].axis
        else:
            x = np.arange(axes[0].size)
        ax.step(x, loadings[idx],
                label='%s' % idx)
        if comp_label and not same_window:
            plt.title('%s #%s' % (comp_label, idx))
        plt.ylabel('Score (a. u.)')
        if calibrate:
            if axes[0].units is not Undefined:
                plt.xlabel(axes[0].units)
            else:
                plt.xlabel('depth')
        else:
            plt.xlabel('depth')
    else:
        raise ValueError('View not supported')
