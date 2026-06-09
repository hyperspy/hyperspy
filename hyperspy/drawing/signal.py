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

# This file contains plotting code generic to the BaseSignal class.

import numpy as np
from traits.api import Undefined

from hyperspy.drawing.backends import get_backend
from hyperspy.drawing.utils import set_axes_decor
from hyperspy.misc.utils import to_numpy


def _plot_1D_component(
    factors,
    idx,
    axes_manager,
    ax=None,
    calibrate=True,
    comp_label=None,
    same_window=False,
):
    backend = get_backend()
    if ax is None:
        import matplotlib.pyplot as plt

        ax = plt.gca()
    axis = axes_manager.signal_axes[0]
    if calibrate:
        x = axis.axis
        backend.set_xlabel(ax, axis.units)
    else:
        x = np.arange(axis.size)
        backend.set_xlabel(ax, "Channel index")
    backend.plot_line(ax, x, to_numpy(factors[:, idx]), label=f"{idx}")
    if comp_label and not same_window:
        backend.set_title(ax, f"{comp_label}")
    return ax


def _plot_2D_component(
    factors,
    idx,
    axes_manager,
    calibrate=True,
    ax=None,
    comp_label=None,
    cmap="gray",
    axes_decor="all",
):
    backend = get_backend()
    shape = axes_manager._signal_shape_in_array
    factors = to_numpy(factors[:, idx].reshape(shape))
    if ax is None:
        import matplotlib.pyplot as plt

        ax = plt.gca()
    axes = axes_manager.signal_axes[::-1]
    extent = None
    if calibrate:
        extent = (
            axes[1].low_value,
            axes[1].high_value,
            axes[0].high_value,
            axes[0].low_value,
        )
    if comp_label:
        backend.set_title(ax, f"{idx}")
    im = backend.plot_image(ax, factors, cmap=cmap, extent=extent)

    # Set axes decorations based on user input
    set_axes_decor(ax, axes_decor)

    backend.add_colorbar(backend.get_figure_from_ax(ax), im, ax)
    return ax


def _plot_loading(
    loadings,
    idx,
    axes_manager,
    ax=None,
    comp_label=None,
    no_nans=True,
    calibrate=True,
    cmap="gray",
    same_window=False,
    axes_decor="all",
):
    backend = get_backend()
    loadings = to_numpy(loadings[idx])
    if ax is None:
        import matplotlib.pyplot as plt

        ax = plt.gca()
    if no_nans:
        loadings = np.nan_to_num(loadings)
    axes = axes_manager.navigation_axes
    if axes_manager.navigation_dimension == 2:
        extent = None
        # get calibration from a passed axes_manager
        shape = axes_manager._navigation_shape_in_array
        if calibrate:
            extent = (
                axes[0].low_value,
                axes[0].high_value,
                axes[1].high_value,
                axes[1].low_value,
            )
        im = backend.plot_image(ax, loadings.reshape(shape), cmap=cmap, extent=extent)
        if calibrate:
            backend.set_xlabel(ax, axes[0].units)
            backend.set_ylabel(ax, axes[1].units)
        else:
            backend.set_xlabel(ax, "pixels")
            backend.set_ylabel(ax, "pixels")
        if comp_label:
            if same_window:
                backend.set_title(ax, f"{idx}")
            else:
                backend.set_title(ax, f"{idx} #{idx}")

        # Set axes decorations based on user input
        set_axes_decor(ax, axes_decor)

        backend.add_colorbar(backend.get_figure_from_ax(ax), im, ax)
    elif axes_manager.navigation_dimension == 1:
        if calibrate:
            x = axes[0].axis
        else:
            x = np.arange(axes[0].size)
        backend.plot_step(ax, x, loadings, label=f"{idx}")
        if comp_label and not same_window:
            backend.set_title(ax, f"{comp_label} #{idx}")
        backend.set_ylabel(ax, "Score (a. u.)")
        if calibrate:
            if axes[0].units is not Undefined:
                backend.set_xlabel(ax, axes[0].units)
            else:
                backend.set_xlabel(ax, "depth")
        else:
            backend.set_xlabel(ax, "depth")
    else:
        raise ValueError("View not supported")
