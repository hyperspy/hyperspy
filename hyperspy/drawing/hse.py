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

"""Backend-agnostic HyperSignal1D_Explorer base."""

import copy
import warnings
from abc import abstractmethod

from traits.api import Undefined

from hyperspy.drawing.backends._protocol import BackendCapabilityError
from hyperspy.drawing.he import HyperExplorer


class HyperSignal1D_Explorer(HyperExplorer):
    """Base for 1-D signal explorers.  Backend subclasses implement
    _make_signal_figure() and _connect_key_handler()."""

    def __init__(self):
        super().__init__()
        self.xlabel = ""
        self.ylabel = ""
        self.right_pointer = None
        self._right_pointer_on = False
        self._auto_update_plot = True

    @property
    def auto_update_plot(self):
        return self._auto_update_plot

    @auto_update_plot.setter
    def auto_update_plot(self, value):
        if self._auto_update_plot is value:
            return
        for line in self.signal_plot.ax_lines + self.signal_plot.right_ax_lines:
            line.auto_update = value
        if self.pointer is not None:
            if value is True:
                self._connect_pointer(self.pointer, self.navigator_plot)
            else:
                self.pointer.disconnect()

    @property
    def right_pointer_on(self):
        """I'm the 'x' property."""
        return self._right_pointer_on

    @right_pointer_on.setter
    def right_pointer_on(self, value):
        if value == self._right_pointer_on:
            return
        self._right_pointer_on = value
        if value is True:
            self.add_right_pointer()
        else:
            self.remove_right_pointer()

    def plot_signal(self, **kwargs):
        super().plot_signal()
        # Create the figure
        self.axis = self.axes_manager.signal_axes[0]
        self.xlabel = "{}".format(self.axes_manager.signal_axes[0])
        if self.axes_manager.signal_axes[0].units is not Undefined:
            self.xlabel += " ({})".format(self.axes_manager.signal_axes[0].units)
        self.ylabel = self.quantity_label if self.quantity_label != "" else "Intensity"

        sf = self._make_signal_figure(**kwargs)
        self.signal_plot = sf
        self._connect_key_nav(sf)
        if self.navigator_plot is not None:
            self._connect_key_nav_switch(sf)
            self._connect_key_nav_switch(self.navigator_plot)
            self._connect_key_nav(self.navigator_plot)

    @abstractmethod
    def _make_signal_figure(self, **kwargs):
        raise NotImplementedError

    def key2switch_right_pointer(self, event):
        if event.key == "e":
            self.right_pointer_on = not self.right_pointer_on

    def _connect_key_nav_switch(self, figure):
        self._connect_key_handler(figure, self.key2switch_right_pointer)

    @abstractmethod
    def _connect_key_handler(self, figure, fn):
        raise NotImplementedError

    def add_right_pointer(self, **kwargs):
        try:
            self._do_add_right_pointer(**kwargs)
        except BackendCapabilityError as e:
            warnings.warn(
                f"Right pointer not available with the current backend: {e}",
                UserWarning,
                stacklevel=2,
            )
            self._right_pointer_on = False

    def _do_add_right_pointer(self, **kwargs):
        if self.signal_plot.right_axes_manager is None:
            self.signal_plot.right_axes_manager = copy.deepcopy(self.axes_manager)
        if self.right_pointer is None:
            pointer_cls = self.assign_pointer()
            if pointer_cls is None:
                # No pointer widget (e.g. slider navigator): proceed to the
                # line only, without a right-axis pointer.
                self._add_right_line(**kwargs)
                self.right_pointer_on = True
                self._redraw_signal_figure()
                return
            self.right_pointer = pointer_cls(self.signal_plot.right_axes_manager)
            # The following is necessary because e.g. a line pointer does not
            # have size
            if hasattr(self.pointer, "size"):
                self.right_pointer.size = self.pointer.size
            self.right_pointer.color = "blue"
            self.right_pointer.connect_navigate()
            self._connect_pointer(self.right_pointer, self.navigator_plot)

        if self.right_pointer is not None:
            for axis in self.axes_manager.navigation_axes[self._pointer_nav_dim :]:
                self.signal_plot.right_axes_manager._axes[axis.index_in_array] = axis

        self._add_right_line(**kwargs)
        self.right_pointer_on = True
        self._redraw_signal_figure()

    @abstractmethod
    def _add_right_line(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _redraw_signal_figure(self):
        raise NotImplementedError

    def remove_right_pointer(self):
        for line in list(self.signal_plot.right_ax_lines):
            line.close()
        self.right_pointer.close()
        self.right_pointer = None
