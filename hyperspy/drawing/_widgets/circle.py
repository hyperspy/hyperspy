# -*- coding: utf-8 -*-
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
import matplotlib.pyplot as plt

from hyperspy.drawing.widgets import Widget2DBase, ResizersMixin


class CircleWidget(Widget2DBase, ResizersMixin):

    """CircleWidget is a symmetric, Cicle-patch based widget, which can
    be dragged, and resized by keystrokes/code.
    """

    def __init__(self, axes_manager, **kwargs):
        super(CircleWidget, self).__init__(axes_manager, **kwargs)
        self.size_step = 1.0
        self.size_snap_offset = (0.5 + 1e-8)

    def _set_axes(self, axes):
        super(CircleWidget, self)._set_axes(axes)
        if self.axes:
            self._size[0] = (0.5 + 1e-8) * self.axes[0].scale
            if len(self.axes) > 1:
                self._size[1] = 0

    def _do_snap_size(self, value=None):
        # Snap to odd diameters = ?.5 radius
        value = np.array(value) if value is not None else self._size
        snap_offset = self.size_snap_offset * self.axes[0].scale
        snap_spacing = self.axes[0].scale * self.size_step
        for i in range(2):
            value[i] = max(0, (round((value[i] - snap_offset) / snap_spacing) *
                               snap_spacing + snap_offset))
        return value

    def _set_size(self, value):
        """Setter for the 'size' property. Calls _size_changed to handle size
        change, if the value has changed.
        """
        # Override so that r_inner can be 0
        value = np.minimum(value,
                           [0.5 * ax.size * ax.scale for ax in self.axes])
        # Changed from base:
        min_sizes = np.array(((0.5 + 1e-8) * self.axes[0].scale, 0))
        value = np.maximum(value, min_sizes)
        if value[0] < value[1]:
            self._set_size(value[::-1])
        else:
            if self.snap_size:
                value = self._do_snap_size(value)
            if np.any(self._size != value):
                self._size = value
                self._size_changed()

    def increase_size(self):
        """Increment all sizes by one step. Applied via 'size' property.
        """
        s = np.array(self.size)
        if self.size[1] > 0:
            s += self.size_step * self.axes[0].scale
        else:
            s[0] += self.size_step * self.axes[0].scale
        self.size = s

    def decrease_size(self):
        """Decrement all sizes by one step. Applied via 'size' property.
        """
        s = np.array(self.size)
        if self.size[1] > 0:
            s -= self.size_step * self.axes[0].scale
        else:
            s[0] -= self.size_step * self.axes[0].scale
        self.size = s

    def get_centre(self):
        return self.position

    def _get_patch_xy(self):
        """Returns the xy coordinates of the patch. In this implementation, the
        patch is centered on the position.
        """
        return self.position

    def _set_patch(self):
        """Sets the patch to a matplotlib Circle with the correct geometry.
        The geometry is defined by _get_patch_xy, and size.
        """
        super(CircleWidget, self)._set_patch()
        xy = self._get_patch_xy()
        ro, ri = self.size
        self.patch = [plt.Circle(
            xy, radius=ro,
            animated=self.blit,
            fill=False,
            lw=self.border_thickness,
            ec=self.color,
            alpha=self.alpha,
            picker=True,)]
        if ri > 0:
            self.patch.append(
                plt.Circle(
                    xy, radius=ro,
                    animated=self.blit,
                    fill=False,
                    lw=self.border_thickness,
                    ec=self.color,
                    alpha=self.alpha,
                    picker=True,))

    def _validate_pos(self, value):
        """Constrict the position within bounds.
        """
        value = (min(value[0], self.axes[0].high_value - self._size[0] +
                     (0.5 + 1e-8) * self.axes[0].scale),
                 min(value[1], self.axes[1].high_value - self._size[0] +
                     (0.5 + 1e-8) * self.axes[1].scale))
        value = (max(value[0], self.axes[0].low_value + self._size[0] -
                     (0.5 + 1e-8) * self.axes[0].scale),
                 max(value[1], self.axes[1].low_value + self._size[0] -
                     (0.5 + 1e-8) * self.axes[1].scale))
        return super(CircleWidget, self)._validate_pos(value)

    def get_size_in_indices(self):
        return np.array(self._size / self.axes[0].scale)

    def _update_patch_position(self):
        if self.is_on() and self.patch:
            self.patch[0].center = self._get_patch_xy()
            if self.size[1] > 0:
                self.patch[1].center = self.patch[0].center
            self._update_resizers()
            self.draw_patch()

    def _update_patch_size(self):
        if self.is_on() and self.patch:
            ro, ri = self.size
            self.patch[0].radius = ro
            if ri > 0:
                # Add the inner circle
                if len(self.patch) == 1:
                    # Need to remove the previous patch before using
                    # `_add_patch_to`
                    self.ax.artists.remove(self.patch[0])
                    self.patch = []
                    self._add_patch_to(self.ax)
                self.patch[1].radius = ri
            self._update_resizers()
            self.draw_patch()

    def _update_patch_geometry(self):
        if self.is_on() and self.patch:
            ro, ri = self.size
            self.patch[0].center = self._get_patch_xy()
            self.patch[0].radius = ro
            if ri > 0:
                self.patch[1].center = self.patch[0].center
                self.patch[1].radius = ri
            self._update_resizers()
            self.draw_patch()

    def _onmousemove(self, event):
        'on mouse motion move the patch if picked'
        if self.picked is True and event.inaxes:
            x = event.xdata
            y = event.ydata
            if self.resizer_picked is False:
                x -= self.pick_offset[0]
                y -= self.pick_offset[1]
                self.position = (x, y)
            else:
                rad_vect = np.array((x, y)) - self._pos
                radius = np.sqrt(np.sum(rad_vect**2))
                s = list(self.size)
                if self.resizer_picked < 4:
                    s[0] = radius
                else:
                    s[1] = radius
                self.size = s

    def _get_resizer_pos(self):
        positions = []
        indices = (0, 1) if self.size[1] > 0 else (0, )
        for i in indices:
            r = self._size[i]
            rsize = self._get_resizer_size() / 2
            rp = np.array(self._get_patch_xy())
            p = rp - (r, 0) - rsize             # Left
            positions.append(p)
            p = rp - (0, r) - rsize             # Top
            positions.append(p)
            p = rp + (r, 0) - rsize             # Right
            positions.append(p)
            p = rp + (0, r) - rsize             # Bottom
            positions.append(p)
        return positions
