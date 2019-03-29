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
import logging

from hyperspy.drawing.widgets import ResizableDraggableWidgetBase


_logger = logging.getLogger(__name__)


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between @D vectors 'v1' and 'v2'::

            >>> angle_between((1, 0), (0, 1))
            1.5707963267948966
            >>> angle_between((1, 0), (1, 0))
            0.0
            >>> angle_between((1, 0), (-1, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arctan2(v2_u[1], v2_u[0]) - np.arctan2(v1_u[1], v1_u[0])
    # angle = np.arccos(np.dot(v1_u, v2_u))
    if np.isnan(angle):
        if (v1_u == v2_u).all():
            return 0.0
        else:
            return np.pi
    return angle


class Line2DWidget(ResizableDraggableWidgetBase):

    """A free-form line on a 2D plot. Enables dragging and moving the end
    points, but also allows rotation of the widget by moving the mouse beyond
    the end points of the line.

    The widget adds the 'linewidth' attribute, which is different from the size
    in the following regards: 'linewidth' is simply the width of the patch
    drawn from point to point. If 'size' is greater than 1, it will in
    principle select a rotated rectangle. If 'size' is greater than 4, the
    bounds of this rectangle will be visualized by two dashed lines along the
    outline of this rectangle, instead of a single line in the center.

    The widget also adds the attributes 'radius_resize', 'radius_move' and
    'radius_rotate' (defaults: 5, 5, 10), which determines the picker radius
    for resizing, aka. moving the edge points (by picking within
    'radius_resize' from an edge point); for moving (by picking within
    'radius_move' from the body of the line); and for rotation (by picking
    within 'radius_rotate' of the edge points on the "outside" of the line).
    The priority is in the order resize, rotate, move; so the 'radius_rotate'
    should always be larger than 'radius_resize' if the function is to be
    accessible (putting it lower is an easy way to disable the functionality).


    NOTE: This widget's internal position does not lock to axes points by
          default.
    NOTE: The 'position' is now a 2D tuple: tuple(tuple(x1, x2), tuple(y1, y2))
    NOTE: The 'size' property corresponds to line width, so it has a len() of
    only one.
    """

    # Bitfield values for different mouse interaction functions
    FUNC_NONE = 0       # Do nothing
    FUNC_MOVE = 1       # Move the widget
    FUNC_RESIZE = 2     # Move a vertex
    FUNC_ROTATE = 4     # Rotate
    FUNC_SIZERS = 8     # Change linewidth by indicators
    FUNC_A = 32         # Resize/rotate by first vertex
    FUNC_B = 64         # Resize/rotate by second vertex

    def __init__(self, axes_manager, **kwargs):
        super(Line2DWidget, self).__init__(axes_manager, **kwargs)
        self.linewidth = 1
        self.radius_move = self.radius_resize = 5
        self.radius_rotate = 15
        self._mfunc = self.FUNC_NONE    # Mouse interaction function
        self._prev_pos = None
        self._orig_pos = None
        self.snap_all = False
        self._width_indicator_patches = []
        self._size = np.array([0])
        # Set default axes
        if self.axes_manager is not None:
            if self.axes_manager.navigation_dimension > 1:
                self.axes = self.axes_manager.navigation_axes[0:2]
            else:
                self.axes = self.axes_manager.signal_axes[0:2]
        value = self.axes[0].scale if self.axes_manager else 1
        # [[x0, y0], [x1, y1]]
        self._pos = np.array([[0, 0], [value, 0]])

    def _set_size(self, value):
        """Setter for the 'size' property.

        Calls _size_changed to handle size change, if the value has changed.

        """
        value = value[0]  # in this method, value is a float/int
        if value < 0:
            value = 0
        elif value:
            # The size must not be smaller than the scale
            value = np.maximum(value, self.axes[0].scale)
            if self.snap_size:
                value = self._do_snap_size(value)[0]
        if self._size[0] != value:
            if not value and self._size:
                self._size = np.array((0,))
                self._remove_size_patch()
            elif value and not self._size:
                self._size = np.array((value,))
                self._set_size_patch()
            else:
                self._size = np.array((value,))
            self._size_changed()

    def _set_axes(self, axes):
        # _set_axes overwrites self._size so we back it up
        size = self._size
        position = self._pos
        super(Line2DWidget, self)._set_axes(axes)
        # Restore self._size
        self._size = size
        self._pos = position

    def connect_navigate(self):
        raise NotImplementedError("2D lines cannot be used to navigate (yet?)")

    def _validate_pos(self, pos):
        """Make sure all vertices are within axis bounds.
        """
        if np.shape(pos)[1] != len(self.axes):
            raise ValueError()

        ndim = np.shape(pos)[1]
        if ndim != len(self.axes):
            raise ValueError()
        pos = np.maximum(pos, [ax.low_value for ax in self.axes])
        pos = np.minimum(pos, [ax.high_value for ax in self.axes])
        if self.snap_position:
            pos = self._do_snap_position(pos)
        return pos

    def _do_snap_position(self, value=None):
        value = np.array(value) if value is not None else self._pos

        ret1 = super(Line2DWidget, self)._do_snap_position(value[0, :])
        ret2 = super(Line2DWidget, self)._do_snap_position(value[1, :])

        return np.array([ret1, ret2])

    def _set_snap_size(self, value):
        if value and self.axes[0].scale != self.axes[1].scale:
            _logger.warning('Snapping the width of the line is not supported '
                            'for axes with different scale.')
            return
        super()._set_snap_size(value)

    def _do_snap_size(self, value=None):
        if value is None:
            value = self._size[0]
        if hasattr(value, '__len__'):
            value = value[0]
        ax = self.axes[0]  # take one axis, different axis scale not supported
        value = round(value / ax.scale) * ax.scale

        # must return an array to be consistent with the widget API
        return np.array([value])

    def _get_line_normal(self):
        v = np.diff(self._pos, axis=0)   # Line vector
        x = -v[:, 1]
        y = v[:, 0]
        n = np.array([x, y]).T                    # Normal vector
        return n / np.linalg.norm(n)            # Normalized

    def get_line_length(self):
        """Returns line length in axes coordinates. Requires units on all axes
        to be the same to make any physical sense.
        """
        return np.linalg.norm(np.diff(self._pos, axis=0), axis=1)

    def get_centre(self):
        """Get the line center, which is simply the mean position of its
        vertices.
        """
        return np.mean(self._pos, axis=0)

    def _get_width_indicator_coords(self):
        """
        Get coordinates of width indicators.

        The returned format is:
            [[[x0A, y0A], [x1A, y1A]], [[x0B, y0B], [x1B, y1B]]]
            Where A and B refer to the two lines
        """
        n = self.size[0] * self._get_line_normal() / 2.
        c = np.array(self._pos)
        return c + n, c - n

    def _update_patch_position(self):
        self._update_patch_geometry()

    def _update_patch_size(self):
        self._update_patch_geometry()

    def _update_patch_geometry(self):
        """Set line position, and set width indicator's if appropriate
        """
        if self.is_on() and self.patch:
            self.patch[0].set_data(np.array(self._pos).T)
            # Update width indicator if present
            if self._width_indicator_patches:
                wc = self._get_width_indicator_coords()
                for i in range(2):
                    self.patch[1 + i].set_data(wc[i].T)
            self.draw_patch()

    def _set_patch(self):
        """Creates the line, and also creates the width indicators if
        appropriate.
        """
        self.ax.autoscale(False)   # Prevent plotting from rescaling
        xy = np.array(self._pos)
        max_r = max(self.radius_move, self.radius_resize,
                    self.radius_rotate)
        self.patch = self.ax.plot(
            xy[:, 0], xy[:, 1],
            linestyle='-',
            animated=self.blit,
            lw=self.linewidth,
            c=self.color,
            alpha=self.alpha,
            marker='s',
            markersize=self.radius_resize,
            mew=0.1,
            mfc='lime',
            picker=max_r,)[0:1]

    def _set_size_patch(self):
        wc = self._get_width_indicator_coords()
        for i in range(2):
            wi, = self.ax.plot(
                wc[i][0], wc[i][1],
                linestyle=':',
                animated=self.blit,
                lw=self.linewidth,
                c=self.color,
                picker=self.radius_move)
            self.patch.append(wi)
            self._width_indicator_patches.append(wi)

    def _remove_size_patch(self):
        if not self._width_indicator_patches:
            return
        for patch in self._width_indicator_patches:
            self.patch.remove(patch)
            patch.remove()
        self._width_indicator_patches = []

    def _get_vertex(self, event):
        """Check bitfield on self.func, and return vertex index.
        """
        if self.func & self.FUNC_A:
            return 0
        elif self.func & self.FUNC_B:
            return 1
        else:
            return None

    def _get_func_from_pos(self, cx, cy):
        """Get interaction function from pixel position (cx,cy)
        """
        if not self.patch:
            return self.FUNC_NONE

        trans = self.ax.transData
        p = np.array(trans.transform(self._pos))

        # Calculate the distances to the vertecies, and find nearest one
        r2 = np.sum(np.power(p - np.array((cx, cy)), 2), axis=1)
        mini = np.argmin(r2)    # Index of nearest vertex
        minr2 = r2[mini]        # Distance squared to nearest vertex
        del r2
        # Check for resize: Click within radius_resize from edge points
        radius = self.radius_resize
        if minr2 <= radius ** 2:
            ret = self.FUNC_RESIZE
            ret |= self.FUNC_A if mini == 0 else self.FUNC_B
            return ret

        # Check for rotate: Click within radius_rotate on outside of edgepts
        radius = self.radius_rotate
        A = p[0, :]  # Vertex A
        B = p[1, :]  # Vertex B. Assumes one line segment only.
        c = np.array((cx, cy))   # mouse click position
        t = np.dot(c - A, B - A)    # t[0]: A->click, t[1]: A->B
        bas = np.linalg.norm(B - A)**2
        if minr2 <= radius**2:   # If within rotate radius
            if t < 0.0 and mini == 0:   # "Before" A on the line
                return self.FUNC_ROTATE | self.FUNC_A
            elif t > bas and mini == 1:  # "After" B on the line
                return self.FUNC_ROTATE | self.FUNC_B

        # Check for move: Click within radius_move from any point on the line
        radius = self.radius_move
        if 0 < t < bas:
            # A + (t/bas)*(B-A) is closest point on line
            if np.linalg.norm(A + (t / bas) * (B - A) - c) < radius:
                return self.FUNC_MOVE

        # Check for line width resize: Click within radius_move from width
        # indicator lines
        radius = self.radius_move
        wc = self._get_width_indicator_coords()
        for i in range(2):
            A = np.array(trans.transform(wc[i][0]))
            B = np.array(trans.transform(wc[i][1]))
            t = np.dot(c - A, B - A)
            bas = np.linalg.norm(B - A)**2
            if 0 < t < bas:
                # A + (t/bas)*(B-A) is closest point on line
                if np.linalg.norm(A + (t / bas) * (B - A) - c) < radius:
                    return self.FUNC_SIZERS
        return self.FUNC_NONE

    def onpick(self, event):
        """Pick, and if picked, figure out which function to apply. Also store
        pouse position for use by _onmousemove. As rotation does not work very
        well with incremental rotations, the original points are stored if
        we're rotating.
        """
        super(Line2DWidget, self).onpick(event)
        if self.picked:
            me = event.mouseevent
            self.func = self._get_func_from_pos(me.x, me.y)
            self._drag_start = [me.xdata, me.ydata]

    def _onmousemove(self, event):
        """Delegate to _move(), _resize() or _rotate().
        """
        if self.picked is True:
            if self.func & self.FUNC_MOVE and event.inaxes:
                self._move(event)
            elif self.func & self.FUNC_RESIZE and event.inaxes:
                self._resize(event)
            elif self.func & self.FUNC_ROTATE:
                self._rotate(event)
            elif self.func & self.FUNC_SIZERS and event.inaxes:
                self._width_resize(event)

    def _get_diff(self, event):
        """Get difference in position in event and what is stored in _prev_pos,
        in value space.
        """
        if event.xdata is None:
            dx = 0
        else:
            dx = event.xdata - self._drag_start[0]
        if event.ydata is None:
            dy = 0
        else:
            dy = event.ydata - self._drag_start[1]
        return np.array((dx, dy))

    def _move(self, event):
        """Move line by drag start position + difference in mouse post from
        when dragging started (picked).
        """
        dx = self._get_diff(event)
        self.position = self._drag_store[0] + dx

    def _resize(self, event):
        """Move vertex by difference from pick / last mouse move. Update
        '_prev_pos'.
        """
        ip = self._get_vertex(event)
        dx = self._get_diff(event)
        p = np.array(self._pos)     # Copy
        p[ip, 0:2] = self._drag_store[0][ip] + dx
        self.position = p

    def _rotate(self, event):
        """Rotate original points by the angle between mouse position and
        rotation start position (rotation center = line center).
        """
        if None in (event.xdata, event.ydata):
            return
        # Get difference in mouse pos since drag start (picked)
        dx = self._get_diff(event)

        # Rotation should happen in screen position, as anything else will
        # mix units
        trans = self.ax.transData
        scr_zero = np.array(trans.transform((0, 0)))
        dx = np.array(trans.transform(dx)) - scr_zero

        # Get center point = center of original line
        c = trans.transform(np.mean(self._drag_store[0], axis=0))

        # Figure out theta
        v1 = (event.x, event.y) - c     # Center to mouse
        v2 = v1 - dx                    # Center to start pos
        theta = angle_between(v2, v1)   # Rotation between start and mouse

        if event.key is not None and 'shift' in event.key:
            base = 30 * np.pi / 180
            theta = base * round(float(theta) / base)

        # vector from points to center
        w1 = c - trans.transform(self._drag_store[0])
        # rotate into w2 for next point
        w2 = np.array((w1[:, 0] * np.cos(theta) - w1[:, 1] * np.sin(theta),
                       w1[:, 1] * np.cos(theta) + w1[:, 0] * np.sin(theta)))
        self.position = trans.inverted().transform(c + np.rot90(w2))

    def _width_resize(self, event):
        if None in (event.xdata, event.ydata) or self.size[0] == 0:
            return
        # Get difference in mouse pos since drag start (picked)
        dx = self._get_diff(event)
        # Project onto normal axis (dot product onto normal)
        n = self._get_line_normal()
        dn = 2 * np.dot(n, dx)
        if self._selected_artist is self.patch[2]:
            dn *= -1
        self.size = np.abs(self._drag_store[1] + dn)
