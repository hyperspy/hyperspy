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
import logging

from hyperspy.drawing.widgets import Widget2DBase, ResizersMixin


_logger = logging.getLogger(__name__)
# Track if we have already warned when the widget is out of range
already_warn_out_of_range = False


class SquareWidget(Widget2DBase):

    """SquareWidget is a symmetric, Rectangle-patch based widget, which can be
    dragged, and resized by keystrokes/code. As the widget is normally only
    meant to indicate position, the sizing is deemed purely visual, but there
    is nothing that forces this use. However, it should be noted that the outer
    bounds only correspond to pure indices for odd sizes.
    """

    def __init__(self, axes_manager, **kwargs):
        super(SquareWidget, self).__init__(axes_manager, **kwargs)

    def _set_patch(self):
        """Sets the patch to a matplotlib Rectangle with the correct geometry.
        The geometry is defined by _get_patch_xy, and get_size_in_axes.
        """
        xy = self._get_patch_xy()
        xs, ys = self.size
        self.patch = [plt.Rectangle(
            xy, xs, ys,
            animated=self.blit,
            fill=False,
            lw=self.border_thickness,
            ec=self.color,
            alpha=self.alpha,
            picker=True,)]
        super(SquareWidget, self)._set_patch()

    def _onmousemove(self, event):
        """on mouse motion move the patch if picked"""
        if self.picked is True and event.inaxes:
            self.position = (event.xdata, event.ydata)


class RectangleWidget(SquareWidget, ResizersMixin):

    """RectangleWidget is a asymmetric, Rectangle-patch based widget, which can
    be dragged and resized by mouse/keys. For resizing by mouse, it adds a
    small Rectangle patch on the outer border of the main patch, to serve as
    resize handles. This feature can be enabled/disabled by the 'resizers'
    property, and the size/color of the handles are set by
    'resize_color'/'resize_pixel_size'.

    For optimized changes of geometry, the class implements two methods
    'set_bounds' and 'set_ibounds', to set the geomtry of the rectangle by
    value and index space coordinates, respectivly. It also adds the 'width'
    and 'height' properties for verbosity.

    For keyboard resizing, 'x'/'c' and 'y'/'u' will increase/decrease the size
    of the rectangle along the first and the second axis, respectively.

    Implements the internal method _validate_geometry to make sure the patch
    will always stay within bounds.
    """

    # --------- External interface ---------

    def _parse_bounds_args(self, args, kwargs):
        """Internal utility function to parse args/kwargs passed to set_bounds
        and set_ibounds.
        """
        if len(args) == 1:
            return args[0]
        elif len(args) == 4:
            return args
        elif len(kwargs) == 1 and 'bounds' in kwargs:
            return kwargs.values()[0]
        else:
            x = kwargs.pop('x', kwargs.pop('left', self._pos[0]))
            y = kwargs.pop('y', kwargs.pop('top', self._pos[1]))
            if 'right' in kwargs:
                w = kwargs.pop('right') - x
            else:
                w = kwargs.pop('w', kwargs.pop('width', self._size[0]))
            if 'bottom' in kwargs:
                h = kwargs.pop('bottom') - y
            else:
                h = kwargs.pop('h', kwargs.pop('height', self._size[1]))
            return x, y, w, h

    def set_ibounds(self, *args, **kwargs):
        """
        Set bounds by indices. Bounds can either be specified in order left,
        bottom, width, height; or by keywords:
         * 'bounds': tuple (left, top, width, height)
         OR
         * 'x'/'left'
         * 'y'/'top'
         * 'w'/'width', alternatively 'right'
         * 'h'/'height', alternatively 'bottom'
        If specifying with keywords, any unspecified dimensions will be kept
        constant (note: width/height will be kept, not right/bottom).
        """

        ix, iy, iw, ih = self._parse_bounds_args(args, kwargs)

        x = self.axes[0].index2value(ix)
        y = self.axes[1].index2value(iy)
        w = self._i2v(self.axes[0], ix + iw) - x
        h = self._i2v(self.axes[1], iy + ih) - y

        old_position, old_size = self.position, self.size
        self._pos = np.array([x, y])
        self._size = np.array([w, h])
        self._apply_changes(old_size=old_size, old_position=old_position)

    def set_bounds(self, *args, **kwargs):
        """
        Set bounds by values. Bounds can either be specified in order left,
        bottom, width, height; or by keywords:
         * 'bounds': tuple (left, top, width, height)
         OR
         * 'x'/'left'
         * 'y'/'top'
         * 'w'/'width', alternatively 'right' (x+w)
         * 'h'/'height', alternatively 'bottom' (y+h)
        If specifying with keywords, any unspecified dimensions will be kept
        constant (note: width/height will be kept, not right/bottom).
        """
        global already_warn_out_of_range
        x, y, w, h = self._parse_bounds_args(args, kwargs)

        def warn(obj, parameter, value):
            global already_warn_out_of_range
            if not already_warn_out_of_range:
                _logger.info('{}: {} is out of range. It is therefore set '
                             'to the value of {}'.format(obj, parameter, value))
                already_warn_out_of_range = True

        scale = [axis.scale for axis in self.axes]
        l0, h0 = self.axes[0].low_value, self.axes[0].high_value
        l1, h1 = self.axes[1].low_value, self.axes[1].high_value

        in_range = 0
        if x < l0:
            x = l0
            warn(self, '`x`', x)
        elif h0 <= x:
            x = h0 - scale[0]
            warn(self, '`x`', x)
        else:
            in_range += 1
        if y < l1:
            y = l1
            warn(self, '`y`', y)
        elif h1 <= y:
            warn(self, '`y`', y)
            y = h1 - scale[1]
        else:
            in_range += 1
        if w < scale[0]:
            w = scale[0]
            warn(self, '`width` or `right`', w)
        elif not (l0 + scale[0] <= x + w <= h0 + scale[0]):
            if self.size[0] != w:  # resize
                w = h0 + scale[0] - self.position[0]
                warn(self, '`width` or `right`', w)
            if self.position[0] != x:  # moved
                x = h0 + scale[0] - self.size[0]
                warn(self, '`x`', x)
        else:
            in_range += 1
        if h < scale[1]:
            h = scale[1]
            warn(self, '`height` or `bottom`', h)
        elif not (l1 + scale[1] <= y + h <= h1 + scale[1]):
            if self.size[1] != h:  # resize
                h = h1 + scale[1] - self.position[1]
                warn(self, '`height` or `bottom`', h)
            if self.position[1] != y:  # moved
                y = h1 + scale[1] - self.size[1]
                warn(self, '`y`', y)
        else:
            in_range += 1

        # if we are in range again, reset `already_warn_out_of_range` to False
        if in_range == 4 and already_warn_out_of_range:
            _logger.info('{} back in range.'.format(self.__class__.__name__))
            already_warn_out_of_range = False

        old_position, old_size = self.position, self.size
        self._pos = np.array([x, y])
        self._size = np.array([w, h])
        self._apply_changes(old_size=old_size, old_position=old_position)

    def _validate_pos(self, value):
        """Constrict the position within bounds.
        """
        value = (min(value[0], self.axes[0].high_value - self._size[0] +
                     self.axes[0].scale),
                 min(value[1], self.axes[1].high_value - self._size[1] +
                     self.axes[1].scale))
        return super(RectangleWidget, self)._validate_pos(value)

    @property
    def width(self):
        return self.get_size_in_indices()[0]

    @width.setter
    def width(self, value):
        if value == self.width:
            return
        ix = self.indices[0] + value
        il0, ih0 = self.axes[0].low_index, self.axes[0].high_index
        if value <= 0 or not (il0 < ix <= ih0):
            raise ValueError('`width` value is not in range. The '
                             '`width` is {} and should be in range '
                             '{}-{}.'.format(ix, il0 + 1, ih0))
        self._set_a_size(0, value)

    @property
    def height(self):
        return self.get_size_in_indices()[1]

    @height.setter
    def height(self, value):
        if value == self.height:
            return
        iy = self.indices[1] + value
        il1, ih1 = self.axes[1].low_index, self.axes[1].high_index
        if value <= 0 or not (il1 < iy <= ih1):
            raise ValueError('`height` value is not in range. The '
                             '`height` is {} and should be in range '
                             '{}-{}.'.format(iy, il1 + 1, ih1))
        self._set_a_size(1, value)

    # --------- Internal functions ---------

    # --- Internals that trigger events ---

    def _set_size(self, value):
        value = np.minimum(value, [ax.size * ax.scale for ax in self.axes])
        value = np.maximum(value, [ax.scale for ax in self.axes])
        if np.any(self._size != value):
            old = self._size
            self._size = value
            self._validate_geometry()
            if np.any(self._size != old):
                self._size_changed()

    def _set_a_size(self, idx, value):
        if self._size[idx] == value or value <= 0:
            return
        # If we are pushed "past" an edge, size towards it
        if self._navigating and self.axes[idx].value > self._pos[idx]:
            if value < self._size[idx]:
                self._pos[idx] += self._size[idx] - value

        self._size[idx] = value
        self._validate_geometry()
        self._size_changed()

    def _increase_xsize(self):
        self._set_a_size(0, self._size[0] +
                         self.axes[0].scale * self.size_step)

    def _decrease_xsize(self):
        new_s = self._size[0] - self.axes[0].scale * self.size_step
        new_s = max(new_s, self.axes[0].scale)
        self._set_a_size(0, new_s)

    def _increase_ysize(self):
        self._set_a_size(1, self._size[1] +
                         self.axes[1].scale * self.size_step)

    def _decrease_ysize(self):
        new_s = self._size[1] - self.axes[1].scale * self.size_step
        new_s = max(new_s, self.axes[1].scale)
        self._set_a_size(1, new_s)

    def on_key_press(self, event):
        if self.selected:
            if event.key == "x":
                self._increase_xsize()
            elif event.key == "c":
                self._decrease_xsize()
            elif event.key == "y":
                self._increase_ysize()
            elif event.key == "u":
                self._decrease_ysize()
            else:
                super(RectangleWidget, self).on_key_press(event)

    # --- End internals that trigger events ---

    def _get_patch_xy(self):
        """Get xy value for Rectangle with position being top left. This value
        deviates from the 'position', as 'position' correspond to the center
        value of the pixel. Here, xy corresponds to the top left of the pixel.
        """
        offset = [a.scale for a in self.axes]
        return self._pos - 0.5 * np.array(offset)

    def _update_patch_position(self):
        # Override to include resizer positioning
        if self.is_on() and self.patch:
            self.patch[0].set_xy(self._get_patch_xy())
            self._update_resizers()
            self.draw_patch()

    def _update_patch_geometry(self):
        # Override to include resizer positioning
        if self.is_on() and self.patch:
            self.patch[0].set_bounds(*self._get_patch_bounds())
            self._update_resizers()
            self.draw_patch()

    def _validate_geometry(self, x1=None, y1=None):
        """Make sure the entire patch always stays within bounds. First the
        position (either from position property or from x1/y1 arguments), is
        limited within the bounds. Then, if the bottom/right edges are out of
        bounds, the position is changed so that they will be at the limit.

        The modified geometry is stored, but no change checks are performed.
        Call _apply_changes after this in order to process any changes (the
        size might change if it is set larger than the bounds size).
        """
        xaxis = self.axes[0]
        yaxis = self.axes[1]

        # Make sure widget size is not larger than axes
        self._size[0] = min(self._size[0], xaxis.size * xaxis.scale)
        self._size[1] = min(self._size[1], yaxis.size * yaxis.scale)

        # Make sure x1/y1 is within bounds
        if x1 is None:
            x1 = self._pos[0]  # Get it if not supplied
        elif x1 < xaxis.low_value:
            x1 = xaxis.low_value
        elif x1 > xaxis.high_value:
            x1 = xaxis.high_value

        if y1 is None:
            y1 = self._pos[1]
        elif y1 < yaxis.low_value:
            y1 = yaxis.low_value
        elif y1 > yaxis.high_value:
            y1 = yaxis.high_value

        # Make sure x2/y2 is with upper bound.
        # If not, keep dims, and change x1/y1!
        x2 = x1 + self._size[0]
        y2 = y1 + self._size[1]
        if x2 > xaxis.high_value + xaxis.scale:
            x2 = xaxis.high_value + xaxis.scale
            x1 = x2 - self._size[0]
        if y2 > yaxis.high_value + yaxis.scale:
            y2 = yaxis.high_value + yaxis.scale
            y1 = y2 - self._size[1]

        self._pos = np.array([x1, y1])
        # Apply snaps if appropriate
        if self.snap_position:
            self._do_snap_position()
        if self.snap_size:
            self._do_snap_size()

    def _onmousemove(self, event):
        """on mouse motion draw the cursor if picked"""
        # Simple checks to make sure we are dragging our patch:
        if self.picked is True and event.inaxes:
            # Setup reused parameters
            xaxis = self.axes[0]
            yaxis = self.axes[1]
            # Mouse position
            x = event.xdata
            y = event.ydata
            p = self._get_patch_xy()
            # Old bounds
            bounds = [p[0], p[1],
                      p[0] + self._size[0],
                      p[1] + self._size[1]]

            # Store geometry for _apply_changes at end
            old_position, old_size = self.position, self.size

            if self.resizer_picked is False:
                # Simply dragging main patch. Offset mouse position by
                # pick_offset to get new position, then validate it.
                x -= self.pick_offset[0]
                y -= self.pick_offset[1]
                self._validate_geometry(x, y)
            else:
                posx = None     # New x pos. If None, the old pos will be used
                posy = None     # Same for y
                corner = self.resizer_picked
                # Adjust for resizer position:
                offset = self._get_resizer_offset()
                x += offset[0] * (0.5 - 1 * (corner % 2))
                y += offset[1] * (0.5 - 1 * (corner // 2))

                if corner % 2 == 0:         # Left side start
                    if x > bounds[2]:       # flipped to right
                        posx = bounds[2]    # New left is old right
                        # New size is mouse position - new left
                        self._size[0] = x - posx
                        self.resizer_picked += 1         # Switch pick to right
                    elif bounds[2] - x < xaxis.scale:   # Width too small
                        posx = bounds[2] - xaxis.scale      # So move pos left
                        self._size[0] = bounds[2] - posx    # Should be scale
                    else:                   # Moving left edge
                        posx = x            # Set left to mouse position
                        # Keep right still by changing size:
                        self._size[0] = bounds[2] - x
                else:                       # Right side start
                    if x < bounds[0]:       # Flipped to left
                        if bounds[0] - x < xaxis.scale:
                            posx = bounds[0] - xaxis.scale
                        else:
                            posx = x            # Set left to mouse
                        # Set size to old left - new left
                        self._size[0] = bounds[0] - posx
                        self.resizer_picked -= 1     # Switch pick to left
                    else:                   # Moving right edge
                        # Left should be left as it is, only size updates:
                        self._size[0] = x - bounds[0]  # mouse - old left
                if corner // 2 == 0:        # Top side start
                    if y > bounds[3]:       # flipped to botton
                        posy = bounds[3]    # New top is old bottom
                        # New size is mouse position - new top
                        self._size[1] = y - posy
                        self.resizer_picked += 2     # Switch pick to bottom
                    elif bounds[3] - y < yaxis.scale:  # Height too small
                        posy = bounds[3] - yaxis.scale      # So move pos up
                        self._size[1] = bounds[3] - posy    # Should be scale
                    else:                   # Moving top edge
                        posy = y           # Set top to mouse index
                        # Keep bottom still by changing size:
                        self._size[1] = bounds[3] - y  # old bottom - new top
                else:                       # Bottom side start
                    if y < bounds[1]:     # Flipped to top
                        if bounds[1] - y < yaxis.scale:
                            posy = bounds[1] - yaxis.scale
                        else:
                            posy = y           # Set top to mouse
                        # Set size to old top - new top
                        self._size[1] = bounds[1] - posy
                        self.resizer_picked -= 2     # Switch pick to top
                    else:                   # Moving bottom edge
                        self._size[1] = y - bounds[1]  # mouse - old top
                # Bound size to scale:
                if self._size[0] < xaxis.scale:
                    self._size[0] = xaxis.scale
                if self._size[1] < yaxis.scale:
                    self._size[1] = yaxis.scale
                if posx is not None:
                    posx += 0.5 * xaxis.scale
                if posy is not None:
                    posy += 0.5 * yaxis.scale
                # Validate the geometry
                self._validate_geometry(posx, posy)
            # Finally, apply any changes and trigger events/redraw:
            self._apply_changes(old_size=old_size, old_position=old_position)
