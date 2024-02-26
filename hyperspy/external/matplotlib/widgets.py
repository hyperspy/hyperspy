# Copied from matplotlib 3.6, remove when minimum requirement of matplotlib
# is version >= 3.6
# Require small changes to work with matplotlib >=3.1
# - use of _api module is removed because it has been introduced in v3.4
# - API of cbook.normalize_kwargs has changed
# - remove set_cursor/_hover
# - remove use of _cm_set introduced in
#   https://github.com/matplotlib/matplotlib/pull/19369


from contextlib import ExitStack
import copy
from numbers import Integral

import numpy as np

from matplotlib import  cbook
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.widgets import AxesWidget


class _SelectorWidget(AxesWidget):

    def __init__(self, ax, onselect, useblit=False, button=None,
                 state_modifier_keys=None):
        super().__init__(ax)

        self.visible = True
        self.onselect = onselect
        self.useblit = useblit and self.canvas.supports_blit
        self.connect_default_events()

        self.state_modifier_keys = dict(move=' ', clear='escape',
                                        square='shift', center='control')
        self.state_modifier_keys.update(state_modifier_keys or {})

        self.background = None

        if isinstance(button, Integral):
            self.validButtons = [button]
        else:
            self.validButtons = button

        # Set to True when a selection is completed, otherwise is False
        self._selection_completed = False

        # will save the data (position at mouseclick)
        self._eventpress = None
        # will save the data (pos. at mouserelease)
        self._eventrelease = None
        self._prev_event = None
        self._state = set()

    def set_active(self, active):
        super().set_active(active)
        if active:
            self.update_background(None)

    def _get_animated_artists(self):
        """
        Convenience method to get all animated artists of a figure, except
        those already present in self.artists. 'z_order' is ignored.
        """
        return tuple([a for ax_ in self.ax.get_figure().get_axes()
                      for a in ax_.get_children()
                      if a.get_animated() and a not in self.artists])

    def update_background(self, event):
        """Force an update of the background."""
        # If you add a call to `ignore` here, you'll want to check edge case:
        # `release` can call a draw event even when `ignore` is True.
        if not self.useblit:
            return
        # Make sure that widget artists don't get accidentally included in the
        # background, by re-rendering the background if needed (and then
        # re-re-rendering the canvas with the visible widget artists).
        # z_order needs to be respected when redrawing
        artists = sorted(self.artists + self._get_animated_artists(),
                         key=lambda a: a.zorder)
        needs_redraw = any(artist.get_visible() for artist in artists)
        with ExitStack() as stack:
            if needs_redraw:
                for artist in artists:
                    stack.callback(artist.set_visible, artist.get_visible())
                    artist.set_visible(False)
                self.canvas.draw()
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        if needs_redraw:
            for artist in artists:
                self.ax.draw_artist(artist)

    def connect_default_events(self):
        """Connect the major canvas events to methods."""
        self.connect_event('motion_notify_event', self.onmove)
        self.connect_event('button_press_event', self.press)
        self.connect_event('button_release_event', self.release)
        self.connect_event('draw_event', self.update_background)
        self.connect_event('key_press_event', self.on_key_press)
        self.connect_event('key_release_event', self.on_key_release)
        self.connect_event('scroll_event', self.on_scroll)

    def ignore(self, event):
        # docstring inherited
        if not self.active or not self.ax.get_visible():
            return True
        # If canvas was locked
        if not self.canvas.widgetlock.available(self):
            return True
        if not hasattr(event, 'button'):
            event.button = None
        # Only do rectangle selection if event was triggered
        # with a desired button
        if (self.validButtons is not None
                and event.button not in self.validButtons):
            return True
        # If no button was pressed yet ignore the event if it was out
        # of the axes
        if self._eventpress is None:
            return event.inaxes != self.ax
        # If a button was pressed, check if the release-button is the same.
        if event.button == self._eventpress.button:
            return False
        # If a button was pressed, check if the release-button is the same.
        return (event.inaxes != self.ax or
                event.button != self._eventpress.button)

    def update(self):
        """Draw using blit() or draw_idle(), depending on ``self.useblit``."""
        if not self.ax.get_visible() or self.ax.figure._cachedRenderer is None:
            return False
        if self.useblit:
            if self.background is not None:
                self.canvas.restore_region(self.background)
            else:
                self.update_background(None)
            # We need to draw all artists, which are not included in the
            # background, therefore we also draw self._get_animated_artists()
            # and we make sure that we respect z_order
            artists = sorted(self.artists + self._get_animated_artists(),
                             key=lambda a: a.zorder)
            for artist in artists:
                self.ax.draw_artist(artist)
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()
        return False

    def _get_data(self, event):
        """Get the xdata and ydata for event, with limits."""
        if event.xdata is None:
            return None, None
        xdata = np.clip(event.xdata, *self.ax.get_xbound())
        ydata = np.clip(event.ydata, *self.ax.get_ybound())
        return xdata, ydata

    def _clean_event(self, event):
        """
        Preprocess an event:

        - Replace *event* by the previous event if *event* has no ``xdata``.
        - Clip ``xdata`` and ``ydata`` to the axes limits.
        - Update the previous event.
        """
        if event.xdata is None:
            event = self._prev_event
        else:
            event = copy.copy(event)
        event.xdata, event.ydata = self._get_data(event)
        self._prev_event = event
        return event

    def press(self, event):
        """Button press handler and validator."""
        if not self.ignore(event):
            event = self._clean_event(event)
            self._eventpress = event
            self._prev_event = event
            key = event.key or ''
            key = key.replace('ctrl', 'control')
            # move state is locked in on a button press
            if key == self.state_modifier_keys['move']:
                self._state.add('move')
            self._press(event)
            return True
        return False

    def _press(self, event):
        """Button press event handler."""

    def release(self, event):
        """Button release event handler and validator."""
        if not self.ignore(event) and self._eventpress:
            event = self._clean_event(event)
            self._eventrelease = event
            self._release(event)
            self._eventpress = None
            self._eventrelease = None
            self._state.discard('move')
            return True
        return False

    def _release(self, event):
        """Button release event handler."""

    def onmove(self, event):
        """Cursor move event handler and validator."""
        if not self.ignore(event) and self._eventpress:
            event = self._clean_event(event)
            self._onmove(event)
            return True
        return False

    def _onmove(self, event):
        """Cursor move event handler."""

    def on_scroll(self, event):
        """Mouse scroll event handler and validator."""
        if not self.ignore(event):
            self._on_scroll(event)

    def _on_scroll(self, event):
        """Mouse scroll event handler."""

    def on_key_press(self, event):
        """Key press event handler and validator for all selection widgets."""
        if self.active:
            key = event.key or ''
            key = key.replace('ctrl', 'control')
            if key == self.state_modifier_keys['clear']:
                self.clear()
                return
            for (state, modifier) in self.state_modifier_keys.items():
                if modifier in key:
                    self._state.add(state)
            self._on_key_press(event)

    def _on_key_press(self, event):
        """Key press event handler - for widget-specific key press actions."""

    def on_key_release(self, event):
        """Key release event handler and validator."""
        if self.active:
            key = event.key or ''
            for (state, modifier) in self.state_modifier_keys.items():
                if modifier in key:
                    self._state.discard(state)
            self._on_key_release(event)

    def _on_key_release(self, event):
        """Key release event handler."""

    def get_visible(self):
        """Get the visibility of the selector artists."""
        return self.visible

    def set_visible(self, visible):
        """Set the visibility of our artists."""
        self.visible = visible
        for artist in self.artists:
            artist.set_visible(visible)

    def clear(self):
        """Clear the selection and set the selector ready to make a new one."""
        self._selection_completed = False
        self.set_visible(False)
        self.update()

    @property
    def artists(self):
        """Tuple of the artists of the selector."""
        handles_artists = getattr(self, '_handles_artists', ())
        return (self._selection_artist,) + handles_artists

    def set_props(self, **props):
        """
        Set the properties of the selector artist. See the `props` argument
        in the selector docstring to know which properties are supported.
        """
        artist = self._selection_artist
        props = cbook.normalize_kwargs(props, artist._alias_map)
        artist.set(**props)
        if self.useblit:
            self.update()
        self._props.update(props)

    def set_handle_props(self, **handle_props):
        """
        Set the properties of the handles selector artist. See the
        `handle_props` argument in the selector docstring to know which
        properties are supported.
        """
        if not hasattr(self, '_handles_artists'):
            raise NotImplementedError("This selector doesn't have handles.")

        artist = self._handles_artists[0]
        handle_props = cbook.normalize_kwargs(handle_props, artist._alias_map)
        for handle in self._handles_artists:
            handle.set(**handle_props)
        if self.useblit:
            self.update()
        self._handle_props.update(handle_props)


class SpanSelector(_SelectorWidget):
    """
    Visually select a min/max range on a single axis and call a function with
    those values.

    To guarantee that the selector remains responsive, keep a reference to it.

    In order to turn off the SpanSelector, set ``span_selector.active`` to
    False.  To turn it back on, set it to True.

    Press and release events triggered at the same coordinates outside the
    selection will clear the selector, except when
    ``ignore_event_outside=True``.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`

    onselect : callable
        A callback function that is called after a release event and the
        selection is created, changed or removed.
        It must have the signature::

            def on_select(min: float, max: float) -> Any

    direction : {"horizontal", "vertical"}
        The direction along which to draw the span selector.

    minspan : float, default: 0
        If selection is less than or equal to *minspan*, the selection is
        removed (when already existing) or cancelled.

    useblit : bool, default: False
        If True, use the backend-dependent blitting features for faster
        canvas updates.

    props : dict, optional
        Dictionary of `matplotlib.patches.Patch` properties.
        Default:

            ``dict(facecolor='red', alpha=0.5)``

    onmove_callback : func(min, max), min/max are floats, default: None
        Called on mouse move while the span is being selected.

    span_stays : bool, default: False
        If True, the span stays visible after the mouse is released.
        Deprecated, use *interactive* instead.

    interactive : bool, default: False
        Whether to draw a set of handles that allow interaction with the
        widget after it is drawn.

    button : `.MouseButton` or list of `.MouseButton`, default: all buttons
        The mouse buttons which activate the span selector.

    handle_props : dict, default: None
        Properties of the handle lines at the edges of the span. Only used
        when *interactive* is True. See `matplotlib.lines.Line2D` for valid
        properties.

    grab_range : float, default: 10
        Distance in pixels within which the interactive tool handles can be
        activated.

    drag_from_anywhere : bool, default: False
        If `True`, the widget can be moved by clicking anywhere within
        its bounds.

    ignore_event_outside : bool, default: False
        If `True`, the event triggered outside the span selector will be
        ignored.

    snap_values : 1D array-like, default: None
        Snap the extents of the selector to the values defined in
        ``snap_values``.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import matplotlib.widgets as mwidgets
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [10, 50, 100])
    >>> def onselect(vmin, vmax):
    ...     print(vmin, vmax)
    >>> span = mwidgets.SpanSelector(ax, onselect, 'horizontal',
    ...                              props=dict(facecolor='blue', alpha=0.5))
    >>> fig.show()

    See also: :doc:`/gallery/widgets/span_selector`
    """

    def __init__(self, ax, onselect, direction, minspan=0, useblit=False,
                 props=None, onmove_callback=None, interactive=False,
                 button=None, handle_props=None, grab_range=10,
                 drag_from_anywhere=False, ignore_event_outside=False,
                 snap_values=None):

        super().__init__(ax, onselect, useblit=useblit, button=button)

        if props is None:
            props = dict(facecolor='red', alpha=0.5)

        props['animated'] = self.useblit

        self.direction = direction

        self.visible = True
        self._extents_on_press = None

        # self._pressv is deprecated and we don't use it internally anymore
        # but we maintain it until it is removed
        self._pressv = None

        self._props = props
        self.onmove_callback = onmove_callback
        self.minspan = minspan

        self.grab_range = grab_range
        self._interactive = interactive
        self._edge_handles = None
        self.drag_from_anywhere = drag_from_anywhere
        self.ignore_event_outside = ignore_event_outside
        self.snap_values = snap_values

        # Reset canvas so that `new_axes` connects events.
        self.canvas = None
        self.new_axes(ax)

        # Setup handles
        if handle_props is None:
            handle_props = {}
        self._handle_props = {
            'color': props.get('facecolor', 'r'),
            **cbook.normalize_kwargs(handle_props, Line2D._alias_map)}

        if self._interactive:
            self._edge_order = ['min', 'max']
            self._setup_edge_handles(self._handle_props)

        self._active_handle = None

        # prev attribute is deprecated but we still need to maintain it
        self._prev = (0, 0)

    def new_axes(self, ax):
        """Set SpanSelector to operate on a new Axes."""
        self.ax = ax
        if self.canvas is not ax.figure.canvas:
            if self.canvas is not None:
                self.disconnect_events()

            self.canvas = ax.figure.canvas
            self.connect_default_events()

        # Reset
        self._selection_completed = False

        if self.direction == 'horizontal':
            trans = ax.get_xaxis_transform()
            w, h = 0, 1
        else:
            trans = ax.get_yaxis_transform()
            w, h = 1, 0
        rect_artist = Rectangle((0, 0), w, h,
                                transform=trans,
                                visible=False,
                                **self._props)

        self.ax.add_patch(rect_artist)
        self._selection_artist = rect_artist

    def _setup_edge_handles(self, props):
        # Define initial position using the axis bounds to keep the same bounds
        if self.direction == 'horizontal':
            positions = self.ax.get_xbound()
        else:
            positions = self.ax.get_ybound()
        self._edge_handles = ToolLineHandles(self.ax, positions,
                                             direction=self.direction,
                                             line_props=props,
                                             useblit=self.useblit)

    @property
    def _handles_artists(self):
        if self._edge_handles is not None:
            return self._edge_handles.artists
        else:
            return ()

    def _press(self, event):
        """Button press event handler."""
        if self._interactive and self._selection_artist.get_visible():
            self._set_active_handle(event)
        else:
            self._active_handle = None

        if self._active_handle is None or not self._interactive:
            # Clear previous rectangle before drawing new rectangle.
            self.update()

        v = event.xdata if self.direction == 'horizontal' else event.ydata
        # self._pressv and self._prev are deprecated but we still need to
        # maintain them
        self._pressv = v
        self._prev = self._get_data(event)

        if self._active_handle is None and not self.ignore_event_outside:
            # when the press event outside the span, we initially set the
            # visibility to False and extents to (v, v)
            # update will be called when setting the extents
            self.visible = False
            self._set_extents((v, v))
            # We need to set the visibility back, so the span selector will be
            # drawn when necessary (span width > 0)
            self.visible = True
        else:
            self.set_visible(True)

        return False

    @property
    def direction(self):
        """Direction of the span selector: 'vertical' or 'horizontal'."""
        return self._direction

    @direction.setter
    def direction(self, direction):
        """Set the direction of the span selector."""
        if hasattr(self, '_direction') and direction != self._direction:
            # remove previous artists
            self._selection_artist.remove()
            if self._interactive:
                self._edge_handles.remove()
            self._direction = direction
            self.new_axes(self.ax)
            if self._interactive:
                self._setup_edge_handles(self._handle_props)
        else:
            self._direction = direction

    def _release(self, event):
        """Button release event handler."""
        # self._pressv is deprecated but we still need to maintain it
        self._pressv = None

        if not self._interactive:
            self._selection_artist.set_visible(False)

        if (self._active_handle is None and self._selection_completed and
                self.ignore_event_outside):
            return

        vmin, vmax = self.extents
        span = vmax - vmin

        if span <= self.minspan:
            # Remove span and set self._selection_completed = False
            self.set_visible(False)
            if self._selection_completed:
                # Call onselect, only when the span is already existing
                self.onselect(vmin, vmax)
            self._selection_completed = False
        else:
            self.onselect(vmin, vmax)
            self._selection_completed = True

        self.update()

        self._active_handle = None

        return False

    def _onmove(self, event):
        """Motion notify event handler."""

        # self._prev are deprecated but we still need to maintain it
        self._prev = self._get_data(event)

        v = event.xdata if self.direction == 'horizontal' else event.ydata
        if self.direction == 'horizontal':
            vpress = self._eventpress.xdata
        else:
            vpress = self._eventpress.ydata

        # move existing span
        # When "dragging from anywhere", `self._active_handle` is set to 'C'
        # (match notation used in the RectangleSelector)
        if self._active_handle == 'C' and self._extents_on_press is not None:
            vmin, vmax = self._extents_on_press
            dv = v - vpress
            vmin += dv
            vmax += dv

        # resize an existing shape
        elif self._active_handle and self._active_handle != 'C':
            vmin, vmax = self._extents_on_press
            if self._active_handle == 'min':
                vmin = v
            else:
                vmax = v
        # new shape
        else:
            # Don't create a new span if there is already one when
            # ignore_event_outside=True
            if self.ignore_event_outside and self._selection_completed:
                return
            vmin, vmax = vpress, v
            if vmin > vmax:
                vmin, vmax = vmax, vmin

        self._set_extents((vmin, vmax))

        if self.onmove_callback is not None:
            self.onmove_callback(vmin, vmax)

        return False

    def _draw_shape(self, vmin, vmax):
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        if self.direction == 'horizontal':
            self._selection_artist.set_x(vmin)
            self._selection_artist.set_width(vmax - vmin)
        else:
            self._selection_artist.set_y(vmin)
            self._selection_artist.set_height(vmax - vmin)

    def _set_active_handle(self, event):
        """Set active handle based on the location of the mouse event."""
        # Note: event.xdata/ydata in data coordinates, event.x/y in pixels
        e_idx, e_dist = self._edge_handles.closest(event.x, event.y)

        # Prioritise center handle over other handles
        # Use 'C' to match the notation used in the RectangleSelector
        if 'move' in self._state:
            self._active_handle = 'C'
        elif e_dist > self.grab_range:
            # Not close to any handles
            self._active_handle = None
            if self.drag_from_anywhere and self._contains(event):
                # Check if we've clicked inside the region
                self._active_handle = 'C'
                self._extents_on_press = self.extents
            else:
                self._active_handle = None
                return
        else:
            # Closest to an edge handle
            self._active_handle = self._edge_order[e_idx]

        # Save coordinates of rectangle at the start of handle movement.
        self._extents_on_press = self.extents

    def _contains(self, event):
        """Return True if event is within the patch."""
        return self._selection_artist.contains(event, radius=0)[0]

    @staticmethod
    def _snap(values, snap_values):
        """Snap values to a given array values (snap_values)."""
        indices = np.empty_like(values, dtype="uint")
        # take into account machine precision
        eps = np.min(np.abs(np.diff(snap_values))) * 1e-12
        for i, v in enumerate(values):
            indices[i] = np.abs(snap_values - v + np.sign(v) * eps).argmin()
        return snap_values[indices]

    @property
    def extents(self):
        """Return extents of the span selector."""
        if self.direction == 'horizontal':
            vmin = self._selection_artist.get_x()
            vmax = vmin + self._selection_artist.get_width()
        else:
            vmin = self._selection_artist.get_y()
            vmax = vmin + self._selection_artist.get_height()
        return vmin, vmax

    @extents.setter
    def extents(self, extents):
        self._set_extents(extents)
        self._selection_completed = True

    def _set_extents(self, extents):
        # Update displayed shape
        if self.snap_values is not None:
            extents = tuple(self._snap(extents, self.snap_values))
        self._draw_shape(*extents)
        if self._interactive:
            # Update displayed handles
            self._edge_handles.set_data(self.extents)
        self.set_visible(self.visible)
        self.update()


class ToolLineHandles:
    """
    Control handles for canvas tools.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Matplotlib axes where tool handles are displayed.
    positions : 1D array
        Positions of handles in data coordinates.
    direction : {"horizontal", "vertical"}
        Direction of handles, either 'vertical' or 'horizontal'
    line_props : dict, optional
        Additional line properties. See `matplotlib.lines.Line2D`.
    useblit : bool, default: True
        Whether to use blitting for faster drawing (if supported by the
        backend).
    """

    def __init__(self, ax, positions, direction, line_props=None,
                 useblit=True):
        self.ax = ax
        self._direction = direction

        if line_props is None:
            line_props = {}
        line_props.update({'visible': False, 'animated': useblit})

        line_fun = ax.axvline if self.direction == 'horizontal' else ax.axhline

        self._artists = [line_fun(p, **line_props) for p in positions]

    @property
    def artists(self):
        return tuple(self._artists)

    @property
    def positions(self):
        """Positions of the handle in data coordinates."""
        method = 'get_xdata' if self.direction == 'horizontal' else 'get_ydata'
        return [getattr(line, method)()[0] for line in self.artists]

    @property
    def direction(self):
        """Direction of the handle: 'vertical' or 'horizontal'."""
        return self._direction

    def set_data(self, positions):
        """
        Set x or y positions of handles, depending if the lines are vertical
        of horizontal.

        Parameters
        ----------
        positions : tuple of length 2
            Set the positions of the handle in data coordinates
        """
        method = 'set_xdata' if self.direction == 'horizontal' else 'set_ydata'
        for line, p in zip(self.artists, positions):
            getattr(line, method)([p, p])

    def set_visible(self, value):
        """Set the visibility state of the handles artist."""
        for artist in self.artists:
            artist.set_visible(value)

    def set_animated(self, value):
        """Set the animated state of the handles artist."""
        for artist in self.artists:
            artist.set_animated(value)

    def remove(self):
        """Remove the handles artist from the figure."""
        for artist in self._artists:
            artist.remove()

    def closest(self, x, y):
        """
        Return index and pixel distance to closest handle.

        Parameters
        ----------
        x, y : float
            x, y position from which the distance will be calculated to
            determinate the closest handle

        Returns
        -------
        index, distance : index of the handle and its distance from
            position x, y
        """
        if self.direction == 'horizontal':
            p_pts = np.array([
                self.ax.transData.transform((p, 0))[0] for p in self.positions
                ])
            dist = abs(p_pts - x)
        else:
            p_pts = np.array([
                self.ax.transData.transform((0, p))[1] for p in self.positions
                ])
            dist = abs(p_pts - y)
        index = np.argmin(dist)
        return index, dist[index]
