import textwrap
import matplotlib.pyplot as plt

from hyperspy.events import Event, Events


class BlittedFigure(object):

    def __init__(self):
        self._background = None
        self.events = Events()
        self.events.closed = Event("""
            Event that triggers when the figure window is closed.

            Arguments:
                obj:  SpectrumFigure instances
                    The instance that triggered the event.
            """, arguments=["obj"])

    def _set_background(self):
        if self.figure:
            canvas = self.figure.canvas
            if canvas.supports_blit:
                self._background = canvas.copy_from_bbox(self.figure.bbox)

    def _on_draw(self, *args):
        if self.figure:
            canvas = self.figure.canvas
            if canvas.supports_blit:
                self._set_background()
                self._draw_animated()
            else:
                canvas.draw_idle()

    def _draw_animated(self):
        if self.ax.figure and self.figure.axes:
            canvas = self.ax.figure.canvas
            if canvas.supports_blit:
                canvas.restore_region(self._background)
            for ax in self.figure.axes:
                artists = []
                artists.extend(ax.images)
                artists.extend(ax.collections)
                artists.extend(ax.patches)
                artists.extend(ax.lines)
                artists.extend(ax.texts)
                artists.extend(ax.artists)
                artists.append(ax.get_yaxis())
                artists.append(ax.get_xaxis())
                try:
                    [ax.draw_artist(a) for a in artists if
                     a.get_animated() is True]
                except AttributeError:
                     # The method was called before draw. This is a quick
                     # fix. Properly fixing the issue involves avoiding
                     # calling this method too early in the code.
                    pass
            if canvas.supports_blit:
                canvas.blit(self.figure.bbox)

    def add_marker(self, marker):
        marker.ax = self.ax
        if marker.axes_manager is None:
            marker.axes_manager = self.axes_manager
        self.ax_markers.append(marker)
        marker.events.closed.connect(lambda obj: self.ax_markers.remove(obj))

    def _on_close(self):
        if self.figure is None:
            return  # Already closed
        for marker in self.ax_markers:
            marker.close(update_plot=False)
        self.events.closed.trigger(obj=self)
        for f in self.events.closed.connected:
            self.events.closed.disconnect(f)
        self.figure = None

    def close(self):
        figure = self.figure
        self._on_close()   # Needs to trigger serially for a well defined state
        plt.close(figure)

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, value):
        # Wrap the title so that each line is not longer than 60 characters.
        self._title = textwrap.fill(value, 60)
