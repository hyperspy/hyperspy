import textwrap


class BlittedFigure(object):

    def _on_draw(self, *args):
        canvas = self.figure.canvas
        self._background = canvas.copy_from_bbox(self.figure.bbox)
        self._draw_animated()

    def _draw_animated(self):
        canvas = self.ax.figure.canvas
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
            [ax.draw_artist(a) for a in artists if
             a.get_animated() is True]
        canvas.blit(self.figure.bbox)

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, value):
        # Wrap the title so that each line is not longer than 60 characters.
        self._title = textwrap.fill(value, 60)
