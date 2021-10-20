"""Markers that can be added to `Signal` plots.

Example
-------

>>> import scipy.misc
>>> im = hs.signals.Signal2D(scipy.misc.ascent())
>>> m = hs.plot.markers.rectangle(x1=150, y1=100, x2=400, y2=400, color='red')
>>> im.add_marker(m)

"""

from hyperspy.drawing._markers.horizontal_line \
    import HorizontalLine as horizontal_line
from hyperspy.drawing._markers.vertical_line \
    import VerticalLine as vertical_line
from hyperspy.drawing._markers.vertical_line_segment \
    import VerticalLineSegment as vertical_line_segment
from hyperspy.drawing._markers.line_segment import LineSegment as line_segment
from hyperspy.drawing._markers.horizontal_line_segment \
    import HorizontalLineSegment as horizontal_line_segment
from hyperspy.drawing._markers.rectangle import Rectangle as rectangle
from hyperspy.drawing._markers.point import Point as point
from hyperspy.drawing._markers.text import Text as text
