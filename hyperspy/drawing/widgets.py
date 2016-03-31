"""Interactive widgets that can be added to `Signal` plots.

Example
-------

"""


from hyperspy.drawing.widget import (
    WidgetBase, DraggableWidgetBase, ResizableDraggableWidgetBase,
    Widget2DBase, Widget1DBase, ResizersMixin)
from hyperspy.drawing._widgets.horizontal_line import HorizontalLineWidget
from hyperspy.drawing._widgets.vertical_line import VerticalLineWidget
from hyperspy.drawing._widgets.label import LabelWidget
from hyperspy.drawing._widgets.scalebar import ScaleBar
from hyperspy.drawing._widgets.circle import CircleWidget
from hyperspy.drawing._widgets.rectangles import RectangleWidget, SquareWidget
from hyperspy.drawing._widgets.range import ModifiableSpanSelector, RangeWidget
from hyperspy.drawing._widgets.line2d import Line2DWidget
