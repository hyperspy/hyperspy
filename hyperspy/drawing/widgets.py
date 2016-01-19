"""Interactive widgets that can be added to `Signal` plots.

Example
-------

"""


from hyperspy.drawing.widget import (
    WidgetBase, DraggableWidgetBase, ResizableDraggableWidgetBase,
    Widget2DBase, ResizersMixin)
from hyperspy.drawing._widgets.horizontal_line \
    import DraggableHorizontalLine as HorizontalLine
from hyperspy.drawing._widgets.vertical_line \
    import DraggableVerticalLine as VerticalLine
from hyperspy.drawing._widgets.label \
    import DraggableLabel as Label
from hyperspy.drawing._widgets.scalebar import ScaleBar
from hyperspy.drawing._widgets.circle \
    import Draggable2DCircle as Circle
from hyperspy.drawing._widgets.rectangles \
    import ResizableDraggableRectangle as Rectangle, DraggableSquare as Square
from hyperspy.drawing._widgets.range \
    import ModifiableSpanSelector, DraggableResizableRange as Range

