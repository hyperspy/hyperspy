import ipywidgets
import traitlets

from hyperspy.gui_ipywidgets.utils import (
    labelme, labelme_sandwich, enum2dropdown, add_display_arg,
    register_ipy_widget)
from hyperspy.link_traits.link_traits import link_bidirectional


@register_ipy_widget(toolkey="SpanROI")
@add_display_arg
def span_roi_ipy(obj, **kwargs):
    wdict = {}
    left = ipywidgets.FloatText(description="Left")
    right = ipywidgets.FloatText(description="Right")
    link_bidirectional((obj, "left"), (left, "value"))
    link_bidirectional((obj, "right"), (right, "value"))
    wdict["left"] = left
    wdict["right"] = right
    container = ipywidgets.HBox([left, right])
    return {
        "widget": container,
        "wdict": wdict,
    }


@register_ipy_widget(toolkey="Point1DROI")
@add_display_arg
def point1d_roi_ipy(obj, **kwargs):
    wdict = {}
    value = ipywidgets.FloatText(description="value")
    wdict["value"] = value
    link_bidirectional((obj, "value"), (value, "value"))
    return {
        "widget": value,
        "wdict": wdict,
    }


@register_ipy_widget(toolkey="Point2DROI")
@add_display_arg
def point_2d_ipy(obj, **kwargs):
    wdict = {}
    x = ipywidgets.FloatText(description="x")
    y = ipywidgets.FloatText(description="y")
    wdict["x"] = x
    wdict["y"] = y
    link_bidirectional((obj, "x"), (x, "value"))
    link_bidirectional((obj, "y"), (y, "value"))
    container = ipywidgets.HBox([x, y])
    return {
        "widget": container,
        "wdict": wdict,
    }


@register_ipy_widget(toolkey="RectangularROI")
@add_display_arg
def rectangular_roi_ipy(obj, **kwargs):
    wdict = {}
    left = ipywidgets.FloatText(description="left")
    right = ipywidgets.FloatText(description="right")
    link_bidirectional((obj, "left"), (left, "value"))
    link_bidirectional((obj, "right"), (right, "value"))
    container1 = ipywidgets.HBox([left, right])
    top = ipywidgets.FloatText(description="top")
    bottom = ipywidgets.FloatText(description="bottom")
    link_bidirectional((obj, "top"), (top, "value"))
    link_bidirectional((obj, "bottom"), (bottom, "value"))
    container2 = ipywidgets.HBox([top, bottom])
    container = ipywidgets.VBox([container1, container2])
    wdict["left"] = left
    wdict["right"] = right
    wdict["top"] = top
    wdict["bottom"] = bottom
    return {
        "widget": container,
        "wdict": wdict,
    }


@register_ipy_widget(toolkey="CircleROI")
@add_display_arg
def circle_roi_ipy(obj, **kwargs):
    wdict = {}
    x = ipywidgets.FloatText(description="x")
    y = ipywidgets.FloatText(description="y")
    link_bidirectional((obj, "cx"), (x, "value"))
    link_bidirectional((obj, "cy"), (y, "value"))
    container1 = ipywidgets.HBox([x, y])
    radius = ipywidgets.FloatText(description="radius")
    inner_radius = ipywidgets.FloatText(description="inner_radius")
    link_bidirectional((obj, "r"), (radius, "value"))
    link_bidirectional((obj, "r_inner"), (inner_radius, "value"))
    container2 = ipywidgets.HBox([radius, inner_radius])
    container = ipywidgets.VBox([container1, container2])
    wdict["cx"] = x
    wdict["cy"] = y
    wdict["radius"] = radius
    wdict["inner_radius"] = inner_radius
    return {
        "widget": container,
        "wdict": wdict,
    }


@register_ipy_widget(toolkey="Line2DROI")
@add_display_arg
def line2d_roi_ipy(obj, **kwargs):
    wdict = {}
    x1 = ipywidgets.FloatText(description="x1")
    y1 = ipywidgets.FloatText(description="x2")
    link_bidirectional((obj, "x1"), (x1, "value"))
    link_bidirectional((obj, "y1"), (y1, "value"))
    container1 = ipywidgets.HBox([x1, y1])
    x2 = ipywidgets.FloatText(description="x2")
    y2 = ipywidgets.FloatText(description="y2")
    link_bidirectional((obj, "x2"), (x2, "value"))
    link_bidirectional((obj, "y2"), (y2, "value"))
    container2 = ipywidgets.HBox([x2, y2])
    linewidth = ipywidgets.FloatText(description="linewidth")
    link_bidirectional((obj, "linewidth"), (linewidth, "value"))
    container = ipywidgets.VBox([container1, container2, linewidth])
    wdict["x1"] = x1
    wdict["x2"] = x2
    wdict["y1"] = y1
    wdict["y2"] = y2
    wdict["linewidth"] = linewidth
    return {
        "widget": container,
        "wdict": wdict,
    }
