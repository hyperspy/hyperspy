import ipywidgets

from hyperspy.gui_ipywidgets.utils import (
    labelme, register_ipy_widget, add_display_arg)
from hyperspy.misc.link_traits import link_traits


@register_ipy_widget(toolkey="navigation_sliders")
@add_display_arg
def ipy_navigation_sliders(obj, **kwargs):
    widgets = []
    for axis in obj:
        iwidget = ipywidgets.IntSlider(
            min=0,
            max=axis.size - 1,
            readout=True,
        )
        link_traits((axis, "index"), (iwidget, "value"))
        vwidget = ipywidgets.FloatSlider(
            min=axis.low_value,
            max=axis.high_value,
            # readout_format=".lf"
        )
        link_traits((axis, "value"), (vwidget, "value"))
        link_traits((axis, "high_value"), (vwidget, "max"))
        link_traits((axis, "low_value"), (vwidget, "min"))
        bothw = ipywidgets.VBox([iwidget, vwidget])
        labeled_widget = labelme(str(axis).replace(" ", "_"), bothw)
        link_traits((axis, "name"), (labeled_widget.children[0], "value"))
        widgets.append(labeled_widget)
    box = ipywidgets.VBox(widgets)
    return box


def _get_axis_widgets(axis):
    widgets = []

    widget = ipywidgets.Text()
    widgets.append(labelme(str(axis).replace(" ", "_"), widget))
    link_traits((axis, "name"), (widget, "value"))

    widget = ipywidgets.IntText(disabled=True)
    widgets.append(labelme("Size", widget))
    link_traits((axis, "size"), (widget, "value"))

    widget = ipywidgets.IntText(disabled=True)
    widgets.append(labelme("Index in array", widget))
    link_traits((axis, "index_in_array"), (widget, "value"))

    widget = ipywidgets.IntSlider(min=0, max=axis.size - 1)
    widgets.append(labelme("Index", widget))
    link_traits((axis, "index"), (widget, "value"))

    widget = ipywidgets.FloatSlider(
        min=axis.low_value,
        max=axis.high_value,
    )
    widgets.append(labelme("Value", widget))
    link_traits((axis, "value"), (widget, "value"))
    link_traits((axis, "high_value"), (widget, "max"))
    link_traits((axis, "low_value"), (widget, "min"))

    widget = ipywidgets.Text()
    widgets.append(labelme("Units", widget))
    link_traits((axis, "units"), (widget, "value"))

    widget = ipywidgets.Checkbox(disabled=True)
    widgets.append(labelme("Navigate", widget))
    link_traits((axis, "navigate"), (widget, "value"))

    widget = ipywidgets.FloatText()
    widgets.append(labelme("Scale", widget))
    link_traits((axis, "scale"), (widget, "value"))

    widget = ipywidgets.FloatText()
    widgets.append(labelme("Offset", widget))
    link_traits((axis, "offset"), (widget, "value"))

    return widgets


@register_ipy_widget(toolkey="AxesManager")
@add_display_arg
def ipy_axes_gui(obj, **kwargs):
    nav_widgets = [ipywidgets.VBox(_get_axis_widgets(axis))
                   for axis in obj.navigation_axes]
    sig_widgets = [ipywidgets.VBox(_get_axis_widgets(axis))
                   for axis in obj.signal_axes]
    nav_accordion = ipywidgets.Accordion(nav_widgets)
    sig_accordion = ipywidgets.Accordion(sig_widgets)
    i = 0  # For when there is not navigation axes
    for i in range(obj.navigation_dimension):
        nav_accordion.set_title(i, "Axis %i" % i)
    for j in range(obj.signal_dimension):
        sig_accordion.set_title(j, "Axis %i" % (i + j + 1))
    tabs = ipywidgets.HBox([nav_accordion, sig_accordion])
    return tabs
