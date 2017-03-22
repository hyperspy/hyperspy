import ipywidgets
from IPython.display import display

from hyperspy.gui_ipywidgets.utils import labelme
from hyperspy.misc.link_traits import link_traits


def ipy_navigation_sliders(axes):
    widgets = []
    for axis in axes:
        widget = ipywidgets.IntSlider(
            min=0,
            max=axis.size - 1,
            readout=True,
        )
        link_traits((axis, "index"), (widget, "value"))
        widgets.append(labelme(str(axis).replace(" ", "_"), widget))
    box = ipywidgets.VBox(widgets)
    display(box)

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

def ipy_axes_gui(axes_manager):
    nav_widgets = [ipywidgets.VBox(_get_axis_widgets(axis))
                   for axis in axes_manager.navigation_axes]
    sig_widgets = [ipywidgets.VBox(_get_axis_widgets(axis))
                   for axis in axes_manager.signal_axes]
    nav_accordion = ipywidgets.Accordion(nav_widgets)
    sig_accordion = ipywidgets.Accordion(sig_widgets)
    for i in range(axes_manager.navigation_dimension):
        nav_accordion.set_title(i, "Axis %i" % i)
    for j in range(axes_manager.signal_dimension):
        sig_accordion.set_title(j, "Axis %i" % (i + j + 1))
    tabs = ipywidgets.HBox([nav_accordion, sig_accordion])
    display(tabs)
