import ipywidgets
from IPython.display import display

from hyperspy.gui_ipywidgets.utils import labelme
from hyperspy.misc.link_traits import link_traits


def ipy_navigation_sliders(axes):
    widgets = []
    for axis in axes:
        widget = ipywidgets.IntSlider(
            min=axis.low_value,
            max=axis.high_value,
            readout=True,
        )
        link_traits((axis, "index"), (widget, "value"))
        widgets.append(labelme(str(axis).replace(" ", "_"), widget))
    box = ipywidgets.VBox(widgets)
    display(box)
