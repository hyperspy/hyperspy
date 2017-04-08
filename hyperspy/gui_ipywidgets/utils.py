import functools

import ipywidgets
from traits.api import Undefined
import IPython.display

from hyperspy.ui_registry import register_widget


register_ipy_widget = functools.partial(register_widget, toolkit="ipywidgets")

FORM_ITEM_LAYOUT = ipywidgets.Layout(
    display='flex',
    flex_flow='row',
    justify_content='space-between',
)


def labelme(label, widget):
    if label is Undefined:
        label = ""
    if not isinstance(label, ipywidgets.Label):
        label = ipywidgets.Label(label,
                                 layout=ipywidgets.Layout(width="auto"))
    return ipywidgets.HBox(
        [label, widget],
        layout=FORM_ITEM_LAYOUT,
    )


def labelme_sandwich(label1, widget, label2):
    if label1 is Undefined:
        label1 = ""
    if label2 is Undefined:
        label2 = ""
    if not isinstance(label1, ipywidgets.Label):
        label1 = ipywidgets.Label(label1)
    if not isinstance(label2, ipywidgets.Label):
        label2 = ipywidgets.Label(label2)
    return ipywidgets.HBox(
        [label1, widget, label2],
        layout=FORM_ITEM_LAYOUT)


def enum2dropdown(trait):
    tooltip = trait.desc if trait.desc else ""
    widget = ipywidgets.Dropdown(
        options=trait.trait_type.values,
        tooltip=tooltip,)
    return widget


def add_display_arg(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        display = kwargs.pop("display", True)
        widget = f(*args, **kwargs)
        if display:
            IPython.display.display(widget)
        else:
            return widget
    return wrapper

