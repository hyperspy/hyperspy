import ipywidgets
from traits.api import Undefined


FORM_ITEM_LAYOUT = ipywidgets.Layout(
    display='flex',
    flex_flow='row',
    justify_content='space-between',
)


def labelme(label, widget):
    if label is Undefined:
        label = ""
    if not isinstance(label, ipywidgets.Label):
        label = ipywidgets.Label(label)
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
