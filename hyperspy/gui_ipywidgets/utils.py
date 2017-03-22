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
    return ipywidgets.HBox(
        [ipywidgets.Label(label), widget], layout=FORM_ITEM_LAYOUT)
