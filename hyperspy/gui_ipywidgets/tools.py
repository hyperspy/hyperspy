import ipywidgets
from IPython.display import display

from hyperspy.gui_ipywidgets.utils import labelme
from hyperspy.misc.link_traits import link_traits


def interactive_range_ipy(range_selector):
    # Define widgets
    axis = range_selector.axis
    left = ipywidgets.FloatText(disabled=True)
    right = ipywidgets.FloatText(disabled=True)
    units = ipywidgets.Label()
    help = ipywidgets.Label(
        "Click on the signal figure and drag to the left to select a signal "
        "range. Press `Apply` to perform the operation or `Close` to cancel.")
    help = ipywidgets.Accordion(children=[help])
    help.set_title(0, "Help")
    close = ipywidgets.Button(
        description="Close",
        tooltip="Close widget and remove span selector from the signal figure.")
    apply = ipywidgets.Button(
        description="Apply",
        tooltip="Perform the operation using the selected range.")

    # Connect
    link_traits((range_selector, "ss_left_value"), (left, "value"))
    link_traits((range_selector, "ss_right_value"), (right, "value"))
    link_traits((axis, "units"), (units, "value"))

    def on_apply_clicked(b):
        obj = range_selector
        if obj.ss_left_value != obj.ss_right_value:
            obj.span_selector_switch(False)
            for method, cls in obj.on_close:
                method(cls, obj.ss_left_value, obj.ss_right_value)
            obj.span_selector_switch(True)
    apply.on_click(on_apply_clicked)

    box = ipywidgets.VBox([
        ipywidgets.HBox([left, units, ipywidgets.Label("-"), right, units]),
        help,
        ipywidgets.HBox((apply, close))
    ])
    display(box)

    def on_close_clicked(b):
        range_selector.span_selector_switch(False)
        box.close()
    close.on_click(on_close_clicked)
