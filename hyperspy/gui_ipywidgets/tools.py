import ipywidgets
from IPython.display import display
import traitlets

from hyperspy.gui_ipywidgets.utils import (
    labelme, labelme_sandwich, enum2dropdown)
from hyperspy.misc.link_traits import link_traits
from hyperspy.gui_ipywidgets.custom_widgets import OddIntSlider


def interactive_range_ipy(obj):
    # Define widgets
    axis = obj.axis
    left = ipywidgets.FloatText(disabled=True)
    right = ipywidgets.FloatText(disabled=True)
    units = ipywidgets.Label()
    help = ipywidgets.Label(
        "Click on the signal figure and drag to the right to select a signal "
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
    link_traits((obj, "ss_left_value"), (left, "value"))
    link_traits((obj, "ss_right_value"), (right, "value"))
    link_traits((axis, "units"), (units, "value"))

    def on_apply_clicked(b):
        obj = obj
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
        obj.span_selector_switch(False)
        box.close()
    close.on_click(on_close_clicked)


def calibrate_ipy(obj):
    # Define widgets
    axis = obj.axis
    left = ipywidgets.FloatText(disabled=True)
    right = ipywidgets.FloatText(disabled=True)
    offset = ipywidgets.FloatText(disabled=True)
    scale = ipywidgets.FloatText(disabled=True)
    new_left = ipywidgets.FloatText(disabled=False)
    new_right = ipywidgets.FloatText(disabled=False)
    units = ipywidgets.Label()
    help = ipywidgets.Label(
        "Click on the signal figure and drag to the right to select a signal "
        "range. Set the new left and right values and press `Apply` to update "
        "the calibration of the axis with the new values or press "
        " `Close` to cancel.")
    help = ipywidgets.Accordion(children=[help])
    help.set_title(0, "Help")
    close = ipywidgets.Button(
        description="Close",
        tooltip="Close widget and remove span selector from the signal figure.")
    apply = ipywidgets.Button(
        description="Apply",
        tooltip="Set the axis calibration with the `offset` and `scale` values "
        "above.")

    # Connect
    link_traits((obj, "ss_left_value"), (left, "value"))
    link_traits((obj, "ss_right_value"), (right, "value"))
    link_traits((obj, "left_value"), (new_left, "value"))
    link_traits((obj, "right_value"), (new_right, "value"))
    link_traits((axis, "units"), (units, "value"))
    link_traits((axis, "offset"), (offset, "value"))
    link_traits((axis, "scale"), (scale, "value"))

    def on_apply_clicked(b):
        axis.scale = obj.scale
        axis.offset = obj.offset
        obj.span_selector_switch(on=False)
        obj.signal._plot.signal_plot.update()
        obj.span_selector_switch(on=True)
    apply.on_click(on_apply_clicked)

    box = ipywidgets.VBox([
        labelme_sandwich("New left", new_left, units),
        labelme_sandwich("New right", new_right, units),
        labelme_sandwich("Left", left, units),
        labelme_sandwich("Right", right, units),
        labelme_sandwich("Scale", scale, ""),  # No units, but padding
        labelme_sandwich("Offset", offset, units),
        help,
        ipywidgets.HBox((apply, close))
    ])
    display(box)

    def on_close_clicked(b):
        obj.span_selector_switch(False)
        box.close()
    close.on_click(on_close_clicked)


def smooth_savitzky_golay_ipy(obj):
    window_length = OddIntSlider(
        value=3, step=2, min=3, max=max(int(obj.axis.size * 0.25), 3))
    polynomial_order = ipywidgets.IntSlider(value=3, min=1,
                                            max=window_length.value - 1)
    # Polynomial order must be less than window length

    def update_bound(change):
        polynomial_order.max = change.new - 1
    window_length.observe(update_bound, "value")
    differential_order = ipywidgets.IntSlider(value=0, min=0, max=10)
    color = ipywidgets.ColorPicker()
    link_traits((obj, "polynomial_order"), (polynomial_order, "value"))
    link_traits((obj, "window_length"), (window_length, "value"))
    link_traits((obj, "differential_order"), (differential_order, "value"))
    # Differential order must be less or equal to polynomial_order
    link_traits((polynomial_order, "value"), (differential_order, "max"))
    link_traits((obj, "line_color_ipy"), (color, "value"))
    box = ipywidgets.VBox([
        labelme("Window length", window_length),
        labelme("polynomial order", polynomial_order),
        labelme("Differential order", differential_order),
        labelme("Color", color),
    ])
    display(box)


def smooth_lowess_ipy(obj):
    smoothing_parameter = ipywidgets.FloatSlider(min=0, max=1)
    number_of_iterations = ipywidgets.IntText()
    color = ipywidgets.ColorPicker()
    link_traits((obj, "smoothing_parameter"), (smoothing_parameter, "value"))
    link_traits((obj, "number_of_iterations"), (number_of_iterations, "value"))
    link_traits((obj, "line_color_ipy"), (color, "value"))
    box = ipywidgets.VBox([
        labelme("Smoothing parameter", smoothing_parameter),
        labelme("Number of iterations", number_of_iterations),
        labelme("Color", color),
    ])
    display(box)


def smooth_tv_ipy(obj):
    smoothing_parameter = ipywidgets.FloatSlider(min=0.1, max=1000)
    smoothing_parameter_max = ipywidgets.FloatText(
        value=smoothing_parameter.max)
    color = ipywidgets.ColorPicker()
    link_traits((obj, "smoothing_parameter"), (smoothing_parameter, "value"))
    link_traits((smoothing_parameter_max, "value"),
                (smoothing_parameter, "max"))
    link_traits((obj, "line_color_ipy"), (color, "value"))
    box = ipywidgets.VBox([
        labelme("Weight", smoothing_parameter),
        labelme("Weight max", smoothing_parameter_max),
        labelme("Color", color),
    ])
    display(box)


def image_constast_editor_ipy(obj):
    left = ipywidgets.FloatText(disabled=True)
    right = ipywidgets.FloatText(disabled=True)
    help = ipywidgets.Label(
        "Click on the histogram figure and drag to the right to select a"
        "range. Press `Apply` to set the new contrast limits, `Reset` to reset "
        "them or `Close` to cancel.")
    help = ipywidgets.Accordion(children=[help])
    help.set_title(0, "Help")
    close = ipywidgets.Button(
        description="Close",
        tooltip="Close widget and remove span selector from the signal figure.")
    apply = ipywidgets.Button(
        description="Apply",
        tooltip="Perform the operation using the selected range.")
    reset = ipywidgets.Button(
        description="Reset",
        tooltip="Reset the contrast to the previous value.")

    # Connect
    link_traits((obj, "ss_left_value"), (left, "value"))
    link_traits((obj, "ss_right_value"), (right, "value"))

    def on_apply_clicked(b):
        obj.apply()
    apply.on_click(on_apply_clicked)

    def on_reset_clicked(b):
        obj.reset()
    reset.on_click(on_reset_clicked)

    box = ipywidgets.VBox([
        labelme("vmin", left),
        labelme("vmax", right),
        help,
        ipywidgets.HBox((apply, reset, close))
    ])
    display(box)

    def on_close_clicked(b):
        obj.close()
        box.close()
    close.on_click(on_close_clicked)


def fit_component_ipy(obj):
    only_current = ipywidgets.Checkbox()
    help = ipywidgets.Label(
        "Click on the signal figure and drag to the right to select a"
        "range. Press `Fit` to fit the component in that range. If only "
        "current is unchecked the fit is performed in the whole dataset.")
    help = ipywidgets.Accordion(children=[help])
    help.set_title(0, "Help")
    link_traits((obj, "only_current"), (only_current, "value"))
    fit = ipywidgets.Button(
        description="Fit",
        tooltip="Fit in the selected signal range")
    close = ipywidgets.Button(
        description="Close",
        tooltip="Close widget and remove span selector from the signal figure.")

    def on_fit_clicked(b):
        obj._fit_fired()
    fit.on_click(on_fit_clicked)
    box = ipywidgets.VBox([
        labelme("Only current", only_current),
        help,
        ipywidgets.HBox((fit, close))
    ])
    display(box)

    def on_close_clicked(b):
        obj.span_selector_switch(False)
        box.close()
    close.on_click(on_close_clicked)


def remove_background_ipy(obj):
    fast = ipywidgets.Checkbox()
    help = ipywidgets.Label(
        "Click on the signal figure and drag to the right to select a"
        "range. Press `Fit` to fit the component in that range. If only "
        "current is unchecked the fit is performed in the whole dataset.")
    help = ipywidgets.Accordion(children=[help])
    help.set_title(0, "Help")
    close = ipywidgets.Button(
        description="Close",
        tooltip="Close widget and remove span selector from the signal figure.")
    apply = ipywidgets.Button(
        description="Apply",
        tooltip="Remove the background in the whole dataset.")

    def on_apply_clicked(b):
        obj.apply()
    apply.on_click(on_apply_clicked)
    polynomial_order = ipywidgets.IntSlider(min=1, max=10)
    labeled_polyorder = labelme("Polynomial order", polynomial_order)
    background_type = enum2dropdown(obj.traits()["background_type"])

    def enable_poly_order(change):
        if change.new == "Polynomial":
            for child in labeled_polyorder.children:
                child.layout.display = ""
        else:
            for child in labeled_polyorder.children:
                child.layout.display = "none"
    background_type.observe(enable_poly_order, "value")
    link_traits((obj, "background_type"), (background_type, "value"))
    # Trigger the function that controls the visibility of poly order as
    # setting the default value doesn't trigger it.

    class Dummy:
        new = background_type.value
    enable_poly_order(change=Dummy())
    link_traits((obj, "polynomial_order"),
                (background_type, "polynomial_order"))
    link_traits((obj, "fast"), (fast, "value"))
    box = ipywidgets.VBox([
        labelme("Background type", background_type),
        labeled_polyorder,
        labelme("Fast", fast),
        help,
        ipywidgets.HBox((apply, close)),
    ])
    display(box)

    def on_close_clicked(b):
        obj.span_selector_switch(False)
        box.close()
    close.on_click(on_close_clicked)
