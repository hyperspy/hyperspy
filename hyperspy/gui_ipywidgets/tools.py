import ipywidgets
import traitlets

from hyperspy.gui_ipywidgets.utils import (
    labelme, labelme_sandwich, enum2dropdown, add_display_arg,
    register_ipy_widget)
from hyperspy.link_traits.link_traits import link
from hyperspy.gui_ipywidgets.custom_widgets import OddIntSlider
from hyperspy.signal_tools import SPIKES_REMOVAL_INSTRUCTIONS


@register_ipy_widget(toolkey="interactive_range_selector")
@add_display_arg
def interactive_range_ipy(obj, **kwargs):
    # Define widgets
    wdict = {}
    axis = obj.axis
    left = ipywidgets.FloatText(disabled=True)
    right = ipywidgets.FloatText(disabled=True)
    units = ipywidgets.Label()
    help = ipywidgets.HTML(
        "Click on the signal figure and drag to the right to select a signal "
        "range. Press `Apply` to perform the operation or `Close` to cancel.",)
    help = ipywidgets.Accordion(children=[help])
    help.set_title(0, "Help")
    close = ipywidgets.Button(
        description="Close",
        tooltip="Close widget and remove span selector from the signal figure.")
    apply = ipywidgets.Button(
        description="Apply",
        tooltip="Perform the operation using the selected range.")
    wdict["left"] = left
    wdict["right"] = right
    wdict["units"] = units
    wdict["help"] = help
    wdict["close_button"] = close
    wdict["apply_button"] = apply

    # Connect
    link((obj, "ss_left_value"), (left, "value"))
    link((obj, "ss_right_value"), (right, "value"))
    link((axis, "units"), (units, "value"))

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

    def on_close_clicked(b):
        obj.span_selector_switch(False)
        box.close()
    close.on_click(on_close_clicked)
    return {
        "widget": box,
        "wdict": wdict,
    }


@register_ipy_widget(toolkey="Signal1D.calibrate")
@add_display_arg
def calibrate_ipy(obj, **kwargs):
    # Define widgets
    wdict = {}
    axis = obj.axis
    left = ipywidgets.FloatText(disabled=True, description="Left")
    right = ipywidgets.FloatText(disabled=True, description="Right")
    offset = ipywidgets.FloatText(disabled=True, description="Offset")
    scale = ipywidgets.FloatText(disabled=True, description="Scale")
    new_left = ipywidgets.FloatText(disabled=False, description="New left")
    new_right = ipywidgets.FloatText(disabled=False, description="New right")
    units = ipywidgets.Text(description="Units", )
    unitsl = ipywidgets.Label(layout=ipywidgets.Layout(width="10%"))
    help = ipywidgets.HTML(
        "Click on the signal figure and drag to the right to select a signal "
        "range. Set the new left and right values and press `Apply` to update "
        "the calibration of the axis with the new values or press "
        " `Close` to cancel.",)
    wdict["help"] = help
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
    link((obj, "ss_left_value"), (left, "value"))
    link((obj, "ss_right_value"), (right, "value"))
    link((obj, "left_value"), (new_left, "value"))
    link((obj, "right_value"), (new_right, "value"))
    link((obj, "units"), (units, "value"))
    link((obj, "units"), (unitsl, "value"))
    link((obj, "offset"), (offset, "value"))
    link((obj, "scale"), (scale, "value"))

    def on_apply_clicked(b):
        obj.apply()
    apply.on_click(on_apply_clicked)

    box = ipywidgets.VBox([
        ipywidgets.HBox([new_left, unitsl]),
        ipywidgets.HBox([new_right, unitsl]),
        ipywidgets.HBox([left, unitsl]),
        ipywidgets.HBox([right, unitsl]),
        ipywidgets.HBox([offset, unitsl]),
        scale,
        units,
        help,
        ipywidgets.HBox((apply, close))
    ])

    def on_close_clicked(b):
        obj.span_selector_switch(False)
        box.close()
    close.on_click(on_close_clicked)

    wdict["left"] = left
    wdict["right"] = right
    wdict["offset"] = offset
    wdict["scale"] = scale
    wdict["new_left"] = new_left
    wdict["new_right"] = new_right
    wdict["units"] = units
    wdict["close_button"] = close
    wdict["apply_button"] = apply

    return {
        "widget": box,
        "wdict": wdict,
    }


@register_ipy_widget(toolkey="Signal1D.smooth_savitzky_golay")
@add_display_arg
def smooth_savitzky_golay_ipy(obj, **kwargs):
    wdict = {}
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
    close = ipywidgets.Button(
        description="Close",
        tooltip="Close widget and remove the smoothed line from the signal figure.")
    apply = ipywidgets.Button(
        description="Apply",
        tooltip="Perform the operation using the selected range.")
    link((obj, "polynomial_order"), (polynomial_order, "value"))
    link((obj, "window_length"), (window_length, "value"))
    link((obj, "differential_order"),
         (differential_order, "value"))
    # Differential order must be less or equal to polynomial_order
    link((polynomial_order, "value"),
         (differential_order, "max"))
    link((obj, "line_color_ipy"), (color, "value"))
    box = ipywidgets.VBox([
        labelme("Window length", window_length),
        labelme("polynomial order", polynomial_order),
        labelme("Differential order", differential_order),
        labelme("Color", color),
        ipywidgets.HBox((apply, close))
    ])

    wdict["window_length"] = window_length
    wdict["polynomial_order"] = polynomial_order
    wdict["differential_order"] = differential_order
    wdict["color"] = color
    wdict["close_button"] = close
    wdict["apply_button"] = apply

    def on_apply_clicked(b):
        obj.apply()
    apply.on_click(on_apply_clicked)

    def on_close_clicked(b):
        obj.close()
        box.close()
    close.on_click(on_close_clicked)
    return {
        "widget": box,
        "wdict": wdict,
    }


@register_ipy_widget(toolkey="Signal1D.smooth_lowess")
@add_display_arg
def smooth_lowess_ipy(obj, **kwargs):
    wdict = {}
    smoothing_parameter = ipywidgets.FloatSlider(min=0, max=1)
    number_of_iterations = ipywidgets.IntText()
    color = ipywidgets.ColorPicker()
    close = ipywidgets.Button(
        description="Close",
        tooltip="Close widget and remove the smoothed line from the signal figure.")
    apply = ipywidgets.Button(
        description="Apply",
        tooltip="Perform the operation using the selected range.")
    link((obj, "smoothing_parameter"),
         (smoothing_parameter, "value"))
    link((obj, "number_of_iterations"),
         (number_of_iterations, "value"))
    link((obj, "line_color_ipy"), (color, "value"))
    box = ipywidgets.VBox([
        labelme("Smoothing parameter", smoothing_parameter),
        labelme("Number of iterations", number_of_iterations),
        labelme("Color", color),
        ipywidgets.HBox((apply, close))
    ])
    wdict["smoothing_parameter"] = smoothing_parameter
    wdict["number_of_iterations"] = number_of_iterations
    wdict["color"] = color
    wdict["close_button"] = close
    wdict["apply_button"] = apply

    def on_apply_clicked(b):
        obj.apply()
    apply.on_click(on_apply_clicked)

    def on_close_clicked(b):
        obj.close()
        box.close()
    close.on_click(on_close_clicked)
    return {
        "widget": box,
        "wdict": wdict,
    }


@register_ipy_widget(toolkey="Signal1D.smooth_total_variation")
@add_display_arg
def smooth_tv_ipy(obj, **kwargs):
    wdict = {}
    smoothing_parameter = ipywidgets.FloatSlider(min=0.1, max=1000)
    smoothing_parameter_max = ipywidgets.FloatText(
        value=smoothing_parameter.max)
    color = ipywidgets.ColorPicker()
    close = ipywidgets.Button(
        description="Close",
        tooltip="Close widget and remove the smoothed line from the signal figure.")
    apply = ipywidgets.Button(
        description="Apply",
        tooltip="Perform the operation using the selected range.")
    link((obj, "smoothing_parameter"),
         (smoothing_parameter, "value"))
    link((smoothing_parameter_max, "value"),
         (smoothing_parameter, "max"))
    link((obj, "line_color_ipy"), (color, "value"))
    wdict["smoothing_parameter"] = smoothing_parameter
    wdict["smoothing_parameter_max"] = smoothing_parameter_max
    wdict["color"] = color
    wdict["close_button"] = close
    wdict["apply_button"] = apply
    box = ipywidgets.VBox([
        labelme("Weight", smoothing_parameter),
        labelme("Weight max", smoothing_parameter_max),
        labelme("Color", color),
        ipywidgets.HBox((apply, close))
    ])

    def on_apply_clicked(b):
        obj.apply()
    apply.on_click(on_apply_clicked)

    def on_close_clicked(b):
        obj.close()
        box.close()
    close.on_click(on_close_clicked)
    return {
        "widget": box,
        "wdict": wdict,
    }


@register_ipy_widget(toolkey="Signal1D.smooth_butterworth")
@add_display_arg
def smooth_butterworth(obj, **kwargs):
    wdict = {}
    cutoff = ipywidgets.FloatSlider(min=0.01, max=1.)
    order = ipywidgets.IntText()
    type_ = ipywidgets.Dropdown(options=("low", "high"))
    color = ipywidgets.ColorPicker()
    close = ipywidgets.Button(
        description="Close",
        tooltip="Close widget and remove the smoothed line from the signal figure.")
    apply = ipywidgets.Button(
        description="Apply",
        tooltip="Perform the operation using the selected range.")
    link((obj, "cutoff_frequency_ratio"), (cutoff, "value"))
    link((obj, "type"), (type_, "value"))
    link((obj, "order"), (order, "value"))
    wdict["cutoff"] = cutoff
    wdict["order"] = order
    wdict["type"] = type_
    wdict["color"] = color
    wdict["close_button"] = close
    wdict["apply_button"] = apply
    box = ipywidgets.VBox([
        labelme("Cutoff frequency ration", cutoff),
        labelme("Type", type_),
        labelme("Order", order),
        ipywidgets.HBox((apply, close))
    ])

    def on_apply_clicked(b):
        obj.apply()
    apply.on_click(on_apply_clicked)

    def on_close_clicked(b):
        obj.close()
        box.close()
    close.on_click(on_close_clicked)
    return {
        "widget": box,
        "wdict": wdict,
    }


@register_ipy_widget(toolkey="Signal1D.contrast_editor")
@add_display_arg
def image_constast_editor_ipy(obj, **kwargs):
    wdict = {}
    left = ipywidgets.FloatText(disabled=True)
    right = ipywidgets.FloatText(disabled=True)
    help = ipywidgets.HTML(
        "Click on the histogram figure and drag to the right to select a"
        "range. Press `Apply` to set the new contrast limits, `Reset` to reset "
        "them or `Close` to cancel.",)
    wdict["help"] = help
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
    wdict["left"] = left
    wdict["right"] = right
    wdict["close_button"] = close
    wdict["apply_button"] = apply
    wdict["reset_button"] = reset

    # Connect
    link((obj, "ss_left_value"), (left, "value"))
    link((obj, "ss_right_value"), (right, "value"))

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

    def on_close_clicked(b):
        obj.close()
        box.close()
    close.on_click(on_close_clicked)
    return {
        "widget": box,
        "wdict": wdict,
    }


@register_ipy_widget(toolkey="Signal1D.remove_background")
@add_display_arg
def remove_background_ipy(obj, **kwargs):
    wdict = {}
    left = ipywidgets.FloatText(disabled=True, description="Left")
    right = ipywidgets.FloatText(disabled=True, description="Right")
    link((obj, "ss_left_value"), (left, "value"))
    link((obj, "ss_right_value"), (right, "value"))
    fast = ipywidgets.Checkbox(description="Fast")
    help = ipywidgets.HTML(
        "Click on the signal figure and drag to the right to select a"
        "range. Press `Apply` to remove the background in the whole dataset. "
        "If fast is checked, the background parameters are estimated using a "
        "fast (analytical) method that can compromise accuray. When unchecked "
        "non linear least squares is employed instead.",)
    wdict["help"] = help
    help = ipywidgets.Accordion(children=[help])
    help.set_title(0, "Help")
    close = ipywidgets.Button(
        description="Close",
        tooltip="Close widget and remove span selector from the signal figure.")
    apply = ipywidgets.Button(
        description="Apply",
        tooltip="Remove the background in the whole dataset.")

    polynomial_order = ipywidgets.IntText(description="Polynomial order")
    background_type = enum2dropdown(obj.traits()["background_type"])
    background_type.description = "Background type"

    def enable_poly_order(change):
        if change.new == "Polynomial":
            polynomial_order.layout.display = ""
        else:
            polynomial_order.layout.display = "none"
    background_type.observe(enable_poly_order, "value")
    link((obj, "background_type"), (background_type, "value"))
    # Trigger the function that controls the visibility of poly order as
    # setting the default value doesn't trigger it.

    class Dummy:
        new = background_type.value
    enable_poly_order(change=Dummy())
    link((obj, "polynomial_order"), (polynomial_order, "value"))
    link((obj, "fast"), (fast, "value"))
    wdict["left"] = left
    wdict["right"] = right
    wdict["fast"] = fast
    wdict["polynomial_order"] = polynomial_order
    wdict["background_type"] = background_type
    wdict["apply_button"] = apply
    box = ipywidgets.VBox([
        left, right,
        background_type,
        polynomial_order,
        fast,
        help,
        ipywidgets.HBox((apply, close)),
    ])

    def on_apply_clicked(b):
        obj.apply()
        box.close()
    apply.on_click(on_apply_clicked)

    def on_close_clicked(b):
        obj.span_selector_switch(False)
        box.close()
    close.on_click(on_close_clicked)
    return {
        "widget": box,
        "wdict": wdict,
    }


@register_ipy_widget(toolkey="Signal1D.spikes_removal_tool")
@add_display_arg
def spikes_removal_ipy(obj, **kwargs):
    wdict = {}
    threshold = ipywidgets.FloatText()
    add_noise = ipywidgets.Checkbox()
    default_spike_width = ipywidgets.IntText()
    interpolator_kind = enum2dropdown(obj.traits()["interpolator_kind"])
    spline_order = ipywidgets.IntSlider(min=1, max=10)
    progress_bar = ipywidgets.IntProgress(max=len(obj.coordinates) - 1)
    help = ipywidgets.HTML(
        value=SPIKES_REMOVAL_INSTRUCTIONS.replace('\n', '<br/>'))
    help = ipywidgets.Accordion(children=[help])
    help.set_title(0, "Help")

    show_diff = ipywidgets.Button(
        description="Show derivative histogram",
        tooltip="This figure is useful to estimate the threshold.",
        layout=ipywidgets.Layout(width="auto"))
    close = ipywidgets.Button(
        description="Close",
        tooltip="Close widget and remove span selector from the signal figure.")
    next = ipywidgets.Button(
        description="Find next",
        tooltip="Find next spike")
    previous = ipywidgets.Button(
        description="Find previous",
        tooltip="Find previous spike")
    remove = ipywidgets.Button(
        description="Remove spike",
        tooltip="Remove spike and find next one.")
    wdict["threshold"] = threshold
    wdict["add_noise"] = add_noise
    wdict["default_spike_width"] = default_spike_width
    wdict["interpolator_kind"] = interpolator_kind
    wdict["spline_order"] = spline_order
    wdict["progress_bar"] = progress_bar
    wdict["show_diff_button"] = show_diff
    wdict["close_button"] = close
    wdict["next_button"] = next
    wdict["previous_button"] = previous
    wdict["remove_button"] = remove

    def on_show_diff_clicked(b):
        obj._show_derivative_histogram_fired()
    show_diff.on_click(on_show_diff_clicked)

    def on_next_clicked(b):
        obj.find()
    next.on_click(on_next_clicked)

    def on_previous_clicked(b):
        obj.find(back=True)
    previous.on_click(on_previous_clicked)

    def on_remove_clicked(b):
        obj.apply()
    remove.on_click(on_remove_clicked)
    labeled_spline_order = labelme("Spline order", spline_order)

    def enable_interpolator_kind(change):
        if change.new == "Spline":
            for child in labeled_spline_order.children:
                child.layout.display = ""
        else:
            for child in labeled_spline_order.children:
                child.layout.display = "none"
    interpolator_kind.observe(enable_interpolator_kind, "value")
    link((obj, "interpolator_kind"),
         (interpolator_kind, "value"))
    link((obj, "threshold"), (threshold, "value"))
    link((obj, "add_noise"), (add_noise, "value"))
    link((obj, "default_spike_width"),
         (default_spike_width, "value"))
    link((obj, "spline_order"), (spline_order, "value"))
    link((obj, "index"), (progress_bar, "value"))
    # Trigger the function that controls the visibility  as
    # setting the default value doesn't trigger it.

    class Dummy:
        new = interpolator_kind.value
    enable_interpolator_kind(change=Dummy())
    advanced = ipywidgets.Accordion((
        ipywidgets.VBox([
            labelme("Add noise", add_noise),
            labelme("Interpolator kind", interpolator_kind),
            labelme("Default spike width", default_spike_width),
            labelme("Spline order", spline_order), ]),))

    advanced.set_title(0, "Advanced settings")
    box = ipywidgets.VBox([
        ipywidgets.VBox([
            show_diff,
            labelme("Threshold", threshold),
            labelme("Progress", progress_bar), ]),
        advanced,
        help,
        ipywidgets.HBox([previous, next, remove, close])
    ])

    def on_close_clicked(b):
        obj.span_selector_switch(False)
        box.close()
    close.on_click(on_close_clicked)
    return {
        "widget": box,
        "wdict": wdict,
    }
