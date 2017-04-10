import functools

from traitlets import TraitError as TraitletError
import ipywidgets
from ipywidgets import (
    Accordion, FloatSlider, FloatText, Layout, HBox, VBox, Checkbox, Label,
    Button)
import numpy as np

from hyperspy.misc.link_traits import link_traits, directional_link
from hyperspy.gui_ipywidgets.utils import (
    add_display_arg, register_ipy_widget, labelme)


def _interactive_slider_bounds(obj, index=None):
    """Guesstimates the bounds for the slider. They will probably have to
    be changed later by the user.

    """
    pad = 10.
    _min, _max, step = None, None, None
    value = obj.value if index is None else obj.value[index]
    if obj.bmin is not None:
        _min = obj.bmin
    if obj.bmax is not None:
        _max = obj.bmax
    if _max is None and _min is not None:
        _max = value + pad
    if _min is None and _max is not None:
        _min = value - pad
    if _min is None and _max is None:
        if obj is obj.component._position:
            axis = obj._axes_manager.signal_axes[-1]
            _min = axis.axis.min()
            _max = axis.axis.max()
            step = np.abs(axis.scale)
        else:
            _max = value + pad
            _min = value - pad
    if step is None:
        step = (_max - _min) * 0.001
    return {'min': _min, 'max': _max, 'step': step}


def _get_value_widget(obj, index=None):
    wdict = {}
    widget_bounds = _interactive_slider_bounds(obj, index=index)
    thismin = FloatText(value=widget_bounds['min'],
                        description='min',
                        layout=Layout(flex='0 1 auto',
                                      width='auto'),)
    thismax = FloatText(value=widget_bounds['max'],
                        description='max',
                        layout=Layout(flex='0 1 auto',
                                      width='auto'),)
    current_value = obj.value if index is None else obj.value[index]
    if index is None:
        current_name = obj.name
    else:
        current_name = '{}'.format(index)
    widget = FloatSlider(value=current_value,
                         min=thismin.value,
                         max=thismax.value,
                         step=widget_bounds['step'],
                         description=current_name,
                         layout=Layout(flex='1 1 auto', width='auto'))

    def on_min_change(change):
        if widget.max > change['new']:
            widget.min = change['new']
            widget.step = np.abs(widget.max - widget.min) * 0.001

    def on_max_change(change):
        if widget.min < change['new']:
            widget.max = change['new']
            widget.step = np.abs(widget.max - widget.min) * 0.001

    thismin.observe(on_min_change, names='value')
    thismax.observe(on_max_change, names='value')
    # We store the link in the widget so that they are not deleted by the
    # garbage collector
    thismin._link = directional_link((obj, "bmin"), (thismin, "value"))
    thismax._link = directional_link((obj, "bmax"), (thismax, "value"))
    if index is not None:  # value is tuple, expanding
        def _interactive_tuple_update(value):
            """Callback function for the widgets, to update the value
            """
            obj.value = obj.value[:index] + (value['new'],) +\
                obj.value[index + 1:]
        widget.observe(_interactive_tuple_update, names='value')
    else:
        link_traits((obj, "value"), (widget, "value"))

    container = HBox((thismin, widget, thismax))
    wdict["value"] = widget
    wdict["min"] = thismin
    wdict["max"] = thismax
    return {
        "widget": container,
        "wdict": wdict,
    }


@register_ipy_widget(toolkey="Parameter")
@add_display_arg
def get_parameter_widget(obj, **kwargs):
    """Creates interactive notebook widgets for the parameter, if
    available.

    """
    if obj._number_of_elements == 1:
        return _get_value_widget(obj)
    else:
        wdict = {}
        par_widgets = []
        for i in range(obj._number_of_elements):
            thiswd = _get_value_widget(obj=obj, index=i)
            par_widgets.append(thiswd["widget"])
            wdict["element{}".format(i)] = thiswd["wdict"]
        update = Button(
            description="Update",
            tooltip="Unlike most other widgets, the multivalue parameter "
            "widgets do not update automatically when the value of the "
            "changes by other means. Use this button to update the values"
            "manually")

        def on_update_clicked(b):
            for value, container in zip(obj.value, par_widgets):

                minwidget = container.children[0]
                vwidget = container.children[1]
                maxwidget = container.children[2]
                if value < vwidget.min:
                    minwidget.value = value
                elif value > vwidget.max:
                    maxwidget.value = value
                vwidget.value = value
        update.on_click(on_update_clicked)
        wdict["update_button"] = update
        container = Accordion([VBox([update] + par_widgets)],
                              descrition=obj.name)
        container.set_title(0, obj.name)

    return {
        "widget": container,
        "wdict": wdict,
    }


@register_ipy_widget(toolkey="Component")
@add_display_arg
def get_component_widget(obj, **kwargs):
    """Creates interactive notebook widgets for all component parameters,
    if available.

    """
    wdict = {}
    active = Checkbox(description='active', value=obj.active)
    wdict["active"] = active
    link_traits((obj, "active"), (active, "value"))
    container = VBox([active])
    for parameter in obj.parameters:
        pardict = parameter.gui(
            toolkit="ipywidgets", display=False)["ipywidgets"]
        wdict["parameter_{}".format(parameter.name)] = pardict["wdict"]
        container.children += pardict["widget"],
    return {
        "widget": container,
        "wdict": wdict,
    }


@register_ipy_widget(toolkey="Model")
@add_display_arg
def get_model_widget(obj, **kwargs):
    """Creates interactive notebook widgets for all components and
    parameters, if available.

    """
    children = []
    wdict = {}
    for component in obj:
        idict = component.gui(
            display=False,
            toolkit="ipywidgets")["ipywidgets"]
        children.append(idict["widget"])
        wdict["component_{}".format(component.name)] = idict["wdict"]
    accordion = Accordion(children=children)
    for i, comp in enumerate(obj):
        accordion.set_title(i, comp.name)
    return {
        "widget": accordion,
        "wdict": wdict
    }


@register_ipy_widget(toolkey="EELSCLEdge_Component")
@add_display_arg
def get_eelscl_widget(obj, **kwargs):
    """Create ipywidgets for the EELSCLEDge component.

    """
    wdict = {}
    active = Checkbox(description='active', value=obj.active)
    fine_structure = Checkbox(description='Fine structure',
                              value=obj.fine_structure_active)
    fs_smoothing = FloatSlider(description='Fine structure smoothing',
                               min=0, max=1, step=0.001,
                               value=obj.fine_structure_smoothing)
    container = VBox([active, fine_structure, fs_smoothing])
    wdict["active"] = active
    wdict["fine_structure"] = fine_structure
    wdict["fs_smoothing"] = fs_smoothing
    for parameter in [obj.intensity, obj.effective_angle,
                      obj.onset_energy]:
        pdict = parameter.gui(
            toolkit="ipywidgets", display=False)["ipywidgets"]
        container.children += pdict["widget"],
        wdict["parameter_{}".format(parameter.name)] = pdict["wdict"]
    return {
        "widget": container,
        "wdict": wdict,
    }


@register_ipy_widget(toolkey="ScalableFixedPattern_Component")
@add_display_arg
def get_scalable_fixed_patter_widget(obj, **kwargs):
    cdict = get_component_widget(obj, display=False)
    wdict = cdict["wdict"]
    container = cdict["widget"]
    interpolate = Checkbox(description='interpolate',
                           value=obj.interpolate)
    wdict["interpolate"] = interpolate
    link_traits((obj, "interpolate"), (interpolate, "value"))
    container.children = (container.children[0], interpolate) + \
        container.children[1:]
    return {
        "widget": container,
        "wdict": wdict,
    }


@register_ipy_widget(toolkey="Model1D.fit_component")
@add_display_arg
def fit_component_ipy(obj, **kwargs):
    wdict = {}
    only_current = Checkbox()
    wdict["only_current"] = only_current
    help = Label(
        "Click on the signal figure and drag to the right to select a"
        "range. Press `Fit` to fit the component in that range. If only "
        "current is unchecked the fit is performed in the whole dataset.",
        layout=ipywidgets.Layout(width="auto"))
    wdict["help"] = only_current
    help = Accordion(children=[help])
    help.set_title(0, "Help")
    link_traits((obj, "only_current"), (only_current, "value"))
    fit = Button(
        description="Fit",
        tooltip="Fit in the selected signal range")
    close = Button(
        description="Close",
        tooltip="Close widget and remove span selector from the signal figure.")
    wdict["close_button"] = close
    wdict["fit_button"] = fit

    def on_fit_clicked(b):
        obj._fit_fired()
    fit.on_click(on_fit_clicked)
    box = VBox([
        labelme("Only current", only_current),
        help,
        HBox((fit, close))
    ])

    def on_close_clicked(b):
        obj.span_selector_switch(False)
        box.close()
    close.on_click(on_close_clicked)
    return {
        "widget": box,
        "wdict": wdict,
    }
