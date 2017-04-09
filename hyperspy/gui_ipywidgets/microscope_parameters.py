
import ipywidgets

from hyperspy.gui_ipywidgets.utils import (
    labelme, register_ipy_widget, add_display_arg, float2floattext, get_label)

from hyperspy.misc.link_traits import link_traits


def _set_microscope_parameters(obj, **kwargs):
    traits = obj.traits()
    widgets = []
    for trait_name in obj.editable_traits():
        if trait_name in ("mapping", "signal"):
            continue
        trait = traits[trait_name]
        widget = float2floattext(
            trait, get_label(trait, trait_name))
        widgets.append(widget)
        link_traits((obj, trait_name),
                    (widget.children[1], "value"))
    store_button = ipywidgets.Button(
        description="Store",
        tooltip="Store the values in metadata")
    store_button.on_click(obj.store)
    return ipywidgets.VBox([ipywidgets.VBox(widgets),
                            store_button])


@register_ipy_widget(toolkey="microscope_parameters_EELS")
@add_display_arg
def eels_microscope_parameter_ipy(obj, **kwargs):
    return(_set_microscope_parameters(obj=obj, **kwargs))


@register_ipy_widget(toolkey="microscope_parameters_EDS_SEM")
@add_display_arg
def eds_sem_microscope_parameter_ipy(obj, **kwargs):
    return(_set_microscope_parameters(obj=obj, **kwargs))


@register_ipy_widget(toolkey="microscope_parameters_EDS_TEM")
@add_display_arg
def eds_tem_microscope_parameter_ipy(obj, **kwargs):
    return(_set_microscope_parameters(obj=obj, **kwargs))
