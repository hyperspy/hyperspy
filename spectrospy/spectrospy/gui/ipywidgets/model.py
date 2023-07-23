from ipywidgets import  FloatSlider, VBox, Checkbox,
from hyperspy_gui_ipywidgets.model import add_display_arg

@add_display_arg
def get_eelscl_widget(obj, **kwargs):
    """Create ipywidgets for the EELSCLEdge component.

    """
    wdict = {}
    active = Checkbox(description='active', value=obj.active)
    fine_structure = Checkbox(description='Fine structure',
                              value=obj.fine_structure_active)
    fs_smoothing = FloatSlider(description='Fine structure smoothing',
                               min=0, max=1, step=0.001,
                               value=obj.fine_structure_smoothing)
    link((obj, "active"), (active, "value"))
    link((obj, "fine_structure_active"),
         (fine_structure, "value"))
    link((obj, "fine_structure_smoothing"),
         (fs_smoothing, "value"))
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