import traitsui.api as tu
from traitsui.menu import OKButton, CancelButton

from hyperspy.gui_traitsui.utils import (
    add_display_arg, register_traitsui_widget)

@register_traitsui_widget(toolkey="microscope_parameters_EDS_SEM")
@add_display_arg
def microscope_parameters_EDS_SEM(obj, **kwargs):
    view = tu.View(
        tu.Group('beam_energy',
                 'tilt_stage',
                 label='SEM', show_border=True),
        tu.Group('live_time', 'azimuth_angle',
                 'elevation_angle', 'energy_resolution_MnKa',
                 label='EDS', show_border=True),
        kind='modal', buttons=[OKButton, CancelButton],
        title='SEM parameters definition wizard')
    return obj, {"view": view}

@register_traitsui_widget(toolkey="microscope_parameters_EDS_TEM")
@add_display_arg
def microscope_parameters_EDS_TEM(obj, **kwargs):
    view = tu.View(
        tu.Group('beam_energy',
                 'tilt_stage',
                 'probe_area',
                 'beam_current',
                 label='TEM', show_border=True),
        tu.Group('real_time', 'live_time', 'azimuth_angle',
                 'elevation_angle', 'energy_resolution_MnKa',
                 label='EDS', show_border=True),
        kind='modal', buttons=[OKButton, CancelButton],
        title='TEM parameters definition wizard')
    return obj, {"view": view}

@register_traitsui_widget(toolkey="microscope_parameters_EELS")
@add_display_arg
def microscope_parameters_EELS(obj, **kwargs):
    view = tu.View(
        tu.Group('beam_energy',
                 'convergence_angle',
                 label='TEM', show_border=True),
        tu.Group('collection_angle',
                 label='EELS', show_border=True),
        kind='modal', buttons=[OKButton, CancelButton],
        title='TEM parameters definition wizard')
    return obj, {"view": view}
