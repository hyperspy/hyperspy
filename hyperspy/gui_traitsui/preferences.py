import traitsui.api as tui
from traitsui.menu import CancelButton

from hyperspy.gui_traitsui.buttons import SaveButton
from hyperspy.gui_traitsui.utils import (
    register_traitsui_widget, add_display_arg)


class PreferencesHandler(tui.Handler):

    def save(self, info):
        # Removes the span selector from the plot
        info.object.save()
        return True

PREFERENCES_VIEW = tui.View(
    tui.Group(tui.Item('General', style='custom', show_label=False, ),
              label='General'),
    tui.Group(tui.Item('Model', style='custom', show_label=False, ),
              label='Model'),
    tui.Group(tui.Item('EELS', style='custom', show_label=False, ),
              label='EELS'),
    tui.Group(tui.Item('EDS', style='custom', show_label=False, ),
              label='EDS'),
    tui.Group(tui.Item('MachineLearning', style='custom',
                       show_label=False,),
              label='Machine Learning'),
    tui.Group(tui.Item('Plot', style='custom', show_label=False, ),
              label='Plot'),
    title='Preferences',
    buttons=[SaveButton, CancelButton],
    handler=PreferencesHandler,)

EELS_VIEW = tui.View(
    tui.Group(
        'synchronize_cl_with_ll',
        label='General'),
    tui.Group(
        'eels_gos_files_path',
        'preedge_safe_window_width',
        tui.Group(
            'fine_structure_width',
            'fine_structure_active',
            'fine_structure_smoothing',
            'min_distance_between_edges_for_fine_structure',
            label='Fine structure'),
        label='Model')
)


@register_traitsui_widget(toolkey="Preferences")
@add_display_arg
def preferences_traitsui(obj, **kwargs):
    obj.EELS.trait_view("traits_view", EELS_VIEW)
    obj.trait_view("traits_view", PREFERENCES_VIEW)
    return obj, {}
