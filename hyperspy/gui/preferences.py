import traitsui.api as tui


class PreferencesHandler(tui.Handler):

    def close(self, info, is_ok):
        # Removes the span selector from the plot
        info.object.save()
        return True
preferences_view = tui.View(
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
    handler=PreferencesHandler,)

eels_view = tui.View(
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
