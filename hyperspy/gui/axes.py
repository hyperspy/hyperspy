
import traits.api as t
import traitsui.api as tui


def navigation_sliders(data_axes, title=None):
    """Raises a windows with sliders to control the index of DataAxis

    Parameters
    ----------
    data_axes : list of DataAxis instances

    """

    class NavigationSliders(t.HasTraits):
        pass

    nav = NavigationSliders()
    view_tuple = ()
    for axis in data_axes:
        name = str(axis).replace(" ", "_")
        nav.add_class_trait(name, axis)
        nav.trait_set([name, axis])
        view_tuple += (
            tui.Item(name,
                     style="custom",
                     editor=tui.InstanceEditor(
                         view=tui.View(
                             tui.Item(
                                 "index",
                                 show_label=False,
                                 # The following is commented out
                                 # due to a traits ui bug
                                 # editor=tui.RangeEditor(mode="slider"),
                             ),
                         ),
                     ),
                     ),
        )

    view = tui.View(tui.VSplit(view_tuple), title="Navigation sliders"
                    if title is None
                    else title)

    nav.edit_traits(view=view)


data_axis_view = tui.View(
    tui.Group(
        tui.Group(
            tui.Item(name='name'),
            tui.Item(name='size', style='readonly'),
            tui.Item(name='index_in_array', style='readonly'),
            tui.Item(name='index'),
            tui.Item(name='value', style='readonly'),
            tui.Item(name='units'),
            tui.Item(name='navigate', label='navigate'),
            show_border=True,),
        tui.Group(
            tui.Item(name='scale'),
            tui.Item(name='offset'),
            label='Calibration',
            show_border=True,),
        label="Data Axis properties",
        show_border=True,),
    title='Axis configuration',)


def get_axis_group(n, label=''):
    group = tui.Group(
        tui.Group(
            tui.Item('axis%i.name' % n),
            tui.Item('axis%i.size' % n, style='readonly'),
            tui.Item('axis%i.index_in_array' % n, style='readonly'),
            tui.Item('axis%i.low_index' % n, style='readonly'),
            tui.Item('axis%i.high_index' % n, style='readonly'),
            # The style of the index is chosen to be readonly because of
            # a bug in Traits 4.0.0 when using context with a Range traits
            # where the limits are defined by another traits_view
            tui.Item('axis%i.index' % n, style='readonly'),
            tui.Item('axis%i.value' % n, style='readonly'),
            tui.Item('axis%i.units' % n),
            tui.Item('axis%i.navigate' % n, label='navigate'),
            show_border=True,),
        tui.Group(
            tui.Item('axis%i.scale' % n),
            tui.Item('axis%i.offset' % n),
            label='Calibration',
            show_border=True,),
        label=label,
        show_border=True,)
    return group
