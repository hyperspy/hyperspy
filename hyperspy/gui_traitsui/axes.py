import traits.api as t
import traitsui.api as tui

from hyperspy.gui_traitsui.utils import register_traitsui_widget, add_display_arg
from hyperspy.misc.utils import isiterable, ordinal


def get_axis_label(axis):
    idx = ordinal(axis.index_in_axes_manager)
    type_ = "navigation" if axis.navigate else "signal"
    label = "{} ({}) axis".format(idx, type_)
    return label


@register_traitsui_widget(toolkey="navigation_sliders")
@add_display_arg
def navigation_sliders(obj, title=None, **kwargs):
    """Raises a windows with sliders to control the index of DataAxis

    Parameters
    ----------
    obj : list of DataAxis instances

    """

    class NavigationSliders(t.HasTraits):
        pass

    nav = NavigationSliders()
    view_tuple = ()
    for axis in obj:
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
    nav.trait_view("traits_view", view)
    return nav, {}


def get_data_axis_view(navigate, label):
    group_args = [
        tui.Item(name='name'),
        tui.Item(name='size', style='readonly'),
        tui.Item(name='index_in_array', style='readonly'),
        tui.Item(name='units'),

    ]
    if navigate:
        group_args.extend([
            tui.Item(name='index'),
            tui.Item(name='value', style='readonly'), ])
    data_axis_view = tui.View(
        tui.Group(
            tui.Group(*group_args,
                      show_border=True,),
            tui.Group(
                tui.Item(name='scale'),
                tui.Item(name='offset'),
                label='Calibration',
                show_border=True,),
            # label="Data Axis properties",
            show_border=True,),
        title=label,)
    return data_axis_view


@register_traitsui_widget(toolkey="DataAxis")
@add_display_arg
def data_axis_traitsui(obj, **kwargs):
    return obj, {"view": get_data_axis_view(
        navigate=obj.navigate,
        label=get_axis_label(obj))}


def get_axis_group(n, navigate, label=''):
    group_args = [
        tui.Item('axis%i.name' % n),
        tui.Item('axis%i.size' % n, style='readonly'),
        tui.Item('axis%i.index_in_array' % n, style='readonly'),
        tui.Item('axis%i.low_index' % n, style='readonly'),
        tui.Item('axis%i.high_index' % n, style='readonly'),
        tui.Item('axis%i.units' % n),
    ]
    # The style of the index is chosen to be readonly because of
    # a bug in Traits 4.0.0 when using context with a Range traits
    # where the limits are defined by another traits_view
    if navigate:
        group_args.extend([
            tui.Item('axis%i.index' % n, style='readonly'),
            tui.Item('axis%i.value' % n, style='readonly'), ])
    group = tui.Group(
        tui.Group(*group_args,
                  show_border=True,),
        tui.Group(
            tui.Item('axis%i.scale' % n),
            tui.Item('axis%i.offset' % n),
            label='Calibration',
            show_border=True,),
        label=label,
        show_border=True,)
    return group


@register_traitsui_widget(toolkey="AxesManager")
@add_display_arg
def axes_gui(obj, **kwargs):
    context = {}
    ag = []
    for n, axis in enumerate(obj._get_axes_in_natural_order()):
        ag.append(get_axis_group(
            n, label=get_axis_label(axis), navigate=axis.navigate))
        context['axis%i' % n] = axis
    ag = tuple(ag)
    obj.trait_view("traits_view", tui.View(*ag, title="Axes GUI"))
    return obj, {"context": context}
