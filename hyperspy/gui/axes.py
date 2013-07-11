
import traits.api as t
import traitsui.api as tui

def navigation_sliders(data_axes):
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
                             style = "custom",
                             editor = tui.InstanceEditor(
                                 view=tui.View(
                                     tui.Item(
                                         "index",
                                         show_label=False,
                                         # The following is commented out
                                         # due to a traits ui bug
                                         #editor=tui.RangeEditor(mode="slider"),
                                          ),
                                     ),
                                 ),
                             ),
                         )
                               
                                  



    view = tui.View(tui.VSplit(view_tuple), title="Navigation sliders")

    nav.edit_traits(view=view)
