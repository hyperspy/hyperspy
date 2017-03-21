from IPython.display import display
import traitlets
import traits.api as t
import traits
import ipywidgets

from hyperspy.misc.link_traits import link_traits

FORM_ITEM_LAYOUT = ipywidgets.Layout(
    display='flex',
    flex_flow='row',
    justify_content='space-between',
)


def labelme(label, widget):
    return ipywidgets.HBox(
        [ipywidgets.Label(label), widget], layout=FORM_ITEM_LAYOUT)


def bool2checkbox(trait, label):
    tooltip = trait.desc if trait.desc else ""
    widget = ipywidgets.Checkbox(
        tooltip=tooltip,
    )
    return labelme(widget=widget, label=label)


def float2floattext(trait, label):
    tooltip = trait.desc if trait.desc else ""

    widget = ipywidgets.FloatText(
        tooltip=tooltip,
    )
    return labelme(widget=widget, label=label)


def directory2unicode(trait, label):
    tooltip = trait.desc if trait.desc else ""
    widget = ipywidgets.Text(
        tooltip=tooltip,)
    return labelme(widget=widget, label=label)


def range2floatrangeslider(trait, label):
    tooltip = trait.desc if trait.desc else ""

    widget = ipywidgets.FloatSlider(
        min=trait.trait_type._low,
        max=trait.trait_type._high,
        tooltip=tooltip,)
    return labelme(widget=widget, label=label)


def enum2dropdown(trait, label):
    tooltip = trait.desc if trait.desc else ""
    widget = ipywidgets.Dropdown(
        options=trait.trait_type.values,
        tooltip=tooltip,)
    return labelme(widget=widget, label=label)

traits2ipywidgets = {
    traits.trait_types.CBool: bool2checkbox,
    traits.trait_types.Bool: bool2checkbox,
    traits.trait_types.CFloat: float2floattext,
    traits.trait_types.Directory: directory2unicode,
    traits.trait_types.Range: range2floatrangeslider,
    traits.trait_types.Enum: enum2dropdown,
}


def get_label(trait, label):
    label = (label.replace("_", " ").capitalize()
             if not trait.label else trait.label)
    return label


def show_preferences_widget(preferences):
    ipytabs = {}
    for tab in preferences.editable_traits():
        ipytab = []
        tabtraits = getattr(preferences, tab).traits()
        for trait_name in getattr(preferences, tab).editable_traits():
            trait = tabtraits[trait_name]
            widget = traits2ipywidgets[type(trait.trait_type)](
                trait, get_label(trait, trait_name))
            ipytab.append(widget)
            link_traits((getattr(preferences, tab), trait_name),
                        (widget.children[1], "value"))
        ipytabs[tab] = ipywidgets.VBox(ipytab)
    titles = ["General", "Plot", "Model", "EELS", "EDS", "Machine Learning"]
    ipytabs_ = ipywidgets.Tab(
        children=[ipytabs[title.replace(" ", "")] for title in titles], titles=titles)
    for i, title in enumerate(titles):
        ipytabs_.set_title(i, title)
    save_button = ipywidgets.Button(
        description="Save",
        tooltip="Make changes permanent")
    def on_button_clicked(b):
        preferences.save()

    save_button.on_click(on_button_clicked)


    display(ipywidgets.VBox([ipytabs_, save_button]))
