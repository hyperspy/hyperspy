import ipywidgets

@add_display_arg
def print_edges_table_ipy(obj, **kwargs):
    # Define widgets
    wdict = {}
    axis = obj.axis
    style_d = {'description_width': 'initial'}
    layout_d = {'width': '50%'}
    left = ipywidgets.FloatText(disabled=True, layout={'width': '25%'})
    right = ipywidgets.FloatText(disabled=True, layout={'width': '25%'})
    units = ipywidgets.Label(style=style_d)
    major = ipywidgets.Checkbox(value=False, description='Only major edge',
                                indent=False, layout=layout_d)
    complmt = ipywidgets.Checkbox(value=False, description='Complementary edge',
                                 indent=False, layout=layout_d)
    order = ipywidgets.Dropdown(options=['closest', 'ascending', 'descending'],
                                value='closest',
                                description='Sort energy by: ',
                                disabled=False,
                                style=style_d
                                )
    update = ipywidgets.Button(description='Refresh table', layout={'width': 'initial'})
    gb = ipywidgets.GridBox(layout=ipywidgets.Layout(
            grid_template_columns="70px 125px 75px 250px"))
    help_text = ipywidgets.HTML(
        "Click on the signal figure and drag to the right to select a signal "
        "range. Drag the rectangle or change its border to display edges in "
        "different signal range. Select edges to show their positions "
        "on the signal.",)
    help = ipywidgets.Accordion(children=[help_text], selected_index=None)
    set_title_container(help, ["Help"])
    close = ipywidgets.Button(description="Close", tooltip="Close the widget.")
    reset = ipywidgets.Button(description="Reset",
                              tooltip="Reset the span selector.")

    header = ('<p style="padding-left: 1em; padding-right: 1em; '
              'text-align: center; vertical-align: top; '
              'font-weight:bold">{}</p>')
    entry = ('<p style="padding-left: 1em; padding-right: 1em; '
             'text-align: center; vertical-align: top">{}</p>')

    wdict["left"] = left
    wdict["right"] = right
    wdict["units"] = units
    wdict["help"] = help
    wdict["major"] = major
    wdict["update"] = update
    wdict["complmt"] = complmt
    wdict["order"] = order
    wdict["gb"] = gb
    wdict["reset"] = reset
    wdict["close"] = close

    # Connect
    link((obj, "ss_left_value"), (left, "value"))
    link((obj, "ss_right_value"), (right, "value"))
    link((axis, "units"), (units, "value"))
    link((obj, "only_major"), (major, "value"))
    link((obj, "complementary"), (complmt, "value"))
    link((obj, "order"), (order, "value"))

    def update_table(change):
        edges, energy, relevance, description = obj.update_table()

        # header
        items = [ipywidgets.HTML(header.format('edge')),
                 ipywidgets.HTML(header.format('onset energy (eV)')),
                 ipywidgets.HTML(header.format('relevance')),
                 ipywidgets.HTML(header.format('description'))]

        # rows
        obj.btns = []
        for k, edge in enumerate(edges):
            if edge in obj.active_edges or \
                edge in obj.active_complementary_edges:
                btn_state = True
            else:
                btn_state = False

            btn = ipywidgets.ToggleButton(value=btn_state,
                                          description=edge,
                                          layout=ipywidgets.Layout(width='70px'))
            btn.observe(obj.update_active_edge,  names='value')
            obj.btns.append(btn)

            wenergy = ipywidgets.HTML(entry.format(str(energy[k])))
            wrelv = ipywidgets.HTML(entry.format(str(relevance[k])))
            wdes = ipywidgets.HTML(entry.format(str(description[k])))
            items.extend([btn, wenergy, wrelv, wdes])

        gb.children = items
    update.on_click(update_table)
    major.observe(update_table)

    def on_complementary_toggled(change):
        obj.update_table()
        obj.check_btn_state()
    complmt.observe(on_complementary_toggled)

    def on_order_changed(change):
        obj._get_edges_info_within_energy_axis()
        update_table(change)
    order.observe(on_order_changed)

    def on_close_clicked(b):
        obj.span_selector_switch(False)
        box.close()
    close.on_click(on_close_clicked)

    def on_reset_clicked(b):
        # ss_left_value is linked with left.value, this can prevent cyclic
        # referencing
        obj._clear_markers()
        obj.span_selector_switch(False)
        left.value = 0
        right.value = 0
        obj.span_selector_switch(True)
        update_table(b)
    reset.on_click(on_reset_clicked)

    energy_box = ipywidgets.HBox([left, units, ipywidgets.Label("-"), right,
                                   units])
    check_box = ipywidgets.HBox([major, complmt])
    control_box = ipywidgets.VBox([energy_box, update, order, check_box])

    box = ipywidgets.VBox([
        ipywidgets.HBox([gb, control_box]),
        help,
        ipywidgets.HBox([reset, close]),
    ])

    return {
        "widget": box,
        "wdict": wdict,
    }