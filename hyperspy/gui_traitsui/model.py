
fit_component_view = tu.View(
        tu.Item('only_current', show_label=True,),
        buttons=[OurFitButton, OurCloseButton],
        title='Fit single component',
        handler=ComponentFitHandler,
    )
