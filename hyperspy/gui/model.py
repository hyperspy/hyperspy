import traits.api as t

from hyperspy.gui.tools import SpanSelectorInSignal1D


class ComponentFit(SpanSelectorInSignal1D):

    only_current = t.Bool(True)

    def __init__(self, model, component, signal_range=None,
                 estimate_parameters=True, fit_independent=False,
                 only_current=True, **kwargs):
        if model.signal.axes_manager.signal_dimension != 1:
            raise SignalDimensionError(
                model.signal.axes_manager.signal_dimension, 1)

        self.signal = model.signal
        self.axis = self.signal.axes_manager.signal_axes[0]
        self.span_selector = None
        self.model = model
        self.component = component
        self.signal_range = signal_range
        self.estimate_parameters = estimate_parameters
        self.fit_independent = fit_independent
        self.fit_kwargs = kwargs
        if signal_range == "interactive":
            if not hasattr(self.model, '_plot'):
                self.model.plot()
            elif self.model._plot is None:
                self.model.plot()
            elif self.model._plot.is_active() is False:
                self.model.plot()
            self.span_selector_switch(on=True)

    def _fit_fired(self):
        if (self.signal_range != "interactive" and
                self.signal_range is not None):
            self.model.set_signal_range(*self.signal_range)
        elif self.signal_range == "interactive":
            self.model.set_signal_range(self.ss_left_value,
                                        self.ss_right_value)

        # Backup "free state" of the parameters and fix all but those
        # of the chosen component
        if self.fit_independent:
            active_state = []
            for component_ in self.model:
                active_state.append(component_.active)
                if component_ is not self.component:
                    component_.active = False
                else:
                    component_.active = True
        else:
            free_state = []
            for component_ in self.model:
                for parameter in component_.parameters:
                    free_state.append(parameter.free)
                    if component_ is not self.component:
                        parameter.free = False

        # Setting reasonable initial value for parameters through
        # the components estimate_parameters function (if it has one)
        only_current = self.only_current
        if self.estimate_parameters:
            if hasattr(self.component, 'estimate_parameters'):
                if (self.signal_range != "interactive" and
                        self.signal_range is not None):
                    self.component.estimate_parameters(
                        self.signal,
                        self.signal_range[0],
                        self.signal_range[1],
                        only_current=only_current)
                elif self.signal_range == "interactive":
                    self.component.estimate_parameters(
                        self.signal,
                        self.ss_left_value,
                        self.ss_right_value,
                        only_current=only_current)

        if only_current:
            self.model.fit(**self.fit_kwargs)
        else:
            self.model.multifit(**self.fit_kwargs)

        # Restore the signal range
        if self.signal_range is not None:
            self.model.channel_switches = (
                self.model.backup_channel_switches.copy())

        self.model.update_plot()

        if self.fit_independent:
            for component_ in self.model:
                component_.active = active_state.pop(0)
        else:
            # Restore the "free state" of the components
            for component_ in self.model:
                for parameter in component_.parameters:
                    parameter.free = free_state.pop(0)

    def apply(self):
        self._fit_fired()
