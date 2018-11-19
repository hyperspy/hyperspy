# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.

import copy

import numpy as np

from hyperspy.model import BaseModel, ModelComponents, ModelSpecialSlicers
import hyperspy.drawing.signal1d
from hyperspy.axes import generate_axis
from hyperspy.exceptions import WrongObjectError, SignalDimensionError
from hyperspy.decorators import interactive_range_selector
from hyperspy.drawing.widgets import VerticalLineWidget, LabelWidget
from hyperspy.events import EventSuppressor
from hyperspy.signal_tools import SpanSelectorInSignal1D
from hyperspy.ui_registry import add_gui_method, DISPLAY_DT, TOOLKIT_DT
from hyperspy.misc.utils import signal_range_from_roi


@add_gui_method(toolkey="Model1D.fit_component")
class ComponentFit(SpanSelectorInSignal1D):

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
        self.only_current = only_current
        if signal_range == "interactive":
            if (not hasattr(self.model, '_plot') or self.model._plot is None or
                    not self.model._plot.is_active):
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


class Model1D(BaseModel):

    """Model and data fitting for one dimensional signals.

    A model is constructed as a linear combination of :mod:`components` that
    are added to the model using :meth:`append` or :meth:`extend`. There
    are many predifined components available in the in the :mod:`components`
    module. If needed, new components can be created easily using the code of
    existing components as a template.

    Once defined, the model can be fitted to the data using :meth:`fit` or
    :meth:`multifit`. Once the optimizer reaches the convergence criteria or
    the maximum number of iterations the new value of the component parameters
    are stored in the components.

    It is possible to access the components in the model by their name or by
    the index in the model. An example is given at the end of this docstring.

    Attributes
    ----------

    signal : Signal1D instance
        It contains the data to fit.
    chisq : A Signal of floats
        Chi-squared of the signal (or np.nan if not yet fit)
    dof : A Signal of integers
        Degrees of freedom of the signal (0 if not yet fit)
    red_chisq : Signal instance
        Reduced chi-squared.
    components : `ModelComponents` instance
        The components of the model are attributes of this class. This provides
        a convinient way to access the model components when working in IPython
        as it enables tab completion.

    Methods
    -------

    append
        Append one component to the model.
    extend
        Append multiple components to the model.
    remove
        Remove component from model.
    as_signal
        Generate a Signal1D instance (possible multidimensional)
        from the model.
    store_current_values
        Store the value of the parameters at the current position.
    fetch_stored_values
        fetch stored values of the parameters.
    update_plot
        Force a plot update. (In most cases the plot should update
        automatically.)
    set_signal_range, remove_signal range, reset_signal_range,
    add signal_range.
        Customize the signal range to fit.
    fit, multifit
        Fit the model to the data at the current position or the
        full dataset.
    save_parameters2file, load_parameters_from_file
        Save/load the parameter values to/from a file.
    plot
        Plot the model and the data.
    enable_plot_components, disable_plot_components
        Plot each component separately. (Use after `plot`.)
    set_current_values_to
        Set the current value of all the parameters of the given component as
        the value for all the dataset.
    export_results
        Save the value of the parameters in separate files.
    plot_results
        Plot the value of all parameters at all positions.
    print_current_values
        Print the value of the parameters at the current position.
    enable_adjust_position, disable_adjust_position
        Enable/disable interactive adjustment of the position of the components
        that have a well defined position. (Use after `plot`).
    fit_component
        Fit just the given component in the given signal range, that can be
        set interactively.
    set_parameters_not_free, set_parameters_free
        Fit the `free` status of several components and parameters at once.
    set_parameters_value
        Set the value of a parameter in components in a model to a specified
        value.
    as_dictionary
        Exports the model to a dictionary that can be saved in a file.

    Examples
    --------
    In the following example we create a histogram from a normal distribution
    and fit it with a gaussian component. It demonstrates how to create
    a model from a :class:`~._signals.signal1d.Signal1D` instance, add
    components to it, adjust the value of the parameters of the components,
    fit the model to the data and access the components in the model.

    >>> s = hs.signals.Signal1D(
            np.random.normal(scale=2, size=10000)).get_histogram()
    >>> g = hs.model.components1D.Gaussian()
    >>> m = s.create_model()
    >>> m.append(g)
    >>> m.print_current_values()
    Components	Parameter	Value
    Gaussian
                sigma	1.000000
                A	1.000000
                centre	0.000000
    >>> g.centre.value = 3
    >>> m.print_current_values()
    Components	Parameter	Value
    Gaussian
                sigma	1.000000
                A	1.000000
                centre	3.000000
    >>> g.sigma.value
    1.0
    >>> m.fit()
    >>> g.sigma.value
    1.9779042300856682
    >>> m[0].sigma.value
    1.9779042300856682
    >>> m["Gaussian"].centre.value
    -0.072121936813224569

    """

    def __init__(self, signal1D, dictionary=None):
        super(Model1D, self).__init__()
        self.signal = signal1D
        self.axes_manager = self.signal.axes_manager
        self._plot = None
        self._position_widgets = {}
        self._adjust_position_all = None
        self._plot_components = False
        self._suspend_update = False
        self._model_line = None
        self._adjust_position_all = None
        self.axis = self.axes_manager.signal_axes[0]
        self.axes_manager.events.indices_changed.connect(
            self.fetch_stored_values, [])
        self.channel_switches = np.array([True] * len(self.axis.axis))
        self.chisq = signal1D._get_navigation_signal()
        self.chisq.change_dtype("float")
        self.chisq.data.fill(np.nan)
        self.chisq.metadata.General.title = (
            self.signal.metadata.General.title + ' chi-squared')
        self.dof = self.chisq._deepcopy_with_new_data(
            np.zeros_like(self.chisq.data, dtype='int'))
        self.dof.metadata.General.title = (
            self.signal.metadata.General.title + ' degrees of freedom')
        self.free_parameters_boundaries = None
        self._low_loss = None
        self.convolved = False
        self.components = ModelComponents(self)
        if dictionary is not None:
            self._load_dictionary(dictionary)
        self.inav = ModelSpecialSlicers(self, True)
        self.isig = ModelSpecialSlicers(self, False)
        self._whitelist = {
            'channel_switches': None,
            'convolved': None,
            'free_parameters_boundaries': None,
            'low_loss': ('sig', None),
            'chisq.data': None,
            'dof.data': None}
        self._slicing_whitelist = {
            'channel_switches': 'isig',
            'low_loss': 'inav',
            'chisq.data': 'inav',
            'dof.data': 'inav'}

    @property
    def signal(self):
        return self._signal

    @signal.setter
    def signal(self, value):
        from hyperspy._signals.signal1d import Signal1D
        if isinstance(value, Signal1D):
            self._signal = value
        else:
            raise WrongObjectError(str(type(value)), 'Signal1D')

    @property
    def low_loss(self):
        return self._low_loss

    @low_loss.setter
    def low_loss(self, value):
        if value is not None:
            if (value.axes_manager.navigation_shape !=
                    self.signal.axes_manager.navigation_shape):
                raise ValueError('The low-loss does not have '
                                 'the same navigation dimension as the '
                                 'core-loss')
            self._low_loss = value
            self.set_convolution_axis()
            self.convolved = True
        else:
            self._low_loss = value
            self.convolution_axis = None
            self.convolved = False

    # Extend the list methods to call the _touch when the model is modified

    def set_convolution_axis(self):
        """
        Creates an axis to use to generate the data of the model in the precise
        scale to obtain the correct axis and origin after convolution with the
        lowloss spectrum.
        """
        ll_axis = self.low_loss.axes_manager.signal_axes[0]
        dimension = self.axis.size + ll_axis.size - 1
        step = self.axis.scale
        knot_position = ll_axis.size - ll_axis.value2index(0) - 1
        self.convolution_axis = generate_axis(self.axis.offset, step,
                                              dimension, knot_position)

    def append(self, thing):
        super(Model1D, self).append(thing)
        if self._plot_components:
            self._plot_component(thing)
        if self._adjust_position_all:
            self._make_position_adjuster(thing, self._adjust_position_all[0],
                                         self._adjust_position_all[1])

    def remove(self, things):
        things = self._get_component(things)
        if not np.iterable(things):
            things = [things]
        for thing in things:
            parameter = thing._position
            if parameter in self._position_widgets:
                for pw in reversed(self._position_widgets[parameter]):
                    pw.close()
            if hasattr(thing, '_model_plot_line'):
                line = thing._model_plot_line
                line.close()
        super(Model1D, self).remove(things)
        self._disconnect_parameters2update_plot(things)

    remove.__doc__ = BaseModel.remove.__doc__

    def __call__(self, non_convolved=False, onlyactive=False,
                 component_list=None):
        """Returns the corresponding model for the current coordinates

        Parameters
        ----------
        non_convolved : bool
            If True it will return the deconvolved model
        only_active : bool
            If True, only the active components will be used to build the
            model.
        component_list : list or None
            If None, the sum of all the components is returned. If list, only
            the provided components are returned

        cursor: 1 or 2

        Returns
        -------
        numpy array
        """

        if component_list is None:
            component_list = self
        if not isinstance(component_list, (list, tuple)):
            raise ValueError(
                "'Component_list' parameter need to be a list or None")

        if onlyactive:
            component_list = [
                component for component in component_list if component.active]

        if self.convolved is False or non_convolved is True:
            axis = self.axis.axis[self.channel_switches]
            sum_ = np.zeros(len(axis))
            for component in component_list:
                sum_ += component.function(axis)
            to_return = sum_

        else:  # convolved
            sum_convolved = np.zeros(len(self.convolution_axis))
            sum_ = np.zeros(len(self.axis.axis))
            for component in component_list:
                if component.convolved:
                    sum_convolved += component.function(self.convolution_axis)
                else:
                    sum_ += component.function(self.axis.axis)

            to_return = sum_ + np.convolve(
                self.low_loss(self.axes_manager),
                sum_convolved, mode="valid")
            to_return = to_return[self.channel_switches]
        if self.signal.metadata.Signal.binned is True:
            to_return *= self.signal.axes_manager[-1].scale
        return to_return

    def _errfunc(self, param, y, weights=None):
        if weights is None:
            weights = 1.
        errfunc = self._model_function(param) - y
        return errfunc * weights

    def _set_signal_range_in_pixels(self, i1=None, i2=None):
        """Use only the selected spectral range in the fitting routine.

        Parameters
        ----------
        i1 : Int
        i2 : Int

        Notes
        -----
        To use the full energy range call the function without arguments.
        """

        self.backup_channel_switches = copy.copy(self.channel_switches)
        self.channel_switches[:] = False
        self.channel_switches[i1:i2] = True
        self.update_plot()

    @interactive_range_selector
    def set_signal_range(self, x1=None, x2=None):
        """Use only the selected spectral range defined in its own units in the
        fitting routine.

        Parameters
        ----------
        E1 : None or float
        E2 : None or float

        """
        try:
            x1, x2 = signal_range_from_roi(x1)
        except TypeError:
            # It was not a ROI, we carry on
            pass
        i1, i2 = self.axis.value_range_to_indices(x1, x2)
        self._set_signal_range_in_pixels(i1, i2)

    def _remove_signal_range_in_pixels(self, i1=None, i2=None):
        """Removes the data in the given range from the data range that
        will be used by the fitting rountine

        Parameters
        ----------
        x1 : None or float
        x2 : None or float
        """
        self.channel_switches[i1:i2] = False
        self.update_plot()

    @interactive_range_selector
    def remove_signal_range(self, x1=None, x2=None):
        """Removes the data in the given range from the data range that
        will be used by the fitting rountine

        Parameters
        ----------
        x1 : None or float
        x2 : None or float

        """
        try:
            x1, x2 = signal_range_from_roi(x1)
        except TypeError:
            # It was not a ROI, we carry on
            pass
        i1, i2 = self.axis.value_range_to_indices(x1, x2)
        self._remove_signal_range_in_pixels(i1, i2)

    def reset_signal_range(self):
        """Resets the data range"""
        self._set_signal_range_in_pixels()

    def _add_signal_range_in_pixels(self, i1=None, i2=None):
        """Adds the data in the given range from the data range that
        will be used by the fitting rountine

        Parameters
        ----------
        x1 : None or float
        x2 : None or float
        """
        self.channel_switches[i1:i2] = True
        self.update_plot()

    @interactive_range_selector
    def add_signal_range(self, x1=None, x2=None):
        """Adds the data in the given range from the data range that
        will be used by the fitting rountine

        Parameters
        ----------
        x1 : None or float
        x2 : None or float

        """
        try:
            x1, x2 = signal_range_from_roi(x1)
        except TypeError:
            # It was not a ROI, we carry on
            pass
        i1, i2 = self.axis.value_range_to_indices(x1, x2)
        self._add_signal_range_in_pixels(i1, i2)

    def reset_the_signal_range(self):
        self.channel_switches[:] = True
        self.update_plot()

    def _jacobian(self, param, y, weights=None):
        if weights is None:
            weights = 1.
        if self.convolved is True:
            counter = 0
            grad = np.zeros(len(self.axis.axis))
            for component in self:  # Cut the parameters list
                if component.active:
                    component.fetch_values_from_array(
                        param[
                            counter:counter +
                            component._nfree_param],
                        onlyfree=True)
                    if component.convolved:
                        for parameter in component.free_parameters:
                            par_grad = np.convolve(
                                parameter.grad(self.convolution_axis),
                                self.low_loss(self.axes_manager),
                                mode="valid")
                            if parameter._twins:
                                for par in parameter._twins:
                                    np.add(par_grad, np.convolve(
                                        par.grad(
                                            self.convolution_axis),
                                        self.low_loss(self.axes_manager),
                                        mode="valid"), par_grad)
                            grad = np.vstack((grad, par_grad))
                    else:
                        for parameter in component.free_parameters:
                            par_grad = parameter.grad(self.axis.axis)
                            if parameter._twins:
                                for par in parameter._twins:
                                    np.add(par_grad, par.grad(
                                        self.axis.axis), par_grad)
                            grad = np.vstack((grad, par_grad))
                    counter += component._nfree_param
            to_return = grad[1:, self.channel_switches] * weights
        else:
            axis = self.axis.axis[self.channel_switches]
            counter = 0
            grad = axis
            for component in self:  # Cut the parameters list
                if component.active:
                    component.fetch_values_from_array(
                        param[
                            counter:counter +
                            component._nfree_param],
                        onlyfree=True)
                    for parameter in component.free_parameters:
                        par_grad = parameter.grad(axis)
                        if parameter._twins:
                            for par in parameter._twins:
                                np.add(par_grad, par.grad(
                                    axis), par_grad)
                        grad = np.vstack((grad, par_grad))
                    counter += component._nfree_param
            to_return = grad[1:, :] * weights
        if self.signal.metadata.Signal.binned is True:
            to_return *= self.signal.axes_manager[-1].scale
        return to_return

    def _function4odr(self, param, x):
        return self._model_function(param)

    def _jacobian4odr(self, param, x):
        return self._jacobian(param, x)

    def _poisson_likelihood_function(self, param, y, weights=None):
        """Returns the likelihood function of the model for the given
        data and parameters
        """
        mf = self._model_function(param)
        with np.errstate(invalid='ignore'):
            return -(y * np.log(mf) - mf).sum()

    def _gradient_ml(self, param, y, weights=None):
        mf = self._model_function(param)
        return -(self._jacobian(param, y) * (y / mf - 1)).sum(1)

    def _gradient_ls(self, param, y, weights=None):
        gls = (2 * self._errfunc(param, y, weights) *
               self._jacobian(param, y)).sum(1)
        return gls

    def _model2plot(self, axes_manager, out_of_range2nans=True):
        old_axes_manager = None
        if axes_manager is not self.axes_manager:
            old_axes_manager = self.axes_manager
            self.axes_manager = axes_manager
            self.fetch_stored_values()
        s = self.__call__(non_convolved=False, onlyactive=True)
        if old_axes_manager is not None:
            self.axes_manager = old_axes_manager
            self.fetch_stored_values()
        if out_of_range2nans is True:
            ns = np.empty(self.axis.axis.shape)
            ns.fill(np.nan)
            ns[np.where(self.channel_switches)] = s
            s = ns
        return s

    def plot(self, plot_components=False, **kwargs):
        """Plots the current spectrum to the screen and a map with a
        cursor to explore the SI.

        Parameters
        ----------
        plot_components : bool
            If True, add a line per component to the signal figure.
        kwargs:
            All extra keyword arguements are passed to ``Signal1D.plot``


        """

        # If new coordinates are assigned
        self.signal.plot(**kwargs)
        _plot = self.signal._plot
        l1 = _plot.signal_plot.ax_lines[0]
        color = l1.line.get_color()
        l1.set_line_properties(color=color, type='scatter')

        l2 = hyperspy.drawing.signal1d.Signal1DLine()
        l2.data_function = self._model2plot
        l2.set_line_properties(color='blue', type='line')
        # Add the line to the figure
        _plot.signal_plot.add_line(l2)
        l2.plot()
        _plot.signal_plot.events.closed.connect(self._close_plot, [])

        self._model_line = l2
        self._plot = self.signal._plot
        self._connect_parameters2update_plot(self)
        if plot_components is True:
            self.enable_plot_components()
        else:
            # If we were plotted before, make sure we reset state here
            self.disable_plot_components()
        # If we were plotted before, make sure we reset state here
        self.disable_adjust_position()

    @staticmethod
    def _connect_component_line(component):
        if hasattr(component, "_model_plot_line"):
            f = component._model_plot_line._auto_update_line
            component.events.active_changed.connect(f, [])
            for parameter in component.parameters:
                parameter.events.value_changed.connect(f, [])

    @staticmethod
    def _disconnect_component_line(component):
        if hasattr(component, "_model_plot_line"):
            f = component._model_plot_line._auto_update_line
            component.events.active_changed.disconnect(f)
            for parameter in component.parameters:
                parameter.events.value_changed.disconnect(f)

    def _connect_component_lines(self):
        for component in self:
            if component.active:
                self._connect_component_line(component)

    def _disconnect_component_lines(self):
        for component in self:
            if component.active:
                self._disconnect_component_line(component)

    def _plot_component(self, component):
        line = hyperspy.drawing.signal1d.Signal1DLine()
        line.data_function = component._component2plot
        # Add the line to the figure
        self._plot.signal_plot.add_line(line)
        line.plot()
        component._model_plot_line = line
        self._connect_component_line(component)

    @staticmethod
    def _update_component_line(component):
        if hasattr(component, "_model_plot_line"):
            component._model_plot_line.update()

    def _disable_plot_component(self, component):
        self._disconnect_component_line(component)
        if hasattr(component, "_model_plot_line"):
            component._model_plot_line.close()
            del component._model_plot_line
        self._plot_components = False

    def _close_plot(self):
        if self._plot_components is True:
            self.disable_plot_components()
        self.disable_adjust_position()
        self._disconnect_parameters2update_plot(components=self)
        self._model_line = None

    def enable_plot_components(self):
        if self._plot is None or self._plot_components:
            return
        self._plot_components = True
        for component in [component for component in self if
                          component.active]:
            self._plot_component(component)

    def disable_plot_components(self):
        self._plot_components = False
        if self._plot is None:
            return
        for component in self:
            self._disable_plot_component(component)

    def enable_adjust_position(
            self, components=None, fix_them=True, show_label=True):
        """Allow changing the *x* position of component by dragging
        a vertical line that is plotted in the signal model figure

        Parameters
        ----------
        components : {None, list of components}
            If None, the position of all the active components of the
            model that has a well defined *x* position with a value
            in the axis range will get a position adjustment line.
            Otherwise the feature is added only to the given components.
            The components can be specified by name, index or themselves.
        fix_them : bool
            If True the position parameter of the components will be
            temporarily fixed until adjust position is disable.
            This can
            be useful to iteratively adjust the component positions and
            fit the model.
        show_label : bool, optional
            If True, a label showing the component name is added to the
            plot next to the vertical line.

        See also
        --------
        disable_adjust_position

        """
        if self._plot is None or not self._plot.is_active:
            self.plot()
        if self._position_widgets:
            self.disable_adjust_position()
        if components:
            components = [self._get_component(x) for x in components]
        else:
            self._adjust_position_all = (fix_them, show_label)

        components = components if components else self
        if not components:
            # The model does not have components so we do nothing
            return
        components = [
            component for component in components if component.active]
        for component in components:
            self._make_position_adjuster(component, fix_them, show_label)

    def _make_position_adjuster(self, component, fix_it, show_label):
        if (component._position is None or component._position.twin):
            return
        axis = self.axes_manager.signal_axes[0]
        # Create the vertical line and labels
        widgets = [VerticalLineWidget(self.axes_manager)]
        if show_label:
            label = LabelWidget(self.axes_manager)
            label.string = component._get_short_description().replace(
                ' component', '')
            widgets.append(label)

        self._position_widgets[component._position] = widgets
        for w in widgets:
            # Setup widget
            w.axes = (axis,)
            w.snap_position = False
            w.position = (component._position.value,)
            w.set_mpl_ax(self._plot.signal_plot.ax)
            # Create widget -> parameter connection
            w.events.moved.connect(self._on_widget_moved, {'obj': 'widget'})
            # Create parameter -> widget connection
            component._position.events.value_changed.connect(
                w._set_position, dict(value='position'))
            # Map relation for close event
            w.events.closed.connect(self._on_position_widget_close,
                                    {'obj': 'widget'})

    def _reverse_lookup_position_widget(self, widget):
        for parameter, widgets in self._position_widgets.items():
            if widget in widgets:
                return parameter
        raise KeyError()

    def _on_widget_moved(self, widget):
        parameter = self._reverse_lookup_position_widget(widget)
        es = EventSuppressor()
        for w in self._position_widgets[parameter]:
            es.add((w.events.moved, w._set_position))
        with es.suppress():
            parameter.value = widget.position[0]

    def _on_position_widget_close(self, widget):
        widget.events.closed.disconnect(self._on_position_widget_close)
        parameter = self._reverse_lookup_position_widget(widget)
        self._position_widgets[parameter].remove(widget)
        if len(self._position_widgets[parameter]) == 0:
            self._position_widgets.pop(parameter)
        parameter.events.value_changed.disconnect(widget._set_position)
        widget.events.moved.disconnect(self._on_widget_moved)

    def disable_adjust_position(self):
        """Disables the interactive adjust position feature

        See also
        --------
        enable_adjust_position

        """
        self._adjust_position_all = False
        for pws in list(self._position_widgets.values()):
            # Iteration works on a copied collection, so changes during
            # iteration should be ok
            for pw in reversed(pws):    # pws is reference, so work in reverse
                pw.close()

    def fit_component(
            self,
            component,
            signal_range="interactive",
            estimate_parameters=True,
            fit_independent=False,
            only_current=True,
            display=True,
            toolkit=None,
            **kwargs):
        signal_range = signal_range_from_roi(signal_range)
        component = self._get_component(component)
        cf = ComponentFit(self, component, signal_range,
                          estimate_parameters, fit_independent,
                          only_current, **kwargs)
        if signal_range == "interactive":
            return cf.gui(display=display, toolkit=toolkit)
        else:
            cf.apply()
    fit_component.__doc__ = \
        """
        Fit just the given component in the given signal range.

        This method is useful to obtain starting parameters for the
        components. Any keyword arguments are passed to the fit method.

        Parameters
        ----------
        component : component instance
            The component must be in the model, otherwise an exception
            is raised. The component can be specified by name, index or itself.
        signal_range : {'interactive', (left_value, right_value), None}
            If 'interactive' the signal range is selected using the span
             selector on the spectrum plot. The signal range can also
             be manually specified by passing a tuple of floats. If None
             the current signal range is used.
        estimate_parameters : bool, default True
            If True will check if the component has an
            estimate_parameters function, and use it to estimate the
            parameters in the component.
        fit_independent : bool, default False
            If True, all other components are disabled. If False, all other
            component paramemeters are fixed.
        %s
        %s

        Examples
        --------
        Signal range set interactivly

        >>> s = hs.signals.Signal1D([0,1,2,4,8,4,2,1,0])
        >>> m = s.create_model()
        >>> g1 = hs.model.components1D.Gaussian()
        >>> m.append(g1)
        >>> m.fit_component(g1)

        Signal range set through direct input

        >>> m.fit_component(g1, signal_range=(1,7))

        """ % (DISPLAY_DT, TOOLKIT_DT)
