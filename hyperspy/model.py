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
import os
import tempfile
import warnings
import numbers
import numpy as np
import scipy.odr as odr
from scipy.optimize import (leastsq,
                            fmin,
                            fmin_cg,
                            fmin_ncg,
                            fmin_bfgs,
                            fmin_l_bfgs_b,
                            fmin_tnc,
                            fmin_powell)
from traits.trait_errors import TraitError

from hyperspy import messages
import hyperspy.drawing.signal1d
from hyperspy.drawing.utils import on_figure_window_close
from hyperspy.external import progressbar
from hyperspy._signals.signal1d import Signal1D
from hyperspy.defaults_parser import preferences
from hyperspy.axes import generate_axis
from hyperspy.exceptions import WrongObjectError
from hyperspy.decorators import interactive_range_selector
from hyperspy.external.mpfit.mpfit import mpfit
from hyperspy.axes import AxesManager
from hyperspy.drawing.widgets import (DraggableVerticalLine,
                                      DraggableLabel)
from hyperspy.gui.tools import ComponentFit
from hyperspy.component import Component
from hyperspy.signal import BaseSignal
from hyperspy.misc.utils import slugify, shorten_name


class ModelComponents(object):

    """Container for model components.

    Useful to provide tab completion when running in IPython.

    """

    def __init__(self, model):
        self._model = model

    def __repr__(self):
        signature = "%4s | %25s | %25s | %25s"
        ans = signature % ('#',
                           'Attribute Name',
                           'Component Name',
                           'Component Type')
        ans += "\n"
        ans += signature % ('-' * 4, '-' * 25, '-' * 25, '-' * 25)
        if self._model:
            for i, c in enumerate(self._model):
                ans += "\n"
                name_string = c.name
                variable_name = slugify(name_string, valid_variable_name=True)
                component_type = c._id_name

                variable_name = shorten_name(variable_name, 25)
                name_string = shorten_name(name_string, 25)
                component_type = shorten_name(component_type, 25)

                ans += signature % (i,
                                    variable_name,
                                    name_string,
                                    component_type)
        return ans


class Model(list):

    """One-dimensional model and data fitting.

    A model is constructed as a linear combination of :mod:`components` that
    are added to the model using :meth:`append` or :meth:`extend`. There
    are many predifined components available in the in the :mod:`components`
    module. If needed, new components can easyly created using the code of
    existing components as a template.

    Once defined, the model can be fitted to the data using :meth:`fit` or
    :meth:`multifit`. Once the optimizer reaches the convergence criteria or
    the maximum number of iterations the new value of the component parameters
    are stored in the components.

    It is possible to access the components in the model by their name or by
    the index in the model. An example is given at the end of this docstring.

    Attributes
    ----------

    signal1D : Signal1D instance
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
        Fetch stored values of the parameters.
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

    Examples
    --------
    In the following example we create a histogram from a normal distribution
    and fit it with a gaussian component. It demonstrates how to create
    a model from a :class:`~._signals.signal1D.Signal1D` instance, add
    components to it, adjust the value of the parameters of the components,
    fit the model to the data and access the components in the model.

    >>> s = hs.signals.Signal1D(
            np.random.normal(scale=2, size=10000)).get_histogram()
    >>> g = hs.model.components.Gaussian()
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

    _firstimetouch = True

    def __hash__(self):
        # This is needed to simulate a hashable object so that PySide does not
        # raise an exception when using windows.connect
        return id(self)

    def __init__(self, signal1D):
        self.convolved = False
        self.signal = signal1D
        self.axes_manager = self.signal.axes_manager
        self.axis = self.axes_manager.signal_axes[0]
        self.axes_manager.connect(self.fetch_stored_values)

        self.free_parameters_boundaries = None
        self.channel_switches = np.array([True] * len(self.axis.axis))
        self._low_loss = None
        self._position_widgets = []
        self._plot = None
        self._model_line = None

        self.chisq = signal1D._get_navigation_signal()
        self.chisq.change_dtype("float")
        self.chisq.data.fill(np.nan)
        self.chisq.metadata.General.title = \
            self.signal.metadata.General.title + ' chi-squared'
        self.dof = self.chisq._deepcopy_with_new_data(
            np.zeros_like(
                self.chisq.data,
                dtype='int'))
        self.dof.metadata.General.title = \
            self.signal.metadata.General.title + ' degrees of freedom'
        self._suspend_update = False
        self._adjust_position_all = None
        self._plot_components = False
        self.components = ModelComponents(self)

    def __repr__(self):
        title = self.signal.metadata.General.title
        class_name = str(self.__class__).split("'")[1].split('.')[-1]

        if len(title):
            return "<%s, title: %s>" % (
                class_name, self.signal.metadata.General.title)
        else:
            return "<%s>" % class_name

    def _get_component(self, thing):
        if isinstance(thing, int) or isinstance(thing, str):
            thing = self[thing]
        elif not isinstance(thing, Component):
            raise ValueError("Not a component or component id.")
        if thing in self:
            return thing
        else:
            raise ValueError("The component is not in the model.")

    def insert(self, **kwargs):
        raise NotImplementedError

    @property
    def signal(self):
        return self._signal

    @signal.setter
    def signal(self, value):
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

    def append(self, thing):
        # Check if any of the other components in the model has the same name
        if thing in self:
            raise ValueError("Component already in model")
        component_name_list = []
        for component in self:
            component_name_list.append(component.name)
        name_string = ""
        if thing.name:
            name_string = thing.name
        else:
            name_string = thing._id_name

        if name_string in component_name_list:
            temp_name_string = name_string
            index = 0
            while temp_name_string in component_name_list:
                temp_name_string = name_string + "_" + str(index)
                index += 1
            name_string = temp_name_string
        thing.name = name_string

        thing._axes_manager = self.axes_manager
        thing._create_arrays()
        list.append(self, thing)
        thing.model = self
        setattr(self.components, slugify(name_string,
                                         valid_variable_name=True), thing)
        self._touch()
        if self._plot_components:
            self._plot_component(thing)
        if self._adjust_position_all is not None:
            self._make_position_adjuster(thing, self._adjust_position_all[0],
                                         self._adjust_position_all[1])

    def extend(self, iterable):
        for object in iterable:
            self.append(object)

    def __delitem__(self, things):
        things = self.__getitem__(things)
        if not isinstance(things, list):
            things = [things, ]
        for thing in things:
            self.remove(thing)

    def remove(self, thing, touch=True):
        """Remove component from model.

        Examples
        --------

        >>> s = hs.signals.Signal1D(np.empty(1))
        >>> m = s.create_model()
        >>> g = hs.model.components.Gaussian()
        >>> m.append(g)

        You could remove `g` like this

        >>> m.remove(g)

        Like this:

        >>> m.remove("Gaussian")

        Or like this:

        >>> m.remove(0)

        """
        thing = self._get_component(thing)
        for pw in self._position_widgets:
            if hasattr(pw, 'component') and pw.component is thing:
                pw.component._position.twin = None
                del pw.component
                pw.close()
                del pw
        if hasattr(thing, '_model_plot_line'):
            line = thing._model_plot_line
            line.close()
            del line
            idx = self.index(thing)
            self.signal._plot.signal_plot.ax_lines.remove(
                self.signal._plot.signal_plot.ax_lines[2 + idx])
        list.remove(self, thing)
        thing.model = None
        if touch is True:
            self._touch()
        if self._plot_active:
            self.update_plot()

    def _touch(self):
        """Run model setup tasks

        This function is called everytime that we add or remove components
        from the model.

        """
        if self._plot_active is True:
            self._connect_parameters2update_plot()

    __touch = _touch

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

    def _connect_parameters2update_plot(self):
        if self._plot_active is False:
            return
        for i, component in enumerate(self):
            component.connect(
                self._model_line.update)
            for parameter in component.parameters:
                parameter.connect(self._model_line.update)
        if self._plot_components is True:
            self._connect_component_lines()

    def _disconnect_parameters2update_plot(self):
        if self._model_line is None:
            return
        for component in self:
            component.disconnect(self._model_line.update)
            for parameter in component.parameters:
                parameter.disconnect(self._model_line.update)
        if self._plot_components is True:
            self._disconnect_component_lines()

    def as_signal(self, component_list=None, out_of_range_to_nan=True,
                  show_progressbar=None):
        """Returns a recreation of the dataset using the model.
        the spectral range that is not fitted is filled with nans.

        Parameters
        ----------
        component_list : list of hyperspy components, optional
            If a list of components is given, only the components given in the
            list is used in making the returned spectrum. The components can
            be specified by name, index or themselves.
        out_of_range_to_nan : bool
            If True the spectral range that is not fitted is filled with nans.
        show_progressbar : None or bool
            If True, display a progress bar. If None the default is set in
            `preferences`.

        Returns
        -------
        signal : An instance of the same class as `signal`.

        Examples
        --------
        >>> s = hs.signals.Signal1D(np.random.random((10,100)))
        >>> m = s.create_model()
        >>> l1 = hs.model.components.Lorentzian()
        >>> l2 = hs.model.components.Lorentzian()
        >>> m.append(l1)
        >>> m.append(l2)
        >>> s1 = m.as_signal()
        >>> s2 = m.as_signal(component_list=[l1])

        """
        # change actual values to whatever except bool
        _multi_on_ = '_multi_on_'
        _multi_off_ = '_multi_off_'
        if show_progressbar is None:
            show_progressbar = preferences.General.show_progressbar

        if component_list:
            component_list = [self._get_component(x) for x in component_list]
            active_state = []
            for component_ in self:
                if component_.active_is_multidimensional:
                    if component_ not in component_list:
                        active_state.append(_multi_off_)
                        component_._toggle_connect_active_array(False)
                        component_.active = False
                    else:
                        active_state.append(_multi_on_)
                else:
                    active_state.append(component_.active)
                    if component_ in component_list:
                        component_.active = True
                    else:
                        component_.active = False
        data = np.empty(self.signal.data.shape, dtype='float')
        data.fill(np.nan)
        if out_of_range_to_nan is True:
            channel_switches_backup = copy.copy(self.channel_switches)
            self.channel_switches[:] = True
        maxval = self.axes_manager.navigation_size
        pbar = progressbar.progressbar(maxval=maxval,
                                       disabled=not show_progressbar)
        i = 0
        for index in self.axes_manager:
            self.fetch_stored_values(only_fixed=False)
            data[self.axes_manager._getitem_tuple][
                self.channel_switches] = self.__call__(
                non_convolved=not self.convolved, onlyactive=True)
            i += 1
            if maxval > 0:
                pbar.update(i)
        pbar.finish()
        if out_of_range_to_nan is True:
            self.channel_switches[:] = channel_switches_backup
        signal1D = self.signal.__class__(
            data,
            axes=self.signal.axes_manager._get_axes_dicts())
        signal1D.metadata.General.title = (
            self.signal.metadata.General.title + " from fitted model")
        signal1D.metadata.Signal.binned = self.signal.metadata.Signal.binned

        if component_list:
            for component_ in self:
                active_s = active_state.pop(0)
                if isinstance(active_s, bool):
                    component_.active = active_s
                else:
                    if active_s == _multi_off_:
                        component_._toggle_connect_active_array(True)
        return signal1D

    @property
    def _plot_active(self):
        if self._plot is not None and self._plot.is_active() is True:
            return True
        else:
            return False

    def _set_p0(self):
        self.p0 = ()
        for component in self:
            if component.active:
                for parameter in component.free_parameters:
                    self.p0 = (self.p0 + (parameter.value,)
                               if parameter._number_of_elements == 1
                               else self.p0 + parameter.value)

    def set_boundaries(self):
        """Generate the boundary list.

        Necessary before fitting with a boundary aware optimizer.

        """
        self.free_parameters_boundaries = []
        for component in self:
            if component.active:
                for param in component.free_parameters:
                    if param._number_of_elements == 1:
                        self.free_parameters_boundaries.append((
                            param._bounds))
                    else:
                        self.free_parameters_boundaries.extend((
                            param._bounds))

    def set_mpfit_parameters_info(self):
        self.mpfit_parinfo = []
        for component in self:
            if component.active:
                for param in component.free_parameters:
                    limited = [False, False]
                    limits = [0, 0]
                    if param.bmin is not None:
                        limited[0] = True
                        limits[0] = param.bmin
                    if param.bmax is not None:
                        limited[1] = True
                        limits[1] = param.bmax
                    if param._number_of_elements == 1:
                        self.mpfit_parinfo.append(
                            {'limited': limited,
                             'limits': limits})
                    else:
                        self.mpfit_parinfo.extend((
                            {'limited': limited,
                             'limits': limits},) * param._number_of_elements)

    def ensure_parameters_in_bounds(self):
        """For all active components, snaps their free parameter values to
        be within their boundaries (if bounded). Does not touch the array of
        values.
        """
        for component in self:
            if component.active:
                for param in component.free_parameters:
                    bmin = -np.inf if param.bmin is None else param.bmin
                    bmax = np.inf if param.bmax is None else param.bmax
                    if not bmin <= param.value <= bmax:
                        min_d = np.abs(param.value - bmin)
                        max_d = np.abs(param.value - bmax)
                        if min_d < max_d:
                            param.value = bmin
                        else:
                            param.value = bmax

    def store_current_values(self):
        """ Store the parameters of the current coordinates into the
        parameters array.

        If the parameters array has not being defined yet it creates it filling
        it with the current parameters."""
        for component in self:
            if component.active:
                component.store_current_parameters_in_map()

    def fetch_stored_values(self, only_fixed=False):
        """Fetch the value of the parameters that has been previously stored.

        Parameters
        ----------
        only_fixed : bool
            If True, only the fixed parameters are fetched.

        See Also
        --------
        store_current_values

        """
        switch_aap = self._plot_active is not False
        if switch_aap is True:
            self._disconnect_parameters2update_plot()
        for component in self:
            component.fetch_stored_values(only_fixed=only_fixed)
        if switch_aap is True:
            self._connect_parameters2update_plot()
            self.update_plot()

    def update_plot(self, *args, **kwargs):
        """Update model plot.

        The updating can be suspended using `suspend_update`.

        See Also
        --------
        suspend_update
        resume_update

        """
        if self._plot_active is True and self._suspend_update is False:
            try:
                self._update_model_line()
                for component in [component for component in self if
                                  component.active is True]:
                    self._update_component_line(component)
            except:
                self._disconnect_parameters2update_plot()

    def suspend_update(self):
        """Prevents plot from updating until resume_update() is called

        See Also
        --------
        resume_update
        update_plot
        """
        if self._suspend_update is False:
            self._suspend_update = True
            self._disconnect_parameters2update_plot()
        else:
            warnings.warn("Update already suspended, does nothing.")

    def resume_update(self, update=True):
        """Resumes plot update after suspension by suspend_update()

        Parameters
        ----------
        update : bool, optional
            If True, also updates plot after resuming (default).

        See Also
        --------
        suspend_update
        update_plot
        """
        if self._suspend_update is True:
            self._suspend_update = False
            self._connect_parameters2update_plot()
            if update is True:
                # Ideally, the update flag should in stead work like this:
                # If update is true, update_plot is called if any action
                # would have called it while updating was suspended.
                # However, this is prohibitively difficult to track, so
                # in stead it is simply assume that a change has happened
                # between suspend and resume, and therefore that the plot
                # needs to update. As we do not know what has changed,
                # all components need to update. This can however be
                # suppressed by setting update to false
                self.update_plot()
        else:
            warnings.warn("Update not suspended, nothing to resume.")

    def _update_model_line(self):
        if (self._plot_active is True and
                self._model_line is not None):
            self._model_line.update()

    def _fetch_values_from_p0(self, p_std=None):
        """Fetch the parameter values from the output of the optimzer `self.p0`

        Parameters
        ----------
        p_std : array
            array containing the corresponding standard deviatio
            n

        """
        comp_p_std = None
        counter = 0
        for component in self:  # Cut the parameters list
            if component.active is True:
                if p_std is not None:
                    comp_p_std = p_std[
                        counter: counter +
                        component._nfree_param]
                component.fetch_values_from_array(
                    self.p0[counter: counter + component._nfree_param],
                    comp_p_std, onlyfree=True)
                counter += component._nfree_param

    # Defines the functions for the fitting process -------------------------
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
            ns[self.channel_switches] = s
            s = ns
        return s

    def __call__(self, non_convolved=False, onlyactive=False):
        """Returns the corresponding model for the current coordinates

        Parameters
        ----------
        non_convolved : bool
            If True it will return the deconvolved model
        only_active : bool
            If True, only the active components will be used to build the
            model.

        cursor: 1 or 2

        Returns
        -------
        numpy array
        """

        if self.convolved is False or non_convolved is True:
            axis = self.axis.axis[self.channel_switches]
            sum_ = np.zeros(len(axis))
            if onlyactive is True:
                for component in self:  # Cut the parameters list
                    if component.active:
                        np.add(sum_, component.function(axis),
                               sum_)
            else:
                for component in self:  # Cut the parameters list
                    np.add(sum_, component.function(axis),
                           sum_)
            to_return = sum_

        else:  # convolved
            counter = 0
            sum_convolved = np.zeros(len(self.convolution_axis))
            sum_ = np.zeros(len(self.axis.axis))
            for component in self:  # Cut the parameters list
                if onlyactive:
                    if component.active:
                        if component.convolved:
                            np.add(sum_convolved,
                                   component.function(
                                       self.convolution_axis), sum_convolved)
                        else:
                            np.add(sum_,
                                   component.function(self.axis.axis), sum_)
                        counter += component._nfree_param
                else:
                    if component.convolved:
                        np.add(sum_convolved,
                               component.function(self.convolution_axis),
                               sum_convolved)
                    else:
                        np.add(sum_, component.function(self.axis.axis),
                               sum_)
                    counter += component._nfree_param
            to_return = sum_ + np.convolve(
                self.low_loss(self.axes_manager),
                sum_convolved, mode="valid")
            to_return = to_return[self.channel_switches]
        if self.signal.metadata.Signal.binned is True:
            to_return *= self.signal.axes_manager[-1].scale
        return to_return

    # TODO: the way it uses the axes
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

        Notes
        -----
        To use the full energy range call the function without arguments.
        """
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
        i1, i2 = self.axis.value_range_to_indices(x1, x2)
        self._add_signal_range_in_pixels(i1, i2)

    def reset_the_signal_range(self):
        self.channel_switches[:] = True
        self.update_plot()

    def _model_function(self, param):

        if self.convolved is True:
            counter = 0
            sum_convolved = np.zeros(len(self.convolution_axis))
            sum = np.zeros(len(self.axis.axis))
            for component in self:  # Cut the parameters list
                if component.active:
                    if component.convolved is True:
                        np.add(sum_convolved, component.__tempcall__(param[
                            counter:counter + component._nfree_param],
                            self.convolution_axis), sum_convolved)
                    else:
                        np.add(
                            sum,
                            component.__tempcall__(
                                param[
                                    counter:counter +
                                    component._nfree_param],
                                self.axis.axis),
                            sum)
                    counter += component._nfree_param

            to_return = (sum + np.convolve(self.low_loss(self.axes_manager),
                                           sum_convolved, mode="valid"))[
                self.channel_switches]

        else:
            axis = self.axis.axis[self.channel_switches]
            counter = 0
            first = True
            for component in self:  # Cut the parameters list
                if component.active:
                    if first is True:
                        sum = component.__tempcall__(
                            param[
                                counter:counter +
                                component._nfree_param],
                            axis)
                        first = False
                    else:
                        sum += component.__tempcall__(
                            param[
                                counter:counter +
                                component._nfree_param],
                            axis)
                    counter += component._nfree_param
            to_return = sum

        if self.signal.metadata.Signal.binned is True:
            to_return *= self.signal.axes_manager[-1].scale
        return to_return

    # noinspection PyAssignmentToLoopOrWithParameter
    def _jacobian(self, param, y, weights=None):
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
                        counter += component._nfree_param
                    else:
                        for parameter in component.free_parameters:
                            par_grad = parameter.grad(self.axis.axis)
                            if parameter._twins:
                                for par in parameter._twins:
                                    np.add(par_grad, par.grad(
                                        self.axis.axis), par_grad)
                            grad = np.vstack((grad, par_grad))
                        counter += component._nfree_param
            if weights is None:
                to_return = grad[1:, self.channel_switches]
            else:
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
            if weights is None:
                to_return = grad[1:, :]
            else:
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

    def _errfunc(self, param, y, weights=None):
        errfunc = self._model_function(param) - y
        if weights is None:
            return errfunc
        else:
            return errfunc * weights

    def _errfunc2(self, param, y, weights=None):
        if weights is None:
            return ((self._errfunc(param, y)) ** 2).sum()
        else:
            return ((weights * self._errfunc(param, y)) ** 2).sum()

    def _gradient_ls(self, param, y, weights=None):
        gls = (2 * self._errfunc(param, y, weights) *
               self._jacobian(param, y)).sum(1)
        return gls

    def _errfunc4mpfit(self, p, fjac=None, x=None, y=None,
                       weights=None):
        if fjac is None:
            errfunc = self._model_function(p) - y
            if weights is not None:
                errfunc *= weights
            status = 0
            return [status, errfunc]
        else:
            return [0, self._jacobian(p, y).T]

    def _calculate_chisq(self):
        if self.signal.metadata.has_item('Signal.Noise_properties.variance'):

            variance = self.signal.metadata.Signal.Noise_properties.variance
            if isinstance(variance, BaseSignal):
                variance = variance.data.__getitem__(
                    self.signal.axes_manager._getitem_tuple
                )[self.channel_switches]
        else:
            variance = 1.0
        d = self(onlyactive=True) - self.signal()[self.channel_switches]
        d *= d / (1. * variance)  # d = difference^2 / variance.
        self.chisq.data[self.signal.axes_manager.indices[::-1]] = sum(d)

    def _set_current_degrees_of_freedom(self):
        self.dof.data[self.signal.axes_manager.indices[::-1]] = len(self.p0)

    @property
    def red_chisq(self):
        """Reduced chi-squared. Calculated from self.chisq and self.dof
        """
        tmp = self.chisq / (- self.dof + sum(self.channel_switches) - 1)
        tmp.metadata.General.title = self.signal.metadata.General.title + \
            ' reduced chi-squared'
        return tmp

    def fit(self, fitter=None, method='ls', grad=False,
            bounded=False, ext_bounding=False, update_plot=False,
            **kwargs):
        """Fits the model to the experimental data.

        The chi-squared, reduced chi-squared and the degrees of freedom are
        computed automatically when fitting. They are stored as signals, in the
        `chisq`, `red_chisq`  and `dof`. Note that,
        unless ``metadata.Signal.Noise_properties.variance`` contains an
        accurate estimation of the variance of the data, the chi-squared and
        reduced chi-squared cannot be computed correctly. This is also true for
        homocedastic noise.

        Parameters
        ----------
        fitter : {None, "leastsq", "odr", "mpfit", "fmin"}
            The optimizer to perform the fitting. If None the fitter
            defined in `preferences.Model.default_fitter` is used.
            "leastsq" performs least squares using the Levenberg–Marquardt
            algorithm.
            "mpfit"  performs least squares using the Levenberg–Marquardt
            algorithm and, unlike "leastsq", support bounded optimization.
            "fmin" performs curve fitting using a downhill simplex algorithm.
            It is less robust than the Levenberg-Marquardt based optimizers,
            but, at present, it is the only one that support maximum likelihood
            optimization for poissonian noise.
            "odr" performs the optimization using the orthogonal distance
            regression algorithm. It does not support bounds.
            "leastsq", "odr" and "mpfit" can estimate the standard deviation of
            the estimated value of the parameters if the
            "metada.Signal.Noise_properties.variance" attribute is defined.
            Note that if it is not defined the standard deviation is estimated
            using variance equal 1, what, if the noise is heterocedatic, will
            result in a biased estimation of the parameter values and errors.
            If `variance` is a `Signal` instance of the same
            `navigation_dimension` as the signal, and `method` is "ls"
            weighted least squares is performed.
        method : {'ls', 'ml'}
            Choose 'ls' (default) for least squares and 'ml' for poissonian
            maximum-likelihood estimation.  The latter is only available when
            `fitter` is "fmin".
        grad : bool
            If True, the analytical gradient is used if defined to
            speed up the optimization.
        bounded : bool
            If True performs bounded optimization if the fitter
            supports it. Currently only "mpfit" support it.
        update_plot : bool
            If True, the plot is updated during the optimization
            process. It slows down the optimization but it permits
            to visualize the optimization progress.
        ext_bounding : bool
            If True, enforce bounding by keeping the value of the
            parameters constant out of the defined bounding area.

        **kwargs : key word arguments
            Any extra key word argument will be passed to the chosen
            fitter. For more information read the docstring of the optimizer
            of your choice in `scipy.optimize`.

        See Also
        --------
        multifit

        """

        if fitter is None:
            fitter = preferences.Model.default_fitter
        switch_aap = (update_plot != self._plot_active)
        if switch_aap is True and update_plot is False:
            self._disconnect_parameters2update_plot()

        if bounded is True:
            if fitter not in ("mpfit", "tnc", "l_bfgs_b"):
                raise NotImplementedError("Bounded optimization is only"
                                          "available for the mpfit "
                                          "optimizer.")
            else:
                # this has to be done before setting the p0, so moved things
                # around
                self.ensure_parameters_in_bounds()

        self.p_std = None
        self._set_p0()
        if ext_bounding:
            self._enable_ext_bounding()
        if grad is False:
            approx_grad = True
            jacobian = None
            odr_jacobian = None
            grad_ml = None
            grad_ls = None
        else:
            approx_grad = False
            jacobian = self._jacobian
            odr_jacobian = self._jacobian4odr
            grad_ml = self._gradient_ml
            grad_ls = self._gradient_ls

        if method == 'ml':
            weights = None
            if fitter != "fmin":
                raise NotImplementedError("Maximum likelihood estimation "
                                          'is only implemented for the "fmin" '
                                          'optimizer')
        elif method == "ls":
            if ("Signal.Noise_properties.variance" not in
                    self.signal.metadata):
                variance = 1
            else:
                variance = self.signal.metadata.Signal.\
                    Noise_properties.variance
                if isinstance(variance, BaseSignal):
                    if (variance.axes_manager.navigation_shape ==
                            self.signal.axes_manager.navigation_shape):
                        variance = variance.data.__getitem__(
                            self.axes_manager._getitem_tuple)[
                            self.channel_switches]
                    else:
                        raise AttributeError(
                            "The `navigation_shape` of the variance signals "
                            "is not equal to the variance shape of the "
                            "signal")
                elif not isinstance(variance, numbers.Number):
                    raise AttributeError(
                        "Variance must be a number or a `Signal` instance but "
                        "currently it is a %s" % type(variance))

            weights = 1. / np.sqrt(variance)
        else:
            raise ValueError(
                'method must be "ls" or "ml" but %s given' %
                method)
        args = (self.signal()[self.channel_switches],
                weights)

        # Least squares "dedicated" fitters
        if fitter == "leastsq":
            output = \
                leastsq(self._errfunc, self.p0[:], Dfun=jacobian,
                        col_deriv=1, args=args, full_output=True, **kwargs)

            self.p0, pcov = output[0:2]

            if (self.axis.size > len(self.p0)) and pcov is not None:
                pcov *= ((self._errfunc(self.p0, *args) ** 2).sum() /
                         (len(args[0]) - len(self.p0)))
                self.p_std = np.sqrt(np.diag(pcov))
            self.fit_output = output

        elif fitter == "odr":
            modelo = odr.Model(fcn=self._function4odr,
                               fjacb=odr_jacobian)
            mydata = odr.RealData(
                self.axis.axis[self.channel_switches],
                self.signal()[self.channel_switches],
                sx=None,
                sy=(1 / weights if weights is not None else None))
            myodr = odr.ODR(mydata, modelo, beta0=self.p0[:])
            myoutput = myodr.run()
            result = myoutput.beta
            self.p_std = myoutput.sd_beta
            self.p0 = result
            self.fit_output = myoutput

        elif fitter == 'mpfit':
            autoderivative = 1
            if grad is True:
                autoderivative = 0
            if bounded is True:
                self.set_mpfit_parameters_info()
            elif bounded is False:
                self.mpfit_parinfo = None
            m = mpfit(self._errfunc4mpfit, self.p0[:],
                      parinfo=self.mpfit_parinfo, functkw={
                          'y': self.signal()[self.channel_switches],
                          'weights': weights}, autoderivative=autoderivative,
                      quiet=1)
            self.p0 = m.params
            if (self.axis.size > len(self.p0)) and m.perror is not None:
                self.p_std = m.perror * np.sqrt(
                    (self._errfunc(self.p0, *args) ** 2).sum() /
                    (len(args[0]) - len(self.p0)))
            self.fit_output = m
        else:
            # General optimizers (incluiding constrained ones(tnc,l_bfgs_b)
            # Least squares or maximum likelihood
            if method == 'ml':
                tominimize = self._poisson_likelihood_function
                fprime = grad_ml
            elif method in ['ls', "wls"]:
                tominimize = self._errfunc2
                fprime = grad_ls

            # OPTIMIZERS

            # Simple (don't use gradient)
            if fitter == "fmin":
                self.p0 = fmin(
                    tominimize, self.p0, args=args, **kwargs)
            elif fitter == "powell":
                self.p0 = fmin_powell(tominimize, self.p0, args=args,
                                      **kwargs)

            # Make use of the gradient
            elif fitter == "cg":
                self.p0 = fmin_cg(tominimize, self.p0, fprime=fprime,
                                  args=args, **kwargs)
            elif fitter == "ncg":
                self.p0 = fmin_ncg(tominimize, self.p0, fprime=fprime,
                                   args=args, **kwargs)
            elif fitter == "bfgs":
                self.p0 = fmin_bfgs(
                    tominimize, self.p0, fprime=fprime,
                    args=args, **kwargs)

            # Constrainded optimizers

            # Use gradient
            elif fitter == "tnc":
                if bounded is True:
                    self.set_boundaries()
                elif bounded is False:
                    self.free_parameters_boundaries = None
                self.p0 = fmin_tnc(
                    tominimize,
                    self.p0,
                    fprime=fprime,
                    args=args,
                    bounds=self.free_parameters_boundaries,
                    approx_grad=approx_grad,
                    **kwargs)[0]
            elif fitter == "l_bfgs_b":
                if bounded is True:
                    self.set_boundaries()
                elif bounded is False:
                    self.free_parameters_boundaries = None
                self.p0 = fmin_l_bfgs_b(tominimize, self.p0,
                                        fprime=fprime, args=args,
                                        bounds=self.free_parameters_boundaries,
                                        approx_grad=approx_grad, **kwargs)[0]
            else:
                print("""
                The %s optimizer is not available.

                Available optimizers:
                Unconstrained:
                --------------
                Only least Squares: leastsq and odr
                General: fmin, powell, cg, ncg, bfgs

                Cosntrained:
                ------------
                tnc and l_bfgs_b
                """ % fitter)
        if np.iterable(self.p0) == 0:
            self.p0 = (self.p0,)
        self._fetch_values_from_p0(p_std=self.p_std)
        self.store_current_values()
        self._calculate_chisq()
        self._set_current_degrees_of_freedom()
        if ext_bounding is True:
            self._disable_ext_bounding()
        if switch_aap is True and update_plot is False:
            self._connect_parameters2update_plot()
            self.update_plot()

    def multifit(self, mask=None, fetch_only_fixed=False,
                 autosave=False, autosave_every=10, show_progressbar=None,
                 **kwargs):
        """Fit the data to the model at all the positions of the
        navigation dimensions.

        Parameters
        ----------

        mask : {None, numpy.array}
            To mask (do not fit) at certain position pass a numpy.array
            of type bool where True indicates that the data will not be
            fitted at the given position.
        fetch_only_fixed : bool
            If True, only the fixed parameters values will be updated
            when changing the positon.
        autosave : bool
            If True, the result of the fit will be saved automatically
            with a frequency defined by autosave_every.
        autosave_every : int
            Save the result of fitting every given number of spectra.

        show_progressbar : None or bool
            If True, display a progress bar. If None the default is set in
            `preferences`.

        **kwargs : key word arguments
            Any extra key word argument will be passed to
            the fit method. See the fit method documentation for
            a list of valid arguments.

        See Also
        --------
        fit

        """
        if show_progressbar is None:
            show_progressbar = preferences.General.show_progressbar

        if autosave is not False:
            fd, autosave_fn = tempfile.mkstemp(
                prefix='hyperspy_autosave-',
                dir='.', suffix='.npz')
            os.close(fd)
            autosave_fn = autosave_fn[:-4]
            messages.information(
                "Autosaving each %s pixels to %s.npz" % (autosave_every,
                                                         autosave_fn))
            messages.information(
                "When multifit finishes its job the file will be deleted")
        if mask is not None and (
            mask.shape != tuple(
                self.axes_manager._navigation_shape_in_array)):
            messages.warning_exit(
                "The mask must be a numpy array of boolen type with "
                " shape: %s" +
                str(self.axes_manager._navigation_shape_in_array))
        masked_elements = 0 if mask is None else mask.sum()
        maxval = self.axes_manager.navigation_size - masked_elements
        if maxval > 0:
            pbar = progressbar.progressbar(maxval=maxval,
                                           disabled=not show_progressbar)
        if 'bounded' in kwargs and kwargs['bounded'] is True:
            if kwargs['fitter'] not in ("tnc", "l_bfgs_b", "mpfit"):
                messages.information(
                    "The chosen fitter does not suppport bounding."
                    "If you require bounding please select one of the "
                    "following fitters instead: mpfit, tnc, l_bfgs_b")
                kwargs['bounded'] = False
        i = 0
        self.axes_manager.disconnect(self.fetch_stored_values)
        for index in self.axes_manager:
            if mask is None or not mask[index[::-1]]:
                self.fetch_stored_values(only_fixed=fetch_only_fixed)
                self.fit(**kwargs)
                i += 1
                if maxval > 0:
                    pbar.update(i)
            if autosave is True and i % autosave_every == 0:
                self.save_parameters2file(autosave_fn)
        if maxval > 0:
            pbar.finish()
        self.axes_manager.connect(self.fetch_stored_values)
        if autosave is True:
            messages.information(
                'Deleting the temporary file %s pixels' % (
                    autosave_fn + 'npz'))
            os.remove(autosave_fn + '.npz')

    def save_parameters2file(self, filename):
        """Save the parameters array in binary format.

        The data is saved to a single file in numpy's uncompressed ``.npz``
        format.

        Parameters
        ----------
        filename : str

        See Also
        --------
        load_parameters_from_file, export_results

        Notes
        -----
        This method can be used to save the current state of the model in a way
        that can be loaded back to recreate the it using `load_parameters_from
        file`. Actually, as of HyperSpy 0.8 this is the only way to do so.
        However, this is known to be brittle. For example see
        https://github.com/hyperspy/hyperspy/issues/341.

        """
        kwds = {}
        i = 0
        for component in self:
            cname = component.name.lower().replace(' ', '_')
            for param in component.parameters:
                pname = param.name.lower().replace(' ', '_')
                kwds['%s_%s.%s' % (i, cname, pname)] = param.map
            i += 1
        np.savez(filename, **kwds)

    def load_parameters_from_file(self, filename):
        """Loads the parameters array from  a binary file written with the
        'save_parameters2file' function.

        Parameters
        ---------
        filename : str

        See Also
        --------
        save_parameters2file, export_results

        Notes
        -----
        In combination with `save_parameters2file`, this method can be used to
        recreate a model stored in a file. Actually, before HyperSpy 0.8 this
        is the only way to do so.  However, this is known to be brittle. For
        example see https://github.com/hyperspy/hyperspy/issues/341.

        """
        f = np.load(filename)
        i = 0
        for component in self:  # Cut the parameters list
            cname = component.name.lower().replace(' ', '_')
            for param in component.parameters:
                pname = param.name.lower().replace(' ', '_')
                param.map = f['%s_%s.%s' % (i, cname, pname)]
            i += 1

        self.fetch_stored_values()

    def plot(self, plot_components=False):
        """Plots the current signal to the screen and a map with a
        cursor to explore the SI.

        Parameters
        ----------
        plot_components : bool
            If True, add a line per component to the signal figure.

        """

        # If new coordinates are assigned
        self.signal.plot()
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
        on_figure_window_close(_plot.signal_plot.figure,
                               self._close_plot)

        self._model_line = l2
        self._plot = self.signal._plot
        self._connect_parameters2update_plot()
        if plot_components is True:
            self.enable_plot_components()

    @staticmethod
    def _connect_component_line(component):
        if hasattr(component, "_model_plot_line"):
            component.connect(component._model_plot_line.update)
            for parameter in component.parameters:
                parameter.connect(component._model_plot_line.update)

    @staticmethod
    def _disconnect_component_line(component):
        if hasattr(component, "_model_plot_line"):
            component.disconnect(component._model_plot_line.update)
            for parameter in component.parameters:
                parameter.disconnect(component._model_plot_line.update)

    def _connect_component_lines(self):
        for component in [component for component in self if
                          component.active]:
            self._connect_component_line(component)

    def _disconnect_component_lines(self):
        for component in [component for component in self if
                          component.active]:
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
        self._disconnect_parameters2update_plot()
        self._model_line = None

    def enable_plot_components(self):
        if self._plot is None or self._plot_components:
            return
        self._plot_components = True
        for component in [component for component in self if
                          component.active]:
            self._plot_component(component)

    def disable_plot_components(self):
        if self._plot is None:
            return
        for component in self:
            self._disable_plot_component(component)
        self._plot_components = False

    def assign_current_values_to_all(self, components_list=None, mask=None):
        """Set parameter values for all positions to the current ones.

        Parameters
        ----------
        component_list : list of components, optional
            If a list of components is given, the operation will be performed
            only in the value of the parameters of the given components.
            The components can be specified by name, index or themselves.
        mask : boolean numpy array or None, optional
            The operation won't be performed where mask is True.

        """
        if components_list is None:
            components_list = []
            for comp in self:
                if comp.active:
                    components_list.append(comp)
        else:
            components_list = [self._get_component(x) for x in components_list]

        for comp in components_list:
            for parameter in comp.parameters:
                parameter.assign_current_value_to_all(mask=mask)

    def _enable_ext_bounding(self, components=None):
        """
        """
        if components is None:
            components = self
        for component in components:
            for parameter in component.parameters:
                parameter.ext_bounded = True

    def _disable_ext_bounding(self, components=None):
        """
        """
        if components is None:
            components = self
        for component in components:
            for parameter in component.parameters:
                parameter.ext_bounded = False

    def export_results(self, folder=None, format=None, save_std=False,
                       only_free=True, only_active=True):
        """Export the results of the parameters of the model to the desired
        folder.

        Parameters
        ----------
        folder : str or None
            The path to the folder where the file will be saved. If `None` the
            current folder is used by default.
        format : str
            The format to which the data will be exported. It must be the
            extension of any format supported by HyperSpy. If None, the default
            format for exporting as defined in the `Preferences` will be used.
        save_std : bool
            If True, also the standard deviation will be saved.
        only_free : bool
            If True, only the value of the parameters that are free will be
            exported.
        only_active : bool
            If True, only the value of the active parameters will be exported.

        Notes
        -----
        The name of the files will be determined by each the Component and
        each Parameter name attributes. Therefore, it is possible to customise
        the file names modify the name attributes.

        """
        for component in self:
            if only_active is False or component.active:
                component.export(folder=folder, format=format,
                                 save_std=save_std, only_free=only_free)

    def plot_results(self, only_free=True, only_active=True):
        """Plot the value of the parameters of the model

        Parameters
        ----------

        only_free : bool
            If True, only the value of the parameters that are free will be
            plotted.
        only_active : bool
            If True, only the value of the active parameters will be plotted.

        Notes
        -----
        The name of the files will be determined by each the Component and
        each Parameter name attributes. Therefore, it is possible to customise
        the file names modify the name attributes.

        """
        for component in self:
            if only_active is False or component.active:
                component.plot(only_free=only_free)

    def print_current_values(self, only_free=True):
        """Print the value of each parameter of the model.

        Parameters
        ----------
        only_free : bool
            If True, only the value of the parameters that are free will
             be printed.

        """
        print("Components\tParameter\tValue")
        for component in self:
            if component.active:
                if component.name:
                    print(component.name)
                else:
                    print(component._id_name)
                parameters = component.free_parameters if only_free \
                    else component.parameters
                for parameter in parameters:
                    if not hasattr(parameter.value, '__iter__'):
                        print("\t\t%s\t%g" % (
                            parameter.name, parameter.value))

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
        if (self._plot is None or
                self._plot.is_active() is False):
            self.plot()
        if self._position_widgets:
            self.disable_adjust_position()
        on_figure_window_close(self._plot.signal_plot.figure,
                               self.disable_adjust_position)
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
        if (component._position is not None and
                not component._position.twin):
            set_value = component._position._set_value
            get_value = component._position._get_value
        else:
            return
        # Create an AxesManager for the widget
        axis_dict = self.axes_manager.signal_axes[0].get_axis_dictionary()
        am = AxesManager([axis_dict, ])
        am._axes[0].navigate = True
        try:
            am._axes[0].value = get_value()
        except TraitError:
            # The value is outside of the axis range
            return
        # Create the vertical line and labels
        if show_label:
            self._position_widgets.extend((
                DraggableVerticalLine(am),
                DraggableLabel(am),))
            # Store the component for bookkeeping, and to reset
            # its twin when disabling adjust position
            self._position_widgets[-2].component = component
            self._position_widgets[-1].component = component
            w = self._position_widgets[-1]
            w.string = component._get_short_description().replace(
                ' component', '')
            w.set_mpl_ax(self._plot.signal_plot.ax)
            self._position_widgets[-2].set_mpl_ax(
                self._plot.signal_plot.ax)
        else:
            self._position_widgets.extend((
                DraggableVerticalLine(am),))
            # Store the component for bookkeeping, and to reset
            # its twin when disabling adjust position
            self._position_widgets[-1].component = component
            self._position_widgets[-1].set_mpl_ax(
                self._plot.signal_plot.ax)
        # Create widget -> parameter connection
        am._axes[0].continuous_value = True
        am._axes[0].on_trait_change(set_value, 'value')
        # Create parameter -> widget connection
        # This is done with a duck typing trick
        # We disguise the AxesManager axis of Parameter by adding
        # the _twin attribute
        am._axes[0]._twins = set()
        component._position.twin = am._axes[0]

    def disable_adjust_position(self):
        """Disables the interactive adjust position feature

        See also
        --------
        enable_adjust_position

        """
        self._adjust_position_all = None
        while self._position_widgets:
            pw = self._position_widgets.pop()
            if hasattr(pw, 'component'):
                pw.component._position.twin = None
                del pw.component
            pw.close()
            del pw

    def fit_component(
            self,
            component,
            signal_range="interactive",
            estimate_parameters=True,
            fit_independent=False,
            only_current=True,
            **kwargs):
        """Fit just the given component in the given signal range.

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
            component parameters are fixed.
        only_current : bool, default True
            If True, will only fit the currently selected index in the model.
            If False, will fit the full dataset.
            component paramemeters are fixed.

        Examples
        --------
        Signal range set interactivly

        >>> g1 = hs.model.components.Gaussian()
        >>> m.append(g1)
        >>> m.fit_component(g1)

        Signal range set through direct input

        >>> m.fit_component(g1, signal_range=(50,100))
        """
        component = self._get_component(component)
        cf = ComponentFit(self, component, signal_range,
                          estimate_parameters, fit_independent,
                          only_current, **kwargs)
        if signal_range == "interactive":
            cf.edit_traits()
        else:
            cf.apply()

    def set_parameters_not_free(self, component_list=None,
                                parameter_name_list=None):
        """
        Sets the parameters in a component in a model to not free.

        Parameters
        ----------
        component_list : None, or list of hyperspy components, optional
            If None, will apply the function to all components in the model.
            If list of components, will apply the functions to the components
            in the list.  The components can be specified by name, index or
            themselves.
        parameter_name_list : None or list of strings, optional
            If None, will set all the parameters to not free.
            If list of strings, will set all the parameters with the same name
            as the strings in parameter_name_list to not free.

        Examples
        --------
        >>> v1 = hs.model.components.Voigt()
        >>> m.append(v1)
        >>> m.set_parameters_not_free()

        >>> m.set_parameters_not_free(component_list=[v1],
                                      parameter_name_list=['area','centre'])

        See also
        --------
        set_parameters_free
        hyperspy.component.Component.set_parameters_free
        hyperspy.component.Component.set_parameters_not_free
        """

        if not component_list:
            component_list = []
            for _component in self:
                component_list.append(_component)
        else:
            component_list = [self._get_component(x) for x in component_list]

        for _component in component_list:
            _component.set_parameters_not_free(parameter_name_list)

    def set_parameters_free(self, component_list=None,
                            parameter_name_list=None):
        """
        Sets the parameters in a component in a model to free.

        Parameters
        ----------
        component_list : None, or list of hyperspy components, optional
            If None, will apply the function to all components in the model.
            If list of components, will apply the functions to the components
            in the list. The components can be specified by name, index or
            themselves.

        parameter_name_list : None or list of strings, optional
            If None, will set all the parameters to not free.
            If list of strings, will set all the parameters with the same name
            as the strings in parameter_name_list to not free.

        Examples
        --------
        >>> v1 = hs.model.components.Voigt()
        >>> m.append(v1)
        >>> m.set_parameters_free()
        >>> m.set_parameters_free(component_list=[v1],
                                  parameter_name_list=['area','centre'])

        See also
        --------
        set_parameters_not_free
        hyperspy.component.Component.set_parameters_free
        hyperspy.component.Component.set_parameters_not_free
        """

        if not component_list:
            component_list = []
            for _component in self:
                component_list.append(_component)
        else:
            component_list = [self._get_component(x) for x in component_list]

        for _component in component_list:
            _component.set_parameters_free(parameter_name_list)

    def set_parameters_value(
            self,
            parameter_name,
            value,
            component_list=None,
            only_current=False):
        """
        Sets the value of a parameter in components in a model to a specified
        value

        Parameters
        ----------
        parameter_name : string
            Name of the parameter whos value will be changed
        value : number
            The new value of the parameter
        component_list : list of hyperspy components, optional
            A list of components whos parameters will changed. The components
            can be specified by name, index or themselves.

        only_current : bool, default False
            If True, will only change the parameter value at the current
            position in the model.
            If False, will change the parameter value for all the positions.

        Examples
        --------
        >>> v1 = hs.model.components.Voigt()
        >>> v2 = hs.model.components.Voigt()
        >>> m.extend([v1,v2])
        >>> m.set_parameters_value('area', 5)
        >>> m.set_parameters_value('area', 5, component_list=[v1])
        >>> m.set_parameters_value('area', 5, component_list=[v1],
                                   only_current=True)

        """

        if not component_list:
            component_list = []
            for _component in self:
                component_list.append(_component)
        else:
            component_list = [self._get_component(x) for x in component_list]

        for _component in component_list:
            for _parameter in _component.parameters:
                if _parameter.name == parameter_name:
                    if only_current:
                        _parameter.value = value
                        _parameter.store_current_value_in_array()
                    else:
                        _parameter.value = value
                        _parameter.assign_current_value_to_all()

    def set_component_active_value(
            self, value, component_list=None, only_current=False):
        """
        Sets the component 'active' parameter to a specified value

        Parameters
        ----------
        value : bool
            The new value of the 'active' parameter
        component_list : list of hyperspy components, optional
            A list of components whos parameters will changed. The components
            can be specified by name, index or themselves.

        only_current : bool, default False
            If True, will only change the parameter value at the current
            position in the model.
            If False, will change the parameter value for all the positions.

        Examples
        --------
        >>> v1 = hs.model.components.Voigt()
        >>> v2 = hs.model.components.Voigt()
        >>> m.extend([v1,v2])
        >>> m.set_component_active_value(False)
        >>> m.set_component_active_value(True, component_list=[v1])
        >>> m.set_component_active_value(False, component_list=[v1],
                                         only_current=True)

        """

        if not component_list:
            component_list = []
            for _component in self:
                component_list.append(_component)
        else:
            component_list = [self._get_component(x) for x in component_list]

        for _component in component_list:
            _component.active = value
            if _component.active_is_multidimensional:
                if only_current:
                    _component._active_array[
                        self.axes_manager.indices[
                            ::-
                            1]] = value
                else:
                    _component._active_array.fill(value)

    def __getitem__(self, value):
        """x.__getitem__(y) <==> x[y]"""
        if isinstance(value, str):
            component_list = []
            for component in self:
                if component.name:
                    if component.name == value:
                        component_list.append(component)
                elif component._id_name == value:
                    component_list.append(component)
            if component_list:
                if len(component_list) == 1:
                    return component_list[0]
                else:
                    raise ValueError(
                        "There are several components with "
                        "the name \"" + str(value) + "\"")
            else:
                raise ValueError(
                    "Component name \"" + str(value) +
                    "\" not found in model")
        else:
            return list.__getitem__(self, value)

    def notebook_interaction(self):
        """Creates interactive notebook widgets for all components and
        parameters, if available.

        Requires `ipywidgets` to be installed.
        """
        from ipywidgets import Accordion
        from traitlets import TraitError as TraitletError
        from IPython.display import display as ip_display

        try:
            children = [component.notebook_interaction(False) for component in
                        self]
            accord = Accordion(children=children)
            for i, comp in enumerate(self):
                accord.set_title(i, comp.name)
            ip_display(accord)
        except TraitletError:
            print('This function is only avialable when running in a notebook')
