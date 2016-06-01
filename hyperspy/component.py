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

import os

import numpy as np
import functools
import warnings

import traits.api as t
from traits.trait_numeric import Array

from hyperspy.defaults_parser import preferences
from hyperspy.misc.utils import slugify
from hyperspy.misc.io.tools import (incremental_filename,
                                    append2pathname,)
from hyperspy.exceptions import NavigationDimensionError


class NoneFloat(t.CFloat):   # Lazy solution, but usable
    default_value = None

    def validate(self, object, name, value):
        if value == "None" or value == b"None":
            value = None
        if value is None:
            super(NoneFloat, self).validate(object, name, 0)
            return None
        return super(NoneFloat, self).validate(object, name, value)


class Parameter(t.HasTraits):

    """Model parameter

    Attributes
    ----------
    value : float or array
        The value of the parameter for the current location. The value
        for other locations is stored in map.
    bmin, bmax: float
        Lower and upper bounds of the parameter value.
    twin : {None, Parameter}
        If it is not None, the value of the current parameter is
        a function of the given Parameter. The function is by default
        the identity function, but it can be defined by twin_function
    twin_function : function
        Function that, if selt.twin is not None, takes self.twin.value
        as its only argument and returns a float or array that is
        returned when getting Parameter.value
    twin_inverse_function : function
        The inverse of twin_function. If it is None then it is not
        possible to set the value of the parameter twin by setting
        the value of the current parameter.
    ext_force_positive : bool
        If True, the parameter value is set to be the absolute value
        of the input value i.e. if we set Parameter.value = -3, the
        value stored is 3 instead. This is useful to bound a value
        to be positive in an optimization without actually using an
        optimizer that supports bounding.
    ext_bounded : bool
        Similar to ext_force_positive, but in this case the bounds are
        defined by bmin and bmax. It is a better idea to use
        an optimizer that supports bounding though.

    Methods
    -------
    as_signal(field = 'values')
        Get a parameter map as a signal object
    plot()
        Plots the value of the Parameter at all locations.
    export(folder=None, name=None, format=None, save_std=False)
        Saves the value of the parameter map to the specified format
    connect, disconnect(function)
        Call the functions connected when the value attribute changes.

    """
    __number_of_elements = 1
    __value = 0
    __free = True
    _bounds = (None, None)
    __twin = None
    _axes_manager = None
    __ext_bounded = False
    __ext_force_positive = False

    # traitsui bugs out trying to make an editor for this, so always specify!
    # (it bugs out, because both editor shares the object, and Array editors
    # don't like non-sequence objects). TextEditor() works well, so does
    # RangeEditor() as it works with bmin/bmax.
    value = t.Property(t.Either([t.CFloat(0), Array()]))

    units = t.Str('')
    free = t.Property(t.CBool(True))

    bmin = t.Property(NoneFloat(), label="Lower bounds")
    bmax = t.Property(NoneFloat(), label="Upper bounds")

    def __init__(self):
        self._twins = set()
        self.connected_functions = list()
        self.twin_function = lambda x: x
        self.twin_inverse_function = lambda x: x
        self.std = None
        self.component = None
        self.grad = None
        self.name = ''
        self.map = None
        self.model = None

    def __repr__(self):
        text = ''
        text += 'Parameter %s' % self.name
        if self.component is not None:
            text += ' of %s' % self.component._get_short_description()
        text = '<' + text + '>'
        return text

    def __len__(self):
        return self._number_of_elements

    def connect(self, f):
        if f not in self.connected_functions:
            self.connected_functions.append(f)
            if self.twin:
                self.twin.connect(f)

    def disconnect(self, f):
        if f in self.connected_functions:
            self.connected_functions.remove(f)
            if self.twin:
                self.twin.disconnect(f)

    def _get_value(self):
        if self.twin is None:
            return self.__value
        else:
            return self.twin_function(self.twin.value)

    def _set_value(self, arg):
        try:
            # Use try/except instead of hasattr("__len__") because a numpy
            # memmap has a __len__ wrapper even for numbers that raises a
            # TypeError when calling. See issue #349.
            if len(arg) != self._number_of_elements:
                raise ValueError(
                    "The length of the parameter must be ",
                    self._number_of_elements)
            else:
                if not isinstance(arg, tuple):
                    arg = tuple(arg)
        except TypeError:
            if self._number_of_elements != 1:
                raise ValueError(
                    "The length of the parameter must be ",
                    self._number_of_elements)
        old_value = self.__value

        if self.twin is not None:
            if self.twin_inverse_function is not None:
                self.twin.value = self.twin_inverse_function(arg)
            return

        if self.ext_bounded is False:
            self.__value = arg
        else:
            if self.ext_force_positive is True:
                arg = np.abs(arg)
            if self._number_of_elements == 1:
                if self.bmin is not None and arg <= self.bmin:
                    self.__value = self.bmin
                elif self.bmax is not None and arg >= self.bmax:
                    self.__value = self.bmax
                else:
                    self.__value = arg
            else:
                bmin = (self.bmin if self.bmin is not None
                        else -np.inf)
                bmax = (self.bmax if self.bmin is not None
                        else np.inf)
                self.__value = np.clip(arg, bmin, bmax)

        if (self._number_of_elements != 1 and
                not isinstance(self.__value, tuple)):
            self.__value = tuple(self.__value)
        if old_value != self.__value:
            for f in self.connected_functions:
                try:
                    f()
                except:
                    self.disconnect(f)
        self.trait_property_changed('value', old_value, self.__value)

    # Fix the parameter when coupled
    def _get_free(self):
        if self.twin is None:
            return self.__free
        else:
            return False

    def _set_free(self, arg):
        old_value = self.__free
        self.__free = arg
        if self.component is not None:
            self.component._update_free_parameters()
        self.trait_property_changed('free', old_value, self.__free)

    def _set_twin(self, arg):
        if arg is None:
            if self.twin is not None:
                # Store the value of the twin in order to set the
                # value of the parameter when it is uncoupled
                twin_value = self.value
                if self in self.twin._twins:
                    self.twin._twins.remove(self)
                    for f in self.connected_functions:
                        self.twin.disconnect(f)

                self.__twin = arg
                self.value = twin_value
        else:
            if self not in arg._twins:
                arg._twins.add(self)
                for f in self.connected_functions:
                    arg.connect(f)
            self.__twin = arg

        if self.component is not None:
            self.component._update_free_parameters()

    def _get_twin(self):
        return self.__twin
    twin = property(_get_twin, _set_twin)

    def _get_bmin(self):
        if self._number_of_elements == 1:
            return self._bounds[0]
        else:
            return self._bounds[0][0]

    def _set_bmin(self, arg):
        old_value = self.bmin
        if self._number_of_elements == 1:
            self._bounds = (arg, self.bmax)
        else:
            self._bounds = ((arg, self.bmax),) * self._number_of_elements
        # Update the value to take into account the new bounds
        self.value = self.value
        self.trait_property_changed('bmin', old_value, arg)

    def _get_bmax(self):
        if self._number_of_elements == 1:
            return self._bounds[1]
        else:
            return self._bounds[0][1]

    def _set_bmax(self, arg):
        old_value = self.bmax
        if self._number_of_elements == 1:
            self._bounds = (self.bmin, arg)
        else:
            self._bounds = ((self.bmin, arg),) * self._number_of_elements
        # Update the value to take into account the new bounds
        self.value = self.value
        self.trait_property_changed('bmax', old_value, arg)

    @property
    def _number_of_elements(self):
        return self.__number_of_elements

    @_number_of_elements.setter
    def _number_of_elements(self, arg):
        # Do nothing if the number of arguments stays the same
        if self.__number_of_elements == arg:
            return
        if arg <= 1:
            raise ValueError("Please provide an integer number equal "
                             "or greater to 1")
        self._bounds = ((self.bmin, self.bmax),) * arg
        self.__number_of_elements = arg

        if arg == 1:
            self._Parameter__value = 0
        else:
            self._Parameter__value = (0,) * arg
        if self.component is not None:
            self.component.update_number_parameters()

    @property
    def ext_bounded(self):
        return self.__ext_bounded

    @ext_bounded.setter
    def ext_bounded(self, arg):
        if arg is not self.__ext_bounded:
            self.__ext_bounded = arg
            # Update the value to take into account the new bounds
            self.value = self.value

    @property
    def ext_force_positive(self):
        return self.__ext_force_positive

    @ext_force_positive.setter
    def ext_force_positive(self, arg):
        if arg is not self.__ext_force_positive:
            self.__ext_force_positive = arg
            # Update the value to take into account the new bounds
            self.value = self.value

    def store_current_value_in_array(self):
        """Store the value and std attributes.

        See also
        --------
        fetch, assign_current_value_to_all

        """
        indices = self._axes_manager.indices[::-1]
        # If it is a single spectrum indices is ()
        if not indices:
            indices = (0,)
        self.map['values'][indices] = self.value
        self.map['is_set'][indices] = True
        if self.std is not None:
            self.map['std'][indices] = self.std

    def fetch(self):
        """Fetch the stored value and std attributes.


        See Also
        --------
        store_current_value_in_array, assign_current_value_to_all

        """
        indices = self._axes_manager.indices[::-1]
        # If it is a single spectrum indices is ()
        if not indices:
            indices = (0,)
        if self.map['is_set'][indices]:
            self.value = self.map['values'][indices]
            self.std = self.map['std'][indices]

    def assign_current_value_to_all(self, mask=None):
        """Assign the current value attribute to all the  indices

        Parameters
        ----------
        mask: {None, boolean numpy array}
            Set only the indices that are not masked i.e. where
            mask is False.

        See Also
        --------
        store_current_value_in_array, fetch

        """
        if mask is None:
            mask = np.zeros(self.map.shape, dtype='bool')
        self.map['values'][mask == False] = self.value
        self.map['is_set'][mask == False] = True

    def _create_array(self):
        """Create the map array to store the information in
        multidimensional datasets.

        """
        shape = self._axes_manager._navigation_shape_in_array
        if not shape:
            shape = [1, ]
        dtype_ = np.dtype([
            ('values', 'float', self._number_of_elements),
            ('std', 'float', self._number_of_elements),
            ('is_set', 'bool', 1)])
        if (self.map is None or self.map.shape != shape or
                self.map.dtype != dtype_):
            self.map = np.zeros(shape, dtype_)
            self.map['std'].fill(np.nan)
            # TODO: in the future this class should have access to
            # axes manager and should be able to fetch its own
            # values. Until then, the next line is necessary to avoid
            # erros when self.std is defined and the shape is different
            # from the newly defined arrays
            self.std = None

    def as_signal(self, field='values'):
        """Get a parameter map as a signal object.

        Please note that this method only works when the navigation
        dimension is greater than 0.

        Parameters
        ----------
        field : {'values', 'std', 'is_set'}

        Raises
        ------

        NavigationDimensionError : if the navigation dimension is 0

        """
        from hyperspy.signal import BaseSignal
        if self._axes_manager.navigation_dimension == 0:
            raise NavigationDimensionError(0, '>0')

        s = BaseSignal(data=self.map[field],
                       axes=self._axes_manager._get_navigation_axes_dicts())
        if self.component.active_is_multidimensional:
            s.data[np.logical_not(self.component._active_array)] = np.nan
        s.metadata.General.title = ("%s parameter" % self.name
                                    if self.component is None
                                    else "%s parameter of %s component" %
                                    (self.name, self.component.name))
        for axis in s.axes_manager._axes:
            axis.navigate = False
        if self._number_of_elements > 1:
            s.axes_manager._append_axis(
                size=self._number_of_elements,
                name=self.name,
                navigate=True)
        return s

    def plot(self):
        self.as_signal().plot()

    def export(self, folder=None, name=None, format=None,
               save_std=False):
        """Save the data to a file.

        All the arguments are optional.

        Parameters
        ----------
        folder : str or None
            The path to the folder where the file will be saved.
             If `None` the current folder is used by default.
        name : str or None
            The name of the file. If `None` the Components name followed
             by the Parameter `name` attributes will be used by default.
              If a file with the same name exists the name will be
              modified by appending a number to the file path.
        save_std : bool
            If True, also the standard deviation will be saved

        """
        if format is None:
            format = preferences.General.default_export_format
        if name is None:
            name = self.component.name + '_' + self.name
        filename = incremental_filename(slugify(name) + '.' + format)
        if folder is not None:
            filename = os.path.join(folder, filename)
        self.as_signal().save(filename)
        if save_std is True:
            self.as_signal(field='std').save(append2pathname(
                filename, '_std'))

    def default_traits_view(self):
        # As mentioned above, the default editor for
        # value = t.Property(t.Either([t.CFloat(0), Array()]))
        # gives a ValueError. We therefore implement default_traits_view so
        # that configure/edit_traits will still work straight out of the box.
        # A whitelist controls which traits to include in this view.
        from traitsui.api import RangeEditor, View, Item
        whitelist = ['bmax', 'bmin', 'free', 'name', 'std', 'units', 'value']
        editable_traits = [trait for trait in self.editable_traits()
                           if trait in whitelist]
        if 'value' in editable_traits:
            i = editable_traits.index('value')
            v = editable_traits.pop(i)
            editable_traits.insert(i, Item(
                v, editor=RangeEditor(low_name='bmin', high_name='bmax')))
        view = View(editable_traits, buttons=['OK', 'Cancel'])
        return view

    def _interactive_slider_bounds(self, index=None):
        """Guesstimates the bounds for the slider. They will probably have to
        be changed later by the user.
        """
        fraction = 10.
        _min, _max, step = None, None, None
        value = self.value if index is None else self.value[index]
        if self.bmin is not None:
            _min = self.bmin
        if self.bmax is not None:
            _max = self.bmax
        if _max is None and _min is not None:
            _max = value + fraction * (value - _min)
        if _min is None and _max is not None:
            _min = value - fraction * (_max - value)
        if _min is None and _max is None:
            if self is self.component._position:
                axis = self._axes_manager.signal_axes[-1]
                _min = axis.axis.min()
                _max = axis.axis.max()
                step = np.abs(axis.scale)
            else:
                _max = value + np.abs(value * fraction)
                _min = value - np.abs(value * fraction)
        if step is None:
            step = (_max - _min) * 0.001
        return {'min': _min, 'max': _max, 'step': step}

    def _interactive_update(self, value=None, index=None):
        """Callback function for the widgets, to update the value
        """
        if value is not None:
            if index is None:
                self.value = value['new']
            else:
                self.value = self.value[:index] + (value['new'],) +\
                    self.value[index + 1:]

    def notebook_interaction(self, display=True):
        """Creates interactive notebook widgets for the parameter, if
        available.

        Requires `ipywidgets` to be installed.

        Parameters
        ----------
        display : bool
            if True (default), attempts to display the parameter widget.
            Otherwise returns the formatted widget object.
        """
        from ipywidgets import VBox
        from traitlets import TraitError as TraitletError
        from IPython.display import display as ip_display
        try:
            if self._number_of_elements == 1:
                container = self._create_notebook_widget()
            else:
                children = [self._create_notebook_widget(index=i) for i in
                            range(self._number_of_elements)]
                container = VBox(children)
            if not display:
                return container
            ip_display(container)
        except TraitletError:
            if display:
                print('This function is only avialable when running in a'
                      ' notebook')
            else:
                raise

    def _create_notebook_widget(self, index=None):

        from ipywidgets import (FloatSlider, FloatText, Layout, HBox)

        widget_bounds = self._interactive_slider_bounds(index=index)
        thismin = FloatText(value=widget_bounds['min'],
                            description='min',
                            layout=Layout(flex='0 1 auto',
                                          width='auto'),)
        thismax = FloatText(value=widget_bounds['max'],
                            description='max',
                            layout=Layout(flex='0 1 auto',
                                          width='auto'),)
        current_value = self.value if index is None else self.value[index]
        current_name = self.name
        if index is not None:
            current_name += '_{}'.format(index)
        widget = FloatSlider(value=current_value,
                             min=thismin.value,
                             max=thismax.value,
                             step=widget_bounds['step'],
                             description=current_name,
                             layout=Layout(flex='1 1 auto', width='auto'))

        def on_min_change(change):
            if widget.max > change['new']:
                widget.min = change['new']
                widget.step = np.abs(widget.max - widget.min) * 0.001

        def on_max_change(change):
            if widget.min < change['new']:
                widget.max = change['new']
                widget.step = np.abs(widget.max - widget.min) * 0.001

        thismin.observe(on_min_change, names='value')
        thismax.observe(on_max_change, names='value')

        this_observed = functools.partial(self._interactive_update,
                                          index=index)

        widget.observe(this_observed, names='value')
        container = HBox((thismin, widget, thismax))
        return container


class Component(t.HasTraits):
    __axes_manager = None

    active = t.Property(t.CBool(True))
    name = t.Property(t.Str(''))

    def __init__(self, parameter_name_list):
        self.connected_functions = list()
        self.parameters = []
        self.init_parameters(parameter_name_list)
        self._update_free_parameters()
        self.active = True
        self._active_array = None
        self.isbackground = False
        self.convolved = True
        self.parameters = tuple(self.parameters)
        self._id_name = self.__class__.__name__
        self._id_version = '1.0'
        self._position = None
        self.model = None
        self.name = ''

    _name = ''
    _active_is_multidimensional = False
    _active = True

    @property
    def active_is_multidimensional(self):
        return self._active_is_multidimensional

    @active_is_multidimensional.setter
    def active_is_multidimensional(self, value):
        if not isinstance(value, bool):
            raise ValueError('Only boolean values are permitted')

        if value == self.active_is_multidimensional:
            warnings.warn(
                '`active_is_multidimensional` already %s for %s' %
                (str(value), self.name), RuntimeWarning)
            return

        if value:  # Turn on
            if self._axes_manager.navigation_size < 2:
                warnings.warn(
                    '`navigation_size` < 2, skipping',
                    RuntimeWarning)
                return
            # Store value at current position
            self._create_active_array()
            self._store_active_value_in_array(self._active)
            self._active_is_multidimensional = True
        else:  # Turn off
            # Get the value at the current position before switching it off
            self._active = self.active
            self._active_array = None
            self._active_is_multidimensional = False

    def _get_name(self):
        return self._name

    def _set_name(self, value):
        old_value = self._name
        if self.model:
            for component in self.model:
                if value == component.name:
                    if not (component is self):
                        raise ValueError(
                            "Another component already has "
                            "the name " + str(value))
            self._name = value
            setattr(self.model.components, slugify(
                value, valid_variable_name=True), self)
            self.model.components.__delattr__(
                slugify(old_value, valid_variable_name=True))
        else:
            self._name = value
        self.trait_property_changed('name', old_value, self._name)

    @property
    def _axes_manager(self):
        return self.__axes_manager

    @_axes_manager.setter
    def _axes_manager(self, value):
        for parameter in self.parameters:
            parameter._axes_manager = value
        self.__axes_manager = value

    def connect(self, f):
        if f not in self.connected_functions:
            self.connected_functions.append(f)

    def disconnect(self, f):
        if f in self.connected_functions:
            self.connected_functions.remove(f)

    def _toggle_connect_active_array(self, if_on):
        # nothing to do (was never multidimensional)
        if self._active_array is None:
            return
        # as it should be (both True)
        if self.active_is_multidimensional and if_on:
            return
        # as it should be (both False)
        if not self.active_is_multidimensional and not if_on:
            return
        # active_is_multidimensional = True, want to set to False
        if not if_on:
            self._active_is_multidimensional = False
            self.active = self._active
            return
        if if_on:  # a_i_m = False, want to set to False
            # check that dimensions are correct
            shape = self._axes_manager._navigation_shape_in_array
            if self._active_array.shape != shape:
                warnings.warn(
                    '`_active_array` of wrong shape, skipping',
                    RuntimeWarning)
                return
            self._active_is_multidimensional = True
            self.active = self.active

    def _get_active(self):
        if self.active_is_multidimensional is True:
            # The following should set
            self.active = self._active_array[self._axes_manager.indices[::-1]]
        return self._active

    def _store_active_value_in_array(self, value):
        self._active_array[self._axes_manager.indices[::-1]] = value

    def _set_active(self, arg):
        if self._active == arg:
            return
        old_value = self._active
        self._active = arg
        if self.active_is_multidimensional is True:
            self._store_active_value_in_array(arg)

        for f in self.connected_functions:
            try:
                f()
            except:
                self.disconnect(f)
        self.trait_property_changed('active', old_value, self._active)

    def init_parameters(self, parameter_name_list):
        for name in parameter_name_list:
            parameter = Parameter()
            self.parameters.append(parameter)
            parameter.name = name
            setattr(self, name, parameter)
            if hasattr(self, 'grad_' + name):
                parameter.grad = getattr(self, 'grad_' + name)
            parameter.component = self
            self.add_trait(name, t.Instance(Parameter))

    def _get_long_description(self):
        if self.name:
            text = '%s (%s component)' % (self.name, self._id_name)
        else:
            text = '%s component' % self._id_name
        return text

    def _get_short_description(self):
        text = ''
        if self.name:
            text += self.name
        else:
            text += self._id_name
        text += ' component'
        return text

    def __repr__(self):
        text = '<%s>' % self._get_long_description()
        return text

    def _update_free_parameters(self):
        self.free_parameters = set()
        for parameter in self.parameters:
            if parameter.free:
                self.free_parameters.add(parameter)
        # update_number_free_parameters(self):
        i = 0
        for parameter in self.free_parameters:
            i += parameter._number_of_elements
        self._nfree_param = i

    def update_number_parameters(self):
        i = 0
        for parameter in self.parameters:
            i += parameter._number_of_elements
        self.nparam = i
        self._update_free_parameters()

    def fetch_values_from_array(self, p, p_std=None, onlyfree=False):
        if onlyfree is True:
            parameters = self.free_parameters
        else:
            parameters = self.parameters
        i = 0
        for parameter in parameters:
            length = parameter._number_of_elements
            parameter.value = (p[i] if length == 1 else p[i:i + length])
            if p_std is not None:
                parameter.std = (p_std[i] if length == 1 else
                                 tuple(p_std[i:i + length]))

            i += length

    def _create_active_array(self):
        shape = self._axes_manager._navigation_shape_in_array
        if len(shape) == 1 and shape[0] == 0:
            shape = [1, ]
        if (not isinstance(self._active_array, np.ndarray)
                or self._active_array.shape != shape):
            self._active_array = np.ones(shape, dtype=bool)

    def _create_arrays(self):
        if self.active_is_multidimensional:
            self._create_active_array()
        for parameter in self.parameters:
            parameter._create_array()

    def store_current_parameters_in_map(self):
        for parameter in self.parameters:
            parameter.store_current_value_in_array()

    def fetch_stored_values(self, only_fixed=False):
        if self.active_is_multidimensional:
            # Store the stored value in self._active and trigger the connected
            # functions.
            self.active = self.active
        if only_fixed is True:
            parameters = (set(self.parameters) -
                          set(self.free_parameters))
        else:
            parameters = self.parameters
        parameters = [parameter for parameter in parameters
                      if (parameter.twin is None or
                          not isinstance(parameter.twin, Parameter))]
        for parameter in parameters:
            parameter.fetch()

    def plot(self, only_free=True):
        """Plot the value of the parameters of the model

        Parameters
        ----------
        only_free : bool
            If True, only the value of the parameters that are free will
             be plotted

        """
        if only_free:
            parameters = self.free_parameters
        else:
            parameters = self.parameters

        parameters = [k for k in parameters if k.twin is None]
        for parameter in parameters:
            parameter.plot()

    def export(self, folder=None, format=None, save_std=False,
               only_free=True):
        """Plot the value of the parameters of the model

        Parameters
        ----------
        folder : str or None
            The path to the folder where the file will be saved. If
            `None` the
            current folder is used by default.
        format : str
            The format to which the data will be exported. It must be
            the
            extension of any format supported by HyperSpy. If None, the
            default
            format for exporting as defined in the `Preferences` will be
             used.
        save_std : bool
            If True, also the standard deviation will be saved.
        only_free : bool
            If True, only the value of the parameters that are free will
             be
            exported.

        Notes
        -----
        The name of the files will be determined by each the Component
        and
        each Parameter name attributes. Therefore, it is possible to
        customise
        the file names modify the name attributes.

        """
        if only_free:
            parameters = self.free_parameters
        else:
            parameters = self.parameters

        parameters = [k for k in parameters if k.twin is None]
        for parameter in parameters:
            parameter.export(folder=folder, format=format,
                             save_std=save_std,)

    def summary(self):
        for parameter in self.parameters:
            dim = len(parameter.map.squeeze().shape) if parameter.map \
                is not None else 0
            if parameter.twin is None:
                if dim <= 1:
                    print('%s = %s ± %s %s' % (parameter.name,
                                               parameter.value,
                                               parameter.std,
                                               parameter.units))

    def __tempcall__(self, p, x, onlyfree=True):
        self.fetch_values_from_array(p, onlyfree=onlyfree)
        return self.function(x)

    def __call__(self):
        """Returns the corresponding model for the current coordinates

        Returns
        -------
        numpy array
        """

        axis = self.model.axis.axis[self.model.channel_switches]
        component_array = self.function(axis)
        return component_array

    def _component2plot(self, axes_manager, out_of_range2nans=True):
        old_axes_manager = None
        if axes_manager is not self.model.axes_manager:
            old_axes_manager = self.model.axes_manager
            self.model.axes_manager = axes_manager
            self.fetch_stored_values()
        s = self.__call__()
        if not self.active:
            s.fill(np.nan)
        if self.model.signal.metadata.Signal.binned is True:
            s *= self.model.signal.axes_manager.signal_axes[0].scale
        if old_axes_manager is not None:
            self.model.axes_manager = old_axes_manager
            self.fetch_stored_values()
        if out_of_range2nans is True:
            ns = np.empty(self.model.axis.axis.shape)
            ns.fill(np.nan)
            ns[self.model.channel_switches] = s
            s = ns
        return s

    def set_parameters_free(self, parameter_name_list=None):
        """
        Sets parameters in a component to free.

        Parameters
        ----------
        parameter_name_list : None or list of strings, optional
            If None, will set all the parameters to free.
            If list of strings, will set all the parameters with the same name
            as the strings in parameter_name_list to free.

        Examples
        --------
        >>> v1 = hs.model.components.Voigt()
        >>> v1.set_parameters_free()
        >>> v1.set_parameters_free(parameter_name_list=['area','centre'])

        See also
        --------
        set_parameters_not_free
        hyperspy.model.Model.set_parameters_free
        hyperspy.model.Model.set_parameters_not_free
        """

        parameter_list = []
        if not parameter_name_list:
            parameter_list = self.parameters
        else:
            for _parameter in self.parameters:
                if _parameter.name in parameter_name_list:
                    parameter_list.append(_parameter)

        for _parameter in parameter_list:
            _parameter.free = True

    def set_parameters_not_free(self, parameter_name_list=None):
        """
        Sets parameters in a component to not free.

        Parameters
        ----------
        parameter_name_list : None or list of strings, optional
            If None, will set all the parameters to not free.
            If list of strings, will set all the parameters with the same name
            as the strings in parameter_name_list to not free.

        Examples
        --------
        >>> v1 = hs.model.components.Voigt()
        >>> v1.set_parameters_not_free()
        >>> v1.set_parameters_not_free(parameter_name_list=['area','centre'])

        See also
        --------
        set_parameters_free
        hyperspy.model.Model.set_parameters_free
        hyperspy.model.Model.set_parameters_not_free
        """

        parameter_list = []
        if not parameter_name_list:
            parameter_list = self.parameters
        else:
            for _parameter in self.parameters:
                if _parameter.name in parameter_name_list:
                    parameter_list.append(_parameter)

        for _parameter in parameter_list:
            _parameter.free = False

    def _estimate_parameters(self, signal):
        if self._axes_manager != signal.axes_manager:
            self._axes_manager = signal.axes_manager
            self._create_arrays()

    def notebook_interaction(self, display=True):
        """Creates interactive notebook widgets for all component parameters,
        if available.

        Requires `ipywidgets` to be installed.

        Parameters
        ----------
        display : bool
            if True (default), attempts to display the widgets.
            Otherwise returns the formatted widget object.
        """
        from ipywidgets import (Checkbox, VBox)
        from traitlets import TraitError as TraitletError
        from IPython.display import display as ip_display
        try:
            active = Checkbox(description='active', value=self.active)

            def on_active_change(change):
                self.active = change['new']
            active.observe(on_active_change, names='value')

            container = VBox([active])
            for parameter in self.parameters:
                container.children += parameter.notebook_interaction(False),

            if not display:
                return container
            ip_display(container)
        except TraitletError:
            if display:
                print('This function is only avialable when running in a'
                      ' notebook')
            else:
                raise
