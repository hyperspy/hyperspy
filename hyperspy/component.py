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
from dask.array import Array as dArray
import traits.api as t
from traits.trait_numeric import Array
import sympy
from sympy.utilities.lambdify import lambdify

from hyperspy.misc.utils import slugify
from hyperspy.misc.io.tools import (incremental_filename,
                                    append2pathname,)
from hyperspy.misc.export_dictionary import export_to_dictionary, \
    load_from_dictionary
from hyperspy.events import Events, Event
from hyperspy.ui_registry import add_gui_method

import logging

_logger = logging.getLogger(__name__)


class NoneFloat(t.CFloat):   # Lazy solution, but usable
    default_value = None

    def validate(self, object, name, value):
        if value == "None" or value == b"None":
            value = None
        if value is None:
            super(NoneFloat, self).validate(object, name, 0)
            return None
        return super(NoneFloat, self).validate(object, name, value)


@add_gui_method(toolkey="Parameter")
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
    twin_function_expr: str
        Expression of the ``twin_function`` that enables setting a functional
        relationship between the parameter and its twin. If ``twin`` is not
        ``None``, the parameter value is calculated as the output of calling the
        twin function with the value of the twin parameter. The string is
        parsed using sympy, so permitted values are any valid sympy expressions
        of one variable. If the function is invertible the twin inverse function
        is set automatically.
    twin_inverse_function : str
        Expression of the ``twin_inverse_function`` that enables setting the
        value of the twin parameter. If ``twin`` is not
        ``None``, its value is set to the output of calling the
        twin inverse function with the value provided. The string is
        parsed using sympy, so permitted values are any valid sympy expressions
        of one variable.
    twin_function : function
        **Setting this attribute manually
        is deprecated in HyperSpy newer than 1.1.2. It will become private in
        HyperSpy 2.0. Please use ``twin_function_expr`` instead.**
    twin_inverse_function : function
        **Setting this attribute manually
        is deprecated in HyperSpy newer than 1.1.2. It will become private in
        HyperSpy 2.0. Please use ``twin_inverse_function_expr`` instead.**
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
    _twin_function_expr = ""
    _twin_inverse_function_expr = ""
    twin_function = None
    _twin_inverse_function = None
    _twin_inverse_sympy = None

    def __init__(self):
        self._twins = set()
        self.events = Events()
        self.events.value_changed = Event("""
            Event that triggers when the `Parameter.value` changes.

            The event triggers after the internal state of the `Parameter` has
            been updated.

            Arguments
            ---------
            obj : Parameter
                The `Parameter` that the event belongs to
            value : {float | array}
                The new value of the parameter
            """, arguments=["obj", 'value'])
        self.std = None
        self.component = None
        self.grad = None
        self.name = ''
        self.units = ''
        self.map = None
        self.model = None
        self._whitelist = {'_id_name': None,
                           'value': None,
                           'std': None,
                           'free': None,
                           'units': None,
                           'map': None,
                           '_bounds': None,
                           'ext_bounded': None,
                           'name': None,
                           'ext_force_positive': None,
                           'twin_function_expr': None,
                           'twin_inverse_function_expr': None,
                           'self': ('id', None),
                           }
        self._slicing_whitelist = {'map': 'inav'}

    def _load_dictionary(self, dictionary):
        """Load data from dictionary

        Parameters
        ----------
        dict : dictionary
            A dictionary containing at least the following items:
            _id_name : string
                _id_name of the original parameter, used to create the
                dictionary. Has to match with the self._id_name
            _whitelist : dictionary
                a dictionary, which keys are used as keywords to match with the
                parameter attributes.  For more information see
                :meth:`hyperspy.misc.export_dictionary.load_from_dictionary`
            * any field from _whitelist.keys() *
        Returns
        -------
        id_value : int
            the ID value of the original parameter, to be later used for setting
            up the correct twins

        """
        if dictionary['_id_name'] == self._id_name:
            load_from_dictionary(self, dictionary)
            return dictionary['self']
        else:
            raise ValueError("_id_name of parameter and dictionary do not match, \nparameter._id_name = %s\
                    \ndictionary['_id_name'] = %s" % (self._id_name, dictionary['_id_name']))

    def __repr__(self):
        text = ''
        text += 'Parameter %s' % self.name
        if self.component is not None:
            text += ' of %s' % self.component._get_short_description()
        text = '<' + text + '>'
        return text

    def __len__(self):
        return self._number_of_elements

    @property
    def twin_function_expr(self):
        return self._twin_function_expr

    @twin_function_expr.setter
    def twin_function_expr(self, value):
        if not value:
            self.twin_function = None
            self.twin_inverse_function = None
            self._twin_function_expr = ""
            self._twin_inverse_sympy = None
            return
        expr = sympy.sympify(value)
        if len(expr.free_symbols) > 1:
            raise ValueError("The expression must contain only one variable.")
        elif len(expr.free_symbols) == 0:
            raise ValueError("The expression must contain one variable, "
                             "it contains none.")
        x = tuple(expr.free_symbols)[0]
        self.twin_function = lambdify(x, expr.evalf())
        self._twin_function_expr = value
        if not self.twin_inverse_function:
            y = sympy.Symbol(x.name + "2")
            try:
                inv = sympy.solveset(sympy.Eq(y, expr), x)
                self._twin_inverse_sympy = lambdify(y, inv)
                self._twin_inverse_function = None
            except BaseException:
                # Not all may have a suitable solution.
                self._twin_inverse_function = None
                self._twin_inverse_sympy = None
                _logger.warning(
                    "The function {} is not invertible. Setting the value of "
                    "{} will raise an AttributeError unless you set manually "
                    "``twin_inverse_function_expr``. Otherwise, set the "
                    "value of its twin parameter instead.".format(value, self))

    @property
    def twin_inverse_function_expr(self):
        if self.twin:
            return self._twin_inverse_function_expr
        else:
            return ""

    @twin_inverse_function_expr.setter
    def twin_inverse_function_expr(self, value):
        if not value:
            self.twin_inverse_function = None
            self._twin_inverse_function_expr = ""
            return
        expr = sympy.sympify(value)
        if len(expr.free_symbols) > 1:
            raise ValueError("The expression must contain only one variable.")
        elif len(expr.free_symbols) == 0:
            raise ValueError("The expression must contain one variable, "
                             "it contains none.")
        x = tuple(expr.free_symbols)[0]
        self._twin_inverse_function = lambdify(x, expr.evalf())
        self._twin_inverse_function_expr = value

    @property
    def twin_inverse_function(self):
        if (not self.twin_inverse_function_expr and
                self.twin_function_expr and self._twin_inverse_sympy):
            return lambda x: self._twin_inverse_sympy(x).pop()
        else:
            return self._twin_inverse_function

    @twin_inverse_function.setter
    def twin_inverse_function(self, value):
        self._twin_inverse_function = value

    def _get_value(self):
        if self.twin is None:
            return self.__value
        else:
            if self.twin_function:
                return self.twin_function(self.twin.value)
            else:
                return self.twin.value

    def _set_value(self, value):
        try:
            # Use try/except instead of hasattr("__len__") because a numpy
            # memmap has a __len__ wrapper even for numbers that raises a
            # TypeError when calling. See issue #349.
            if len(value) != self._number_of_elements:
                raise ValueError(
                    "The length of the parameter must be ",
                    self._number_of_elements)
            else:
                if not isinstance(value, tuple):
                    value = tuple(value)
        except TypeError:
            if self._number_of_elements != 1:
                raise ValueError(
                    "The length of the parameter must be ",
                    self._number_of_elements)
        old_value = self.__value

        if self.twin is not None:
            if self.twin_function is not None:
                if self.twin_inverse_function is not None:
                    self.twin.value = self.twin_inverse_function(value)
                    return
                else:
                    raise AttributeError(
                        "This parameter has a ``twin_function`` but"
                        "its ``twin_inverse_function`` is not defined.")
            else:
                self.twin.value = value
                return

        if self.ext_bounded is False:
            self.__value = value
        else:
            if self.ext_force_positive is True:
                value = np.abs(value)
            if self._number_of_elements == 1:
                if self.bmin is not None and value <= self.bmin:
                    self.__value = self.bmin
                elif self.bmax is not None and value >= self.bmax:
                    self.__value = self.bmax
                else:
                    self.__value = value
            else:
                bmin = (self.bmin if self.bmin is not None
                        else -np.inf)
                bmax = (self.bmax if self.bmin is not None
                        else np.inf)
                self.__value = np.clip(value, bmin, bmax)

        if (self._number_of_elements != 1 and
                not isinstance(self.__value, tuple)):
            self.__value = tuple(self.__value)
        if old_value != self.__value:
            self.events.value_changed.trigger(value=self.__value,
                                              obj=self)
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

    def _on_twin_update(self, value, twin=None):
        if (twin is not None
                and hasattr(twin, 'events')
                and hasattr(twin.events, 'value_changed')):
            with twin.events.value_changed.suppress_callback(
                    self._on_twin_update):
                self.events.value_changed.trigger(value=value, obj=self)
        else:
            self.events.value_changed.trigger(value=value, obj=self)

    def _set_twin(self, arg):
        if arg is None:
            if self.twin is not None:
                # Store the value of the twin in order to set the
                # value of the parameter when it is uncoupled
                twin_value = self.value
                if self in self.twin._twins:
                    self.twin._twins.remove(self)
                    self.twin.events.value_changed.disconnect(
                        self._on_twin_update)

                self.__twin = arg
                self.value = twin_value
        else:
            if self not in arg._twins:
                arg._twins.add(self)
                arg.events.value_changed.connect(self._on_twin_update,
                                                 ["value"])
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
            value = self.map['values'][indices]
            std = self.map['std'][indices]
            if isinstance(value, dArray):
                value = value.compute()
            if isinstance(std, dArray):
                std = std.compute()
            self.value = value
            self.std = std

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

        s = BaseSignal(data=self.map[field],
                       axes=self._axes_manager._get_navigation_axes_dicts())
        if self.component is not None and \
                self.component.active_is_multidimensional:
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
        s._assign_subclass()
        if field == "values":
            # Add the variance if available
            std = self.as_signal(field="std")
            if not np.isnan(std.data).all():
                std.data = std.data ** 2
                std.metadata.General.title = "Variance"
                s.metadata.set_item(
                    "Signal.Noise_properties.variance", std)
        return s

    def plot(self, **kwargs):
        """Plot parameter signal.

        Parameters
        ----------
        **kwargs
            Any extra keyword arguments are passed to the signal plot.

        Example
        -------
        >>> parameter.plot() #doctest: +SKIP

        Set the minimum and maximum displayed values

        >>> parameter.plot(vmin=0, vmax=1) #doctest: +SKIP
        """
        self.as_signal().plot(**kwargs)

    def export(self, folder=None, name=None, format="hspy",
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
        format: str
            The extension of any file format supported by HyperSpy, default hspy

        """
        if format is None:
            format = "hspy"
        if name is None:
            name = self.component.name + '_' + self.name
        filename = incremental_filename(slugify(name) + '.' + format)
        if folder is not None:
            filename = os.path.join(folder, filename)
        self.as_signal().save(filename)
        if save_std is True:
            self.as_signal(field='std').save(append2pathname(
                filename, '_std'))

    def as_dictionary(self, fullcopy=True):
        """Returns parameter as a dictionary, saving all attributes from
        self._whitelist.keys() For more information see
        :meth:`hyperspy.misc.export_dictionary.export_to_dictionary`

        Parameters
        ----------
        fullcopy : Bool (optional, False)
            Copies of objects are stored, not references. If any found,
            functions will be pickled and signals converted to dictionaries
        Returns
        -------
        dic : dictionary with the following keys:
            _id_name : string
                _id_name of the original parameter, used to create the
                dictionary. Has to match with the self._id_name
            _twins : list
                a list of ids of the twins of the parameter
            _whitelist : dictionary
                a dictionary, which keys are used as keywords to match with the
                parameter attributes.  For more information see
                :meth:`hyperspy.misc.export_dictionary.export_to_dictionary`
            * any field from _whitelist.keys() *

        """
        dic = {'_twins': [id(t) for t in self._twins]}
        export_to_dictionary(self, self._whitelist, dic, fullcopy)
        return dic

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


@add_gui_method(toolkey="Component")
class Component(t.HasTraits):
    __axes_manager = None

    active = t.Property(t.CBool(True))
    name = t.Property(t.Str(''))

    def __init__(self, parameter_name_list):
        self.events = Events()
        self.events.active_changed = Event("""
            Event that triggers when the `Component.active` changes.

            The event triggers after the internal state of the `Component` has
            been updated.

            Arguments
            ---------
            obj : Component
                The `Component` that the event belongs to
            active : bool
                The new active state
            """, arguments=["obj", 'active'])
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
        self._whitelist = {'_id_name': None,
                           'name': None,
                           'active_is_multidimensional': None,
                           '_active_array': None,
                           'active': None
                           }
        self._slicing_whitelist = {'_active_array': 'inav'}
        self._slicing_order = ('active', 'active_is_multidimensional',
                               '_active_array',)

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
            return

        if value:  # Turn on
            if self._axes_manager.navigation_size < 2:
                _logger.info('`navigation_size` < 2, skipping')
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
        if old_value == value:
            return
        if self.model:
            for component in self.model:
                if value == component.name:
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
        self.events.active_changed.trigger(active=self._active, obj=self)
        self.trait_property_changed('active', old_value, self._active)

    def init_parameters(self, parameter_name_list):
        for name in parameter_name_list:
            parameter = Parameter()
            self.parameters.append(parameter)
            parameter.name = name
            parameter._id_name = name
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
        self.free_parameters = sorted([par for par in self.parameters if
                                       par.free], key=lambda x: x.name)
        self._nfree_param = sum([par._number_of_elements for par in
                                 self.free_parameters])

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
        for parameter in sorted(parameters, key=lambda x: x.name):
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
            _logger.debug('Creating _active_array for {}.\n\tCurrent array '
                          'is:\n{}'.format(self, self._active_array))
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

    def export(self, folder=None, format="hspy", save_std=False,
               only_free=True):
        """Plot the value of the parameters of the model

        Parameters
        ----------
        folder : str or None
            The path to the folder where the file will be saved. If
            `None` the
            current folder is used by default.
        format : str
            The extension of the file format, default "hspy".
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
        s = self.model.__call__(component_list=[self])
        if not self.active:
            s.fill(np.nan)
        if old_axes_manager is not None:
            self.model.axes_manager = old_axes_manager
            self.charge()
        if out_of_range2nans is True:
            ns = np.empty(self.model.axis.axis.shape)
            ns.fill(np.nan)
            ns[self.model.channel_switches] = s
            s = ns
        if old_axes_manager is not None:
            self.model.axes_manager = old_axes_manager
            self.fetch_stored_values()
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
        >>> v1 = hs.model.components1D.Voigt()
        >>> v1.set_parameters_free()
        >>> v1.set_parameters_free(parameter_name_list=['area','centre'])

        See also
        --------
        set_parameters_not_free
        hyperspy.model.BaseModel.set_parameters_free
        hyperspy.model.BaseModel.set_parameters_not_free
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
        >>> v1 = hs.model.components1D.Voigt()
        >>> v1.set_parameters_not_free()
        >>> v1.set_parameters_not_free(parameter_name_list=['area','centre'])

        See also
        --------
        set_parameters_free
        hyperspy.model.BaseModel.set_parameters_free
        hyperspy.model.BaseModel.set_parameters_not_free
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
        self.binned = signal.metadata.Signal.binned
        if self._axes_manager != signal.axes_manager:
            self._axes_manager = signal.axes_manager
            self._create_arrays()

    def as_dictionary(self, fullcopy=True):
        """Returns component as a dictionary
        For more information on method and conventions, see
        :meth:`hyperspy.misc.export_dictionary.export_to_dictionary`
        Parameters
        ----------
        fullcopy : Bool (optional, False)
            Copies of objects are stored, not references. If any found,
            functions will be pickled and signals converted to dictionaries
        Returns
        -------
        dic : dictionary
            A dictionary, containing at least the following fields:
            parameters : list
                a list of dictionaries of the parameters, one per
            _whitelist : dictionary
                a dictionary with keys used as references saved attributes, for
                more information, see
                :meth:`hyperspy.misc.export_dictionary.export_to_dictionary`
            * any field from _whitelist.keys() *
        """
        dic = {
            'parameters': [
                p.as_dictionary(fullcopy) for p in self.parameters]}
        export_to_dictionary(self, self._whitelist, dic, fullcopy)
        from hyperspy.model import components
        if self._id_name not in components.__dict__.keys():
            import dill
            dic['_class_dump'] = dill.dumps(self.__class__)
        return dic

    def _load_dictionary(self, dic):
        """Load data from dictionary.
        Parameters
        ----------
        dict : dictionary
            A dictionary containing following items:
            _id_name : string
                _id_name of the original component, used to create the
                dictionary. Has to match with the self._id_name
            parameters : list
                A list of dictionaries, one per parameter of the component (see
                parameter.as_dictionary() documentation for more)
            _whitelist : dictionary
                a dictionary, which keys are used as keywords to match with the
                component attributes.  For more information see
                :meth:`hyperspy.misc.export_dictionary.load_from_dictionary`
            * any field from _whitelist.keys() *
        Returns
        -------
        twin_dict : dictionary
            Dictionary of 'id' values from input dictionary as keys with all of
            the parameters of the component, to be later used for setting up
            correct twins.
        """

        if dic['_id_name'] == self._id_name:
            load_from_dictionary(self, dic)
            id_dict = {}
            for p in dic['parameters']:
                idname = p['_id_name']
                if hasattr(self, idname):
                    par = getattr(self, idname)
                    t_id = par._load_dictionary(p)
                    id_dict[t_id] = par
                else:
                    raise ValueError(
                        "_id_name of parameters in component and dictionary do not match")
            return id_dict
        else:
            raise ValueError("_id_name of component and dictionary do not match, \ncomponent._id_name = %s\
                    \ndictionary['_id_name'] = %s" % (self._id_name, dic['_id_name']))
