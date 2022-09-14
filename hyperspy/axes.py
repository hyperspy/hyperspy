# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

from contextlib import contextmanager
import copy
import math
import logging

import dask.array as da
import numpy as np
import pint
from sympy.utilities.lambdify import lambdify
import traits.api as t
from traits.trait_errors import TraitError

from hyperspy.api_nogui import _ureg
from hyperspy.events import Events, Event
from hyperspy.misc.array_tools import (
    numba_closest_index_round,
    numba_closest_index_floor,
    numba_closest_index_ceil,
    round_half_towards_zero,
    round_half_away_from_zero,
)
from hyperspy.misc.utils import isiterable, ordinal
from hyperspy.misc.math_tools import isfloat
from hyperspy.ui_registry import add_gui_method, get_gui
from hyperspy.defaults_parser import preferences
from hyperspy._components.expression import _parse_substitutions


import warnings
import inspect
from collections.abc import Iterable

_logger = logging.getLogger(__name__)


FACTOR_DOCSTRING = \
    """factor : float (default: 0.25)
            'factor' is an adjustable value used to determine the prefix of
            the units. The product `factor * scale * size` is passed to the
            pint `to_compact` method to determine the prefix."""


class ndindex_nat(np.ndindex):

    def __next__(self):
        return super().__next__()[::-1]


def generate_uniform_axis(offset, scale, size, offset_index=0):
    """Creates a uniform axis vector given the offset, scale and number of
    channels.

    Alternatively, the offset_index of the offset channel can be specified.

    Parameters
    ----------
    offset : float
    scale : float
    size : number of channels
    offset_index : int
        offset_index number of the offset

    Returns
    -------
    Numpy array

    """

    return np.linspace(offset - offset_index * scale,
                       offset + scale * (size - 1 - offset_index),
                       size)


def create_axis(**kwargs):
    """Creates a uniform, a non-uniform axis or a functional axis depending on
    the kwargs provided. If `axis` or  `expression` are provided, a non-uniform
    or a functional axis is created, respectively. Otherwise a uniform axis is
    created, which can be defined by `scale`, `size` and `offset`.

    Alternatively, the offset_index of the offset channel can be specified.

    Parameters
    ----------
    axis : iterable of values (list, tuple or 1D numpy array) (optional)
    expression : Component function in SymPy text expression format (str) (optional)
    offset : float (optional)
    scale : float (optional)
    size : number of channels (optional)

    Returns
    -------
    A DataAxis, FunctionalDataAxis or a UniformDataAxis

    """
    if 'axis' in kwargs.keys():  # non-uniform axis
        axis_class = DataAxis
    elif 'expression' in kwargs.keys():  # Functional axis
        axis_class = FunctionalDataAxis
    else:  # if not argument is provided fall back to uniform axis
        axis_class = UniformDataAxis
    return axis_class(**kwargs)


class UnitConversion:

    def __init__(self, units=None, scale=1.0, offset=0.0):
        if units is None:
            units = t.Undefined
        self.units = units
        self.scale = scale
        self.offset = offset

    def _ignore_conversion(self, units):
        if units == t.Undefined:
            return True
        try:
            _ureg(units)
        except pint.errors.UndefinedUnitError:
            warnings.warn(
                f"Unit {units} not supported for conversion. Nothing done."
                )
            return True
        return False

    def _convert_compact_units(self, factor=0.25, inplace=True):
        """ Convert units to "human-readable" units, which means with a
            convenient prefix.

            Parameters
            ----------
            %s
        """
        if self._ignore_conversion(self.units):
            return
        scale = self.scale * _ureg(self.units)
        scale_size = factor * scale * self.size
        converted_units = '{:~}'.format(scale_size.to_compact().units)
        return self._convert_units(converted_units, inplace=inplace)

    _convert_compact_units.__doc__ %= FACTOR_DOCSTRING

    def _get_value_from_value_with_units(self, value):
        if self.units is t.Undefined:
            raise ValueError("Units conversion can't be perfomed "
                             f"because the axis '{self}' doesn't have "
                             "units.")
        value = _ureg.parse_expression(value)
        if not hasattr(value, 'units'):
            raise ValueError(f"`{value}` should contain an units.")

        return float(value.to(self.units).magnitude)

    def _convert_units(self, converted_units, inplace=True):
        if self._ignore_conversion(converted_units) or \
                self._ignore_conversion(self.units):
            return
        scale_pint = self.scale * _ureg(self.units)
        offset_pint = self.offset * _ureg(self.units)
        scale = float(scale_pint.to(_ureg(converted_units)).magnitude)
        offset = float(offset_pint.to(_ureg(converted_units)).magnitude)
        units = '{:~}'.format(scale_pint.to(_ureg(converted_units)).units)
        if inplace:
            self.scale = scale
            self.offset = offset
            self.units = units
        else:
            return scale, offset, units

    def convert_to_units(self, units=None, inplace=True, factor=0.25):
        """ Convert the scale and the units of the current axis. If the unit
        of measure is not supported by the pint library, the scale and units
        are not modified.

        Parameters
        ----------
        units : {str | None}
            Default = None
            If str, the axis will be converted to the provided units.
            If `"auto"`, automatically determine the optimal units to avoid
            using too large or too small numbers. This can be tweaked by the
            `factor` argument.
        inplace : bool
            If `True`, convert the axis in place. if `False` return the
            `scale`, `offset` and `units`.
        %s
        """
        if units is None:
            out = self._convert_compact_units(factor, inplace=inplace)
        else:
            out = self._convert_units(units, inplace=inplace)
        return out

    convert_to_units.__doc__ %= FACTOR_DOCSTRING

    def _get_quantity(self, attribute='scale'):
        if attribute == 'scale' or attribute == 'offset':
            units = self.units
            if units == t.Undefined:
                units = ''
            return getattr(self, attribute) * _ureg(units)
        else:
            raise ValueError('`attribute` argument can only take the `scale` '
                             'or the `offset` value.')

    def _set_quantity(self, value, attribute='scale'):
        if attribute == 'scale' or attribute == 'offset':
            units = '' if self.units == t.Undefined else self.units
            if isinstance(value, str):
                value = _ureg.parse_expression(value)
            if isinstance(value, float):
                value = value * _ureg(units)

            # to be consistent, we also need to convert the other one
            # (scale or offset) when both units differ.
            if value.units != units and value.units != '' and units != '':
                other = 'offset' if attribute == 'scale' else 'scale'
                other_quantity = self._get_quantity(other).to(value.units)
                setattr(self, other, float(other_quantity.magnitude))

            self.units = '{:~}'.format(value.units)
            setattr(self, attribute, float(value.magnitude))
        else:
            raise ValueError('`attribute` argument can only take the `scale` '
                             'or the `offset` value.')

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, s):
        if s == '':
            self._units = t.Undefined
        self._units = s


@add_gui_method(toolkey="hyperspy.DataAxis")
class BaseDataAxis(t.HasTraits):
    """Parent class defining common attributes for all DataAxis classes.

    Parameters
    ----------
    name : str, optional
        Name string by which the axis can be accessed. `<undefined>` if not set.
    units : str, optional
         String for the units of the axis vector. `<undefined>` if not set.
    navigate : bool, optional
        True for a navigation axis. Default False (signal axis).
    is_binned : bool, optional
        True if data along the axis is binned. Default False.
    """
    name = t.Str()
    units = t.Str()
    t.navigate = t.Bool()

    def __init__(self,
                 name=None,
                 units=None,
                 navigate=False,
                 is_binned=False,
                 **kwargs):
        super().__init__()
        if name is None:
            name = t.Undefined
        if units is None:
            units = t.Undefined
        self.name = name
        self.units = units

        self.navigate = navigate
        self.axes_manager = None

    @property
    def index_in_array(self):
        if self.axes_manager is not None:
            return self.axes_manager._axes.index(self)
        else:
            raise AttributeError(
                "This {} does not belong to an AxesManager"
                " and therefore its index_in_array attribute "
                " is not defined".format(self.__class__.__name__))

    @property
    def index_in_axes_manager(self):
        if self.axes_manager is not None:
            return self.axes_manager._get_axes_in_natural_order().\
                index(self)
        else:
            raise AttributeError(
                "This {} does not belong to an AxesManager"
                " and therefore its index_in_array attribute "
                " is not defined".format(self.__class__.__name__))

    @property
    def slice(self):
        if self.navigate:
            return None
        else:
            return slice(None)

    def __getitem__(self, item):
        raise NotImplementedError

    def __repr__(self):
        if hasattr(self, "size"):
            text = '<%s axis, size: %i' % (self._get_name(),
                                           self.size,)
        else:
            text = '<%s axis, size: - ' % self._get_name()
        if self.navigate is True and hasattr(self, "index"):
            text += ", index: %i" % self.index
        text += ">"
        return text

    def __str__(self):
        return self._get_name() + " axis"

    def __deepcopy__(self, memodict={}):
        new_dict = copy.deepcopy(self.get_axis_dictionary())
        return self.__class__(**new_dict)

    def _get_name(self):
        name = (self.name
                if self.name is not t.Undefined
                else ("Unnamed " +
                      ordinal(self.index_in_axes_manager))
                if self.axes_manager is not None
                else "Unnamed")
        return name

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return self.__deepcopy__()

    def get_axis_dictionary(self):
        return {'_type': self.__class__.__name__,
                'name': _parse_axis_attribute(self.name),
                'units': _parse_axis_attribute(self.units),
                'navigate': self.navigate,
                }

    def to_numpy_index(self, index):
        raise NotImplementedError("This method is only implemented in subclasses")

    def update_dict(self, new_values):
        """Update an axis from a dict of new values.
        Parameters
        ----------
        new_values: dict
            A dictionary of new values to replace the old ones. I.e {"name": "x", "units":"nm"}
        """
        for key in new_values:
            self.__setattr__(key, new_values[key])

    def update_from(self, axis, attributes):
        """Copy values of specified axes fields from the passed AxesManager.

        Parameters
        ----------
        axis : BaseDataAxis
            The BaseDataAxis instance to use as a source for values.
        attributes : iterable container of strings.
            The name of the attribute to update. If the attribute does not
            exist in either of the AxesManagers, an AttributeError will be
            raised.

        Returns
        -------
        A boolean indicating whether any changes were made.

        """
        any_changes = False
        changed = {}
        for f in attributes:
            if getattr(self, f) != getattr(axis, f):
                changed[f] = getattr(axis, f)
        if len(changed) > 0:
            self.update_dict(changed)
            any_changes = True
        return any_changes


class DataAxis(BaseDataAxis):
    """Defines a common class for all axes that are bounded.  Common features include an axis attribute
       a size and an index. Anything that can be in a numpy array can be an axis label including a string
       an unordered set of numbers or even a set of objects as is sometimes the case for a `ColumnAxis` object
    """
    def __init__(self,
                 index_in_array=None,
                 name=None,
                 units=None,
                 navigate=False,
                 is_binned=False,
                 axis=[1],
                 **kwargs):
        super().__init__(
            index_in_array=index_in_array,
            name=name,
            units=units,
            navigate=navigate,
            is_binned=is_binned,
            **kwargs)
        self.add_trait("_axis", t.Array)
        self._axis = np.asarray(axis)
        self.add_trait("index", t.Int)

    def __getitem__(self, item):
        new_axis = self.__deepcopy__()
        new_item = self.to_numpy_index(item)
        axis = self._axis[new_item]
        if not isinstance(axis, np.ndarray):  # returns scalar
            return None
        else:
            new_axis._axis = axis
            return new_axis

    """Additional Class properties"""
    @property
    def axis(self):
        return self._axis

    @axis.setter
    def axis(self, axis):
        self._axis = axis

    @property
    def is_ordered(self):
        try:
            steps = self.axis[1:] - self.axis[:-1]
            if np.all(steps > 0):
                return True
            elif np.all(steps < 0):
                return False
            else:
                # the axis is not ordered
                return None
        except:
            return None

    @property
    def is_uniform(self):
        try:
            steps = self.axis[1:] - self.axis[:-1]
            steps = np.abs(steps - steps[0])
            if np.all(steps < 0.0000001):
                return True
            else:
                return False
        except:
            return None

    @property
    def value(self):
        return self.axis[self.index]

    @property
    def high_value(self):
        if self.is_ordered is True:
            return self.axis[-1]
        elif self.is_ordered is False:
            return self.axis[0]
        else:
            raise NotImplementedError("The axis is not ordered so there is no high/low value")

    @property
    def low_value(self):
        if self.is_ordered is True:
            return self.axis[0]
        elif self.is_ordered is False:
            return self.axis[-1]
        else:
            raise NotImplementedError("The axis is not ordered so there is no high/low value")

    @property
    def low_index(self):
        return 0

    @property
    def high_index(self):
        return self.size - 1

    @property
    def size(self):
        return len(self._axis)

    def get_axis_dictionary(self):
        d = super().get_axis_dictionary()
        d["axis"] = self.axis
        return d

    """Methods for converting floats, strs, etc to integer indexes"""

    def _float2index(self, value, rounding="round"):
        """Converts a float index (or an array of floats) into
        an integer index (or an array of integer indexes"""
        if self.is_ordered:
            if np.all((value >= self.low_value) * (value <= self.high_value)):
                if rounding == "round":
                    index = numba_closest_index_round(self.axis, value).astype(int)
                elif rounding == "ceil":
                    index = numba_closest_index_ceil(self.axis, value).astype(int)
                elif rounding == "floor":
                    index = numba_closest_index_floor(self.axis, value).astype(int)
                else:
                    raise ValueError(
                        'Non-supported rounding function. Use '
                        '"round", "ceil" or "floor".'
                    )
                return index
        else:
            raise NotImplementedError("Slicing with floats not implemented for "
                                      "unordered axes")

    def _neg_slice(self, value):
        try:
            relative_value = float(value)
        except ValueError:
            raise ValueError("A negative float index must be followed by some float i.e. '-.3' ")
        relative_value = self.high_value+relative_value
        relative_value = self._float2index(np.array([relative_value, ]))[0]
        return relative_value

    def _rel_slice(self, value):
        if self.is_ordered:
            try:
                relative_value = float(value[3:])
                if relative_value > 1:
                    raise ValueError
            except ValueError:
                raise ValueError("`rel` must be followed by a number in range [0, 1].")
            value = self.low_value + relative_value * (self.high_value - self.low_value)
            value = self._float2index(np.array([value, ]))[0]
            return value

        else:
            NotImplementedError("Relative slicing not implemented for "
                                "unordered axes")

    def _str2index(self, string):
        """Converts a str index (or an array of str) into
        an integer index (or an array of integer indexes"""
        value = np.empty(string.shape, dtype=int)
        for i, s in enumerate(string):
            if s.startswith('rel'):# relative slicing
                value[i] = self._rel_slice(value=s)
            elif s in self.axis:  # labeled axes
                value[i] = np.argwhere(self.axis == s)[0]
            elif s.startswith("-"):
                value[i] = self._neg_slice(string)
            elif len(s) == 0:
                raise ValueError(f"`{string}` is not a suitable string for slicing.")
            elif s[0].isdigit() and hasattr(self, "_get_value_from_value_with_units"):  # units
                value[i] = self._get_value_from_value_with_units([s, ])
            else:
                raise ValueError(f"`{string}` is not a suitable string for slicing.")
        return value

    def value2index(self, value, **kwargs):
        """Takes a one dimensional array or scalar value and
        returns the index."""
        is_tuple = False
        is_scalar = False

        if value is None:
            return None
        elif isinstance(value, tuple):
            is_tuple = True
            value = np.array(value)
        elif isinstance(value, str):
            value = np.array([value, ])
            is_scalar = True
        elif isiterable(value):
            value = np.asarray(value)
        else:
            value = np.array([value, ])
            is_scalar = True
        dtype = value.dtype
        if np.issubdtype(dtype, np.integer):
            ind = value  # valid numpy index
        elif np.issubdtype(dtype, np.bool):
            ind = value  # valid numpy index
        elif np.issubdtype(dtype, np.float):
            ind = self._float2index(value, **kwargs)  # convert float to index
        elif np.issubdtype(dtype, np.str):
            ind = self._str2index(value)  # convert string to index
        else:
            raise ValueError("Only ints, floats and strings"
                             " can be cast to indexes")
        if is_tuple:
            ind = tuple(ind)  # array back to tuple
        if is_scalar:
            ind = ind[0]
        return ind

    def value_range_to_indices(self, x1, x2):
        if x1 is None:
            ind1 = self.low_index
        else:
            ind1 = self.value2index(float(x1))
        if x2 is None:
            ind2 = self.high_index
        else:
            ind2 = self.value2index(float(x2))
        if ind2 <= ind1:
            raise ValueError("x1 cannot be greater than or equal to x2")
        return ind1, ind2


    def to_numpy_index(self, index):
        """ Takes some index passed by a FancySlicing object and returns the index
        Parameters
        ----------
        index

        Returns
        -------

        """
        v2i = self.value2index
        if isinstance(index, slice):
            if isfloat(index.step):
                raise ValueError(
                    "Float steps are only supported for uniform axes.")
            new_index = slice(*(v2i(i) for i in (index.start, index.stop, index.step)))
        else:
            new_index = v2i(index)
        return new_index

    def convert_to_uniform_axis(self):
        """Convert to a uniform axis."""
        scale = (self.high_value - self.low_value) / self.size
        d = self.get_axis_dictionary()
        axes_manager = self.axes_manager
        del d["axis"]
        if len(self.axis) > 1:
            scale_err = max(self.axis[1:] - self.axis[:-1]) - scale
            _logger.warning('The maximum scale error is {}.'.format(scale_err))
        d["_type"] = 'UniformDataAxis'
        self.__class__ = UniformDataAxis
        self.__init__(**d, size=self.size, scale=scale, offset=self.low_value)
        self.axes_manager = axes_manager


class FunctionalDataAxis(DataAxis):
    """DataAxis class for a non-uniform axis defined through an ``expression``.

    A `FunctionalDataAxis` is defined based on an ``expression`` that is
    evaluated to yield the axis points. The `expression` is a function defined
    as a ``string`` using the `SymPy <https://docs.sympy.org/latest/tutorial/intro.html>`_
    text expression format. An example would be ``expression = a / x + b``.
    Any variables in the expression, in this case ``a`` and ``b`` must be
    defined as additional attributes of the axis. The property ``is_uniform``
    is automatically set to ``False``.

    ``x`` itself is an instance of `BaseDataAxis`. By default, it will be a
    `UniformDataAxis` with ``offset = 0`` and ``scale = 1`` of the given
    ``size``. However, it can also be initialized with custom ``offset`` and
    ``scale`` values. Alternatively, it can be a non-uniform `DataAxis`.

    Parameters
    ----------
    expression: str
        SymPy mathematical expression defining the axis.
    x : BaseDataAxis
        Defines x-values at which `expression` is evaluated.

    Examples
    --------
    Sample dictionary for a FunctionalDataAxis:

    >>> dict0 = {'expression': 'a / (x + 1) + b', 'a': 100, 'b': 10, 'size': 500}
    >>> s = hs.signals.Signal1D(np.ones(500), axes=[dict0])
    >>> s.axes_manager[0].get_axis_dictionary()
    {'_type': 'FunctionalDataAxis',
     'name': <undefined>,
     'units': <undefined>,
     'navigate': False,
     'expression': 'a / (x + 1) + b',
     'size': 500,
     'x': {'_type': 'UniformDataAxis',
      'name': <undefined>,
      'units': <undefined>,
      'navigate': <undefined>,
      'size': 500,
      'scale': 1.0,
      'offset': 0.0},
     'a': 100,
     'b': 10}
    """
    def __init__(self,
                 expression,
                 x=None,
                 index_in_array=None,
                 name=None,
                 units=None,
                 navigate=False,
                 size=1,
                 is_binned=False,
                 **parameters):
        super().__init__(
            index_in_array=index_in_array,
            name=name,
            units=units,
            navigate=navigate,
            is_binned=is_binned,
            **parameters)

        if x is None:
            if size is t.Undefined:
                raise ValueError("Please provide either `x` or `size`.")
            self._axis = range(0, size)
        elif isiterable(x):
            self._axis = np.asarray(x)
        else:
            self._axis = None

        self._expression = expression
        if '_type' in parameters:
            del parameters['_type']
        # Compile function
        expr = _parse_substitutions(self._expression)

        variables = ["x"]
        expr_parameters = [symbol for symbol in expr.free_symbols
                           if symbol.name not in variables]
        if set(parameters) != set([parameter.name for parameter in expr_parameters]):
            raise ValueError(
                "The values of the following expression parameters "
                f"must be given as keywords: {set(expr_parameters) - set(parameters)}")

        self._function = lambdify(
            variables + expr_parameters, expr.evalf(), dummify=False)
        for parameter in parameters.keys():
            self.add_trait(parameter, t.CFloat(parameters[parameter]))
        self.parameters_list = list(parameters.keys())
        self.update_axis()
        self.on_trait_change(self.update_axis, self.parameters_list)

    @property
    def axis(self):
        kwargs = {}
        for key in self.parameters_list:
            kwargs[key] = getattr(self, key)
        self._function(x=self.axis, **kwargs)

    @property
    def x(self):
        return self._axis

    def update_from(self, axis, attributes=None):
        """Copy values of specified axes fields from the passed AxesManager.

        Parameters
        ----------
        axis : FunctionalDataAxis
            The FunctionalDataAxis instance to use as a source for values.
        attributes : iterable container of strings or None.
            A list of the name of the attribute to update. If an attribute does not
            exist in either of the AxesManagers, an AttributeError will be
            raised. If None, the parameters of `expression` are updated.
        Returns
        -------
        A boolean indicating whether any changes were made.

        """
        if attributes is None:
            attributes = self.parameters_list
        return super().update_from(axis, attributes)

    def get_axis_dictionary(self):
        d = super().get_axis_dictionary()
        d['expression'] = self._expression
        d.update({'x': self.x, })
        for kwarg in self.parameters_list:
            d[kwarg] = getattr(self, kwarg)
        return d

    def convert_to_non_uniform_axis(self):
        """Convert to a non-uniform axis."""
        d = super().get_axis_dictionary()
        axes_manager = self.axes_manager
        d["_type"] = 'DataAxis'
        self.__class__ = DataAxis
        self.__init__(**d)
        del self._expression
        del self._function
        self.axes_manager = axes_manager


class UniformDataAxis(DataAxis, UnitConversion):
    """DataAxis class for a uniform axis defined through a ``scale``, an
    ``offset`` and a ``size``.

    The most common type of axis. It is defined by the ``offset``, ``scale``
    and ``size`` parameters, which determine the `initial value`, `spacing` and
    `length` of the axis, respectively. The actual ``axis`` array is
    automatically calculated from these three values. The ``UniformDataAxis``
    is a special case of the ``FunctionalDataAxis`` defined by the function
    ``scale * x + offset``.

    Parameters
    ----------
    offset : float
        The first value of the axis vector.
    scale : float
        The spacing between axis points.
    size : int
        The number of points in the axis.

    Examples
    --------
    Sample dictionary for a `UniformDataAxis`:

    >>> dict0 = {'offset': 300, 'scale': 1, 'size': 500}
    >>> s = hs.signals.Signal1D(np.ones(500), axes=[dict0])
    >>> s.axes_manager[0].get_axis_dictionary()
    {'_type': 'UniformDataAxis',
     'name': <undefined>,
     'units': <undefined>,
     'navigate': False,
     'size': 500,
     'scale': 1.0,
     'offset': 300.0}
    """
    def __init__(self,
                 index_in_array=None,
                 name=None,
                 units=None,
                 navigate=False,
                 size=1,
                 scale=1.,
                 offset=0.,
                 is_binned=False,
                 **kwargs):
        super().__init__(
            name=name,
            units=units,
            navigate=navigate,
            is_binned=is_binned,
            **kwargs
            )
        # These traits need to added dynamically to be removed when necessary
        self.add_trait("scale", t.CFloat)
        self.add_trait("offset", t.CFloat)
        self.add_trait("size", t.CInt)
        self.scale = scale
        self.offset = offset
        self.size = size

    @property
    def axis(self):
        ax = np.add(np.multiply(np.arange(self.size), self.scale), self.offset)
        return ax

    def to_numpy_index(self, index, **kwargs):
        """ Takes some index passed by a FancySlicing object and returns the index
        Parameters
        ----------
        index

        Returns
        -------

        """
        v2i = self.value2index
        if isinstance(index, slice):
            new_index = slice(v2i(index.start, **kwargs), v2i(index.stop, **kwargs), v2i(index.step, False,**kwargs))
        else:
            new_index = v2i(index, **kwargs)
        return new_index

    def __getitem__(self, item):
        new_axis = self.__deepcopy__()
        ind = self.get_numpy_index(item)
        if isinstance(ind, slice):
            if ind.step is not None:
                new_axis.scale = ind.step*self.scale
            new_axis.offset = self.index2value(ind.start)
            self.size = len(self.axis[ind])
        else:
            new_axis = new_axis.convert_to_non_uniform_axis()
            new_axis.axis[ind]
        return new_axis

    def get_axis_dictionary(self):
        d = super(DataAxis, self).get_axis_dictionary()  # don't save the axis object
        d.update({'size': self.size,
                  'scale': self.scale,
                  'offset': self.offset})
        return d

    def _float2index(self, value, rounding="round", include_offset=True):
        """Overwrites previous float to index method. Faster because it doesn't depend on
        the `axis` parameter"""

        if include_offset:
            multiplier = 1E12
            index = 1 / multiplier * np.trunc(
                (value - self.offset) / self.scale * multiplier
            )
        else:
            if include_offset:
                multiplier = 1E12
                index = 1 / multiplier * np.trunc(value / self.scale * multiplier)

        if rounding == "round":
            index = np.round(index)
        elif rounding == "ceil":
            index = np.ceil(index)
        elif rounding == "floor":
            index = np.floor(index)
        else:
            raise ValueError(
                        'Non-supported rounding function. Use '
                        '"round", "ceil" or "floor".'
                    )

        if any(np.isnan(index)):
            raise ValueError("Indexes cannot be np.nan")
        if any(index < 0):
            _logger.warning("Negative slicing using floats requires using `-0.5`"
                            "rather than -0.5 as the second case can be ambiguous")
            index[index < 0] = 0
        if any(index > self.high_index):
            _logger.warning("Value is greater than the max value. Max index is returned")
            index[index > self.high_index] = self.high_index

        np.array(index,dtype=int)

        return index

    def calibrate(self, value_tuple, index_tuple, modify_calibration=True):
        scale = (value_tuple[1] - value_tuple[0]) /\
            (index_tuple[1] - index_tuple[0])
        offset = value_tuple[0] - scale * index_tuple[0]
        if modify_calibration is True:
            self.offset = offset
            self.scale = scale
        else:
            return offset, scale

    def update_from(self, axis, attributes=None):
        """Copy values of specified axes fields from the passed AxesManager.

        Parameters
        ----------
        axis : UniformDataAxis
            The UniformDataAxis instance to use as a source for values.
        attributes : iterable container of strings or None
            The name of the attribute to update. If the attribute does not
            exist in either of the AxesManagers, an AttributeError will be
            raised. If `None`, `scale`, `offset` and `units` are updated.
        Returns
        -------
        A boolean indicating whether any changes were made.

        """
        if attributes is None:
            attributes = ["scale", "offset", "units"]
        return super().update_from(axis, attributes)

    @property
    def scale_as_quantity(self):
        return self._get_quantity('scale')

    @scale_as_quantity.setter
    def scale_as_quantity(self, value):
        self._set_quantity(value, 'scale')

    @property
    def offset_as_quantity(self):
        return self._get_quantity('offset')

    @offset_as_quantity.setter
    def offset_as_quantity(self, value):
        self._set_quantity(value, 'offset')

    def convert_to_functional_data_axis(self, expression, units=None, name=None, **kwargs):
        d = super().get_axis_dictionary()  # Calls DataAxis get_axis_dictionary
        axes_manager = self.axes_manager
        if units:
            d["units"] = units
        if name:
            d["name"] = name

        d["x"] = d.pop("axis")
        d.update(kwargs)
        self.remove_trait('scale')
        self.remove_trait('offset')
        self.remove_trait('size')
        self.__class__ = FunctionalDataAxis
        d["_type"] = 'FunctionalDataAxis'
        self.__init__(expression=expression, **d)
        self.axes_manager = axes_manager

    def convert_to_non_uniform_axis(self):
        d = super().get_axis_dictionary()
        axes_manager = self.axes_manager
        self.__class__ = DataAxis
        d["_type"] = 'DataAxis'
        self.remove_trait('scale')
        self.remove_trait('offset')
        self.remove_trait('size')
        self.__init__(**d)
        self.axes_manager = axes_manager


def _serpentine_iter(shape):
    '''Similar to np.ndindex, but yields indices
    in serpentine pattern, like snake game.
    Takes shape in hyperspy order, not numpy order.

    Code by Stackoverflow user Paul Panzer,
    from https://stackoverflow.com/questions/57366966/

    Note that the [::-1] reversing is necessary to iterate first along
    the x-direction on multidimensional navigators.
    '''
    shape = shape[::-1]
    N = len(shape)
    idx = N*[0]
    drc = N*[1]
    while True:
        yield (*idx,)[::-1]
        for j in reversed(range(N)):
            if idx[j] + drc[j] not in (-1, shape[j]):
                idx[j] += drc[j]
                break
            drc[j] *= -1
        else:  # pragma: no cover
            break

def _flyback_iter(shape):
    "Classic flyback scan pattern generator which yields indices in similar fashion to np.ndindex. Takes shape in hyperspy order, not numpy order."
    shape = shape[::-1]
    class ndindex_reversed(np.ndindex):
        def __next__(self):
            next(self._it)
            return self._it.multi_index[::-1]

    return ndindex_reversed(shape)


@add_gui_method(toolkey="hyperspy.AxesManager")
class AxesManager(t.HasTraits):

    """Contains and manages the data axes.

    It supports indexing, slicing, subscripting and iteration. As an iterator,
    iterate over the navigation coordinates returning the current indices.
    It can only be indexed and sliced to access the DataAxis objects that it
    contains. Standard indexing and slicing follows the "natural order" as in
    Signal, i.e. [nX, nY, ...,sX, sY,...] where `n` indicates a navigation axis
    and `s` a signal axis. In addition, AxesManager supports indexing using
    complex numbers a + bj, where b can be one of 0, 1, 2 and 3 and a valid
    index. If b is 3, AxesManager is indexed using the order of the axes in the
    array. If b is 1(2), indexes only the navigation(signal) axes in the
    natural order. In addition AxesManager supports subscription using
    axis name.

    Attributes
    ----------
    signal_axes, navigation_axes : list
        Contain the corresponding DataAxis objects

    coordinates, indices, iterpath

    Examples
    --------

    Create a spectrum with random data

    >>> s = hs.signals.Signal1D(np.random.random((2,3,4,5)))
    >>> s.axes_manager
    <Axes manager, axes: (4, 3, 2|5)>
                Name |   size |  index |  offset |   scale |  units
    ================ | ====== | ====== | ======= | ======= | ======
         <undefined> |      4 |      0 |       0 |       1 | <undefined>
         <undefined> |      3 |      0 |       0 |       1 | <undefined>
         <undefined> |      2 |      0 |       0 |       1 | <undefined>
    ---------------- | ------ | ------ | ------- | ------- | ------
         <undefined> |      5 |        |       0 |       1 | <undefined>
    >>> s.axes_manager[0]
    <Unnamed 0th axis, size: 4, index: 0>
    >>> s.axes_manager[3j]
    <Unnamed 2nd axis, size: 2, index: 0>
    >>> s.axes_manager[1j]
    <Unnamed 0th axis, size: 4, index: 0>
    >>> s.axes_manager[2j]
    <Unnamed 3rd axis, size: 5>
    >>> s.axes_manager[1].name = "y"
    >>> s.axes_manager["y"]
    <y axis, size: 3, index: 0>
    >>> for i in s.axes_manager:
    ...     print(i, s.axes_manager.indices)
    ...
    (0, 0, 0) (0, 0, 0)
    (1, 0, 0) (1, 0, 0)
    (2, 0, 0) (2, 0, 0)
    (3, 0, 0) (3, 0, 0)
    (0, 1, 0) (0, 1, 0)
    (1, 1, 0) (1, 1, 0)
    (2, 1, 0) (2, 1, 0)
    (3, 1, 0) (3, 1, 0)
    (0, 2, 0) (0, 2, 0)
    (1, 2, 0) (1, 2, 0)
    (2, 2, 0) (2, 2, 0)
    (3, 2, 0) (3, 2, 0)
    (0, 0, 1) (0, 0, 1)
    (1, 0, 1) (1, 0, 1)
    (2, 0, 1) (2, 0, 1)
    (3, 0, 1) (3, 0, 1)
    (0, 1, 1) (0, 1, 1)
    (1, 1, 1) (1, 1, 1)
    (2, 1, 1) (2, 1, 1)
    (3, 1, 1) (3, 1, 1)
    (0, 2, 1) (0, 2, 1)
    (1, 2, 1) (1, 2, 1)
    (2, 2, 1) (2, 2, 1)
    (3, 2, 1) (3, 2, 1)

    """

    _axes = t.List(BaseDataAxis)
    signal_axes = t.Tuple()
    navigation_axes = t.Tuple()
    _step = t.Int(1)

    def __init__(self, axes_list):
        super().__init__()
        self.events = Events()
        self.events.indices_changed = Event("""
            Event that triggers when the indices of the `AxesManager` changes

            Triggers after the internal state of the `AxesManager` has been
            updated.

            Arguments:
            ----------
            obj : The AxesManager that the event belongs to.
            """, arguments=['obj'])
        self.events.any_axis_changed = Event("""
            Event that trigger when the space defined by the axes transforms.

            Specifically, it triggers when one or more of the following
            attributes changes on one or more of the axes:
                `offset`, `size`, `scale`

            Arguments:
            ----------
            obj : The AxesManager that the event belongs to.
            """, arguments=['obj'])

        # Remove all axis for cases, we reinitiliase the AxesManager
        if self._axes:
            self.remove(self._axes)
        self.create_axes(axes_list)

        self._update_attributes()
        self._update_trait_handlers()
        self.iterpath = 'flyback'
        self._ragged = False

    @property
    def ragged(self):
        return self._ragged

    def _update_trait_handlers(self, remove=False):
        things = {self._on_index_changed: '_axes.index',
                  self._on_slice_changed: '_axes.slice',
                  self._on_size_changed: '_axes.size',
                  self._on_scale_changed: '_axes.scale',
                  self._on_offset_changed: '_axes.offset'}

        for k, v in things.items():
            self.on_trait_change(k, name=v, remove=remove)

    def _get_positive_index(self, axis):
        if axis < 0:
            axis += len(self._axes)
            if axis < 0:
                raise IndexError("index out of bounds")
        return axis

    def _array_indices_generator(self):
        shape = (self.navigation_shape[::-1] if self.navigation_size > 0 else
                 [1, ])
        return np.ndindex(*shape)

    def _am_indices_generator(self):
        shape = (self.navigation_shape if self.navigation_size > 0 else
                 [1, ])[::-1]
        return ndindex_nat(*shape)

    def __getitem__(self, y):
        """x.__getitem__(y) <==> x[y]

        """
        if isinstance(y, str) or not np.iterable(y):
            return self[(y,)][0]
        axes = [self._axes_getter(ax) for ax in y]
        _, indices = np.unique(
            [_id for _id in map(id, axes)], return_index=True)
        ans = tuple(axes[i] for i in sorted(indices))
        return ans

    def _axes_getter(self, y):
        if isinstance(y, BaseDataAxis):
            if y in self._axes:
                return y
            else:
                raise ValueError(f"{y} is not in {self}")
        if isinstance(y, str):
            axes = list(self._get_axes_in_natural_order())
            while axes:
                axis = axes.pop()
                if y == axis.name:
                    return axis
            raise ValueError("There is no DataAxis named %s" % y)
        elif (isfloat(y.real) and not y.real.is_integer() or
                isfloat(y.imag) and not y.imag.is_integer()):
            raise TypeError("axesmanager indices must be integers, "
                            "complex integers or strings")
        if y.imag == 0:  # Natural order
            return self._get_axes_in_natural_order()[y]
        elif y.imag == 3:  # Array order
            # Array order
            return self._axes[int(y.real)]
        elif y.imag == 1:  # Navigation natural order
            #
            return self.navigation_axes[int(y.real)]
        elif y.imag == 2:  # Signal natural order
            return self.signal_axes[int(y.real)]
        else:
            raise IndexError("axesmanager imaginary part of complex indices "
                             "must be 0, 1, 2 or 3")

    def __getslice__(self, i=None, j=None):
        """x.__getslice__(i, j) <==> x[i:j]

        """
        return self._get_axes_in_natural_order()[i:j]

    def _get_axes_in_natural_order(self):
        return self.navigation_axes + self.signal_axes

    @property
    def _navigation_shape_in_array(self):
        return self.navigation_shape[::-1]

    @property
    def _signal_shape_in_array(self):
        return self.signal_shape[::-1]

    @property
    def shape(self):
        nav_shape = (self.navigation_shape
                     if self.navigation_shape != (0,)
                     else tuple())
        sig_shape = (self.signal_shape
                     if self.signal_shape != (0,)
                     else tuple())
        return nav_shape + sig_shape

    @property
    def signal_extent(self):
        signal_extent = []
        for signal_axis in self.signal_axes:
            signal_extent.append(signal_axis.low_value)
            signal_extent.append(signal_axis.high_value)
        return tuple(signal_extent)

    @property
    def navigation_extent(self):
        navigation_extent = []
        for navigation_axis in self.navigation_axes:
            navigation_extent.append(navigation_axis.low_value)
            navigation_extent.append(navigation_axis.high_value)
        return tuple(navigation_extent)

    @property
    def all_uniform(self):
        if any([axis.is_uniform == False for axis in self._axes]):
            return False
        else:
            return True

    def remove(self, axes):
        """Remove one or more axes
        """
        axes = self[axes]
        if not np.iterable(axes):
            axes = (axes,)
        for ax in axes:
            self._remove_one_axis(ax)

    def _remove_one_axis(self, axis):
        """Remove the given Axis.

        Raises
        ------
        ValueError
            If the Axis is not present.

        """
        axis = self._axes_getter(axis)
        axis.axes_manager = None
        self._axes.remove(axis)

    def __delitem__(self, i):
        self.remove(self[i])

    def _get_data_slice(self, fill=None):
        """Return a tuple of slice objects to slice the data.

        Parameters
        ----------
        fill: None or iterable of (int, slice)
            If not None, fill the tuple of index int with the given
            slice.

        """
        cslice = [slice(None), ] * len(self._axes)
        if fill is not None:
            for index, slice_ in fill:
                cslice[index] = slice_
        return tuple(cslice)

    def create_axes(self, axes_list):
        """Given a list of either axes dictionaries or axes objects, these are
        added to the AxesManager. In case dictionaries defining the axes
        properties are passed, the DataAxis/UniformDataAxis/FunctionalDataAxis
        instances are first created.

        The index of the axis in the array and in the `_axes` lists
        can be defined by the index_in_array keyword if given
        for all axes. Otherwise, it is defined by their index in the
        list.

        See also
        --------
        _append_axis

        """
        # Reorder axes_list using index_in_array if it is defined
        # for all axes and the indices are not repeated.
        indices = set([axis['index_in_array'] for axis in axes_list if
                       hasattr(axis, 'index_in_array')])
        if len(indices) == len(axes_list):
            axes_list.sort(key=lambda x: x['index_in_array'])
        for axis_dict in axes_list:
            if isinstance(axis_dict, dict):
                self._append_axis(**axis_dict)
            else:
                self._axes.append(axis_dict)

    def set_axis(self, axis, index_in_axes_manager):
        """Replace an axis of current signal with one given in argument.

        Parameters
        ----------
        axis: BaseDataAxis axis to replace the current axis with

        index_in_axes_manager: index of the axis in current signal to remplace
            with axis passed in argument

        See also
        --------
        _append_axis

        """
        self._axes[index_in_axes_manager] = axis

    def _update_max_index(self):
        self._max_index = 1
        for i in self.navigation_shape:
            self._max_index *= i
        if self._max_index != 0:
            self._max_index -= 1

    @property
    def iterpath(self):
        """Sets the order of iterating through the indices in the navigation
        dimension. Can be either "flyback" or "serpentine", or an iterable
        of navigation indices.
        """
        return self._iterpath

    @iterpath.setter
    def iterpath(self, path):
        if isinstance(path, str):
            if path == 'serpentine':
                self._iterpath = 'serpentine'
                self._iterpath_generator = _serpentine_iter(self.navigation_shape)
            elif path == 'flyback':
                self._iterpath = 'flyback'
                self._iterpath_generator = _flyback_iter(self.navigation_shape)
            else:
                raise ValueError(
                    f'The iterpath scan pattern is set to `"{path}"`. '
                    'It must be either "serpentine" or "flyback", or an iterable '
                    'of navigation indices, and is set either as multifit '
                    '`iterpath` argument or `axes_manager.iterpath`'
                    )
        else:
            # Passing a custom indices iterator
            try:
                iter(path) # If this fails, its not an iterable and we raise TypeError
            except TypeError as e:
                raise TypeError(
                    f'The iterpath `{path}` is not an iterable. '
                    'Ensure it is an iterable like a list, array or generator.'
                    ) from e
            try:
                if not (inspect.isgenerator(path) or type(path) is GeneratorLen):
                # If iterpath is a generator, then we can't check its first value, have to trust it
                    first_indices = path[0]
                    if not isinstance(first_indices, Iterable):
                        raise TypeError
                    assert len(first_indices) == self.navigation_dimension
            except TypeError as e:
                raise TypeError(
                    f"Each set of indices in the iterpath should be an iterable, e.g. `(0,)` or `(0,0,0)`. "
                    f"The first entry currently looks like: `{first_indices}`, and does not satisfy this requirement."
                    ) from e
            except AssertionError as e:
                raise ValueError(
                    f"The current iterpath yields indices of length "
                    f"{len(path)}. It should deliver incides with length "
                    f"equal to the navigation dimension, which is {self.navigation_dimension}."
                    ) from e
            else:
                self._iterpath = path
                self._iterpath_generator = iter(self._iterpath)

    def _get_iterpath_size(self, masked_elements=0):
        "Attempts to get the iterpath size, returning None if it is unknown"
        if isinstance(self.iterpath, str):
            # flyback and serpentine have well-defined lengths <- navigation_size
            maxval = self.navigation_size - masked_elements
        else:
            try:
                maxval = len(self.iterpath)
                if masked_elements:
                    # Checking if mask indices exist in the iterpath could take a long time,
                    # or may not be possible in the case of a generator.
                    _logger.info(
                    ("The progressbar length cannot be estimated when using both custom iterpath and a mask."
                    "The progressbar may terminate before it appears complete. This can safely be ignored."),
                    )
            except TypeError:
                # progressbar is shown, so user can monitor "iterations per second"
                # but the length of the bar is unknown
                maxval = None
                _logger.info(
                    ("The AxesManager `iterpath` is missing the `__len__` method, so does not have a known length. "
                    "The progressbar will only show run time and iterations per second, but no actual progress indicator."),
                    )
        return maxval

    def __next__(self):
        """
        Standard iterator method, returns the current coordinates

        Returns
        -------
        self.indices : tuple of ints
            Returns a tuple containing the coordinates of the current
            iteration.

        """
        self.indices = next(self._iterpath_generator)
        return self.indices

    def __iter__(self):
        # re-initialize iterpath as it is set before correct data shape
        # is created before data shape is known
        self.iterpath = self._iterpath
        return self

    @contextmanager
    def switch_iterpath(self, iterpath=None):
        """
        Context manager to change iterpath. The original iterpath is restored
        when exiting the context.

        Parameters
        ----------
        iterpath : str, optional
            The iterpath to use. The default is None.

        Yields
        ------
        None.

        Examples
        --------
        >>> s = hs.signals.Signal1D(np.arange(2*3*4).reshape([3, 2, 4]))
        >>> with s.axes_manager.switch_iterpath('serpentine'):
        >>>     for indices in s.axes_manager:
        >>>         print(indices)

        """
        if iterpath is not None:
            original_iterpath = self._iterpath
            self._iterpath = iterpath
        try:
            yield
        finally:
            # if an error is raised when using this context manager, we
            # reset the original value of _iterpath
            if iterpath is not None:
                self.iterpath = original_iterpath

    def _append_axis(self, **kwargs):
        axis = create_axis(**kwargs)
        axis.axes_manager = self
        self._axes.append(axis)

    def _on_index_changed(self):
        self._update_attributes()
        self.events.indices_changed.trigger(obj=self)

    def _on_slice_changed(self):
        self._update_attributes()

    def _on_size_changed(self):
        self._update_attributes()
        self.events.any_axis_changed.trigger(obj=self)

    def _on_scale_changed(self):
        self.events.any_axis_changed.trigger(obj=self)

    def _on_offset_changed(self):
        self.events.any_axis_changed.trigger(obj=self)

    def convert_units(self, axes=None, units=None, same_units=True,
                      factor=0.25):
        """ Convert the scale and the units of the selected axes. If the unit
        of measure is not supported by the pint library, the scale and units
        are not changed.

        Parameters
        ----------
        axes : {int | string | iterable of `DataAxis` | None}
            Default = None
            Convert to a convenient scale and units on the specified axis.
            If int, the axis can be specified using the index of the
            axis in `axes_manager`.
            If string, argument can be `navigation` or `signal` to select the
            navigation or signal axes. The axis name can also be provided.
            If `None`, convert all axes.
        units : {list of string of the same length than axes | str | None}
            Default = None
            If list, the selected axes will be converted to the provided units.
            If str, the navigation or signal axes will be converted to the
            provided units.
            If `None`, the scale and the units are converted to the appropriate
            scale and units to avoid displaying scalebar with >3 digits or too
            small number. This can be tweaked by the `factor` argument.
        same_units : bool
            If `True`, force to keep the same units if the units of
            the axes differs. It only applies for the same kind of axis,
            `navigation` or `signal`. By default the converted units of the
            first axis is used for all axes. If `False`, convert all axes
            individually.
        %s

        Note
        ----
        Requires a uniform axis.
        """
        convert_navigation = convert_signal = True

        if axes is None:
            axes = self.navigation_axes + self.signal_axes
            convert_navigation = (len(self.navigation_axes) > 0)
        elif axes == 'navigation':
            axes = self.navigation_axes
            convert_signal = False
            convert_navigation = (len(self.navigation_axes) > 0)
        elif axes == 'signal':
            axes = self.signal_axes
            convert_navigation = False
        elif isinstance(axes, (UniformDataAxis, int, str)):
            if not isinstance(axes, UniformDataAxis):
                axes = self[axes]
            axes = (axes, )
            convert_navigation = axes[0].navigate
            convert_signal = not convert_navigation
        else:
            raise TypeError(
                'Axes type `{}` is not correct.'.format(type(axes)))

        for axis in axes:
            if not axis.is_uniform:
                raise NotImplementedError(
                    "This operation is not implemented for non-uniform axes "
                    f"such as {axis}")

        if isinstance(units, str) or units is None:
            units = [units] * len(axes)
        elif isinstance(units, list):
            if len(units) != len(axes):
                raise ValueError('Length of the provided units list {} should '
                                 'be the same than the length of the provided '
                                 'axes {}.'.format(units, axes))
        else:
            raise TypeError('Units type `{}` is not correct. It can be a '
                            '`string`, a `list` of string or `None`.'
                            ''.format(type(units)))

        if same_units:
            if convert_navigation:
                units_nav = units[:self.navigation_dimension]
                self._convert_axes_to_same_units(self.navigation_axes,
                                                 units_nav, factor)
            if convert_signal:
                offset = self.navigation_dimension if convert_navigation else 0
                units_sig = units[offset:]
                self._convert_axes_to_same_units(self.signal_axes,
                                                 units_sig, factor)
        else:
            for axis, unit in zip(axes, units):
                axis.convert_to_units(unit, factor=factor)

    convert_units.__doc__ %= FACTOR_DOCSTRING

    def _convert_axes_to_same_units(self, axes, units, factor=0.25):
        # Check if the units are supported
        for axis in axes:
            if axis._ignore_conversion(axis.units):
                return

        # Set the same units for all axes, use the unit of the first axis
        # as reference
        axes[0].convert_to_units(units[0], factor=factor)
        unit = axes[0].units  # after conversion, in case units[0] was None.
        for axis in axes[1:]:
            # Convert only the units have the same dimensionality
            if _ureg(axis.units).dimensionality == _ureg(unit).dimensionality:
                axis.convert_to_units(unit, factor=factor)

    def update_axes_attributes_from(self, axes,
                                    attributes=None):
        """Update the axes attributes to match those given.

        The axes are matched by their index in the array. The purpose of this
        method is to update multiple axes triggering `any_axis_changed` only
        once.

        Parameters
        ----------
        axes: iterable of `DataAxis` instances.
            The axes to copy the attributes from.
        attributes: iterable of strings.
            The attributes to copy.

        """

        # To only trigger once even with several changes, we suppress here
        # and trigger manually below if there were any changes.
        changes = False
        with self.events.any_axis_changed.suppress():
            for axis in axes:
                changed = self._axes[axis.index_in_array].update_from(
                    axis=axis, attributes=attributes)
                changes = changes or changed
        if changes:
            self.events.any_axis_changed.trigger(obj=self)

    def _update_attributes(self):
        getitem_tuple = []
        values = []
        signal_axes = ()
        navigation_axes = ()
        for axis in self._axes:
            # Until we find a better place, take property of the axes
            # here to avoid difficult to debug bugs.
            axis.axes_manager = self
            if axis.slice is None:
                getitem_tuple += axis.index,
                values.append(axis.value)
                navigation_axes += axis,
            else:
                getitem_tuple += axis.slice,
                signal_axes += axis,
        if not signal_axes and navigation_axes:
            getitem_tuple[-1] = slice(axis.index, axis.index + 1)

        self._signal_axes = signal_axes[::-1]
        self._navigation_axes = navigation_axes[::-1]
        self._getitem_tuple = tuple(getitem_tuple)

        if len(self.signal_axes) == 1 and self.signal_axes[0].size == 1:
            self._signal_dimension = 0
        else:
            self._signal_dimension = len(self.signal_axes)
        self._navigation_dimension = len(self.navigation_axes)

        self._signal_size = (np.prod(self.signal_shape)
                             if self.signal_shape else 0)
        self._navigation_size = (np.prod(self.navigation_shape)
                                 if self.navigation_shape else 0)

        self._update_max_index()

    @property
    def signal_axes(self):
        """The signal axes as a tuple."""
        return self._signal_axes

    @property
    def navigation_axes(self):
        """The navigation axes as a tuple."""
        return self._navigation_axes

    @property
    def signal_shape(self):
        """The shape of the signal space."""
        return tuple([axis.size for axis in self._signal_axes])

    @property
    def navigation_shape(self):
        """The shape of the navigation space."""
        if self.navigation_dimension != 0:
            return tuple([axis.size for axis in self._navigation_axes])
        else:
            return ()

    @property
    def signal_size(self):
        """The size of the signal space."""
        return self._signal_size

    @property
    def navigation_size(self):
        """The size of the navigation space."""
        return self._navigation_size

    @property
    def navigation_dimension(self):
        """The dimension of the navigation space."""
        return self._navigation_dimension

    @property
    def signal_dimension(self):
        """The dimension of the signal space."""
        return self._signal_dimension

    def _set_signal_dimension(self, value):
        if len(self._axes) == 0 or self._signal_dimension == value:
            # Nothing to be done
            return
        elif self.ragged and value > 0:
            raise ValueError("Signal containing ragged array "
                             "must have zero signal dimension.")
        elif value > len(self._axes):
            raise ValueError(
                "The signal dimension cannot be greater "
                f"than the number of axes which is {len(self._axes)}")
        elif value < 0:
            raise ValueError(
                "The signal dimension must be a positive integer")

        # Figure out which axis needs navigate=True
        tl = [True] * len(self._axes)
        if value != 0:
            tl[-value:] = (False,) * value
        for axis in self._axes:
            # Changing navigate attribute will update the axis._slice
            # which in turn will trigger _on_slice_changed and call
            # _update_attribute
            axis.navigate = tl.pop(0)

    def key_navigator(self, event):
        'Set hotkeys for controlling the indices of the navigator plot'

        if self.navigation_dimension == 0:
            # No hotkeys exist that do anything in this case
            return

        # keyDict values are (axis_index, direction)
        # Using arrow keys without Ctrl will be deprecated in 2.0
        mod01 = preferences.Plot.modifier_dims_01
        mod23 = preferences.Plot.modifier_dims_23
        mod45 = preferences.Plot.modifier_dims_45

        dim0_decrease = mod01 + '+' + preferences.Plot.dims_024_decrease
        dim0_increase = mod01 + '+' + preferences.Plot.dims_024_increase
        dim1_decrease = mod01 + '+' + preferences.Plot.dims_135_decrease
        dim1_increase = mod01 + '+' + preferences.Plot.dims_135_increase
        dim2_decrease = mod23 + '+' + preferences.Plot.dims_024_decrease
        dim2_increase = mod23 + '+' + preferences.Plot.dims_024_increase
        dim3_decrease = mod23 + '+' + preferences.Plot.dims_135_decrease
        dim3_increase = mod23 + '+' + preferences.Plot.dims_135_increase
        dim4_decrease = mod45 + '+' + preferences.Plot.dims_024_decrease
        dim4_increase = mod45 + '+' + preferences.Plot.dims_024_increase
        dim5_decrease = mod45 + '+' + preferences.Plot.dims_135_decrease
        dim5_increase = mod45 + '+' + preferences.Plot.dims_135_increase

        keyDict = {
            # axes 0, 1
            **dict.fromkeys(['left', dim0_decrease, '4'], (0, -1)),
            **dict.fromkeys(['right', dim0_increase, '6'], (0, +1)),
            **dict.fromkeys(['up', dim1_decrease, '8'], (1, -1)),
            **dict.fromkeys(['down', dim1_increase, '2'], (1, +1)),
            # axes 2, 3
            **dict.fromkeys([dim2_decrease], (2, -1)),
            **dict.fromkeys([dim2_increase], (2, +1)),
            **dict.fromkeys([dim3_decrease], (3, -1)),
            **dict.fromkeys([dim3_increase], (3, +1)),
            # axes 4, 5
            **dict.fromkeys([dim4_decrease], (4, -1)),
            **dict.fromkeys([dim4_increase], (4, +1)),
            **dict.fromkeys([dim5_decrease], (5, -1)),
            **dict.fromkeys([dim5_increase], (5, +1)),
        }

        if event.key == 'pageup':
            self._step += 1
        elif event.key == 'pagedown':
            if self._step > 1:
                self._step -= 1
        else:
            try:
                # may raise keyerror
                axes_index, direction = keyDict[event.key]
                axes = self.navigation_axes[axes_index]  # may raise indexerror
                axes.index += direction * self._step  # may raise traiterror
            except (KeyError, IndexError, TraitError):
                pass

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)

    def __deepcopy__(self, *args):
        return AxesManager(self._get_axes_dicts())

    def _get_axes_dicts(self, axes=None):
        if axes is None:
            axes = self._axes
        axes_dicts = []
        for axis in axes:
            axes_dicts.append(axis.get_axis_dictionary())
        return axes_dicts

    def as_dictionary(self):
        am_dict = {}
        for i, axis in enumerate(self._axes):
            am_dict['axis-%i' % i] = axis.get_axis_dictionary()
        return am_dict

    def _get_signal_axes_dicts(self):
        return [axis.get_axis_dictionary() for axis in
                self.signal_axes[::-1]]

    def _get_navigation_axes_dicts(self):
        return [axis.get_axis_dictionary() for axis in
                self.navigation_axes[::-1]]

    def _get_dimension_str(self):
        string = "("
        for axis in self.navigation_axes:
            string += str(axis.size) + ", "
        string = string.rstrip(", ")
        string += "|"
        for axis in self.signal_axes:
            string += str(axis.size) + ", "
        string = string.rstrip(", ")
        if self.ragged:
            string += 'ragged'
        string += ")"
        return string

    def __repr__(self):
        text = ('<Axes manager, axes: %s>\n' %
                self._get_dimension_str())
        ax_signature_uniform = "% 16s | %6g | %6s | %7.2g | %7.2g | %6s "
        ax_signature_non_uniform = "% 16s | %6g | %6s | non-uniform axis | %6s "
        signature = "% 16s | %6s | %6s | %7s | %7s | %6s "
        text += signature % ('Name', 'size', 'index', 'offset', 'scale',
                             'units')
        text += '\n'
        text += signature % ('=' * 16, '=' * 6, '=' * 6,
                             '=' * 7, '=' * 7, '=' * 6)

        def axis_repr(ax, ax_signature_uniform, ax_signature_non_uniform):
            if ax.is_uniform:
                return ax_signature_uniform % (str(ax.name)[:16], ax.size,
                                              str(ax.index), ax.offset,
                                              ax.scale, ax.units)
            else:
                return ax_signature_non_uniform % (str(ax.name)[:16], ax.size,
                                                  str(ax.index), ax.units)

        for ax in self.navigation_axes:
            text += '\n'
            text += axis_repr(ax, ax_signature_uniform, ax_signature_non_uniform)
        text += '\n'
        text += signature % ('-' * 16, '-' * 6, '-' * 6,
                             '-' * 7, '-' * 7, '-' * 6)
        for ax in self.signal_axes:
            text += '\n'
            text += axis_repr(ax, ax_signature_uniform, ax_signature_non_uniform)
        if self.ragged:
            text += '\n'
            text += "     Ragged axis |               Variable length"

        return text

    def _repr_html_(self):
        text = ("<style>\n"
                "table, th, td {\n\t"
                "border: 1px solid black;\n\t"
                "border-collapse: collapse;\n}"
                "\nth, td {\n\t"
                "padding: 5px;\n}"
                "\n</style>")
        text += ('\n<p><b>< Axes manager, axes: %s ></b></p>\n' %
                 self._get_dimension_str())

        def format_row(*args, tag='td', bold=False):
            if bold:
                signature = "\n<tr class='bolder_row'> "
            else:
                signature = "\n<tr> "
            signature += " ".join(("{}" for _ in args)) + " </tr>"
            return signature.format(*map(lambda x:
                                         '\n<' + tag +
                                         '>{}</'.format(x) + tag + '>',
                                         args))

        def axis_repr(ax):
            index = ax.index if ax.navigate else ""
            if ax.is_uniform:
                return format_row(ax.name, ax.size, index, ax.offset,
                                  ax.scale, ax.units)
            else:
                return format_row(ax.name, ax.size, index, "non-uniform axis",
                                  "non-uniform axis", ax.units)

        if self.navigation_axes:
            text += "<table style='width:100%'>\n"
            text += format_row('Navigation axis name', 'size', 'index', 'offset',
                               'scale', 'units', tag='th')
            for ax in self.navigation_axes:
                text += axis_repr(ax)
            text += "</table>\n"
        if self.signal_axes:
            text += "<table style='width:100%'>\n"
            text += format_row('Signal axis name', 'size',
                               "", 'offset', 'scale', 'units', tag='th')
            for ax in self.signal_axes:
                text += axis_repr(ax)
            text += "</table>\n"
        return text

    @property
    def coordinates(self):
        """
        Get and set the current coordinates, if the navigation dimension
        is not 0. If the navigation dimension is 0, it raises
        AttributeError when attempting to set its value.
        """
        return tuple([axis.value for axis in self.navigation_axes])

    @coordinates.setter
    def coordinates(self, coordinates):
        # See class docstring
        if len(coordinates) != self.navigation_dimension:
            raise AttributeError(
                "The number of coordinates must be equal to the "
                "navigation dimension that is %i" %
                self.navigation_dimension)
        changes = False
        with self.events.indices_changed.suppress():
            for value, axis in zip(coordinates, self.navigation_axes):
                changes = changes or (axis.value != value)
                axis.value = value
        # Trigger only if the indices are changed
        if changes:
            self.events.indices_changed.trigger(obj=self)

    @property
    def indices(self):
        """
        Get and set the current indices, if the navigation dimension
        is not 0. If the navigation dimension is 0, it raises
        AttributeError when attempting to set its value.
        """
        return tuple([axis.index for axis in self.navigation_axes])

    @indices.setter
    def indices(self, indices):
        # See class docstring
        if len(indices) != self.navigation_dimension:
            raise AttributeError(
                "The number of indices must be equal to the "
                "navigation dimension that is %i" %
                self.navigation_dimension)
        changes = False
        with self.events.indices_changed.suppress():
            for index, axis in zip(indices, self.navigation_axes):
                changes = changes or (axis.index != index)
                axis.index = index
        # Trigger only if the indices are changed
        if changes:
            self.events.indices_changed.trigger(obj=self)

    def _get_axis_attribute_values(self, attr):
        return [getattr(axis, attr) for axis in self._axes]

    def _set_axis_attribute_values(self, attr, values):
        """Set the given attribute of all the axes to the given
        value(s)

        Parameters
        ----------
        attr : string
            The DataAxis attribute to set.
        values : any
            If iterable, it must have the same number of items
            as axes are in this AxesManager instance. If not iterable,
            the attribute of all the axes are set to the given value.

        """
        if not isiterable(values):
            values = [values, ] * len(self._axes)
        elif len(values) != len(self._axes):
            raise ValueError("Values must have the same number"
                             "of items are axes are in this AxesManager")
        for axis, value in zip(self._axes, values):
            setattr(axis, attr, value)

    @property
    def navigation_indices_in_array(self):
        return tuple([axis.index_in_array for axis in self.navigation_axes])

    @property
    def signal_indices_in_array(self):
        return tuple([axis.index_in_array for axis in self.signal_axes])

    @property
    def axes_are_aligned_with_data(self):
        """Verify if the data axes are aligned with the signal axes.

        When the data are aligned with the axes the axes order in `self._axes`
        is [nav_n, nav_n-1, ..., nav_0, sig_m, sig_m-1 ..., sig_0].

        Returns
        -------
        aligned : bool

        """
        nav_iia_r = self.navigation_indices_in_array[::-1]
        sig_iia_r = self.signal_indices_in_array[::-1]
        iia_r = nav_iia_r + sig_iia_r
        aligned = iia_r == tuple(range(len(iia_r)))
        return aligned

    def _sort_axes(self):
        """Sort _axes to align them.

        When the data are aligned with the axes the axes order in `self._axes`
        is [nav_n, nav_n-1, ..., nav_0, sig_m, sig_m-1 ..., sig_0]. This method
        sort the axes in this way. Warning: this doesn't sort the `data` axes.

        """
        am = self
        new_axes = am.navigation_axes[::-1] + am.signal_axes[::-1]
        self._axes = list(new_axes)

    def gui_navigation_sliders(self, title="", display=True, toolkit=None):
        # With traits 6.1 and traitsui 7.0, we have this deprecation warning,
        # which is fine to filter
        # https://github.com/enthought/traitsui/issues/883
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning,
                                    message="'TraitPrefixList'",
                                    module='traitsui')
            warnings.filterwarnings("ignore", category=DeprecationWarning,
                                    message="'TraitMap'",
                                    module='traits')
            return get_gui(self=self.navigation_axes,
                           toolkey="hyperspy.navigation_sliders",
                           display=display,
                           toolkit=toolkit,
                           title=title)
    gui_navigation_sliders.__doc__ = \
        """
        Navigation sliders to control the index of the navigation axes.

        Parameters
        ----------
        title: str
        %s
        %s
        """

class GeneratorLen:
    """
    Helper class for creating a generator-like object with a known length.
    Useful when giving a generator as input to the AxesManager iterpath, so that the
    length is known for the progressbar.

    Found at: https://stackoverflow.com/questions/7460836/how-to-lengenerator/7460986

    Parameters
    ----------
    gen : generator
        The Generator containing hyperspy navigation indices.
    length : int
        The manually-specified length of the generator.
    """
    def __init__(self, gen, length):
        self.gen = gen
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.gen


def _parse_axis_attribute(value):
    """Parse axis attribute"""
    if value is t.Undefined:
        return None
    else:
        return value
