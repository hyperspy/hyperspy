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

from functools import wraps
import numpy as np
import sympy

import warnings

from hyperspy.component import Component, convolve_component_values
from hyperspy.docstrings.parameters import FUNCTION_ND_DOCSTRING


_CLASS_DOC = \
    """%s component (created with Expression).

.. math::

    f(x) = %s

"""


def _fill_function_args(fn):
    @wraps(fn)
    def fn_wrapped(self, x):
        return fn(x, *[p.value for p in self.parameters])

    return fn_wrapped


def _fill_function_args_2d(fn):
    @wraps(fn)
    def fn_wrapped(self, x, y):
        return fn(x, y, *[p.value for p in self.parameters])

    return fn_wrapped


def _parse_substitutions(string):
    splits = map(str.strip, string.split(';'))
    expr = sympy.sympify(next(splits))
    # We substitute one by one manually, as passing all at the same time does
    # not work as we want (substitutions inside other substitutions do not work)
    for sub in splits:
        t = tuple(map(str.strip, sub.split('=')))
        expr = expr.subs(t[0], sympy.sympify(t[1]))
    return expr


class Expression(Component):

    """Create a component from a string expression.

    It automatically generates the partial derivatives and the
    class docstring.

    Parameters
    ----------
    expression : str
        Component function in SymPy text expression format with
        substitutions separated by `;`. See examples and the SymPy
        documentation for details. In order to vary the components along the
        signal dimensions, the variables `x` and `y` must be included for 1D
        or 2D components. Also, if `module` is "numexpr" the
        functions are limited to those that numexpr support. See its
        documentation for details.
    name : str
        Name of the component.
    position : str, optional
        The parameter name that defines the position of the component if
        applicable. It enables interative adjustment of the position of the
        component in the model. For 2D components, a tuple must be passed
        with the name of the two parameters e.g. `("x0", "y0")`.
    module : {"numpy", "numexpr", "scipy"}, default "numpy"
        Module used to evaluate the function. numexpr is often faster but
        it supports fewer functions and requires installing numexpr.
    add_rotation : bool, default False
        This is only relevant for 2D components. If `True` it automatically
        adds `rotation_angle` parameter.
    rotation_center : {None, tuple}
        If None, the rotation center is the center i.e. (0, 0) if `position`
        is not defined, otherwise the center is the coordinates specified
        by `position`. Alternatively a tuple with the (x, y) coordinates
        of the center can be provided.
    rename_pars : dictionary
        The desired name of a parameter may sometimes coincide with e.g.
        the name of a scientific function, what prevents using it in the
        `expression`. `rename_parameters` is a dictionary to map the name
        of the parameter in the `expression`` to the desired name of the
        parameter in the `Component`. For example: {"_gamma": "gamma"}.
    compute_gradients : bool, optional
        If `True`, compute the gradient automatically using sympy. If sympy
        does not support the calculation of the partial derivatives, for
        example in case of expression containing a "where" condition,
        it can be disabled by using `compute_gradients=False`.
    linear_parameter_list : list
        A list of the components parameters that are known to be linear
        parameters.
    check_parameter_linearity : bool
        If `True`, automatically check if each parameter is linear and set
        its corresponding attribute accordingly. If `False`, the default is to 
        set all parameters, except for those who are specified in
        ``linear_parameter_list``.
        
    **kwargs
        Keyword arguments can be used to initialise the value of the
        parameters.

    Note
    ----
    As of version 1.4, Sympy's lambdify function, that the
    :py:class:`~._components.expression.Expression`
    components uses internally, does not support the differentiation of
    some expressions, for example those containing a "where" condition.
    In such cases, the gradients can be set manually if required.

    Examples
    --------
    The following creates a Gaussian component and set the initial value
    of the parameters:

    >>> hs.model.components1D.Expression(
    ... expression="height * exp(-(x - x0) ** 2 * 4 * log(2)/ fwhm ** 2)",
    ... name="Gaussian",
    ... height=1,
    ... fwhm=1,
    ... x0=0,
    ... position="x0",)

    Substitutions for long or complicated expressions are separated by
    semicolumns:

    >>> expr = 'A*B/(A+B) ; A = sin(x)+one; B = cos(y) - two; y = tan(x)'
    >>> comp = hs.model.components1D.Expression(
    ... expression=expr,
    ... name='my function')
    >>> comp.parameters
    (<Parameter one of my function component>,
     <Parameter two of my function component>)

    """

    def __init__(self, expression, name, position=None, module="numpy",
                 autodoc=True, add_rotation=False, rotation_center=None,
                 rename_pars={}, compute_gradients=True,
                 linear_parameter_list=None, check_parameter_linearity=True,
                 **kwargs):

        if linear_parameter_list is None:
            linear_parameter_list = []
        self._add_rotation = add_rotation
        self._str_expression = expression
        self._module = module
        self._rename_pars = rename_pars if rename_pars is not None else {}
        # Since the expression string uses the parameter name before renaming
        # it is useful to have the inverse mapping
        self._rename_pars_inv = {v: k for k, v in self._rename_pars.items()}
        self._compute_gradients = compute_gradients

        if rotation_center is None:
            self.compile_function(module=module, position=position)
        else:
            self.compile_function(module=module, position=rotation_center)

        # Initialise component
        Component.__init__(
            self, self._parameter_strings, linear_parameter_list
            )
        # When creating components using Expression (for example GaussianHF)
        # we shouldn't add anything else to the _whitelist as the
        # component should be initizialized with its own kwargs.
        # An exception is "module"
        self._whitelist['module'] = ('init', module)
        if self.__class__ is Expression:
            self._whitelist['expression'] = ('init', expression)
            self._whitelist['name'] = ('init', name)
            self._whitelist['position'] = ('init', position)
            self._whitelist['rename_pars'] = ('init', rename_pars)
            self._whitelist['linear_parameter_list'] = ('init', linear_parameter_list)
            self._whitelist['compute_gradients'] = ('init', compute_gradients)
            if self._is2D:
                self._whitelist['add_rotation'] = ('init', self._add_rotation)
                self._whitelist['rotation_center'] = ('init', rotation_center)
        self.name = name

        # Set the position parameter
        if position:
            if self._is2D:
                self._position_x = getattr(self, position[0])
                self._position_y = getattr(self, position[1])
            else:
                self._position = getattr(self, position)

        # Set the initial value of the parameters
        if kwargs:
            for kwarg, value in kwargs.items():
                setattr(getattr(self, kwarg), 'value', value)

        if autodoc:
            self.__doc__ = _CLASS_DOC % (
                name, sympy.latex(self._parsed_expr))

        for parameter_name in linear_parameter_list:
            setattr(getattr(self, parameter_name), '_linear', True)

        if check_parameter_linearity:
            for p in self.parameters:
                if p.name not in linear_parameter_list:
                    # _parsed_expr used "non public" parameter name and we
                    # need to use the correct parameter name by using
                    # _rename_pars_inv 
                    p._linear = _check_parameter_linearity(
                        self._parsed_expr,
                        self._rename_pars_inv.get(p.name, p.name)
                        )

    def compile_function(self, module="numpy", position=False):
        """
        Compile the function and calculate the gradient automatically when
        possible.
        Useful to recompile the function and gradient with a different module.
        """
        try:  # Expression is just a constant
            float(self._str_expression)
        except ValueError:
            pass
        else:
            raise ValueError('Expression must contain a symbol, i.e. x, a, '
                             'etc.')
        expr = _parse_substitutions(self._str_expression)
        self._parsed_expr = expr

        # Extract x
        x = [symbol for symbol in expr.free_symbols if symbol.name == "x"]
        if not x: # Expression is just a parameter, no x -> Offset
            # lambdify doesn't support constant
            # https://github.com/sympy/sympy/issues/5642
            # x = [sympy.Symbol('x')]
            raise ValueError('Expression must contain the "x" symbol.')
        x = x[0]
        # Extract y
        y = [symbol for symbol in expr.free_symbols if symbol.name == "y"]

        self._is2D = True if y else False
        if self._is2D:
            y = y[0]
        if self._is2D and self._add_rotation:
            position = position or (0, 0)
            rotx = sympy.sympify(
                "{0} + (x - {0}) * cos(rotation_angle) - (y - {1}) *"
                " sin(rotation_angle)"
                .format(*position))
            roty = sympy.sympify(
                "{1} + (x - {0}) * sin(rotation_angle) + (y - {1}) *"
                "cos(rotation_angle)"
                .format(*position))
            expr = expr.subs({"x": rotx, "y": roty}, simultaneous=False)
        rvars = sympy.symbols([s.name for s in expr.free_symbols], real=True)
        real_expr = expr.subs(
            {orig: real_ for (orig, real_) in zip(expr.free_symbols, rvars)})
        # just replace with the assumption that all our variables are real
        expr = real_expr

        eval_expr = expr.evalf()
        # Extract parameters
        variables = ("x", "y") if self._is2D else ("x", )
        parameters = [
            symbol for symbol in expr.free_symbols
            if symbol.name not in variables]
        parameters.sort(key=lambda x: x.name)  # to have a reliable order
        # Create compiled function
        variables = [x, y] if self._is2D else [x]
        self._f = sympy.utilities.lambdify(variables + parameters, eval_expr,
                           modules=module, dummify=False)

        if self._is2D:
            def f(x, y): return self._f(
                x, y, *[p.value for p in self.parameters])
        else:
            def f(x): return self._f(x, *[p.value for p in self.parameters])
        setattr(self, "function", f)
        parnames = [self._rename_pars.get(symbol.name, symbol.name)
                    for symbol in parameters]
        self._parameter_strings = parnames

        if self._compute_gradients:
            try:
                ffargs = (_fill_function_args_2d if
                          self._is2D else _fill_function_args)
                for p in parameters:
                    grad_expr = sympy.diff(eval_expr, p)
                    name = self._rename_pars.get(p.name, p.name)
                    f_grad = sympy.utilities.lambdify(variables + parameters,
                                      grad_expr.evalf(),
                                      modules=module,
                                      dummify=False)
                    grad_p = ffargs(f_grad).__get__(self, Expression)
                    if len(grad_expr.free_symbols) == 0:
                        # Vectorize in case of constant function
                        # https://github.com/sympy/sympy/issues/5642
                        grad_p = np.vectorize(grad_p)
                    setattr(self, f"grad_{name}", grad_p)

            except (SyntaxError, AttributeError):
                warnings.warn("The gradients can not be computed with sympy.",
                              UserWarning)

    def function_nd(self, *args):
        """%s

        """
        if self._is2D:
            x, y = args[0], args[1]
            # navigation dimension is 0, f_nd same as f
            if not self._is_navigation_multidimensional:
                return self.function(x, y)
            else:
                return self._f(x[np.newaxis, ...], y[np.newaxis, ...],
                               *[p.map['values'][..., np.newaxis, np.newaxis]
                                 for p in self.parameters])
        else:
            x = args[0]
            if not self._is_navigation_multidimensional:
                return self.function(x)
            else:
                return self._f(x[np.newaxis, ...],
                               *[p.map['values'][..., np.newaxis]
                                 for p in self.parameters])
    function_nd.__doc__ %= FUNCTION_ND_DOCSTRING

    @property
    def _constant_term(self):
        """
        Get value of constant term of component, assuming that the nonlinear
        term are fixed.

        The 'constant' part of a component is any part that doesn't change
        when the free parameters are changed.
        """
        free_linear_parameters = [
            # Use `_free` private attribute not to interfere with twin
            self._rename_pars_inv.get(p.name, p.name) for p in self.parameters
            if p._linear and p._free
            ]

        expr = sympy.sympify(self._str_expression)
        args = [
            sympy.sympify(arg, strict=False) for arg in free_linear_parameters
            ]
        constant_expr, _ = expr.as_independent(*args, as_Add=True)

        # Then replace symbols with value of each parameter
        free_symbols = [str(free) for free in constant_expr.free_symbols]
        for p in self.parameters:
            if p.name in free_symbols:
                name = self._rename_pars_inv.get(p.name, p.name)
                constant_expr = constant_expr.subs(name, p.value)

        return float(constant_expr.evalf())

    def _separate_pseudocomponents(self):
        """
        Separate an expression into a group of lambdified functions
        that can compute the free parts of the expression, and a single
        lambdified function that computes the fixed parts of the expression
        
        Used by the _compute_expression_part method.
        """
        expr = self._str_expression
        ex = sympy.sympify(expr)
        remaining_elements = ex.copy()
        free_pseudo_components = {}
        variables = ("x", "y") if self._is2D else ("x", )

        for para in self.free_parameters:
            name = self._rename_pars_inv.get(para.name, para.name)
            symbol = sympy.sympify(name, strict=False)
            element = ex.as_independent(symbol)[-1]
            remaining_elements -= element
            element_names = \
                set([str(p) for p in element.free_symbols]) - set(variables)
            free_pseudo_components[para.name] = {
                'function': sympy.utilities.lambdify(variables + tuple(element_names),
                                     element, modules=self._module),
                'parameters': [getattr(self, self._rename_pars.get(e, e))
                               for e in element_names]
            }

        element_names = \
            set([str(p) for p in remaining_elements.free_symbols]) - \
            set(variables)

        fixed_pseudo_components = {
            'function': sympy.utilities.lambdify(variables + tuple(element_names),
                                 remaining_elements, modules=self._module),
            'parameters': [getattr(self, self._rename_pars.get(e, e))
                           for e in element_names]
        }

        return free_pseudo_components, fixed_pseudo_components,

    def _compute_expression_part(self, part):
        """Compute the expression for a given value or map["values"]."""
        model = self.model
        function = part['function']
        parameters = [para.value for para in part['parameters']]
        
        if model.convolved and self.convolved:
            data = convolve_component_values(
                function(model.convolution_axis, *parameters), model=model)
        else:
            axes = [ax.axis for ax in model.axes_manager.signal_axes]
            mesh = np.meshgrid(*axes)
            data = function(*mesh, *parameters)
            slice_ = np.where(model.channel_switches)
            if len(np.shape(data)) == 0:
                # For calculation of constant term of the component
                signal_shape = model.axes_manager._signal_shape_in_array
                data = data * np.ones(signal_shape)[slice_]
            else:
                data = np.moveaxis(data[slice_], 0, -1)

        return data


def _check_parameter_linearity(expr, name):
    """Check whether expression is linear for a given parameter."""
    symbol = sympy.Symbol(name)
    try:
        if not sympy.diff(expr, symbol, 2) == 0:
            return False
    except AttributeError:
        # AttributeError occurs if the expression cannot be parsed
        # for instance some expressions with where.
        warnings.warn(f"The linearity of the parameter `{name}` can't be "
                      "determined automatically.", UserWarning)
        return False
    return True

