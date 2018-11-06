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

from functools import wraps
import numpy as np

from hyperspy.component import Component
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
    import sympy
    splits = map(str.strip, string.split(';'))
    expr = sympy.sympify(next(splits))
    # We substitute one by one manually, as passing all at the same time does
    # not work as we want (subsitutions inside other substitutions do not work)
    for sub in splits:
        t = tuple(map(str.strip, sub.split('=')))
        expr = expr.subs(t[0], sympy.sympify(t[1]))
    return expr


class Expression(Component):

    """Create a component from a string expression.
    """

    def __init__(self, expression, name, position=None, module="numpy",
                 autodoc=True, add_rotation=False, rotation_center=None,
                 **kwargs):
        """Create a component from a string expression.

        It automatically generates the partial derivatives and the
        class docstring.

        Parameters
        ----------
        expression: str
            Component function in SymPy text expression format with
            substitutions separated by `;`. See examples and the SymPy
            documentation for details. The only additional constraint is that
            the variable(s) must be `x` (for 1D components); or `x` and `y` for
            2D components. Also, if `module` is "numexpr" the
            functions are limited to those that numexpr support. See its
            documentation for details.
        name : str
            Name of the component.
        position: str, optional
            The parameter name that defines the position of the component if
            applicable. It enables interative adjustment of the position of the
            component in the model. For 2D components, a tuple must be passed
            with the name of the two parameters e.g. `("x0", "y0")`.
        module: {"numpy", "numexpr"}, default "numpy"
            Module used to evaluate the function. numexpr is often faster but
            it supports fewer functions and requires installing numexpr.
        add_rotation: bool, default False
            This is only relevant for 2D components. If `True` it automatically
            adds `rotation_angle` parameter.
        rotation_center: {None, tuple}
            If None, the rotation center is the center i.e. (0, 0) if `position`
            is not defined, otherwise the center is the coordinates specified
            by `position`. Alternatively a tuple with the (x, y) coordinates
            of the center can be provided.
        **kwargs
             Keyword arguments can be used to initialise the value of the
             parameters.

        Methods
        -------
        recompile: useful to recompile the function and gradient with a
            a different module.

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

        import sympy
        self._add_rotation = add_rotation
        self._str_expression = expression
        if rotation_center is None:
            self.compile_function(module=module, position=position)
        else:
            self.compile_function(module=module, position=rotation_center)
        # Initialise component
        Component.__init__(self, self._parameter_strings)
        # When creating components using Expression (for example GaussianHF)
        # we shouldn't add anything else to the _whitelist as the
        # component should be initizialized with its own kwargs.
        # An exception is "module"
        self._whitelist['module'] = ('init', module)
        if self.__class__ is Expression:
            self._whitelist['expression'] = ('init', expression)
            self._whitelist['name'] = ('init', name)
            self._whitelist['position'] = ('init', position)
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
                name, sympy.latex(_parse_substitutions(expression)))

    def compile_function(self, module="numpy", position=False):
        import sympy
        from sympy.utilities.lambdify import lambdify
        expr = _parse_substitutions(self._str_expression)
        # Extract x
        x, = [symbol for symbol in expr.free_symbols if symbol.name == "x"]
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
        self._f = lambdify(variables + parameters, eval_expr,
                           modules=module, dummify=False)

        if self._is2D:
            def f(x, y): return self._f(
                x, y, *[p.value for p in self.parameters])
        else:
            def f(x): return self._f(x, *[p.value for p in self.parameters])
        setattr(self, "function", f)
        parnames = [symbol.name for symbol in parameters]
        self._parameter_strings = parnames
        ffargs = _fill_function_args_2d if self._is2D else _fill_function_args
        for parameter in parameters:
            grad_expr = sympy.diff(eval_expr, parameter)
            setattr(self,
                    "_f_grad_%s" % parameter.name,
                    lambdify(variables + parameters,
                             grad_expr.evalf(),
                             modules=module,
                             dummify=False)
                    )

            setattr(self,
                    "grad_%s" % parameter.name,
                    ffargs(
                        getattr(
                            self,
                            "_f_grad_%s" %
                            parameter.name)).__get__(
                        self,
                        Expression)
                    )

    def _is_navigation_multidimensional(self):
        if (self._axes_manager is None or not
                self._axes_manager.navigation_dimension):
            return False
        else:
            return True

    def function_nd(self, *args):
        """%s

        """
        if self._is2D:
            x, y = args[0], args[1]
            # navigation dimension is 0, f_nd same as f
            if not self._is_navigation_multidimensional():
                return self.function(x, y)
            else:
                return self._f(x[np.newaxis, ...], y[np.newaxis, ...],
                               *[p.map['values'][..., np.newaxis, np.newaxis]
                                 for p in self.parameters])
        else:
            x = args[0]
            if not self._is_navigation_multidimensional():
                return self.function(x)
            else:
                return self._f(x[np.newaxis, ...],
                               *[p.map['values'][..., np.newaxis]
                                 for p in self.parameters])
    function_nd.__doc__ %= FUNCTION_ND_DOCSTRING
