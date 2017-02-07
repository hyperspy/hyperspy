from functools import wraps
from hyperspy.component import Component

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


def _parse_substitutions(string, simultaneous=True):
    import sympy
    splits = map(str.strip, string.split(';'))
    expr = sympy.sympify(next(splits))
    # We substitute one by one manually, as passing all at the same time does
    # not work as we want (subsitutions inside other substitutions do not work)
    for sub in splits:
        expr = expr.subs(*tuple(map(str.strip, sub.split('='))))
    return expr

class Expression(Component):

    """Create a component from a string expression.
    """

    def __init__(self, expression, name, position=None, module="numpy",
                 autodoc=True, **kwargs):
        """Create a component from a string expression.

        It automatically generates the partial derivatives and the
        class docstring.

        Parameters
        ----------
        expression: str
            Component function in SymPy text expression format with
            substitutions separated by `;`. See examples and the SymPy
            documentation for details. The only additional constraint is that
            the variable must be `x`. Also, in `module` is "numexpr" the
            functions are limited to those that numexpr support. See its
            documentation for details.
        name : str
            Name of the component.
        position: str, optional
            The parameter name that defines the position of the component if
            applicable. It enables adjusting the position of the component
            interactively in a model.
        module: {"numpy", "numexpr"}, default "numpy"
            Module used to evaluate the function. numexpr is often faster but
            it supports less functions.

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
        self._str_expression = expression
        self.compile_function(module=module)
        # Initialise component
        Component.__init__(self, self._parameter_strings)
        self._whitelist['expression'] = ('init', expression)
        self._whitelist['name'] = ('init', name)
        self._whitelist['position'] = ('init', position)
        self._whitelist['module'] = ('init', module)
        self.name = name
        # Set the position parameter
        if position:
            self._position = getattr(self, position)
        # Set the initial value of the parameters
        if kwargs:
            for kwarg, value in kwargs.items():
                setattr(getattr(self, kwarg), 'value', value)

        if autodoc:
            self.__doc__ = _CLASS_DOC % (
                name, sympy.latex(_parse_substitutions(expression)))

    def function(self, x):
        return self._f(x, *[p.value for p in self.parameters])

    def compile_function(self, module="numpy"):
        import sympy
        from sympy.utilities.lambdify import lambdify
        expr = _parse_substitutions(self._str_expression)

        rvars = sympy.symbols([s.name for s in expr.free_symbols], real=True)
        real_expr = expr.subs(
            {orig: real_ for (orig, real_) in zip(expr.free_symbols, rvars)})
        # just replace with the assumption that all our variables are real
        expr = real_expr

        eval_expr = expr.evalf()
        # Extract parameters
        parameters = [
            symbol for symbol in expr.free_symbols if symbol.name != "x"]
        parameters.sort(key=lambda x: x.name)  # to have a reliable order
        # Extract x
        x, = [symbol for symbol in expr.free_symbols if symbol.name == "x"]
        # Create compiled function
        self._f = lambdify([x] + parameters, eval_expr,
                           modules=module, dummify=False)
        parnames = [symbol.name for symbol in parameters]
        self._parameter_strings = parnames
        for parameter in parameters:
            grad_expr = sympy.diff(eval_expr, parameter)
            setattr(self,
                    "_f_grad_%s" % parameter.name,
                    lambdify([x] + parameters,
                             grad_expr.evalf(),
                             modules=module,
                             dummify=False)
                    )

            setattr(self,
                    "grad_%s" % parameter.name,
                    _fill_function_args(
                        getattr(
                            self,
                            "_f_grad_%s" %
                            parameter.name)).__get__(
                        self,
                        Expression)
                    )
