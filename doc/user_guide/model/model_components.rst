.. _model-components:

Model components
----------------

In HyperSpy a model consists of a sum of individual components. For convenience,
HyperSpy provides a number of pre-defined model components as well as mechanisms
to create your own components.

.. _model_components-label:

Pre-defined model components
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Various components are available in one (:py:mod:`~.api.model.components1D`) and
two-dimensions (:py:mod:`~.api.model.components2D`) to construct a
model.

The following general components are currently available for one-dimensional models:

* :py:class:`~.api.model.components1D.Arctan`
* :py:class:`~.api.model.components1D.Bleasdale`
* :py:class:`~.api.model.components1D.Doniach`
* :py:class:`~.api.model.components1D.Erf`
* :py:class:`~.api.model.components1D.Exponential`
* :py:class:`~.api.model.components1D.Expression`
* :py:class:`~.api.model.components1D.Gaussian`
* :py:class:`~.api.model.components1D.GaussianHF`
* :py:class:`~.api.model.components1D.HeavisideStep`
* :py:class:`~.api.model.components1D.Logistic`
* :py:class:`~.api.model.components1D.Lorentzian`
* :py:class:`~.api.model.components1D.Offset`
* :py:class:`~.api.model.components1D.Polynomial`
* :py:class:`~.api.model.components1D.PowerLaw`
* :py:class:`~.api.model.components1D.SEE`
* :py:class:`~.api.model.components1D.ScalableFixedPattern`
* :py:class:`~.api.model.components1D.SkewNormal`
* :py:class:`~.api.model.components1D.Voigt`
* :py:class:`~.api.model.components1D.SplitVoigt`
* :py:class:`~.api.model.components1D.VolumePlasmonDrude`

The following components developed with specific signal types in mind are
currently available for one-dimensional models:

* :py:class:`~.api.model.components1D.EELSArctan`
* :py:class:`~.api.model.components1D.DoublePowerLaw`
* :py:class:`~.api.model.components1D.EELSCLEdge`
* :py:class:`~.api.model.components1D.PESCoreLineShape`
* :py:class:`~.api.model.components1D.PESVoigt`
* :py:class:`~.api.model.components1D.SEE`
* :py:class:`~.api.model.components1D.Vignetting`

The following components are currently available for two-dimensional models:

* :py:class:`~.api.model.components1D.Expression`
* :py:class:`~.api.model.components2D.Gaussian2D`

However, this doesn't mean that you have to limit yourself to this meagre
list of functions. As discussed below, it is very easy to turn a
mathematical, fixed-pattern or Python function into a component.

.. _expression_component-label:

Define components from a mathematical expression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


The easiest way to turn a mathematical expression into a component is using the
:py:class:`~._components.expression.Expression` component. For example, the
following is all you need to create a
:py:class:`~._components.gaussian.Gaussian` component  with more sensible
parameters for spectroscopy than the one that ships with HyperSpy:

.. code-block:: python

    >>> g = hs.model.components1D.Expression(
    ... expression="height * exp(-(x - x0) ** 2 * 4 * log(2)/ fwhm ** 2)",
    ... name="Gaussian",
    ... position="x0",
    ... height=1,
    ... fwhm=1,
    ... x0=0,
    ... module="numpy")

If the expression is inconvenient to write out in full (e.g. it's long and/or
complicated), multiple substitutions can be given, separated by semicolons.
Both symbolic and numerical substitutions are allowed:

.. code-block:: python

    >>> expression = "h / sqrt(p2) ; p2 = 2 * m0 * e1 * x * brackets;"
    >>> expression += "brackets = 1 + (e1 * x) / (2 * m0 * c * c) ;"
    >>> expression += "m0 = 9.1e-31 ; c = 3e8; e1 = 1.6e-19 ; h = 6.6e-34"
    >>> wavelength = hs.model.components1D.Expression(
    ... expression=expression,
    ... name="Electron wavelength with voltage")

:py:class:`~._components.expression.Expression` uses `Sympy
<https://www.sympy.org>`_ internally to turn the string into
a function. By default it "translates" the expression using
numpy, but often it is possible to boost performance by using
`numexpr <https://github.com/pydata/numexpr>`_ instead.

It can also create 2D components with optional rotation. In the following
example we create a 2D Gaussian that rotates around its center:

.. code-block:: python

    >>> g = hs.model.components2D.Expression(
    ... "k * exp(-((x-x0)**2 / (2 * sx ** 2) + (y-y0)**2 / (2 * sy ** 2)))",
    ... "Gaussian2d", add_rotation=True, position=("x0", "y0"),
    ... module="numpy", )

Define new components from a Python function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Of course :py:class:`~._components.expression.Expression` is only useful for
analytical functions. You can define more general components modifying the
following template to suit your needs:


.. code-block:: python

    from hyperspy.component import Component

    class MyComponent(Component):

        """
        """

        def __init__(self, parameter_1=1, parameter_2=2):
            # Define the parameters
            Component.__init__(self, ('parameter_1', 'parameter_2'))

            # Optionally we can set the initial values
            self.parameter_1.value = parameter_1
            self.parameter_2.value = parameter_2

            # The units (optional)
            self.parameter_1.units = 'Tesla'
            self.parameter_2.units = 'Kociak'

            # Once defined we can give default values to the attribute
            # For example we fix the attribure_1 (optional)
            self.parameter_1.attribute_1.free = False

            # And we set the boundaries (optional)
            self.parameter_1.bmin = 0.
            self.parameter_1.bmax = None

            # Optionally, to boost the optimization speed we can also define
            # the gradients of the function we the syntax:
            # self.parameter.grad = function
            self.parameter_1.grad = self.grad_parameter_1
            self.parameter_2.grad = self.grad_parameter_2

        # Define the function as a function of the already defined parameters,
        # x being the independent variable value
        def function(self, x):
            p1 = self.parameter_1.value
            p2 = self.parameter_2.value
            return p1 + x * p2

        # Optionally define the gradients of each parameter
        def grad_parameter_1(self, x):
            """
            Returns d(function)/d(parameter_1)
            """
            return 0

        def grad_parameter_2(self, x):
            """
            Returns d(function)/d(parameter_2)
            """
            return x

Define components from a fixed-pattern
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:class:`~._components.scalable_fixed_pattern.ScalableFixedPattern`
component enables fitting a pattern (in the form of a
:py:class:`~._signals.signal1d.Signal1D` instance) to data by shifting
(:py:attr:`~._components.scalable_fixed_pattern.ScalableFixedPattern.shift`)
and
scaling it in the x and y directions using the
:py:attr:`~._components.scalable_fixed_pattern.ScalableFixedPattern.xscale`
and
:py:attr:`~._components.scalable_fixed_pattern.ScalableFixedPattern.yscale`
parameters respectively.
