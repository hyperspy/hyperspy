.. _pint_unit_registry:

Unit Handling with Pint Quantity
********************************

HyperSpy uses the `pint <https://pint.readthedocs.io>`_ library to handle unit conversion.
To be interoperatable with other modules, HyperSpy uses the default pint :class:`pint.UnitRegistry` 
provided by the :func:`pint.get_application_registry` function as described in the sections
`having a shared registry <https://pint.readthedocs.io/en/stable/getting/pint-in-your-projects.html>`_
and `serialization <https://pint.readthedocs.io/en/stable/advanced/serialization.html>`_
of the pint user guide.

For example, in the case of the  ``scale_as_quantify``  pint quantity object from :class:`~.axes.UniformDataAxis`,
the default pint :class:`pint.UnitRegistry` is used:

.. code-block:: python

    >>> s = hs.signals.Signal1D(np.arange(10))
    >>> s.axes_manager[0].scale_as_quantity
    <Quantity(1.0, 'dimensionless')>
    >>> s.axes_manager[0].scale_as_quantity = '2.5 µm'
    >>> s.axes_manager[0].scale_as_quantity
    <Quantity(2.5, 'micrometer')>

Then, using :func:`pint.get_application_registry` get the handle of the same instance of :class:`pint.UnitRegistry`
used by HyperSpy and use it to operate on this pint quantity:

    >>> import pint
    >>> ureg = pint.get_application_registry()
    >>> scale = 2E-6 * ureg.meter
    >>> s.axes_manager[0].scale_as_quantity += scale
    >>> s.axes_manager[0].scale_as_quantity
    <Quantity(4.5, 'micrometer')>
