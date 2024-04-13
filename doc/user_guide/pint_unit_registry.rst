.. _pint_unit_registry:

Operation with Pint Quantity
****************************

HyperSpy uses the `pint <https://pint.readthedocs.io>`_ library to handle unit conversion.
To be interoperatable with other modules, hyperspy uses the default pint :class:`pint.UnitRegistry` 
provided by the :func:`pint.get_application_registry` function as described in the sections
`having a shared registry <https://pint.readthedocs.io/en/stable/getting/pint-in-your-projects.html>`_
and the `serialization <https://pint.readthedocs.io/en/stable/advanced/serialization.html>`_
of the pint user guide.

For example, to use pint quantity object from :class:`~.axes.UniformDataAxis`, the same
unit registry needs to be used:

.. code-block:: python

    >>> s = hs.signals.Signal1D(np.arange(10))
    >>> s.axes_manager[0].scale_as_quantity
    <Quantity(1.0, 'dimensionless')>
    >>> s.axes_manager[0].scale_as_quantity = '2.5 µm'
    >>> s.axes_manager[0].scale_as_quantity
    <Quantity(2.5, 'micrometer')>

Use :func:`pint.get_application_registry` to get pint default unit registry:

    >>> import pint
    >>> ureg = pint.get_application_registry()
    >>> scale = 2E-6 * ureg.meter
    >>> s.axes_manager[0].scale_as_quantity += scale
    >>> s.axes_manager[0].scale_as_quantity
    <Quantity(4.5, 'micrometer')>
