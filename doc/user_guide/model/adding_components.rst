.. _adding-components:

Adding components to the model
------------------------------

To print the current components in a model use
:attr:`~.model.BaseModel.components`. A table with component number,
attribute name, component name and component type will be printed:

.. code-block:: python

    >>> s = hs.signals.Signal1D(np.arange(100))
    >>> m = s.create_model()
    >>> m
    <Model1D>
    >>> m.components
       # |      Attribute Name |      Component Name |      Component Type
    ---- | ------------------- | ------------------- | -------------------


.. note:: Sometimes components may be created automatically. For example, if
   the :class:`~._signals.signal1d.Signal1D` is recognised as EELS data, a
   power-law background component may automatically be added to the model.
   Therefore, the table above may not all may empty on model creation.

To add a component to the model, first we have to create an instance of the
component.
Once the instance has been created we can add the component to the model
using the :meth:`~.model.BaseModel.append` and
:meth:`~.model.BaseModel.extend` methods for one or more components
respectively.

As an example, let's add several :class:`~._components.gaussian.Gaussian`
components to the model:

.. code-block:: python

    >>> gaussian = hs.model.components1D.Gaussian() # Create a Gaussian comp.
    >>> m.append(gaussian) # Add it to the model
    >>> m.components # Print the model components
       # |      Attribute Name |      Component Name |      Component Type
    ---- | ------------------- | ------------------- | -------------------
       0 |            Gaussian |            Gaussian |            Gaussian
    >>> gaussian2 = hs.model.components1D.Gaussian() # Create another gaussian
    >>> gaussian3 = hs.model.components1D.Gaussian() # Create a third gaussian


We could use the :meth:`~.model.BaseModel.append` method twice to add the
two Gaussians, but when adding multiple components it is handier to use the
extend method that enables adding a list of components at once.


.. code-block:: python

    >>> m.extend((gaussian2, gaussian3)) # note the double parentheses!
    >>> m.components
       # |      Attribute Name |      Component Name |      Component Type
    ---- | ------------------- | ------------------- | -------------------
       0 |            Gaussian |            Gaussian |            Gaussian
       1 |          Gaussian_0 |          Gaussian_0 |            Gaussian
       2 |          Gaussian_1 |          Gaussian_1 |            Gaussian

We can customise the name of the components.

.. code-block:: python

    >>> gaussian.name = 'Carbon'
    >>> gaussian2.name = 'Long Hydrogen name'
    >>> gaussian3.name = 'Nitrogen'
    >>> m.components
       # |      Attribute Name |      Component Name |      Component Type
    ---- | ------------------- | ------------------- | -------------------
       0 |              Carbon |              Carbon |            Gaussian
       1 |  Long_Hydrogen_name |  Long Hydrogen name |            Gaussian
       2 |            Nitrogen |            Nitrogen |            Gaussian

Notice that two components cannot have the same name:

.. code-block:: python

    >>> gaussian2.name = 'Carbon'
    Traceback (most recent call last):
      File "<ipython-input-5-2b5669fae54a>", line 1, in <module>
        g2.name = "Carbon"
      File "/home/fjd29/Python/hyperspy/hyperspy/component.py", line 466, in
        name "the name " + str(value))
    ValueError: Another component already has the name Carbon


It is possible to access the components in the model by their name or by the
index in the model.

.. code-block:: python

    >>> m.components
       # |      Attribute Name |      Component Name |      Component Type
    ---- | ------------------- | ------------------- | -------------------
       0 |              Carbon |              Carbon |            Gaussian
       1 |  Long_Hydrogen_name |  Long Hydrogen name |            Gaussian
       2 |            Nitrogen |            Nitrogen |            Gaussian
    >>> m[0]
    <Carbon (Gaussian component)>
    >>> m["Long Hydrogen name"]
    <Long Hydrogen name (Gaussian component)>


In addition, the components can be accessed in the
:attr:`~.model.BaseModel.components` `Model` attribute. This is specially
useful when working in interactive data analysis with IPython because it
enables tab completion.

.. code-block:: python

    >>> m.components
       # |      Attribute Name |      Component Name |      Component Type
    ---- | ------------------- | ------------------- | -------------------
       0 |              Carbon |              Carbon |            Gaussian
       1 |  Long_Hydrogen_name |  Long Hydrogen name |            Gaussian
       2 |            Nitrogen |            Nitrogen |            Gaussian
    >>> m.components.Long_Hydrogen_name
    <Long Hydrogen name (Gaussian component)>


It is possible to "switch off" a component by setting its
``active`` attribute to ``False``. When a component is
switched off, to all effects it is as if it was not part of the model. To
switch it back on simply set the ``active`` attribute back to ``True``.

In multi-dimensional signals it is possible to store the value of the
``active`` attribute at each navigation index.
To enable this feature for a given component set the
:attr:`~.component.Component.active_is_multidimensional` attribute to
`True`.

.. code-block:: python

    >>> s = hs.signals.Signal1D(np.arange(100).reshape(10,10))
    >>> m = s.create_model()
    >>> g1 = hs.model.components1D.Gaussian()
    >>> g2 = hs.model.components1D.Gaussian()
    >>> m.extend([g1,g2])
    >>> g1.active_is_multidimensional = True
    >>> m.set_component_active_value(False)
    >>> m.set_component_active_value(True, only_current=True)
