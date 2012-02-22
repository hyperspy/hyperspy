Curve fitting
*************

Hyperspy can perform curve fitting in n-dimensional data sets. It can create a model from a linear combinantion of predefined components and can use multiple optimisation algorithms to fit the model to experimental data. It supports bounds and weights.

Creating a model
----------------

A :py:class:`~.model.Model` can be created using the :py:func:`~.hspy.create_model` function, which first argument is a :py:class:`~.signal.Signal` of any of its subclasses (often it is simply the object returned by the :py:func:`~.io.load` function. e.g.,

.. code-block:: python
    
    # Load the data from a file
    s = load('YourDataFilenameHere')
    # Create the model and asign it to the variable m
    m = create_model()


Adding components to the model
------------------------------

In Hyperspy a model consists of a linear combination of :py:mod:`~.components`. To print the current components in a model simply write the name of the variable a press ``Enter``, e.g.:

.. code-block:: python
    
    # m is the variable in which we have previously stored the model
    m
    >>> []
    
    # [] means that the model is empty
    

To add a component first we have to create an instance of the component. Once the instance has been created we can add the component to the model using the :py:meth:`append` method, e.g.:
    

.. code-block:: python
    
    # Create a Gaussian function component
    gaussian = components.Gaussian()
    # Add it to the model_cube
    m.append(gaussian)
    # Let's print the components
    m
    >>> [Normalized Gaussian]
    # Create two Lorentzian function components
    lorentzian1 = components.Lorentzian()
    lorentzian2 = components.Lorentzian()
    # We could use the append method two times to add the
    # two lorentzians, but when adding multiple components it is handier to used
    # the extend method
    m.extend((lorentzian1, lorentzian2))
    # Let's print the components    
    >>> [Normalized Gaussian, Lorentzian, Lorentzian]
    
