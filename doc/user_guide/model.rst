Curve fitting
*************

Hyperspy can perform curve fitting in n-dimensional data sets. It can create a model from a linear combinantion of predefined components and can use multiple optimisation algorithms to fit the model to experimental data. It supports bounds and weights.

Creating a model
----------------

A :py:class:`~.model.Model` can be created using the :py:func:`~.hspy.create_model` function, which first argument is a :py:class:`~.signal.Signal` of any of its subclasses (often it is simply the object returned by the :py:func:`~.io.load` function. e.g.,

.. code-block:: ipython
    
    In [1]: # Load the data from a file
    In [2]: s = load('YourDataFilenameHere')
    In [3]: #Create the model and asign it to the variable m
    In [4]: m = create_model()


Adding components to the model
------------------------------

In Hyperspy a model consists of a linear combination of :py:mod:`~.components`. These are some of the components which are currently available:


* :py:class:`~.components.eels_cl_edge.EELSCLEdge`
* :py:class:`~.components.power_law.PowerLaw`
* :py:class:`~.components.offset.Offset`
* :py:class:`~.components.exponential.Exponential`
* :py:class:`~.components.scalable_fixed_pattern.ScalableFixedPattern`
* :py:class:`~.components.gaussian.Gaussian`
* :py:class:`~.components.lorentzian.Lorentzian`
* :py:class:`~.components.voigt.Voigt`
* :py:class:`~.components.polynomial.Polynomial`
* :py:class:`~.components.logistic.Logistic`
* :py:class:`~.components.bleasdale.Bleasdale`
* :py:class:`~.components.error_function.Erf`
* :py:class:`~.components.pes_see.SEE`

 
Writing a new component is very easy, so, if the function that you need to fit is not in the list above, by inspecting the code of, for example, the Gaussian component, it should be easy to write your own component. If you need help for the task please submit your question to the :ref:`users mailing list <http://groups.google.com/group/hyperspy-users>`.


To print the current components in a model simply write the name of the variable a press ``Enter``, e.g.:

.. code-block:: ipython
    
    In [5]: # m is the variable in which we have previously stored the model
    In [6]: m
    Out[1]: []
    In [7]: # [] means that the model is empty
    

To add a component first we have to create an instance of the component. Once the instance has been created we can add the component to the model using the :py:meth:`append` method, e.g.:
    

.. code-block:: ipython
    
    In [8]: # Create a Gaussian function component
    In [9]: gaussian = components.Gaussian()
    In [10]: # Add it to the model_cube
    In [11]: m.append(gaussian)
    In [12]: # Let's print the components
    In [13]: m
    Out[2]: [Normalized Gaussian]
    In [14]: # Create two Lorentzian function components
    In [15]:  lorentzian1 = components.Lorentzian()
    In [16]: lorentzian2 = components.Lorentzian()
    In [17]: # We could use the append method two times to add the
    In [18]: # two lorentzians, but when adding multiple components it is handier to used
    In [19]: # the extend method
    In [20]: m.extend((lorentzian1, lorentzian2))
    In [21]: # Let's print the components    
    Out[2]: [Normalized Gaussian, Lorentzian, Lorentzian]
    
    
Fitting the model to the data
-----------------------------

To fit the model to the data at the current coordinates use :py:meth:`~.optimizers.Optimizers.fit`. To fit the model to the data in all the coordinates use :py:meth:`~.model.Model.multifit` and to visualise the result :py:meth:`~.model.Model.plot`, e.g.:

.. code-block:: ipython
    
    In [22]: # Let's fit the data at the current coordinates
    In [23]: m.fit()
    In [24]: # And now let's visualise the results
    In [25]: m.plot()
    In [26]: # Because we like what we see, we will fit the model to the
    In [28]: # data in all the coordinates
    In [29]: m.multifit()
    
Getting and setting the component parameters
--------------------------------------------


Exclude data from the fitting process
-------------------------------------




