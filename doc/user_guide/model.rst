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
    In [15]: gaussian2 = components.Gaussian()
    In [16]: gaussian3 = components.Gaussian()
    In [17]: # We could use the append method two times to add the
    In [18]: # two gaussians, but when adding multiple components it is handier to used
    In [19]: # the extend method
    In [20]: m.extend((gaussian2, gaussian3))
    In [21]: # Let's print the components    
    In [22]: m
    Out[2]: [Normalized Gaussian, Normalized Gaussian, Normalized Gaussian]
    In [23]: # We can customise the name of the components
    In [24]: gaussian.name = 'Carbon'
    In [25]: gaussian2.name = 'Hydrogen'
    In [26]: gaussian3.name = 'Nitrogen'
    In [27]: # Let's print the components of the model once more
    Out[3]:
    [<Carbon (Gaussian component)>,
     <Hydrogen (Gaussian component)>,
     <Nitrogen (Gaussian component)>]
    
    
Fitting the model to the data
-----------------------------

To fit the model to the data at the current coordinates use :py:meth:`~.optimizers.Optimizers.fit`. To fit the model to the data in all the coordinates use :py:meth:`~.model.Model.multifit` and to visualise the result :py:meth:`~.model.Model.plot`, e.g.:

.. code-block:: ipython
    
    In [28]: # Let's fit the data at the current coordinates
    In [29]: m.fit()
    In [30]: # And now let's visualise the results
    In [31]: m.plot()
    In [32]: # Because we like what we see, we will fit the model to the
    In [33]: # data in all the coordinates
    In [34]: m.multifit()
    
Getting and setting parameters value and attributes
--------------------------------------------------------------------

:py:meth:`~.model.Model.print_current_values` prints the value of the parameters of the components in the current coordinates.

:py:attr:`~.component.Component.parameters` contains a list of the parameters of a component and :py:attr:`~.component.Component.free_parameters` lists only the free parameters.

The value of a particular parameter can be accessed in the :py:attr:`~.component.Parameter.value`.

To set the the `free` state of a parameter change the :py:attr:`~.component.Parameter.free` attribute.

The value of a parameter can be coupled to the value of another by setting the :py:attr:`~.component.Parameter.twin` attribute.

The following example clarifies these concepts:

.. code-block:: ipython
    
    In [35]: # Print the parameters of the gaussian components
    In [36]: gaussian.parameters
    Out[4]: (A, sigma, centre)
    In [37]: # Fix the centre
    In [38]: gaussian.centre.free = False
    In [39]: # Print the free parameters
    In [40]: gaussian.parameters
    Out[41]: set([A, sigma])
    In [42]: gaussian.parameters
    Out[5]: set([A, sigma])
    In [43]: # Print the current value of all the free parameters
    In [44]: m.print_current_values()
    Components	Parameter	Value
    Normalized Gaussian
		    A	1.000000
		    sigma	1.000000
    Normalized Gaussian
		    centre	0.000000
		    A	1.000000
		    sigma	1.000000
    Normalized Gaussian
		    A	1.000000
		    sigma	1.000000
		    centre	0.000000
    In [45]: # Couple the A parameter of gaussian2 to the A parameter of gaussian 3
    In [46]: gaussian2.A.twin = gaussian3.A
    In [47]: # Set the gaussian2 centre value to 10
    In [48]: gaussian2.centre.value = 10
    In [50]: # Print the current value of all the free parameters
    In [51]: m.print_current_values()
    Components	Parameter	Value
    Normalized Gaussian
		    A	1.000000
		    sigma	1.000000
    Normalized Gaussian
		    centre	10.000000
		    A	1.000000
		    sigma	1.000000
    Normalized Gaussian
		    A	1.000000
		    sigma	1.000000
		    centre	0.000000   
    


Exclude data from the fitting process
-------------------------------------


The following :py:class:`~.model.Model` methods can be used to exclude undesired spectral channels from the fitting process:

* :py:meth:`~.model.Model.set_data_range_in_units`
* :py:meth:`~.model.Model.set_data_range_in_pixels`
* :py:meth:`~.model.Model.remove_data_range_in_units`
* :py:meth:`~.model.Model.remove_data_range_in_pixels`
* :py:meth:`~.model.Model.reset_data_range`

Visualising the result of the fit
---------------------------------

The :py:class:`~.model.Model`, :py:class:`~.component.Component` and :py:class:`~.component.Parameter` classes have plot methods to visualise
the result of the fit **when fitting multidimensional datasets**.

* :py:meth:`~.model.Model.plot_results`
* :py:meth:`~.component.Component.plot`
* :py:meth:`~.component.Parameter.plot`

Saving and loading the result of the fit
----------------------------------------

To save the result of the fit to a single file use :py:meth:`~.model.Model.save_parameters2file` and :py:meth:`~.model.Model.load_parameters_from_file` to load back the results into the same model structure.

Exporting the result of the fit
-------------------------------

The :py:class:`~.model.Model`, :py:class:`~.component.Component` and :py:class:`~.component.Parameter` classes have export methods:

* :py:meth:`~.model.Model.export_results`
* :py:meth:`~.component.Component.export`
* :py:meth:`~.component.Parameter.export`

