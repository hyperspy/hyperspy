Curve fitting
*************

Hyperspy can perform curve fitting in n-dimensional data sets. It can create a model from a linear combinantion of predefined components and can use multiple optimisation algorithms to fit the model to experimental data. It supports bounds and weights.

Generics tools
--------------

Creating a model
^^^^^^^^^^^^^^^^

A :py:class:`~.model.Model` can be created using the :py:func:`~.hspy.create_model` function, whose first argument is a :py:class:`~.signal.Signal` of any of its subclasses (often it is simply the object returned by the :py:func:`~.io.load` function. e.g.,

.. code-block:: ipython
    
    In [1]: # Load the data from a file
    In [2]: s = load('YourDataFilenameHere')
    In [3]: #Create the model and asign it to the variable m
    In [4]: m = create_model(s)

At this point you may be prompted to provide any necessary information not already included in the datafile, e.g.if s is EELS data, you may be asked for the accellerating voltage, convergence and collection angles etc.

Adding components to the model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In Hyperspy a model consists of a linear combination of :py:mod:`~.components`. These are some of the components which are currently available:


* :py:class:`~.components.eels_cl_edge.EELSCLEdge`
* :py:class:`~.components.components.VolumePlasmonDrude`
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


To print the current components in a model simply enter the name of the variable, e.g.:

.. code-block:: ipython
    
    In [5]: # m is the variable in which we have previously stored the model
    In [6]: m
    Out[1]: []
    In [7]: # [] means that the model is empty
    

In fact, components may be created automatically in some cases. For example, if the s is recognised as EELS data, a power-law background component will automatically be placed in m. To add a component first we have to create an instance of the component. Once the instance has been created we can add the component to the model using the :py:meth:`append` method, e.g. for a type of data that can be modelled using gaussians we might proceed as follows:
    

.. code-block:: ipython
    
    In [8]: # Create a Gaussian function component
    In [9]: gaussian = components.Gaussian()
    In [10]: # Add it to the model_cube
    In [11]: m.append(gaussian)
    In [12]: # Let's print the components
    In [13]: m
    Out[2]: [<Gaussian component>]
    In [14]: # Create two more gaussian function components
    In [15]: gaussian2 = components.Gaussian()
    In [16]: gaussian3 = components.Gaussian()
    In [17]: # We could use the append method two times to add the
    In [18]: # two gaussians, but when adding multiple components it is handier to use
    In [19]: # the extend method
    In [20]: m.extend((gaussian2, gaussian3)) #note the double brackets!
    In [21]: # Let's print the components    
    In [22]: m
    Out[2]: [<Gaussian component>, <Gaussian component>, <Gaussian component>]
    In [23]: # We can customise the name of the components
    In [24]: gaussian.name = 'Carbon'
    In [25]: gaussian2.name = 'Hydrogen'
    In [26]: gaussian3.name = 'Nitrogen'
    In [27]: # Let's print the components of the model once more
    In [28]: m
    Out[3]:
    [<Carbon (Gaussian component)>,
     <Hydrogen (Gaussian component)>,
     <Nitrogen (Gaussian component)>]
    
    
Fitting the model to the data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To fit the model to the data at the current coordinates (e.g. to fit one spectrum at a particular point in a spectrum-image) use :py:meth:`~.optimizers.Optimizers.fit`. To fit the model to the data in all the coordinates use :py:meth:`~.model.Model.multifit` and to visualise the result :py:meth:`~.model.Model.plot`, e.g.:

.. code-block:: ipython
    
    In [28]: # Let's fit the data at the current coordinates
    In [29]: m.fit()
    In [30]: # And now let's visualise the results
    In [31]: m.plot()
    In [32]: # Because we like what we see, we will fit the model to the
    In [33]: # data in all the coordinates
    In [34]: m.multifit() # warning: this can be a lengthy process on large datasets
    
Getting and setting parameter values and attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
    In [40]: gaussian.free_parameters
    Out[4]: set([A, sigma])
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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


The following :py:class:`~.model.Model` methods can be used to exclude undesired spectral channels from the fitting process:

* :py:meth:`~.model.Model.set_signal_range`
* :py:meth:`~.model.Model.remove_signal_range`
* :py:meth:`~.model.Model.reset_signal_range`

Visualising the result of the fit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:class:`~.model.Model` :py:meth:`~.model.Model.plot_results`, :py:class:`~.component.Component` :py:meth:`~.component.Component.plot` and :py:class:`~.component.Parameter` :py:meth:`~.component.Parameter.plot` methods can be used to visualise
the result of the fit **when fitting multidimensional datasets**.


Saving and loading the result of the fit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To save the result of the fit to a single file use :py:meth:`~.model.Model.save_parameters2file` and :py:meth:`~.model.Model.load_parameters_from_file` to load back the results into the same model structure.

Exporting the result of the fit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:class:`~.model.Model` :py:meth:`~.model.Model.export_results`, :py:class:`~.component.Component` :py:meth:`~.component.Component.export` and :py:class:`~.component.Parameter` :py:meth:`~.component.Parameter.export` methods can be used to export the result of the optimization in all supported formats.

EELS curve fitting
------------------

Hyperspy makes it really easy to quantify EELS core-loss spectra by curve fitting as it is shown in the next example of quantification of a boron nitride EELS spectrum from the `The EELS Data Base <http://pc-web.cemes.fr/eelsdb/index.php?page=home.php>`_. 

.. code-block:: ipython
    
    Load the core-loss spectrum
    >>> s = load("BN_(hex)_B_K_Giovanni_Bertoni_100.msa")
    
    Set some important experimental information that is missing
    from the file
    >>> s.set_microscope_parameters(beam_energy=100, convergence_angle=0.2, collection_angle=2.55)
    
    # Define the chemical composition of the sample
    >>> s.add_elements(('B', 'N'))
    
    Load the low-loss spectrum
    >>> ll = load("BN_(hex)_LowLoss_Giovanni_Bertoni_96.msa")
    
    We pass the low-loss spectrum to create_model to model 
    he effect of multiple scattering by Fourier-ratio convolution.
    >>> m = create_model(s, ll=ll)
    
    Hyperspy has created the model and configured it automacally:
    >>> m
    [<background (PowerLaw component)>,
    <N_K (EELSCLEdge component)>,
    <B_K (EELSCLEdge component)>]
    
    Furthermore, the components are available in the user namespace
    >>> N_K
    <N_K (EELSCLEdge component)>
    
    >>> background
    <background (PowerLaw component)>
    
    Conveniently variables named as the element symbol contains 
    all the components associated with the element to enable applying
    some methods to all of them at once. Although in this example
    the lists contains just one component this is not generally
    the case
    >>> N
    [<N_K (EELSCLEdge component)>]
    
    By default the fine structure features are disabled
    and we must enable them to accurately fit this spectrum
    >>> m.enable_fine_structure()
    
    We use smart_fit instead of standard fit method because smart_fit
    is optimized to fit EELS core-loss spectra
    >>> m.smart_fit()
    
    Print the result
    >>> m.quantify()
    Absolute quantification:
    Elem.	Intensity
    B	0.045648
    N	0.048061
    
    Visualize the fit
    >>> m.plot()

.. figure::  images/curve_fitting_BN.png
   :align:   center
   :width:   500    

   Curve fitting quantification of a boron nitride EELS core-loss spectrum from `The EELS Data Base <http://pc-web.cemes.fr/eelsdb/index.php?page=home.php>`_
   
   
The following methods are only available for :py:class:`~.models.EELSModel`: 

* :py:meth:`~.models.EELSModel.quantify`
* :py:meth:`~.models.EELSModel.remove_fine_structure_data`
* :py:meth:`~.models.EELSModel.enable_edges`
* :py:meth:`~.models.EELSModel.enable_background`
* :py:meth:`~.models.EELSModel.disable_background`
* :py:meth:`~.models.EELSModel.enable_fine_structure`
* :py:meth:`~.models.EELSModel.disable_fine_structure`
* :py:meth:`~.models.EELSModel.set_all_edges_intensities_positive`
* :py:meth:`~.models.EELSModel.unset_all_edges_intensities_positive`
* :py:meth:`~.models.EELSModel.enable_free_energy_shift`
* :py:meth:`~.models.EELSModel.disable_free_energy_shift`
* :py:meth:`~.models.EELSModel.fix_edges`
* :py:meth:`~.models.EELSModel.free_edges`
* :py:meth:`~.models.EELSModel.fix_fine_structure`
* :py:meth:`~.models.EELSModel.free_fine_structure`
* :py:meth:`~.models.EELSModel.free_fine_structure`
* :py:meth:`~.models.EELSModel.free_fine_structure`



