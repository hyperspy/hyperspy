Curve fitting
*************

HyperSpy can perform curve fitting in n-dimensional data sets. It can create a
model from a linear combinantion of predefined components and can use multiple
optimisation algorithms to fit the model to experimental data. It supports
bounds and weights.

.. versionadded:: 0.7
   
    Before creating a model verify that the ``Signal.binned`` metadata
    attribute of the signal is set to the correct value because the resulting
    model depends on this parameter. See :ref:`signal.binned` for more details.

Creating a model
^^^^^^^^^^^^^^^^

A :py:class:`~.model.Model` can be created using the
:py:func:`~.hspy.create_model` function, whose first argument is a
:py:class:`~.signal.Signal` of any of its subclasses (often it is simply the
object returned by the :py:func:`~.io.load` function. e.g.,

.. code-block:: python
    
    >>> s = load('YourDataFilenameHere') # Load the data from a file
    >>> m = create_model(s) # Create the model and asign it to the variable m

At this point you may be prompted to provide any necessary information not
already included in the datafile, e.g.if s is EELS data, you may be asked for
the accelerating voltage, convergence and collection angles etc.

Adding components to the model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In HyperSpy a model consists of a linear combination of :py:mod:`~.components`.
These are some of the components which are currently available:


* :py:class:`~._components.eels_cl_edge.EELSCLEdge`
* :py:class:`~._components.volume_plasmon_drude.VolumePlasmonDrude`
* :py:class:`~._components.power_law.PowerLaw`
* :py:class:`~._components.offset.Offset`
* :py:class:`~._components.exponential.Exponential`
* :py:class:`~._components.scalable_fixed_pattern.ScalableFixedPattern`
* :py:class:`~._components.gaussian.Gaussian`
* :py:class:`~._components.lorentzian.Lorentzian`
* :py:class:`~._components.voigt.Voigt`
* :py:class:`~._components.polynomial.Polynomial`
* :py:class:`~._components.logistic.Logistic`
* :py:class:`~._components.bleasdale.Bleasdale`
* :py:class:`~._components.error_function.Erf`
* :py:class:`~._components.pes_see.SEE`
* :py:class:`~._components.arctan.Arctan`


 
Writing a new component is very easy, so, if the function that you need to fit
is not in the list above, by inspecting the code of, for example, the Gaussian
component, it should be easy to write your own component. If you need help for
the task please submit your question to the :ref:`users mailing list
<http://groups.google.com/group/hyperspy-users>`.


To print the current components in a model simply enter the name of the
variable, e.g.:

.. code-block:: python
    
    >>> m # m is the variable in which we have previously stored the model
    []
    >>> # [] means that the model is empty


In fact, components may be created automatically in some cases. For example, if
the s is recognised as EELS data, a power-law background component will
automatically be placed in m. To add a component first we have to create an
instance of the component. Once the instance has been created we can add the
component to the model using the :py:meth:`append` method, e.g. for a type of
data that can be modelled using gaussians we might proceed as follows:
    

.. code-block:: python
    
    >>> gaussian = components.Gaussian() # Create a Gaussian function component
    >>> m.append(gaussian) # Add it to the model
    >>> m # Print the model components 
    [<Gaussian component>]
    >>> gaussian2 = components.Gaussian() # Create another gaussian components
    >>> gaussian3 = components.Gaussian() # Create a third gaussian components
    

We could use the append method two times to add the two gaussians, but when
adding multiple components it is handier to use the extend method that enables
adding a list of components at once.


.. code-block:: python

    >>> m.extend((gaussian2, gaussian3)) #note the double brackets!
    >>> m
    [<Gaussian component>, <Gaussian component>, <Gaussian component>]
    
    
We can customise the name of the components.

.. code-block:: python

    >>> gaussian.name = 'Carbon'
    >>> gaussian2.name = 'Hydrogen'
    >>> gaussian3.name = 'Nitrogen'
    >>> m
    [<Carbon (Gaussian component)>,
     <Hydrogen (Gaussian component)>,
     <Nitrogen (Gaussian component)>]


Two components cannot have the same name.

.. code-block:: python

    >>> gaussian2.name = 'Carbon'
    Traceback (most recent call last):
      File "<ipython-input-5-2b5669fae54a>", line 1, in <module>
        g2.name = "Carbon"
      File "/home/fjd29/Python/hyperspy/hyperspy/component.py", line 466, in name
        "the name " + str(value))
    ValueError: Another component already has the name Carbon


It is possible to access the components in the model by their name or by the
index in the model.
    
.. code-block:: python

    >>> m
    [<Carbon (Gaussian component)>,
     <Hydrogen (Gaussian component)>,
     <Nitrogen (Gaussian component)>]
    >>> m[0]
    <Carbon (Gaussian component)>
    >>> m["Carbon"]
    <Carbon (Gaussian component)>

It is possible to "switch off" a component by setting its
:py:attr:`~.component.Component.active` to `False`. When a components is
switched off, to all effects it is as if it was not part of the model. To
switch it on simply set the :py:attr:`~.component.Component.active` attribute
back to `True`.

.. versionadded:: 0.7.1

    In multidimensional signals it is possible to store the value of the
    :py:attr:`~.component.Component.active` attribute at each navigation index.
    To enable this feature for a given component set the
    :py:attr:`~.component.Component.active_is_multidimensional` attribute to
    `True`. 

    .. code-block:: python
 
        >>> s = signals.Spectrum(np.arange(100).reshape(10,10))
        >>> m = create_model(s)
        >>> g1 = components.Gaussian()
        >>> g2 = components.Gaussian()
        >>> m.extend([g1,g2])
        >>> g1.active_is_multidimensional = True
        >>> g1._active_array
        array([ True,  True,  True,  True,  True,  True,  True,  True,  True,  True], dtype=bool)
        >>> g2._active_array is None
        True
        >>> m.set_component_active_value(False)
        >>> g1._active_array
        array([False, False, False, False, False, False, False, False, False, False], dtype=bool)
        >>> m.set_component_active_value(True, only_current=True)
        >>> g1._active_array
        array([ True, False, False, False, False, False, False, False, False, False], dtype=bool)
        >>> g1.active_is_multidimensional = False
        >>> g1._active_array is None
        True
 

Getting and setting parameter values and attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:py:meth:`~.model.Model.print_current_values` prints the value of the
parameters of the components in the current coordinates.

:py:attr:`~.component.Component.parameters` contains a list of the parameters
of a component and :py:attr:`~.component.Component.free_parameters` lists only
the free parameters.

The value of a particular parameter can be accessed in the
:py:attr:`~.component.Parameter.value`.

If a model contains several components with the same parameters, it is possible
to change them all by using :py:meth:`~.model.Model.set_parameters_value`.
Example:

.. code-block:: python

    >>> s = signals.Spectrum(np.arange(100).reshape(10,10))
    >>> m = create_model(s)
    >>> g1 = components.Gaussian()
    >>> g2 = components.Gaussian()
    >>> m.extend([g1,g2])
    >>> m.set_parameters_value('A', 20)
    >>> g1.A.map['values']
    array([ 20.,  20.,  20.,  20.,  20.,  20.,  20.,  20.,  20.,  20.])
    >>> g2.A.map['values']
    array([ 20.,  20.,  20.,  20.,  20.,  20.,  20.,  20.,  20.,  20.])
    >>> m.set_parameters_value('A', 40, only_current=True)
    >>> g1.A.map['values']
    array([ 40.,  20.,  20.,  20.,  20.,  20.,  20.,  20.,  20.,  20.])
    >>> m.set_parameters_value('A',30, component_list=[g2])
    >>> g2.A.map['values']
    array([ 30.,  30.,  30.,  30.,  30.,  30.,  30.,  30.,  30.,  30.])
    >>> g1.A.map['values']
    array([ 40.,  20.,  20.,  20.,  20.,  20.,  20.,  20.,  20.,  20.])


To set the the `free` state of a parameter change the
:py:attr:`~.component.Parameter.free` attribute. To change the `free` state of
all parameters in a component to `True` use
:py:meth:`~.component.Component.set_parameters_free`, and
:py:meth:`~.component.Component.set_parameters_not_free` for setting them to
`False`. Specific parameter-names can also be specified by using
`parameter_name_list`, shown in the example:

.. code-block:: python

    >>> g = components.Gaussian()
    >>> g.free_parameters
    set([<Parameter A of Gaussian component>,
        <Parameter sigma of Gaussian component>,
        <Parameter centre of Gaussian component>])
    >>> g.set_parameters_not_free()
    set([])
    >>> g.set_parameters_free(parameter_name_list=['A','centre'])
    set([<Parameter A of Gaussian component>,
         <Parameter centre of Gaussian component>])


Similar functions exist for :py:class:`~.model.Model`:
:py:meth:`~.model.Model.set_parameters_free` and
:py:meth:`~.model.Model.set_parameters_not_free`. Which sets the
:py:attr:`~.component.Parameter.free` states for the parameters in components
in a model. Specific components and parameter-names can also be specified. For
example:

.. code-block:: python

    >>> g1 = components.Gaussian()
    >>> g2 = components.Gaussian()
    >>> m.extend([g1,g2])
    >>> m.set_parameters_not_free()
    >>> g1.free_parameters
    set([])
    >>> g2.free_parameters
    set([])
    >>> m.set_parameters_free(parameter_name_list=['A'])
    >>> g1.free_parameters 
    set([<Parameter A of Gaussian component>])
    >>> g2.free_parameters 
    set([<Parameter A of Gaussian component>])
    >>> m.set_parameters_free([g1], parameter_name_list=['sigma'])
    >>> g1.free_parameters 
    set([<Parameter A of Gaussian component>,
         <Parameter sigma of Gaussian component>])
    >>> g2.free_parameters 
    set([<Parameter A of Gaussian component>])


The value of a parameter can be coupled to the value of another by setting the
:py:attr:`~.component.Parameter.twin` attribute.

For example:

.. code-block:: python
    
    >>> gaussian.parameters # Print the parameters of the gaussian components
    (A, sigma, centre)
    >>> gaussian.centre.free = False # Fix the centre
    >>> gaussian.free_parameters  # Print the free parameters
    set([A, sigma])
    >>> m.print_current_values() # Print the current value of all the free parameters
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
    >>> gaussian2.A.twin = gaussian3.A # Couple the A parameter of gaussian2 to the A parameter of gaussian 3
    >>> gaussian2.A.value = 10 # Set the gaussian2 centre value to 10
    >>> m.print_current_values()
    Components	Parameter	Value
    Carbon
            sigma	1.000000
            A	1.000000
            centre	0.000000
    Hydrogen
            sigma	1.000000
            A	10.000000
            centre	10.000000
    Nitrogen
            sigma	1.000000
            A	10.000000
            centre	0.000000

    >>> gaussian3.A.value = 5 # Set the gaussian1 centre value to 5
    >>> m.print_current_values()
    Components	Parameter	Value
    Carbon
            sigma	1.000000
            A	1.000000
            centre	0.000000
    Hydrogen
            sigma	1.000000
            A	5.000000
            centre	10.000000
    Nitrogen
            sigma	1.000000
            A	5.000000
            centre	0.000000


By default the coupling function is the identity function. However it is
possible to set a different coupling function by setting the
:py:attr:`~.component.Parameter.twin_function` and
:py:attr:`~.component.Parameter.twin_inverse_function` attributes.  For
example:
 
    >>> gaussian2.A.twin_function = lambda x: x**2
    >>> gaussian2.A.twin_inverse_function = lambda x: np.sqrt(np.abs(x))
    >>> gaussian2.A.value = 4
    >>> m.print_current_values()
    Components	Parameter	Value
    Carbon
            sigma	1.000000
            A	1.000000
            centre	0.000000
    Hydrogen
            sigma	1.000000
            A	4.000000
            centre	10.000000
    Nitrogen
            sigma	1.000000
            A	2.000000
            centre	0.000000

    >>> gaussian3.A.value = 4
    >>> m.print_current_values()
    Components	Parameter	Value
    Carbon
            sigma	1.000000
            A	1.000000
            centre	0.000000
    Hydrogen
            sigma	1.000000
            A	16.000000
            centre	10.000000
    Nitrogen
            sigma	1.000000
            A	4.000000
            centre	0.000000

.. _model.fitting:            

Fitting the model to the data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To fit the model to the data at the current coordinates (e.g. to fit one
spectrum at a particular point in a spectrum-image) use
:py:meth:`~.model.Model.fit`.

The following table summarizes the features of the currently available
optimizers:


.. table:: Features of curve fitting optimizers.

    +-----------+--------+------------------+-----------------------------------+
    | Optimizer | Bounds | Error estimation | Method                            |
    +===========+========+==================+===================================+
    | "leastsq" |  No    | Yes              | least squares                     |
    +-----------+--------+------------------+-----------------------------------+
    | "mpfit"   |  Yes   | Yes              | least squares                     |
    +-----------+--------+------------------+-----------------------------------+
    | "odr"     |  No    | Yes              | least squares                     |
    +-----------+--------+------------------+-----------------------------------+
    |  "fmin"   |  No    | No               | least squares, maximum likelihood |
    +-----------+--------+------------------+-----------------------------------+

The following example shows how to perfom least squares with error estimation.

First we create data consisting of a line line ``y = a*x + b`` with ``a = 1``
and ``b = 100`` and we add white noise to it:

.. code-block:: python

    >>> s = signals.SpectrumSimulation(
    ...     np.arange(100, 300))
    >>> s.add_gaussian_noise(std=100)

To fit it we create a model consisting of a
:class:`~._components.polynomial.Polynomial` component of order 1 and fit it
to the data.

.. code-block:: python

    >>> m = create_model(s)
    >>> line  = components.Polynomial(order=1)
    >>> m.append(line)
    >>> m.fit()

On fitting completion, the optimized value of the parameters and their estimated standard deviation
are stored in the following line attributes:

.. code-block:: python

    >>> line.coefficients.value
    (0.99246156488437653, 103.67507406125888)
    >>> line.coefficients.std
    (0.11771053738516088, 13.541061301257537)



When the noise is heterocedastic, only if the
``metadata.Signal.Noise_properties.variance`` attribute of the
:class:`~._signals.spectrum.Spectrum` instance is defined can the errors be
estimated accurately. If the variance is not defined, the standard deviation of
the parameters are still computed and stored in the
:attr:`~.component.Parameter.std` attribute by setting variance equal 1.
However, the value won't be correct unless an accurate value of the variance is
defined in ``metadata.Signal.Noise_properties.variance``. See
:ref:`signal.noise_properties` for more information.

In the following example, we add poissonian noise to the data instead of
gaussian noise and proceed to fit as in the previous example.

.. code-block:: python

    >>> s = signals.SpectrumSimulation(
    ...     np.arange(300))
    >>> s.add_poissonian_noise()
    >>> m = create_model(s)
    >>> line  = components.Polynomial(order=1)
    >>> m.append(line)
    >>> m.fit()
    >>> line.coefficients.value
    (1.0052331707848698, -1.0723588390873573)
    >>> line.coefficients.std
    (0.0081710549764721901, 1.4117294994070277)

Because the noise is heterocedastic, the least squares optimizer estimation is
biased. A more accurate result can be obtained by using weighted least squares
instead that, although still biased for poissonian noise, is a good 
approximation in most cases.

.. code-block:: python

   >>> s.estimate_poissonian_noise_variance(expected_value=signals.Spectrum(np.arange(300)))
   >>> m.fit()
   >>> line.coefficients.value
   (1.0004224896604759, -0.46982916592391377)
   >>> line.coefficients.std
   (0.0055752036447948173, 0.46950832982673557)


We can use poissonian maximum likelihood estimation
instead that is an unbiased estimator for poissonian noise.

.. code-block:: python

   >>> m.fit(fitter="fmin", method="ml")
   >>> line.coefficients.value
   (1.0030718094185611, -0.63590210946134107)

Problems of ill-conditioning and divergence can be ameliorated by using bounded
optimization. Currently, only the "mpfit" optimizer supports bounds. In the
following example a gaussian histogram is fitted using a
:class:`~._components.gaussian.Gaussian` component using mpfit and bounds on
the ``centre`` parameter.

.. code-block:: python

    >>> s = signals.Signal(np.random.normal(loc=10, scale=0.01,
    size=1e5)).get_histogram()
    >>> s.metadata.Signal.binned = True
    >>> m = create_model(s)
    >>> g1 = components.Gaussian()
    >>> m.append(g1)
    >>> g1.centre.value = 7
    >>> g1.centre.bmin = 7
    >>> g1.centre.bmax = 14
    >>> g1.centre.bounded = True
    >>> m.fit(fitter="mpfit", bounded=True)
    >>> m.print_current_values()
    Components  Parameter   Value
    Gaussian
            sigma   0.00996345
            A   99918.7
            centre  9.99976



.. versionadded:: 0.7

    The chi-squared, reduced chi-squared and the degrees of freedom are
    computed automatically when fitting. They are stored as signals, in the
    :attr:`~.model.Model.chisq`, :attr:`~.model.Model.red_chisq`  and
    :attr:`~.model.Model.dof` attributes of the model respectively. Note that,
    unless ``metadata.Signal.Noise_properties.variance`` contains an accurate
    estimation of the variance of the data, the chi-squared and reduced
    chi-squared cannot be computed correctly. This is also true for
    homocedastic noise. 
        
.. _model.visualization:

Visualizing the model
^^^^^^^^^^^^^^^^^^^^^

To visualise the result use the :py:meth:`~.model.Model.plot` method:

.. code-block:: python
    
    >>> m.plot() # Visualise the results

.. versionadded:: 0.7

By default only the full model line is displayed in the plot. In addition, it
is possible to display the individual components by calling
:py:meth:`~.model.Model.enable_plot_components` or directly using
:py:meth:`~.model.Model.plot`:

.. code-block:: python
    
    >>> m.plot(plot_components=True) # Visualise the results

To disable this feature call :py:meth:`~.model.Model.disable_plot_components`.

.. versionadded:: 0.7.1

   By default the model plot is automatically updated when any parameter value
   changes. It is possible to suspend this feature with 
   :py:meth:`~.model.Model.suspend_update`. To resume it use
   :py:meth:`~.model.Model.resume_update`. 


.. _model.starting:

Setting the initial parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Non-linear regression often requires setting sensible starting 
parameters. This can be done by plotting the model and adjusting the parameters
by hand. 

.. versionadded:: 0.7

    In addition, it is possible to fit a given component  independently using
    the :py:meth:`~.model.Model.fit_component` method.

    

.. versionadded:: 0.6

    Also, :py:meth:`~.model.Model.enable_adjust_position` provides an
    interactive way of setting the position of the components with a well
    define position.  :py:meth:`~.model.Model.disable_adjust_position` disables
    the tool. 


    .. figure::  images/model_adjust_position.png
        :align:   center
        :width:   500    

        Interactive component position adjustment tool.Drag the vertical lines
        to set the initial value of the position parameter. 



Exclude data from the fitting process
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following :py:class:`~.model.Model` methods can be used to exclude
undesired spectral channels from the fitting process:

* :py:meth:`~.model.Model.set_signal_range`
* :py:meth:`~.model.Model.remove_signal_range`
* :py:meth:`~.model.Model.reset_signal_range`

Fitting multidimensional datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To fit the model to all the elements of a multidimensional datataset use
:py:meth:`~.model.Model.multifit`, e.g.:
    
.. code-block:: python

    >>> m.multifit() # warning: this can be a lengthy process on large datasets

:py:meth:`~.model.Model.multifit` fits the model at the first position, 
store the result of the fit internally and move to the next position until 
reaching the end of the dataset.

Sometimes one may like to store and fetch the value of the parameters at a
given position manually. This is possible using
:py:meth:`~.model.Model.store_current_values` and
:py:meth:`~.model.Model.fetch_stored_values`.


Visualising the result of the fit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:class:`~.model.Model` :py:meth:`~.model.Model.plot_results`,
:py:class:`~.component.Component` :py:meth:`~.component.Component.plot` and
:py:class:`~.component.Parameter` :py:meth:`~.component.Parameter.plot` methods
can be used to visualise the result of the fit **when fitting multidimensional
datasets**.



Saving and loading the result of the fit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As of HyperSpy 0.7.4, the following is the only way to save a model to  a file
and load it back. Note that this method is known to be brittle i.e. there is no
guarantee that a version of HyperSpy different from the one used to save the
model will be able to load it sucessfully.  Also, it is advisable not to use
this method in combination with functions that alter the value of the
parameters interactively (e.g.  `enable_adjust_position`) as the modifications
made by this functions are normally not stored in the IPython notebook or
Python script.

To save a model:

1. Save the parameter arrays to a file using
   :py:meth:`~.model.Model.save_parameters2file`.

2. Save all the commands that used to create the model to a file. This
   can be done in the form of an IPython notebook or a Python script.
   
3.  (Optional) Comment out or delete the fitting commangs (e.g. `multifit`).

To recreate the model:

1. Execute the IPython notebook or Python script.

2. Use :py:meth:`~.model.Model.load_parameters_from_file` to load back the
   parameter values and arrays.


Exporting the result of the fit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:class:`~.model.Model` :py:meth:`~.model.Model.export_results`,
:py:class:`~.component.Component` :py:meth:`~.component.Component.export` and
:py:class:`~.component.Parameter` :py:meth:`~.component.Parameter.export`
methods can be used to export the result of the optimization in all supported
formats.

Batch setting of parameter attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. versionadded:: 0.6

The following methods can be used to ease the task of setting some important
parameter attributes:

* :py:meth:`~.model.Model.set_parameters_not_free`
* :py:meth:`~.model.Model.set_parameters_free`
* :py:meth:`~.model.Model.set_parameters_value`


