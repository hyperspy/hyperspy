Tutorials
++++++++++++++++

The tutorial files are in the tutorial folder of your home directory that will be generated the first time that you start Hyperspy.

In GNOME you can directly start the terminal in the mva folder or, alternatively, navigate to that folder by typing the following in the terminal prompt

.. code-block:: bash

    cd tutorial/mva

Multivariate analysis
=====================


.. _example1-label:

Tutorial 1: Basic PCA/ICA workflow
-------------------------------------------------------

For this example we will use the file `CL1_hyperspy.nc` that contains a simulated EELS SI.

We start by loading and plotting the data:

.. code-block:: python

    s = load('CL1_hyperspy.nc')
    s.plot()


A trained EELS analyst well easily note that the spectra contains the Sr, Ti, O and C ionisation edges in the 100-600 eV energy range.

.. NOTE::

   If you don't remember the position of the ionisation edges of a particular element you can simply type the following to get the list:

   .. code-block:: python

    edges_dict['element_symbol']

We will use principal components analysis (PCA) to improve the SNR. For that we type:

.. code-block:: python

    s.principal_components_analysis(True)


.. NOTE::

    If you want to know why we give the `True` value to the `principal_components_analysis` method you can take a look at the method documentation, see :ref:`getting-help-label`

To check the scree plot simply type:

.. code-block:: python

    s.plot_lev()

As you can observe, there are just four principal components.

To plot the principal components:

.. code-block:: python

    s.plot_principal_components(4)

and to plot the score maps:

.. code-block:: python

    s.plot_principal_components_maps(4)

To save the PCA matrix decomposition:

.. code-block:: python

    s.mva_results.save('filename')

If later on you want to load the PCA file:

.. code-block:: python

    s.mva_results.load('filename.npz')

.. NOTE::

The saving and loading examples above create files that are only
useful to EELSlab, or to people interested in learning Python/Numpy.
MVA results can be saved to a format that can be loaded into Digital
Micrograph or several other programs using the
save_principal_components and save_independent_components functions.


To obtain a model of the SI using only the first four principal components:

.. code-block:: python

    sc =  s.pca_build_SI(4)

You can plot the new Spectrum object sc too see how your PCA model looks like.

.. NOTE::
    
    If you did not close the `s` plots you may have noticed that their cursors are synchronised

To save the new Spectrum file in Hyperspy's netCDF file format:

.. code-block:: python

    sc.save('filename')

Independent Component Analysis
---------------------------------

.. NOTE::

You must have performed PCA before trying to perform ICA.

To perform independent components analysis on the principal components

.. code-block:: python

    s.independent_components_analysis(4)


And to see the result:

.. code-block:: python

    s.plot_independent_components_maps()


Tutorial 2: Better SNR -> Better ICA
-------------------------------------
For this example we will use the file `CL2_hyperspy.nc` that contains a simulated EELS SI.

The SI is identical to the former one, but with higher SNR. Do the full treatment as in :ref:`example1-label`. Is the ICA result any better?

Now you can try to use second order differentation to perform the ICA by
looking at the `independent_components_analysis` method documentation.

.. _tutorial3-label:

Tutorial 3: Correcting energy instabilities
--------------------------------------------
Real data (unlike simulated ones) use to suffer from energy instabilities. In this tutorial we will see how to partially correct its effect by aligning the SI using an spectral feature that is known to be fixed in energy, ideally the zero loss peak (ZLP).

For this example we will use the files `CL3_hyperspy.nc` and `LL3_hyperspy.nc` that contais a simulated EELS SIs.

The SI is identical to `CL1_hyperspy.nc`, but it suffers from poor energy stability.

If we perform the PCA analysis as in :ref:`example1-label` we can observe in the scree plot that the number of principal components has increased. Fortunatelly, we had acquired a low loss SI simultaneously ( LL3_hyperspy.nc ) that we will use to correct the energy instability.

First load the data:

.. code-block:: python

    # Load the CL
    cl = load('CL3_hyperspy.nc')

    # Load the LL
    ll = load('LL3_hyperspy.nc')


.. NOTE::
    
    To easily spot the energy instability you can convert the LL SI in a line spectrum using:

    .. code-block:: python
	
	ll.unfold()
	ll.plot()

    
    To get back your 3D SI:

    .. code-block:: python
	
	ll.fold()


To align the low loss using the -5eV, 5eV energy interval, and apply the same correction to the CL:

.. code-block:: python
    
    # To align
    ll.align((-5,5), sync_SI = cl)

    # To correct the energy origin
    ll.find_low_loss_origin(sync_SI = cl)
    

Once aligned you can perform again the PCA and check that the scree plot gets closer to the one in :ref:`example1-label`


Tutorial 4: Removing spikes
----------------------------
For this example we will use the file `CL4_hyperspy.nc` and  `LL3_hyperspy.nc` that contain a simulated EELS SIs.

The SI is identical to `CL1_hyperspy.nc`, but it suffers from X-rays spikes and
the same energy instabilities found in  :ref:`tutorial3-label`.

If we perform the PCA analysis as in :ref:`example1-label` we can observe in the scree plot that the number of principal components has increased.

Hyperspy has three Spectrum methods to deal with spikes: `spikes_diagnosis`, `plot_spikes` and `remove_spikes`.

The workflow for spikes removal is as follows:

.. code-block:: python

    
    # Plot the energy derivative histogram and 
    # find a threshold for the outliers using spikes_diagnosis 
    s.spikes_diagnosis()

    # By visual inspection we find that the threshold approx. 2000
    # We can check if all the outliers in that region are indeed spikes
    # using the `plot_spikes` method
    s.plot_spikes(2000)

    # If we confirm that all the spectra correspond to spikes
    # we can remove them with the `remove_spikes` function
    s.remove_spikes(2000)

    # Otherwise we can increase the threshold value of use the 
    # `coordinates` parameter to provide a list of the SI 
    # coordinates where there are spikes. See the documentation.

After cleaning the spikes the SI can be processed as in :ref:`tutorial3-label`.


Curve fitting
=============

Setting the microscope parameters
----------------------------------

To quantify EEL spectra it is important to accurately define certain 
experimental parameters. The microscope parameters are stored in the 
``silib.microscope.microscope`` class.
The parameters by default are defined in the :file:`microscopes.csv` file that is
placed in the configuration directory (see :ref:`configuring-hyperspy-label` to
find out where is your configuration directory). Each microscope has a name
associated to it and you can define the default microscope in the :file:`hyperspyrc`
file (see :ref:`configuring-hyperspy-label`).
To modify the parameters from an hyperspy interactive session simply change the
attributes of the microscope class, e.g.:

.. code-block:: python

    microscope.alpha = 15 # convergence semi-angle in mrad
    microscope.beta = 20 #  collection semi-angle in mrad
    microscope.E0 = 100E3 # Beam energy in eV
    microscope.name = 'Pepe'

.. NOTE::

   This settings will be lost once you close your session unless you save a
   file in a format that supports saving the microscope parameters (at the
   moment only netCDF and msa). In that case, the settings will be loaded when
   you load the file.

In the interactive session you can load the parameters of a microscope defined
in file:`microscope.csv` as follows:

.. code-block:: python

    # To print the list of the microscopes defined in the microscope.csv file
    microscope.get_available_microscope_names()
    # To load the parameters of a particular microscope
    microscope.set_microscope('the_name_of_your_microscope')

Tutorial 1: 
-----------

Tutorial 2: 
-----------

Tutorial 3: 
-----------


