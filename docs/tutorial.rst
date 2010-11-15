EELSLab Tutorial
++++++++++++++++

Introduction
============

Starting eelslab
----------------
To start eelslab type in a console:

.. code-block:: bash

    eelslab

or altenatively:

.. code-block:: bash

    python -pylab -wthread

and in the ipython console type:

.. code-block:: python


    import eelslab as el


.. NOTE::

   If you are using GNOME in Linux, you can open a terminal in a folder by 
   choosing "open terminal" in the file menu

Loading a file
--------------

To load a supported file (i.e. NetCDF, dm3, MSA, MRC, ser or emi) simply type:

.. code-block:: python

    s = el.load('filename')

.. NOTE::

   We use the variable `s` but you can choose any (valid) variable name

.. NOTE::

   The filename must include the extension

If the loading was successful, the variable s now contains a python object that 
can be an Image of Spectrum depending on the file

.. _getting-help-label:

Getting help
------------

The documentation can be accessed by adding a question mark to the name of a function. e.g.:

.. code-block:: python
    
    el.load?

.. NOTE::
  
        The documentation of the code is a work in progress, 
        so not all the objects are documented yet.

Autocompletion
--------------

In the Ipython terminal (that eelslab uses) you can conveniently use the tabulator to autocomplete the commands and filenames.

Plotting
--------

To plot an Spectrum or Image objects type:

.. code-block:: python
    
    s.plot()


To navigate in a spectrum image you can either use the pointer of the navigator window or the numpad cursors **when numlock is active**.

You can add an extra cursor to the navigator by pressing 'e' while the navigator figure is focused.

.. NOTE::
    If you prefer that 2D maps in gray scale type:

    .. code-block:: python
	
	gray()


.. NOTE::
    To close all the figures type:
    
    .. code-block:: python
	
	close('all')




Multivariate analysis
=====================

The tutorial files are in the tutorial folder of your home directory that will be generated the first time that you start EELSLab.

In GNOME you can directly start the terminal in the mva folder or, alternatively, navigate util that folder by typing the following in the terminal prompt

.. code-block:: bash

    cd tutorial/mva


.. _example1-label:

Example 1: Basic PCA/ICA workflow
-------------------------------------------------------

For this example we will use the file `CL1_eelslab.nc` that contains a simulated EELS SI.

We start by loading and plotting the data:

.. code-block:: python

    s = el.load('CL1_eelslab.nc')
    s.plot()


As we can observe, the spectra contains the Sr, Ti, O and C ionisation edges in the 100-600 eV energy range.

.. NOTE::

   If you don't remember the position of the ionisation edges of a particular element you can simply type the following to get the list:

   .. code-block:: python

    el.edges_db.edges_dict['element_symbol']

However the spectra are rather noisy. We will use principal components analysis (PCA) to improve the SNR. For that we type:

.. code-block:: python

    s.principal_components_analysis(True)


.. NOTE::

    If you want to know why we give the `True` value to the `principal_components_analysis` method you can take a look at the method documentation, see :ref:`getting-help-label`

To check the scree plot simply type:

.. code-block:: python

    s.plot_lev()

As you can observe, there are clearly just four principal components.

To plot the principal components:

.. code-block:: python

    s.plot_principal_components(4)

or

.. code-block:: python

    s.plot_principal_components_maps(4)

to get their distribution maps.

To save just the PCA matrix decomposition:

.. code-block:: python

    s.pca_results.save('filename')

If later on you want to load the PCA file:

.. code-block:: python

    s.pca_results.load('filename.npz')


To obtain a model of the SI using only the first four principal components:

.. code-block:: python

    sc =  s.pca_build_SI(4)

You can plot the new Spectrum object sc too see how your PCA model looks like.

.. NOTE::
    
    If you did not close the `s` plots you may have noticed that their cursors are synchronised

To save the new Spectrum file in EELSLab's netCDF file format:

.. code-block:: python

    sc.save('filename')

To perform independent components analysis on the principal components

.. code-block:: python

    s.independent_components_analysis(4)


And to see the result:

.. code-block:: python

    s.plot_independent_components_maps()


Example 2: Better SNR -> Better ICA
-----------------------------------
For this example we will use the file `CL2_eelslab.nc` that contains a simulated EELS SI.

The SI is identical to the former one, but with higher SNR. Do the full treatment as in :ref:`example1-label` to see the improvement in the ICA result.


Example 3: Correcting energy instabilities
------------------------------------------
For this example we will use the file `CL3_eelslab.nc` that contains a simulated EELS SI.

The SI is identical to `CL1_eelslab.nc`, but it suffers from poor energy stability.

If we perform the PCA analysis as in :ref:`example1-label` we can observe in the scree plot that the number of principal components has increased. Fortunatelly, we had acquired a low loss SI simultaneously ( LL3_eelslab.nc ) that we will use to correct the energy instability.

First load the data:

.. code-block:: python

    # Load the CL
    cl = el.load('CL3_eelslab.nc')

    # Load the LL
    ll = el.load('LL3_eelslab.nc')


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
    

Once aligned you can check that the scree plot gets closer to the one in :ref:`example1-label`


Example 4: Removing spikes
--------------------------
For this example we will use the file `CL4_eelslab.nc` that contains a simulated EELS SI.

The SI is identical to `CL1_eelslab.nc`, but it suffers from X-rays spikes.

If we perform the PCA analysis as in :ref:`example1-label` we can observe in the scree plot that the number of principal components has increased.

EELSLab has three Spectrum methods to deal with spikes: `spikes_diagnosis`, `plot_spikes` and `remove_spikes`.

The workflow for spikes removal is as follows:

.. code-block:: python

    # Load the file
    s = el.load('CL4_eelslab.nc')
    
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

After cleaning the spikes the SI can be processed as in :ref:`example1-label`


Curve fitting
=============

Example 1: 
----------

Example 2: 
----------

Example 3: 
----------


