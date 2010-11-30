Introduction
============



Starting eelslab
----------------
To start eelslab type in a console:

.. code-block:: bash

    eelslab

.. NOTE::

   If you are using GNOME in Linux, you can open a terminal in a folder by 
   choosing "open terminal" in the file menu if nautilus-open-terminal is 
   installed in your system

.. _configuring-eelslab-label:
Configuring eelslab
-------------------
You can configure some parameters of eelslab by editing the eelslabrc. The
location of the configuration file depends on the system. You can find its path
by calling the get_configuration_directory_path function in the eelslab prompt:

.. code-block:: bash

    get_configuration_directory_path()




Loading a file
--------------

To load a supported file (i.e. NetCDF, dm3, MSA, MRC, ser or emi) simply type:

.. code-block:: python

    s = load('filename')

.. NOTE::

   We use the variable `s` but you can choose any (valid) variable name

.. NOTE::

   The filename *must* include the extension

If the loading was successful, the variable `s` now contains a python object 
that can be an Image of Spectrum.

Setting the microscope parameters
----------------------------------

The microscope parameters are stored in the silib.microscope.microscope class.
The parameters by default are defined in the microscopes.csv file that is
placed in the configuration directory (see :ref:`configuring-eelslab-label` to
find out where is your configuration directory). Each microscope has a name
associated to it and you can define the default microscope in the eelslabrc
file (see :ref:`configuring-eelslab-label`).
To modify the parameters from an eelslab interactive session simply change the
attributes of the microscope class,e.g.:

.. code-block:: python

    microscope.alpha = 15 # convergence semiangle in mrad
    microscope.beta = 20 #  collection semiangle in mrad
    microscope.E0 = 100E3 # Beam energy in eV
    microscope.name = 'Pepe'

.. NOTE::

   This settings will be lost once you close your session unless you save a
   file in a format that supports saving the microscope parameters (at the
   moment only netCDF and msa). In that case, the settings will be loaded when
   you load the file.

In the interactive session you can load the parameters of a microscope defined
in microscope.csv as follows:

.. code-block:: python

    # To print the list of the microscopes defined in the microscope.csv file
    microscope.get_available_microscope_names()
    # To load the parameters of a particular microscope
    microscope.set_microscope('the_name_of_your_microscope')


.. _getting-help-label:

Getting help
------------

The documentation can be accessed by adding a question mark to the name of a function. e.g.:

.. code-block:: python
    
    load?

.. NOTE::
  
        The documentation of the code is a work in progress, 
        so not all the objects are documented yet.

Autocompletion
--------------

In the Ipython terminal (that eelslab uses) you can conveniently use the tabulator to autocomplete the commands and filenames.

Exploring the data
------------------

The Spectrum and Image objects have a plot method.

.. code-block:: python
    
    s = load('YourDataFilenameHere')
    s.plot()

if the object is single spectrum or an image one window will appear when calling the plot method. If the object is a 2D or 3D SI two figures will appear, one containing a plot of a spectrum of the dataset and the other a 2D representation of the data. 

To explore an SI drag the cursor present in the 2D data representation (it can be a line for 2D SIs or a square for 3D SIs). An extra cursor can be added by pressing 'e'  **when numlock is on and the spectrum figure is on focus**. Pressing the 'e' key again will remove the extra cursor.

When exploring a 2D SI of high spatial resolution the default size of the
rectangular cursors can be too small to be dragged or even seen. It is possible to change
the size of the cursors by pressing the '+' and '-' keys  **when the navigator
windows is on focus**.

It is also possible to explore an SI by using the numpad arrows **when numlock is on and the spectrum figure is on focus**. When using the numpad arrows the PageUp and PageDown keys change the size of the step.

The same keys can be used to explore an image stack.






.. NOTE::
    To close all the figures type:
    
    .. code-block:: python
	
	close('all')
    # Note that this is a matplotlib command, not an eelslab one.
