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

Plotting
--------

To plot an Spectrum or Image object type:

.. code-block:: python
    
    s.plot()


To navigate in a spectrum image you can either use the pointer of the navigator window or the numpad cursors **when numlock is active**.

You can add an extra cursor to the navigator by pressing 'e' while the navigator figure is focused.

.. NOTE::
    If you prefer that 2D maps in gray scale type:

    .. code-block:: python
	
	gray()
    # Note that this is a matplotlib command, not an eelslab one.


.. NOTE::
    To close all the figures type:
    
    .. code-block:: python
	
	close('all')
    # Another matplotlib command.
