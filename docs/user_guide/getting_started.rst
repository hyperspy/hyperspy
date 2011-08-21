Getting started
***************

First steps with hyperspy
========================

Starting hyperspy
----------------

To start hyperspy type in a console:

.. code-block:: bash

    $ hyperspy

If everythig goes well Hyperspy should welcome you with a message similar to:

.. code-block:: ipython
    
    H y p e r s p y
    Version 0.3.0
    
    Copyright (C) 2007-2010 Francisco de la Peña
    Copyright (C) 2010-2011 F. de la Peña, S. Mazzucco, M. Sarahan
    
    http://www.hyperspy.org


.. NOTE::

   If you are using GNOME in Linux, you can open a terminal in a folder by 
   choosing :menuselection:`open terminal` in the file menu if 
   :program:`nautilus-open-terminal` is 
   installed in your system.
   A similar feature is available in :program:`Windows Vista` or newer when pressing :kbd:`Shift Right-Mouse-Button`.

   Alternatively, if you are using :program:`Windows Vista` or newer, you can navigate to the
   folder with your data files, and then click in the address bar.
   Enter cmd, then press enter.  A command prompt will be opened in
   that folder.
   
.. NOTE::
       For more comfort in Windows it is recommended to use 
       `Console2 <http://sourceforge.net/projects/console/>`_ instead of the default terminal of that platform.    


Loading data
-----------------------


To load from a supported file format (see :ref:`supported-formats`) simply type:

.. code-block:: python

    s = load('filename')

.. HINT::

   We use the variable :guilabel:`s` but you can choose any (valid) variable name

For more details read :ref:`loading_files`


Saving Files
------------

The data can be saved to several file formats.  The format is specified by
the extension of the filename.

.. code-block:: python

    # load the data
    d=load('example.tif')
    # save the data as a tiff
    d.save('example_processed.tif')
    # save the data as a png
    d.save('example_processed.png')
    # save the data as an hdf5 file
    d.save('example_processed.hdf5')

Some file formats are much better at maintaining the information about
how you processed your data.  The preferred format in EELSlab is hdf5,
the hierarchical data format.  This format keeps the most information
possible.

There are optional flags that may be passed to the save function. See :ref:`saving_files` for more details.


.. _configuring-hyperspy-label:

Configuring hyperspy
-------------------

You can configure some parameters of hyperspy by editing the :file:`hyperspyrc` 
file. The location of the configuration file depends on the system. 
You can find its path by calling the ```get_configuration_directory_path``` 
function in the hyperspy prompt:

.. code-block:: pythons

    get_configuration_directory_path()


Alternatively it is possible to change the same parameters at runtime by changing 
the attributes of the defaults class. For example, to plot automatically the 
data when loading it:

.. code-block:: bash

    # First we load some data
    s = load('YourDataFilenameHere')
    # (in the defaults setting nothing is plotted, unless you can changed the 
    # defaults in the hyperspyrc file)
    #
    # Now we will change the setting at runtime
    defaults.plot_on_load = True
    s = load('YourDataFilenameHere')
    # The data should have been automatically plotted.



.. _getting-help-label:

Getting help
------------

The documentation can be accessed by adding a question mark to the name of a function. e.g.:

.. code-block:: python
    
    load?

This syntax is one of the many features of `IPython <http://ipython.scipy.org/moin/>`_ that is the interactive python shell that Hyperspy uses under the hood.

Please note that the documentation of the code is a work in progress, so not all the objects are documented yet.

Autocompletion
--------------

Another useful `IPython <http://ipython.scipy.org/moin/>`_ feature is the 
autocompletion of commands and filenames. It is highly recommended to read the 
`Ipython documentation <http://ipython.scipy.org/moin/Documentation>`_ for many more useful features that will boost efficiency when working with Hyperspy/Python.

Data visualisation
==================

:py:class:`~.signal.Signal` has a ``plot`` method.

.. code-block:: python
    
    s = load('YourDataFilenameHere')
    s.plot()

if the object is single spectrum or an image one window will appear when calling 
the plot method. If the object is a 2D or 3D SI two figures will appear, 
one containing a plot of a spectrum of the dataset and the other a 2D 
representation of the data. 

To explore a hyperspectrum drag the cursor present in the 2D data representation 
(it can be a line for 1D data exploration or a square for 2D data exploration). 
An extra cursor can be added by pressing the ``e`` key. Pressing ``e`` once more will 
disable the extra cursor.

When exploring a 2D hyperspectral object of high spatial resolution the default size of the rectangular cursors can be too small to be dragged or even seen. It is possible to change the size of the cursors by pressing the ``+`` and ``-`` keys  **when the navigator
windows is on focus**.

It is also possible to move the pointer by using the numpad arrows 
**when numlock is on and the spectrum or navigator figure is on focus**. 
When using the numpad arrows the PageUp and PageDown keys change the size of the step.

The same keys can be used to explore an image stack.



=========   =============================
key         function    
=========   =============================
e           Switch second pointer on/off
Arrows      Change coordinates  
PageUp      Increase step size
PageDown    Decrease step size
``+``           Increase pointer size
``-``           Decrease pointer size
=========   =============================


To close all the figures type:

.. code-block:: python

    plt.close('all')


This is a `matplotlib <http://matplotlib.sourceforge.net/>`_ command. 
Matplotlib is the library that hyperspy uses to produce the plots. You can learn how 
to pan/zoom and more  
`in the matplotlib documentation <http://matplotlib.sourceforge.net/users/navigation_toolbar.html>`_


