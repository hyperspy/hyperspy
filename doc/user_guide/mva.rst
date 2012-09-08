Machine learning
****************

Introduction
============

Hyperspy provides easy access to several "machine learning" algorithms which can
be useful when analysing hyperspectral data. In particular, decomposition 
algorithms such as principal component analysis (PCA) or blind source separation
algorithms such as independent component analysis (ICA) are available through
the methods described in this section.

The behaviour of some machine learning operations can be customised :ref:`customised <configuring-hyperspy-label>` in the Machine Learning section Preferences.

.. _decomposition-nomenclature:

Nomenclature
============

Hyperspy performs the decomposition of a dataset into two new datasets:
one with the dimension of the signal space which we will call `factors` and the other with 
the dimension of the navigation space which we will call `loadings`.
The same nomenclature applies to the result of BSS.

   
   
.. _decomposition:

Decomposition
=============

There are several methods to decompose a matrix or tensor into several factors.
The decomposition is most commonly applied as a means of noise reduction and
dimensionality reduction. One of the most popular decomposition methods is
principal component analysis (PCA). To perform PCA on your data set,
run the :py:meth:`~.learn.mva.MVA.decomposition` method:

.. code-block:: python
    >>> # Note that the s variable must contain a Signal class of any of its
    >>> # subclasses which most likely has been previously loaded with the 
    >>> # load function, e.g. s = load('my_file.hdf5')
    >>> s.decomposition()


Several algorithms exist for performing this analysis. The default algorithm in Hyperspy is :py:const:`SVD`, which performs PCA using an approach called "singular value decomposition". This method has many options. For more details read method documentation.


Poissonian noise
----------------

Most decomposition algorithms assume that the noise of the data follows a
Gaussian distribution. In the case that the data that you are analysing follow
a Poissonian distribution instead, Hyperspy can "normalise" the data by
performing a scaling operation which can greatly enhance the result.

To perform Poissonian noise normalisation:

.. code-block:: python

    >>> # The long way:
    >>> s.decomposition(normalize_poissonian_noise = True
    >>> # Because it is the first argument we cold have simply written:
    >>> s.decomposition(True)
    
For more details about the scaling procedure you can read the 
`following research article <http://onlinelibrary.wiley.com/doi/10.1002/sia.1657/abstract>`_


Principal component analysis
----------------------------

.. _scree-plot:

Scree plot
^^^^^^^^^^

PCA essentially sorts the components in the data in order of decreasing variance. It is often useful to estimate the dimensionality of the data 
by plotting the explained variance against the component index in a
logarithmic y-scale. This plot is sometimes called scree-plot and it should drop
quickly, eventually becoming a slowly descending line. The point at which it
becomes linear (often referred to as an elbow) is generally judged to be a good
estimation of the dimensionality of the data (or equivalently, the number of components that should be retained - see below).

To obtain a scree plot, run the :py:meth:`~.learn.mva.MVA.plot_explained_variance_ratio` method e.g.:

.. code-block:: python

    >>> s.plot_explained_variance_ratio()
    
Data denoising
--------------

One of the most popular uses of PCA is data denoising. The denoising property
is achieved by using a limited set of components to make a model of the
original, omitting the later components that ideally contain only noise.

To perform this operation with Hyperspy, run the :py:meth:`~.learn.mva.MVA.get_decomposition_model` method, usually after estimating the dimension of your data e.g. by using the :ref:`scree-plot`. For example:

.. code-block:: python

    >>> sc = s.get_decomposition_model(components)

.. NOTE:: 
    The components argument can be one of several things (None, int,
    or list of ints):

    * if None, all the components are used to construct the model.
    * if int, only the given number of components (starting from index 0) are used to construct the model.
    * if list of ints, only the components in the given list are used to construct the model.

Usually a low integer (<10) will be the appropriate choice for most types of hyperspectral data

.. HINT::
    Unlike most of the analysis functions, this function returns a new
    object, which in the example above we have called 'sc'. (The name of the variable is totally arbitrary and you can choose it at your will).  You can perform operations on this new object later. It is a copy of the original :py:const:`s` object, except that the data has been replaced by the model constructed using the chosen components.

Sometimes it is useful to examine the residuals between your original
data and the decomposition model. To examine residuals, use the :py:meth:`~.signal.Signal.plot_residual` method on
the reconstructed object, e.g.:

.. code-block:: python

    >>> sc.plot_residual()


Blind Source Separation
=======================

In some cases (it largely depends on the particular application) it is possible
to obtain more physically meaningful components from the result of a data
decomposition by a process called Blind Source Separation (BSS). For more information about the blind source separation you can read the 
`following introductory article  <http://www.sciencedirect.com/science/article/pii/S0893608000000265>`_
or `this other article <http://www.sciencedirect.com/science/article/pii/S030439911000255X>`_
from the authors of Hyperspy for an application to EELS analysis.

To perform BSS on the result of a decomposition, run the :py:meth:`~.learn.mva.MVA.blind_source_separation` method, e.g.:

.. code-block:: python

    s.blind_source_separation(number_of_components)

.. NOTE::
    You must have performed a :ref:`decomposition` before you attempt to 
    perform BSS.

.. NOTE::
    You must pass an integer number of components to ICA.  The best
    way to estimate this number in the case of a PCA decomposition is by
    inspecting the :ref:`scree-plot`.



Visualising results
===================

Plot methods exist for the results of decomposition and blind source separation.
All the methods begin with plot:

* :py:meth:`~.signal.Signal.plot_decomposition_factors`
* :py:meth:`~.signal.Signal.plot_decomposition_loadings`
* :py:meth:`~.signal.Signal.plot_bss_factors`
* :py:meth:`~.signal.Signal.plot_bss_loadings`

In the case of decomposition plots, you should include as a parameter the number of factors or loadings you wish to visualise, since the default is all. For BSS the default is the number you included when running the :py:meth:`~.learn.mva.MVA.blind_source_separation` method.

Saving and loading results
==========================
There are several methods for storing  the result of a machine learning 
analysis.

Saving in the main file
-------------------------
When you save the object on which you've performed machine learning
analysis in the :ref:`hdf5-format` format (the default in Hyperspy)
(see :ref:`saving_files`) the result of the analysis is automatically saved in
the file and it is loaded with the rest of the data when you load the file.

This option is the simplest because everything is stored in the same file and
it does not require any extra command to recover the result of machine learning
analysis when loading a file. However, currently it only supports storing one
decomposition and one BSS result, which may not be enough for your purposes.

Saving to an external files
---------------------------
Alternatively, to save the results of the current machine learning analysis 
to a file you can use the :py:meth:`~.learn.mva.LearningResults.save` method, e.g.:

.. code-block:: python
    
    >>> # To save the result of the analysis
    >>> s.learning_results.save('my_results')
    
    >>> # To load back the results
    >>> s.learning_results.load('my_results.npz')
    
    
Exporting
---------

It is possible to export the results of machine learning to any format supported
by Hyperspy using:

* :py:meth:`~.signal.Signal.export_decomposition_results` or
* :py:meth:`~.signal.Signal.export_bss_results`.

These methods accept many arguments which can be used to customise the way the data is exported,
so please consult the method documentation. The options include the choice of
file format, the prefixes for loadings and factors, saving figures instead of 
data and more.

Please note that the exported data cannot easily be loaded into Hyperspy's
machine learning structure.





