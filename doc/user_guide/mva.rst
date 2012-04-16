Machine learning
****************
.. warning::

   In version 0.4 the syntax of many of the machine-learning functions
   has changed. It follows a summary of the changes:
   
   * `principal_components_analysis` was renamed to `decomposition`
   * `independent_components_analysis` was renamed to `blind_source_separation`
   * `pca_build_SI` was renamed to `get_decomposition_model`
   * `ica_build_SI` was renamed to `get_bss_model`
   * `plot_lev` was renamed to `plot_explained_variance_ratio`
   * `scores` was renamed to `loadings`
   * `mva_result` was renamed to `learning_results`

Introduction
============

Hyperspy provides easy access to several machine learning algorithms which can
be useful when analysing hyperspectral data. In particular, decomposition 
algorithms such as principal component analysis (PCA) or blind source separation
algorithms such as independent component analysis (ICA) are available through
the methods described in this section.

The behaviour of some Machine Learning functionality can be customised :ref:`customised <configuring-hyperspy-label>` in the Machine Learning section Preferences.

.. _decomposition-nomenclature:

Nomenclature
============

Hyperspy performs the decomposition of a dataset into two datasets:
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


The default algorithm is :py:const:`SVD`, which performs PCA using singular value decomposition. This method has many options. For more details read method documentation.


Poissonian noise
----------------

Most decomposition algorithms assume that the noise of the data follows a
Gaussian distribution. In the case that the data that you are analysing follow
a Poissonian distribution instead Hyperspy can "normalise" the data by
performing a scaling operation which can greatly enhance the result.

To perform Poissonian noise normalisation:

.. code-block:: python

    >>> # The long way:
    >>> s.decomposition(normalize_poissonian_noise = True
    >>> # Because it is the first argument we cold have simple written:
    >>> s.decomposition(True)
    
For more details about the scaling procedure you can read the 
`following research article <http://onlinelibrary.wiley.com/doi/10.1002/sia.1657/abstract>`_


Principal component analysis
----------------------------

.. _scree-plot:

Scree plot
^^^^^^^^^^

When using PCA it is often useful to estimate the dimensionality of the data
by plotting the explained variance against the component index in a
logarithmic y-scale. This plot is sometimes called scree-plot and it should drop
quickly, eventually becoming a slowly descending line. The point at which it
becomes linear (often referred to as an elbow) is generally judged to be a good
estimation of the dimensionality of the data.

To obtain a scree plot, run the :py:meth:`~.learn.mva.MVA.plot_explained_variance_ratio` method e.g.:

.. code-block:: python

    >>> s.plot_explained_variance_ratio()
    
Data denoising
--------------

One of the most popular uses of PCA is data denoising. The denoising property
is achieved by using a limited set of components to make a model of the
original, omitting the later components that ideally contain only noise.

To perform this operation with Hyperspy running the :py:meth:`~.learn.mva.MVA.get_decomposition_model` method, usually after estimating the dimension of your data e.g. by using the :ref:`scree-plot` if your algorithm of choice is PCA. For example:

.. code-block:: python

    >>> sc = s.get_decomposition_model(components)

.. NOTE:: 
    The components argument can be one of several things (None, int,
    or list of ints):

    * if None, all the components are used to construct the model.
    * if int, only the given number of components (starting from index 0) are used to construct the model.
    * if list of ints, only the components in the given list are used to
    construct the model.

.. HINT::
    Unlike most of the analysis functions, this function returns a new
    object.  The new object is something that you have to give a
    handle to, so that you can perform operations on that object later.
    That is why we use the `sc = s.get_decomposition_model(components)`,
    which simply assign the object returned by the :py:meth:`get_decomposition_model` 
    method to the variable :py:const:`sc`. The name of the variable is totally arbitrary
    and therefore you can choose it at your will. The returned object is
    a clone of the original :py:const:`s` object, where the data has been replaced by the
    model constructed using the chosen components.

Sometimes it is useful to examine the residuals between your original
data and the decomposition model. To examine residuals, use the :py:meth:`~.signal.Signal.plot_residual` method on
the reconstructed object, e.g.:

.. code-block:: python

    >>> sc.plot_residual()


Blind Source Separation
=======================

In some cases (it largely depends on the particular application) it is possible
to obtain physically meaninful components from the result of a data
decomposition by Blind Source Separation (BSS).

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

For more information about the blind source separation you can read the 
`following introductory article  <http://www.sciencedirect.com/science/article/pii/S0893608000000265>`_
or `this other article <http://www.sciencedirect.com/science/article/pii/S030439911000255X>`_
from the authors of Hyperspy for an application to EELS analysis.

Visualising results
===================

Plot methods exist for the results of decomposition and blind source separation.
All the methods begin with plot:

* :py:meth:`~.signal.Signal.plot_decomposition_factors`
* :py:meth:`~.signal.Signal.plot_decomposition_loadings`
* :py:meth:`~.signal.Signal.plot_bss_factors`
* :py:meth:`~.signal.Signal.plot_bss_loadings`


Saving and loading results
==========================
There are several methods to store  the result of a machine learning 
analysis.

Saving in the main file
-------------------------
When you save the object on which you've performed machine learning
analysis in the :ref:`hdf5-format` format (the default in Hyperspy)
(see :ref:`saving_files`) the result of the analysis is automatically saved in
the file and it is loaded with the rest of the data when you load the file.

This option is the simplest because everything is stored in the same file and
it does not require any extra command to recover the result of machine learning
analysis when loading a file. However, it only supports storing one
decomposition and one BSS result, what may not be enough for your purposes.

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

These methods accept many arguments to customise the way the data is exported,
so please consult the method documentation. The options include the choice of
file format, the prefixes for loadings and factors, saving figures instead of 
data and more.

Please, note that the exported data cannot be easily be loaded into Hyperspy's
machine learning structure.





