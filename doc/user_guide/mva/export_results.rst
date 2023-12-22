.. _mva.export:

Export results
==============

Obtain the results as BaseSignal instances
------------------------------------------

The decomposition and BSS results are internally stored as numpy arrays in the
:class:`~.api.signals.BaseSignal` class. Frequently it is useful to obtain the
decomposition/BSS factors and loadings as HyperSpy signals, and HyperSpy
provides the following methods for that purpose:

* :meth:`~.api.signals.BaseSignal.get_decomposition_loadings`
* :meth:`~.api.signals.BaseSignal.get_decomposition_factors`
* :meth:`~.api.signals.BaseSignal.get_bss_loadings`
* :meth:`~.api.signals.BaseSignal.get_bss_factors`

.. _mva.saving-label:

Save and load results
---------------------

Save in the main file
~~~~~~~~~~~~~~~~~~~~~

If you save the dataset on which you've performed machine learning analysis in
the :external+rsciio:ref:`HSpy-HDF5 <hspy-format>` format (the default in HyperSpy, see
:ref:`saving_files`), the result of the analysis is also saved in the same
file automatically, and it is loaded along with the rest of the data when you
next open the file.

.. note::
   This approach currently supports storing one decomposition and one BSS
   result, which may not be enough for your purposes.

Save to an external file
~~~~~~~~~~~~~~~~~~~~~~~~

Alternatively, you can save the results of the current machine learning
analysis to a separate file with the
:meth:`~.learn.mva.LearningResults.save` method:

.. code-block:: python

   >>> # Save the result of the analysis
   >>> s.learning_results.save('my_results.npz') # doctest: +SKIP

   >>> # Load back the results
   >>> s.learning_results.load('my_results.npz') # doctest: +SKIP

Export in different formats
~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also export the results of a machine learning analysis to any format
supported by HyperSpy with the following methods:

* :meth:`~.api.signals.BaseSignal.export_decomposition_results`
* :meth:`~.api.signals.BaseSignal.export_bss_results`

These methods accept many arguments to customise the way in which the
data is exported, so please consult the method documentation. The options
include the choice of file format, the prefixes for loadings and factors,
saving figures instead of data and more.

.. warning::
   Data exported in this way cannot be easily loaded into HyperSpy's
   machine learning structure.
