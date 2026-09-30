
.. _ml-label:

Machine learning
****************

HyperSpy provides easy access to several "machine learning" algorithms that
can be useful when analysing multi-dimensional data. In particular,
decomposition algorithms, such as principal component analysis (PCA), or
blind source separation (BSS) algorithms, such as independent component
analysis (ICA), are available through the methods described in this section.

.. hint::

   HyperSpy will decompose a dataset, :math:`X`, into two new datasets:
   one with the dimension of the signal space known as **decomposition components** (:math:`A`),
   and the other with the dimension of the navigation space known as **scores**
   (:math:`B`), such that :math:`X = A B^T`.

   For some of the algorithms listed below, the decomposition results in
   an `approximation` of the dataset, i.e. :math:`X \approx A B^T`.

.. note::
   "Decomposition components" (the signal-space profiles from PCA, NMF, SVD,
   etc.) are distinct from "model components" (the fitting functions such
   as :class:`~.model.components1D.Gaussian`, :class:`~.model.components1D.Lorentzian`
   used in model fitting).

.. versionchanged:: 2.5.0
   The terms ``factors`` and ``loadings`` have been renamed to
   ``components`` and ``scores`` respectively across the decomposition
   API. The old method names (e.g. ``get_decomposition_factors()``,
   ``plot_decomposition_loadings()``) still work but emit deprecation
   warnings and will be removed in HyperSpy 3.0.

.. toctree::
    :maxdepth: 2

    decomposition.rst
    bss.rst
    clustering.rst
    visualize_results.rst
    export_results.rst
