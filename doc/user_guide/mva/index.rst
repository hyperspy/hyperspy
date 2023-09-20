
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
   one with the dimension of the signal space known as **factors** (:math:`A`),
   and the other with the dimension of the navigation space known as **loadings**
   (:math:`B`), such that :math:`X = A B^T`.

   For some of the algorithms listed below, the decomposition results in
   an `approximation` of the dataset, i.e. :math:`X \approx A B^T`.

.. toctree::
    :maxdepth: 2

    decomposition.rst
    bss.rst
    clustering.rst
    visualize_results.rst
    export_results.rst
