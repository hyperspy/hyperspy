.. _model_indexing-label:

Indexing the model
------------------

Often it is useful to consider only part of the model - for example at
a particular location (i.e. a slice in the navigation space) or energy range
(i.e. a slice in the signal space). This can be done using exactly the same
syntax that we use for signal :ref:`indexing <signal.indexing>`.
:attr:`~.model.BaseModel.red_chisq` and :attr:`~.model.BaseModel.dof`
are automatically recomputed for the resulting slices.

.. code-block:: python

    >>> s = hs.signals.Signal1D(np.arange(100).reshape(10,10))
    >>> m = s.create_model()
    >>> m.append(hs.model.components1D.Gaussian())
    >>> # select first three navigation pixels and last five signal channels
    >>> m1 = m.inav[:3].isig[-5:]
    >>> m1.signal
    <Signal1D, title: , dimensions: (3|5)>
