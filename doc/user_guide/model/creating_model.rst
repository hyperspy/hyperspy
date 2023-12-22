.. _creating_model:

Creating a model
----------------

A :class:`~.models.model1d.Model1D` can be created for data in the
:class:`~.api.signals.Signal1D` class using the
:meth:`~.api.signals.Signal1D.create_model` method:

.. code-block:: python

    >>> s = hs.signals.Signal1D(np.arange(300).reshape(30, 10))
    >>> m = s.create_model() # Creates the 1D-Model and assign it to m

Similarly, a :class:`~.models.model2d.Model2D` can be created for data
in the :class:`~.api.signals.Signal2D` class using the
:meth:`~.api.signals.Signal2D.create_model` method:

.. code-block:: python

    >>> im = hs.signals.Signal2D(np.arange(300).reshape(3, 10, 10))
    >>> mod = im.create_model() # Create the 2D-Model and assign it to mod

The syntax for creating both one-dimensional and two-dimensional models is thus
identical for the user in practice. When a model is created  you may be
prompted to provide important information not already included in the
datafile, `e.g.` if ``s`` is EELS data, you may be asked for the accelerating
voltage, convergence and collection semi-angles etc.

.. note::

    * Before creating a model verify that the
      :attr:`~.axes.BaseDataAxis.is_binned` attribute
      of the signal axis is set to the correct value because the resulting
      model depends on this parameter. See :ref:`signal.binned` for more details.
    * When importing data that has been binned using other software, in
      particular Gatan's DM, the stored values may be the averages of the
      binned channels or pixels, instead of their sum, as would be required
      for proper statistical analysis. We therefore cannot guarantee that
      the statistics will be valid, and so strongly recommend that all
      pre-fitting binning is performed using Hyperspy.
