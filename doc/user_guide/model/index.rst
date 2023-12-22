.. _model-label:

Model fitting
*************

HyperSpy can perform curve fitting of one-dimensional signals (spectra) and
two-dimensional signals (images) in `n`-dimensional data sets.
Models are defined by adding individual functions (components in HyperSpy's
terminology) to a :class:`~.model.BaseModel` instance. Those individual
components are then summed to create the final model function that can be
fitted to the data, by adjusting the free parameters of the individual
components.


Models can be created and fit to experimental data in both one and two
dimensions i.e. spectra and images respectively. Most of the syntax is
identical in either case. A one-dimensional model is created when a model
is created for a :class:`~._signals.signal1d.Signal1D` whereas a two-
dimensional model is created for a :class:`~._signals.signal2d.Signal2D`.

.. note::

    Plotting and analytical gradient-based fitting methods are not yet
    implemented for the :class:`~.models.model2d.Model2D` class.

.. toctree::
    :maxdepth: 2

    creating_model.rst
    model_components.rst
    adding_components.rst
    indexing_model.rst
    model_parameters.rst
    fitting_model.rst
    storing_models.rst
    fitting_big_data.rst
    SAMFire.rst
