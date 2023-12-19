.. _signal.noise_properties:

Setting the noise properties
----------------------------

Some data operations require the data variance. Those methods use the
``metadata.Signal.Noise_properties.variance`` attribute if it exists. You can
set this attribute as in the following example where we set the variance to be
10:

.. code-block:: python

    >>> s.metadata.Signal.set_item("Noise_properties.variance", 10) # doctest: +SKIP

You can also use the functions :meth:`~.api.signals.BaseSignal.set_noise_variance`
and :meth:`~.api.signals.BaseSignal.get_noise_variance` for convenience:

.. code-block:: python

    >>> s.set_noise_variance(10) # doctest: +SKIP
    >>> s.get_noise_variance() # doctest: +SKIP
    10

For heteroscedastic noise the ``variance`` attribute must be a
:class:`~.api.signals.BaseSignal`.  Poissonian noise is a common case  of
heteroscedastic noise where the variance is equal to the expected value. The
:meth:`~.api.signals.BaseSignal.estimate_poissonian_noise_variance`
method can help setting the variance of data with
semi-Poissonian noise. With the default arguments, this method simply sets the
variance attribute to the given ``expected_value``. However, more generally
(although the noise is not strictly Poissonian), the variance may be
proportional to the expected value. Moreover, when the noise is a mixture of
white (Gaussian) and Poissonian noise, the variance is described by the
following linear model:

    .. math::

        \mathrm{Var}[X] = (a * \mathrm{E}[X] + b) * c

Where `a` is the ``gain_factor``, `b` is the ``gain_offset`` (the Gaussian
noise variance) and `c` the ``correlation_factor``. The correlation
factor accounts for correlation of adjacent signal elements that can
be modelled as a convolution with a Gaussian point spread function.
:meth:`~.api.signals.BaseSignal.estimate_poissonian_noise_variance` can be used to
set the noise properties when the variance can be described by this linear
model, for example:


.. code-block:: python

  >>> s = hs.signals.Signal1D(np.ones(100))
  >>> s.add_poissonian_noise()
  >>> s.metadata
    ├── General
    │   └── title =
    └── Signal
        └── signal_type =

  >>> s.estimate_poissonian_noise_variance()
  >>> s.metadata
    ├── General
    │   └── title =
    └── Signal
        ├── Noise_properties
        │   ├── Variance_linear_model
        │   │   ├── correlation_factor = 1
        │   │   ├── gain_factor = 1
        │   │   └── gain_offset = 0
        │   └── variance = <BaseSignal, title: Variance of , dimensions: (|100)>
        └── signal_type =
