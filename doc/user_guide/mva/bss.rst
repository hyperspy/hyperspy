.. _mva.blind_source_separation:

Blind Source Separation
=======================

In some cases it is possible to obtain more physically interpretable set of
components using a process called Blind Source Separation (BSS). This largely
depends on the particular application. For more information about blind source
separation please see :ref:`[Hyvarinen2000] <Hyvarinen2000>`, and for an
example application to EELS analysis, see :ref:`[Pena2010] <Pena2010>`.

.. warning::

   The BSS algorithms operate on the result of a previous
   decomposition analysis. It is therefore necessary to perform a
   :ref:`decomposition <mva.decomposition>` first before calling
   :meth:`~.api.signals.BaseSignal.blind_source_separation`, otherwise it
   will raise an error.

   You must provide an integer ``number_of_components`` argument,
   or a list of components as the ``comp_list`` argument. This performs
   BSS on the chosen number/list of components from the previous
   decomposition.

To perform blind source separation on the result of a previous decomposition,
run the :meth:`~.api.signals.BaseSignal.blind_source_separation` method, for example:

.. code-block:: python

   >>> s = hs.signals.Signal1D(np.random.randn(10, 10, 200))
   >>> s.decomposition(output_dimension=3)
    Decomposition info:
      normalize_poissonian_noise=False
      algorithm=SVD
      output_dimension=3
      centre=None

   >>> s.blind_source_separation(number_of_components=3)
    Blind source separation info:
      number_of_components=3
      algorithm=sklearn_fastica
      diff_order=1
      reverse_component_criterion=factors
      whiten_method=PCA
    scikit-learn estimator:
    FastICA(tol=1e-10, whiten=False)

   # Perform only on the first and third components
   >>> s.blind_source_separation(comp_list=[0, 2])
    Blind source separation info:
      number_of_components=2
      algorithm=sklearn_fastica
      diff_order=1
      reverse_component_criterion=factors
      whiten_method=PCA
    scikit-learn estimator:
    FastICA(tol=1e-10, whiten=False)


Available algorithms
--------------------

HyperSpy implements a number of BSS algorithms via the ``algorithm`` argument.
The table below lists the algorithms that are currently available, and includes
links to the appropriate documentation for more information on each one.

.. _bss-table:

.. table:: Available blind source separation algorithms in HyperSpy

   +-----------------------------+----------------------------------------------------------------+
   | Algorithm                   | Method                                                         |
   +=============================+================================================================+
   | "sklearn_fastica" (default) | :class:`sklearn.decomposition.FastICA`                         |
   +-----------------------------+----------------------------------------------------------------+
   | "orthomax"                  | :func:`~.learn.orthomax.orthomax`                              |
   +-----------------------------+----------------------------------------------------------------+
   | "FastICA"                   | :class:`mdp.nodes.FastICANode`                                 |
   +-----------------------------+----------------------------------------------------------------+
   | "JADE"                      | :class:`mdp.nodes.JADENode`                                    |
   +-----------------------------+----------------------------------------------------------------+
   | "CuBICA"                    | :class:`mdp.nodes.CuBICANode`                                  |
   +-----------------------------+----------------------------------------------------------------+
   | "TDSEP"                     | :class:`mdp.nodes.TDSEPNode`                                   |
   +-----------------------------+----------------------------------------------------------------+
   | custom object               | An object implementing  ``fit()`` and  ``transform()`` methods |
   +-----------------------------+----------------------------------------------------------------+

.. note::

   Except :func:`~.learn.orthomax.orthomax`, all of the implemented BSS algorithms listed above
   rely on external packages being available on your system. ``sklearn_fastica``, requires
   `scikit-learn <https://scikit-learn.org/>`_ while ``FastICA, JADE, CuBICA, TDSEP``
   require the `Modular toolkit for Data Processing (MDP) <https://mdp-toolkit.github.io/>`_.

.. _mva.orthomax:

Orthomax
--------

Orthomax rotations are a statistical technique used to clarify and highlight the relationship among factors,
by adjusting the coordinates of PCA results. The most common approach is known as
`"varimax" <https://en.wikipedia.org/wiki/Varimax_rotation>`_, which intended to maximize the variance shared
among the components while preserving orthogonality. The results of an orthomax rotation following PCA are
often "simpler" to interpret than just PCA, since each componenthas a more discrete contribution to the data.

.. code-block:: python

   >>> s = hs.signals.Signal1D(np.random.randn(10, 10, 200))
   >>> s.decomposition(output_dimension=3, print_info=False)

   >>> s.blind_source_separation(number_of_components=3, algorithm="orthomax")
    Blind source separation info:
      number_of_components=3
      algorithm=orthomax
      diff_order=1
      reverse_component_criterion=factors
      whiten_method=PCA


.. _mva.ica:

Independent component analysis (ICA)
------------------------------------

One of the most common approaches for blind source separation is
`Independent Component Analysis (ICA) <https://en.wikipedia.org/wiki/Independent_component_analysis>`_.
This separates a signal into subcomponents by assuming that the subcomponents are (a) non-Gaussian,
and (b) that they are statistically independent from each other.

.. _mva.custom_bss:

Custom BSS algorithms
---------------------

As with :ref:`decomposition <mva.decomposition>`, HyperSpy supports passing a custom BSS algorithm,
provided it follows the form of a :external+sklearn:ref:`scikit-learn estimator <develop>`.
Any object that implements ``fit()`` and ``transform()`` methods is acceptable, including
:class:`sklearn.pipeline.Pipeline` and :class:`sklearn.model_selection.GridSearchCV`.
You can access the fitted estimator by passing ``return_info=True``.

.. code-block:: python

   >>> # Passing a custom BSS algorithm
   >>> from sklearn.preprocessing import MinMaxScaler
   >>> from sklearn.pipeline import Pipeline
   >>> from sklearn.decomposition import FastICA

   >>> pipe = Pipeline([("scaler", MinMaxScaler()), ("ica", FastICA())])
   >>> out = s.blind_source_separation(number_of_components=3, algorithm=pipe, return_info=True, print_info=False)

   >>> out
   Pipeline(steps=[('scaler', MinMaxScaler()), ('ica', FastICA())])
