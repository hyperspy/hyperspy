What's new
**********

We only cover here the main highlights, for a detailed list of all the changes
see `the commits in the GITHUB milestones
<https://github.com/hyperspy/hyperspy/milestones?state=closed>`__.

Current Version
===============

.. _changes_1.6:

v1.6
++++

NEW
---

* Support for the following file formats:

  * :ref:`sur-format`
  * :ref:`elid_format-label`
  * :ref:`nexus-format`
  * :ref:`usid-format`
  * :ref:`empad-format`
  * Prismatic EMD format, see :ref:`emd-format`
* :meth:`~._signals.eels.EELSSpectrum_mixin.print_edges_near_energy` method
  that, if the `hyperspy-gui-ipywidgets package
  <https://github.com/hyperspy/hyperspy_gui_ipywidgets>`_
  is installed, includes an
  awesome interactive mode. See :ref:`eels_elemental_composition-label`.
* Model asymmetric line shape components:

  * :py:class:`~._components.doniach.Doniach`
  * :py:class:`~._components.split_pvoigt.SplitVoigt`
* :ref:`EDS absorption correction <eds_absorption-label>`.
* :ref:`Argand diagram for complex signals <complex.argand>`.
* :ref:`Multiple peak finding algorithms for 2D signals <peak_finding-label>`.
* :ref:`cluster_analysis-label`.

Enhancements
------------

* The :py:meth:`~.signal.BaseSignal.get_histogram` now uses numpy's
  `np.histogram_bin_edges()
  <https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html>`_
  and supports all of its ``bins`` keyword values.
* Further improvements to the contrast adjustment tool.
  Test it by pressing the ``h`` key on any image.
* The following components have been rewritten using
  :py:class:`hyperspy._components.expression.Expression`, boosting their
  speeds among other benefits.

  * :py:class:`hyperspy._components.arctan.Arctan`
  * :py:class:`hyperspy._components.voigt.Voigt`
  * :py:class:`hyperspy._components.heaviside.HeavisideStep`
* The model fitting :py:meth:`~.model.BaseModel.fit` and
  :py:meth:`~.model.BaseModel.multifit` methods have been vastly improved. See
  :ref:`model.fitting` and the API changes section below.
* New serpentine iteration path for multi-dimensional fitting.
  See :ref:`model.multidimensional-label`.
* The :py:func:`~.drawing.utils.plot_spectra`  function now listens to
  events to update the figure automatically.
  See :ref:`this example <plot_profiles_interactive-label>`.
* Improve thread-based parallelism. Add ``max_workers`` argument to the
  :py:meth:`~.signal.BaseSignal.map` method, such that the user can directly
  control how many threads they launch.
* Many improvements to the :py:meth:`~.mva.MVA.decomposition` and
  :py:meth:`~.mva.MVA.blind_source_separation` methods, including support for
  scikit-learn like algorithms, better API and much improved documentation.
  See :ref:`ml-label` and the API changes section below.
* Add option to calculate the absolute thickness to the EELS
  :meth:`~._signals.eels.EELSSpectrum_mixin.estimate_thickness` method.
  See :ref:`eels_thickness-label`.
* Vastly improved performance and memory footprint of the
  :py:meth:`~._signals.signal2d.Signal2D.estimate_shift2D` method.
* The :py:meth:`~._signals.signal1d.Signal1D.remove_background` method can
  now remove Doniach, exponential, Lorentzian, skew normal,
  split Voigt and Voigt functions. Furthermore, it can return the background
  model that includes an estimation of the reduced chi-squared.
* The performance of the maximum-likelihood PCA method was greatly improved.
* All ROIs now have a ``__getitem__`` method, enabling e.g. using them with the
  unpack ``*`` operator. See :ref:`roi-slice-label` for an example.
* New syntax to set the contrast when plotting images. In particular, the
  ``vmin`` and ``vmax`` keywords now take values like ``vmin="30th"`` to
  clip the minimum value to the 30th percentile. See :ref:`signal.fft`
  for an example.
* The :py:meth:`~._signals.signal1d.Signal1D.plot` and
  :py:meth:`~._signals.signal2d.Signal2D.plot` methods take a new keyword
  argument ``autoscale``. See :ref:`plot.customize_images` for details.
* The contrast editor and the decomposition methods can now operate on
  complex signals.
* The default colormap can now be set in
  :ref:`preferences <configuring-hyperspy-label>`.


API changes
-----------

* The :py:meth:`~._signals.signal2d.Signal2D.plot` keyword argument
  ``saturated_pixels`` is deprecated. Please use
  ``vmin`` and/or ``vmax`` instead.
* The :py:func:`~.io.load` keyword argument ``dataset_name`` has been
  renamed to ``dataset_path``.
* The :py:meth:`~.signal.BaseSignal.set_signal_type` method no longer takes
  ``None``. Use the empty string ``""`` instead.
* The :py:meth:`~.signal.BaseSignal.get_histogram` ``bins`` keyword values
  have been renamed as follows for consistency with numpy:

    * ``"scotts"`` -> ``"scott"``,
    * ``"freedman"`` -> ``"fd"``
*  Multiple changes to the syntax of the :py:meth:`~.model.BaseModel.fit`
   and :py:meth:`~.model.BaseModel.multifit` methods:

  * The ``fitter`` keyword has been renamed to ``optimizer``.
  * The values that the ``optimizer`` keyword take have been renamed
    for consistency with scipy:

    * ``"fmin"`` -> ``"Nelder-Mead"``,
    * ``"fmin_cg"`` -> ``"CG"``,
    * ``"fmin_ncg"`` -> ``"Newton-CG"``,
    * ``"fmin_bfgs"`` -> ``"BFGS"``,
    * ``"fmin_l_bfgs_b"`` -> ``"L-BFGS-B"``,
    * ``"fmin_tnc"`` -> ``"TNC"``,
    * ``"fmin_powell"`` -> ``"Powell"``,
    * ``"mpfit"`` -> ``"lm"`` (in combination with ``"bounded=True"``),
    * ``"leastsq"`` -> ``"lm"``,

  * Passing integer arguments to ``parallel`` to select the number of
    workers is now deprecated. Use ``parallel=True, max_workers={value}``
    instead.
  * The ``method`` keyword has been renamed to ``loss_function``.
  * The ``loss_function`` value ``"ml"`` has been renamed to ``"ML-poisson"``.
  * The ``grad`` keyword no longer takes boolean values. It takes the
    following values instead: ``"fd"``, ``"analytical"``, callable or ``None``.
  * The ``ext_bounding`` keyword has been deprecated and will be removed. Use
    ``bounded=True`` instead.
  * The ``min_function`` keyword argument has been deprecated and will
    be removed. Use ``loss_function`` instead.,
  * The ``min_function_grad`` keyword arguments has been deprecated and will be
    removed. Use ``grad`` instead.
  * The ``iterpath`` default will change from ``'flyback'`` to
    ``'serpentine'`` in HyperSpy version 2.0.

* The following :py:class:`~.model.BaseModel` methods are now private:

  * :py:meth:`~.model.BaseModel.set_boundaries`
  * :py:meth:`~.model.BaseModel.set_mpfit_parameters_info`
  * :py:meth:`~.model.BaseModel.set_boundaries`

* The ``comp_label`` keyword of the machine learning plotting functions
  has been renamed to ``title``.
* The :py:class:`~.learn.rpca.orpca` constructor's ``learning_rate``
  keyword has been renamed to ``subspace_learning_rate``
* The :py:class:`~.learn.rpca.orpca` constructor's ``momentum``
  keyword has been renamed to ``subspace_momentum``
* The :py:class:`~.learn.svd_pca.svd_pca` constructor's ``centre`` keyword
  values have been renamed as follows:

    * ``"trials"`` -> ``"navigation"``
    * ``"variables"`` -> ``"signal"``
* The ``bounds`` keyword argument of the
  :py:meth:`~._signals.lazy.decomposition` is deprecated and will be removed.
* Several syntax changes in the :py:meth:`~.learn.mva.decomposition` method:

  * Several ``algorithm`` keyword values have been renamed as follows:

    * ``"svd"``: ``"SVD"``,
    * ``"fast_svd"``: ``"SVD"``,
    * ``"nmf"``: ``"NMF"``,
    * ``"fast_mlpca"``: ``"MLPCA"``,
    * ``"mlpca"``: ``"MLPCA"``,
    * ``"RPCA_GoDec"``: ``"RPCA"``,
  * The ``polyfit`` argument has been deprecated and will be removed.
    Use ``var_func`` instead.




Changelog
*********

Previous Versions
=================

.. _changes_1.5.2:


v1.5.2
++++++

This is a maintenance release that adds compatibility with Numpy 1.17 and Dask
2.3.0 and fixes a bug in the Bruker reader. See `the issue tracker
<https://github.com/hyperspy/hyperspy/issues?q=label%3A"type%3A+bug"+is%3Aclosed+milestone%3Av1.5.2>`__
for details.


.. _changes_1.5.1:

v1.5.1
++++++

This is a maintenance release that fixes some regressions introduced in v1.5.
Follow the following links for details on all the `bugs fixed
<https://github.com/hyperspy/hyperspy/issues?q=label%3A"type%3A+bug"+is%3Aclosed+milestone%3Av1.5.1>`__.

.. _changes_1.5:

v1.5
++++

NEW
---

* New method :py:meth:`hyperspy.component.Component.print_current_values`. See
  :ref:`the User Guide for details <Component.print_current_values>`.
* New :py:class:`hyperspy._components.skew_normal.SkewNormal` component.
* New :py:meth:`hyperspy.signal.BaseSignal.apply_apodization` method and
  ``apodization`` keyword for :py:meth:`hyperspy.signal.BaseSignal.fft`. See
  :ref:`signal.fft` for details.
* Estimation of number of significant components by the elbow method.
  See :ref:`mva.scree_plot`.

Enhancements
------------

* The contrast adjustment tool has been hugely improved. Test it by pressing the ``h`` key on any image.
* The :ref:`Developer Guide <dev_guide-label>` has been extended, enhanced and divided into
  chapters.
* Signals with signal dimension equal to 0 and navigation dimension 1 or 2 are
  automatically transposed when using
  :py:func:`hyperspy.drawing.utils.plot_images`
  or :py:func:`hyperspy.drawing.utils.plot_spectra` respectively. This is
  specially relevant when plotting the result of EDS quantification. See
  :ref:`eds-label` for examples.
* The following components have been rewritten using
  :py:class:`hyperspy._components.expression.Expression`, boosting their
  speeds among other benefits. Multiple issues have been fixed on the way.

  * :py:class:`hyperspy._components.lorentzian.Lorentzian`
  * :py:class:`hyperspy._components.exponential.Exponential`
  * :py:class:`hyperspy._components.bleasdale.Bleasdale`
  * :py:class:`hyperspy._components.rc.RC`
  * :py:class:`hyperspy._components.logistic.Logistic`
  * :py:class:`hyperspy._components.error_function.Erf`
  * :py:class:`hyperspy._components.gaussian2d.Gaussian2D`
  * :py:class:`hyperspy._components.volume_plasmon_drude.VolumePlasmonDrude`
  * :py:class:`hyperspy._components.eels_double_power_law.DoublePowerLaw`
  * The :py:class:`hyperspy._components.polynomial_deprecated.Polynomial`
    component will be deprecated in HyperSpy 2.0 in favour of the new
    :py:class:`hyperspy._components.polynomial.Polynomial` component, that is based on
    :py:class:`hyperspy._components.expression.Expression` and has an improved API. To
    start using the new component pass the ``legacy=False`` keyword to the
    the :py:class:`hyperspy._components.polynomial_deprecated.Polynomial` component
    constructor.


For developers
--------------
* Drop support for python 3.5
* New extension mechanism that enables external packages to register HyperSpy
  objects. See :ref:`writing_extensions-label` for details.


.. _changes_1.4.2:

v1.4.2
++++++

This is a maintenance release. Among many other fixes and enhancements, this
release fixes compatibility issues with Matplotlib v 3.1. Follow the
following links for details on all the `bugs fixed
<https://github.com/hyperspy/hyperspy/issues?q=label%3A"type%3A+bug"+is%3Aclosed+milestone%3Av1.4.2>`__
and `enhancements
<https://github.com/hyperspy/hyperspy/issues?q=is%3Aclosed+milestone%3Av1.4.2+label%3A"type%3A+enhancement">`__.


.. _changes_1.4.1:

v1.4.1
++++++

This is a maintenance release. Follow the following links for details on all
the `bugs fixed
<https://github.com/hyperspy/hyperspy/issues?q=label%3A"type%3A+bug"+is%3Aclosed+milestone%3Av1.4.1>`__
and `enhancements
<https://github.com/hyperspy/hyperspy/issues?q=is%3Aclosed+milestone%3Av1.4.1+label%3A"type%3A+enhancement">`__.

This release fixes compatibility issues with Python 3.7.


.. _changes_1.4:

v1.4
++++

This is a minor release. Follow the following links for details on all
the `bugs fixed
<https://github.com/hyperspy/hyperspy/issues?utf8=%E2%9C%93&q=is%3Aclosed+milestone%3Av1.4+label%3A%22type%3A+bug%22+>`__,
`enhancements
<https://github.com/hyperspy/hyperspy/issues?q=is%3Aclosed+milestone%3Av1.4+label%3A%22type%3A+enhancement%22>`__
and `new features
<https://github.com/hyperspy/hyperspy/issues?q=is%3Aclosed+milestone%3Av1.4+label%3A%22type%3A+New+feature%22>`__.

NEW
---

* Support for three new file formats:

    * Reading FEI's Velox EMD file format based on the HDF5 open standard. See :ref:`emd_fei-format`.
    * Reading Bruker's SPX format. See :ref:`spx-format`.
    * Reading and writing the mrcz open format. See :ref:`mrcz-format`.
* New :mod:`~.datasets.artificial_data` module which contains functions for generating
  artificial data, for use in things like docstrings or for people to test
  HyperSpy functionalities. See :ref:`example-data-label`.
* New :meth:`~.signal.BaseSignal.fft` and :meth:`~.signal.BaseSignal.ifft` signal methods. See :ref:`signal.fft`.
* New :meth:`~._signals.hologram_image.HologramImage.statistics` method to compute useful hologram parameters. See :ref:`holography.stats-label`.
* Automatic axes units conversion and better units handling using `pint <https://pint.readthedocs.io/en/latest/>`__.
  See :ref:`quantity_and_converting_units`.
* New :class:`~.roi.Line2DROI` :meth:`~.roi.Line2DROI.angle` method. See :ref:`roi-label` for details.

Enhancements
------------

* :py:func:`~.drawing.utils.plot_images` improvements (see :ref:`plot.images` for details):

    * The ``cmap`` option of :py:func:`~.drawing.utils.plot_images`
      supports iterable types, allowing the user to specify different colormaps
      for the different images that are plotted by providing a list or other
      generator.
    * Clicking on an individual image updates it.
* New customizable keyboard shortcuts to navigate multi-dimensional datasets. See :ref:`visualization-label`.
* The :py:meth:`~._signals.signal1d.Signal1D.remove_background` method now operates much faster
  in multi-dimensional datasets and adds the options to interatively plot the remainder of the operation and
  to set the removed background to zero. See :ref:`signal1D.remove_background` for details.
* The  :py:meth:`~._signals.Signal2D.plot` method now takes a ``norm`` keyword that can be "linear", "log",
  "auto"  or a matplotlib norm. See :ref:`plot.customize_images` for details.
  Moreover, there are three new extra keyword
  arguments, ``fft_shift`` and ``power_spectrum``, that are useful when plotting fourier transforms. See
  :ref:`signal.fft`.
* The :py:meth:`~._signals.signal2d.Signal2D.align2D` and :py:meth:`~._signals.signal2d.Signal2D.estimate_shift2D`
  can operate with sub-pixel accuracy using skimage's upsampled matrix-multiplication DFT. See :ref:`signal2D.align`.


.. _changes_1.3.2:

v1.3.2
++++++

This is a maintenance release. Follow the following links for details on all
the `bugs fixed
<https://github.com/hyperspy/hyperspy/issues?q=label%3A"type%3A+bug"+is%3Aclosed+milestone%3Av1.3.2>`__
and `enhancements <https://github.com/hyperspy/hyperspy/issues?q=is%3Aclosed+milestone%3Av1.3.2+label%3A"type%3A+enhancement">`__.


.. _changes_1.3.1:

v1.3.1
++++++

This is a maintenance release. Follow the following links for details on all
the `bugs fixed
<https://github.com/hyperspy/hyperspy/issues?q=label%3A"type%3A+bug"+is%3Aclosed+milestone%3Av1.3.1>`__
and `enhancements <https://github.com/hyperspy/hyperspy/issues?q=is%3Aclosed+milestone%3Av1.3.1+label%3A"type%3A+enhancement">`__.

Starting with this version, the HyperSpy WinPython Bundle distribution is
no longer released in sync with HyperSpy. For HyperSpy WinPython Bundle
releases see https://github.com/hyperspy/hyperspy-bundle


.. _changes_1.3:

v1.3
++++

This is a minor release. Follow the following links for details on all
the `bugs fixed
<https://github.com/hyperspy/hyperspy/issues?q=label%3A"type%3A+bug"+is%3Aclosed+milestone%3Av1.3>`__,
`feature
<https://github.com/hyperspy/hyperspy/issues?q=is%3Aclosed+milestone%3Av1.3+label%3A"type%3A+enhancement">`__
and `documentation
<https://github.com/hyperspy/hyperspy/issues?utf8=%E2%9C%93&q=is%3Aclosed%20milestone%3Av1.3%20label%3A%22affects%3A%20documentation%22%20>`__ enhancements,
and `new features
<https://github.com/hyperspy/hyperspy/issues?q=is%3Aclosed+milestone%3Av1.3+label%3A"type%3A+New+feature">`__.

NEW
---
* :py:meth:`~.signal.BaseSignal.rebin` supports upscaling and rebinning to
  arbitrary sizes through linear interpolation. See :ref:`rebin-label`. It also runs faster if `numba <http://numba.pydata.org/>`__ is installed.
* :py:attr:`~.axes.AxesManager.signal_extent` and :py:attr:`~.axes.AxesManager.navigation_extent` properties to easily get the extent of each space.
* New IPywidgets Graphical User Interface (GUI) elements for the `Jupyter Notebook <http://jupyter.org>`__.
  See the new `hyperspy_gui_ipywidgets <https://github.com/hyperspy/hyperspy_gui_ipywidgets>`__ package.
  It is not installed by default, see :ref:`install-label` for details.
* All the :ref:`roi-label` now have a :meth:`gui` method to display a GUI if
  at least one of HyperSpy's GUI packgages are installed.

Enhancements
------------
* Creating many markers is now much faster.
* New "Stage" metadata node. See :ref:`metadata_structure` for details.
* The Brucker file reader now supports the new version of the format. See :ref:`bcf-format`.
* HyperSpy is now compatible with all matplotlib backends, including the nbagg which is
  particularly convenient for interactive data analysis in the
  `Jupyter Notebook <http://jupyter.org>`__ in combination with the new
  `hyperspy_gui_ipywidgets <https://github.com/hyperspy/hyperspy_gui_ipywidgets>`__ package.
  See :ref:`importing_hyperspy-label`.
* The ``vmin`` and ``vmax`` arguments of the
  :py:func:`~.drawing.utils.plot_images` function now accept lists to enable
  setting these parameters for each plot individually.
* The :py:meth:`~.signal.MVATools.plot_decomposition_results` and
  :py:meth:`~.signal.MVATools.plot_bss_results` methods now makes a better
  guess of the number of navigators (if any) required to visualise the
  components. (Previously they were always plotting four figures by default.)
* All functions that take a signal range can now take a :py:class:`~.roi.SpanROI`.
* The following ROIs can now be used for indexing or slicing (see :ref:`here <roi-slice-label>` for details):

    * :py:class:`~.roi.Point1DROI`
    * :py:class:`~.roi.Point2DROI`
    * :py:class:`~.roi.SpanROI`
    * :py:class:`~.roi.RectangularROI`


API changes
-----------
* Permanent markers (if any) are now displayed when plotting by default.
* HyperSpy no longer depends on traitsui (fixing many installation issues) and
  ipywidgets as the GUI elements based on these packages have now been splitted
  into separate packages and are not installed by default.
* The following methods now raise a ``ValueError`` when not providing the
  number of components if ``output_dimension`` was not specified when
  performing a decomposition. (Previously they would plot as many figures
  as available components, usually resulting in memory saturation):

    * :py:meth:`~.signal.MVATools.plot_decomposition_results`.
    * :py:meth:`~.signal.MVATools.plot_decomposition_factors`.

* The default extension when saving to HDF5 following HyperSpy's specification
  is now ``hspy`` instead of ``hdf5``. See :ref:`hspy-format`.

* The following methods are deprecated and will be removed in HyperSpy 2.0

    * :py:meth:`~.axes.AxesManager.show`. Use :py:meth:`~.axes.AxesManager.gui`
      instead.
    * All :meth:`notebook_interaction` method. Use the equivalent :meth:`gui` method
      instead.
    * :py:meth:`~._signals.signal1d.Signal1D.integrate_in_range`.
      Use :py:meth:`~._signals.signal1d.Signal1D.integrate1D` instead.

* The following items have been removed from
  :ref:`preferences <configuring-hyperspy-label>`:

    * ``General.default_export_format``
    * ``General.lazy``
    * ``Model.default_fitter``
    * ``Machine_learning.multiple_files``
    * ``Machine_learning.same_window``
    * ``Plot.default_style_to_compare_spectra``
    * ``Plot.plot_on_load``
    * ``Plot.pylab_inline``
    * ``EELS.fine_structure_width``
    * ``EELS.fine_structure_active``
    * ``EELS.fine_structure_smoothing``
    * ``EELS.synchronize_cl_with_ll``
    * ``EELS.preedge_safe_window_width``
    * ``EELS.min_distance_between_edges_for_fine_structure``

* New ``Preferences.GUIs`` section to enable/disable the installed GUI toolkits.

For developers
--------------
* In addition to adding ipywidgets GUI elements, the traitsui GUI elements have
  been splitted into a separate package. See the new
  `hyperspy_gui_traitsui <https://github.com/hyperspy/hyperspy_gui_traitsui>`__
  package.
* The new :py:mod:`~.ui_registry` enables easy connection of external
  GUI elements to HyperSpy. This is the mechanism used to split the traitsui
  and ipywidgets GUI elements.


.. _changes_1.2:

v1.2
++++

This is a minor release. Follow the following links for details on all
the `bugs fixed
<https://github.com/hyperspy/hyperspy/issues?q=label%3A"type%3A+bug"+is%3Aclosed+milestone%3Av1.2>`__,
`enhancements
<https://github.com/hyperspy/hyperspy/issues?q=is%3Aclosed+milestone%3Av1.2+label%3A"type%3A+enhancement">`__
and `new features
<https://github.com/hyperspy/hyperspy/issues?q=is%3Aclosed+milestone%3Av1.2+label%3A"type%3A+New+feature">`__.

NEW
---

* Lazy loading and evaluation. See :ref:`big-data-label`.
* Parallel :py:meth:`~.signal.BaseSignal.map` and all the functions that use
  it internally (a good fraction of HyperSpy's functionaly). See
  :ref:`map-label`.
* :ref:`electron-holography-label` reconstruction.
* Support for reading :ref:`edax-format` files.
* New signal methods :py:meth:`~.signal.BaseSignal.indexmin` and
  :py:meth:`~.signal.BaseSignal.valuemin`.

Enhancements
------------
* Easier creation of :py:class:`~._components.expression.Expression` components
  using substitutions. See the
  :ref:`User Guide for details <expression_component-label>`.
* :py:class:`~._components.expression.Expression` takes two dimensional
  functions that can automatically include a rotation parameter. See the
  :ref:`User Guide for details <expression_component-label>`.
* Better support for EMD files.
* The scree plot got a beauty treatment and some extra features. See
  :ref:`mva.scree_plot`.
* :py:meth:`~.signal.BaseSignal.map` can now take functions that return
  differently-shaped arrays or arbitrary objects, see :ref:`map-label`.
* Add support for stacking multi-signal files. See :ref:`load-multiple-label`.
* Markers can now be saved to hdf5 and creating many markers is easier and
  faster. See :ref:`plot.markers`.
* Add option to save to HDF5 file using the ".hspy" extension instead of
  ".hdf5". See :ref:`hspy-format`. This will be the default extension in
  HyperSpy 1.3.

For developers
--------------
* Most of HyperSpy plotting features are now covered by unittests. See
  :ref:`plot-test-label`.
* unittests migrated from nose to pytest. See :ref:`testing-label`.


.. _changes_1.1.2:

v1.1.2
++++++

This is a maintenance release. Follow the following links for details on all
the `bugs fixed
<https://github.com/hyperspy/hyperspy/issues?q=label%3A"type%3A+bug"+is%3Aclosed+milestone%3Av1.1.2>`__
and `enhancements <https://github.com/hyperspy/hyperspy/issues?q=is%3Aclosed+milestone%3Av1.1.2+label%3A"type%3A+enhancement">`__.


.. _changes_1.1.1:

v1.1.1
++++++

This is a maintenance release. Follow the following link for details on all
the `bugs fixed
<https://github.com/hyperspy/hyperspy/issues?q=label%3A"type%3A+bug"+is%3Aclosed+milestone%3A1.1.1>`__.

Enhancements
------------

* Prettier X-ray lines labels.
* New metadata added to the HyperSpy metadata specifications: ``magnification``,
  ``frame_number``, ``camera_length``, ``authors``, ``doi``, ``notes`` and
  ``quantity``. See :ref:`metadata_structure` for details.
* The y-axis label (for 1D signals) and colorbar label (for 2D signals)
  are now taken from the new ``metadata.Signal.quantity``.
* The ``time`` and ``date`` metadata are now stored in the ISO 8601 format.
* All metadata in the HyperSpy metadata specification is now read from all
  supported file formats when available.

.. _changes_1.1:

v1.1
++++

This is a minor release. Follow the following links for details on all
the `bugs fixed
<https://github.com/hyperspy/hyperspy/issues?q=label%3A"type%3A+bug"+is%3Aclosed+milestone%3A1.1>`__.

NEW
---

* :ref:`signal.transpose`.
* :ref:`protochips-format` reader.

Enhancements
------------


* :py:meth:`~.model.BaseModel.fit` takes a new algorithm, the global optimizer
  `differential evolution`.
* :py:meth:`~.model.BaseModel.fit` algorithm, `leastsq`, inherits SciPy's bound
  constraints support (requires SciPy >= 0.17).
* :py:meth:`~.model.BaseModel.fit` algorithm names changed to be consistent
  `scipy.optimze.minimize()` notation.



v1.0.1
++++++

This is a maintenance release. Follow the following links for details on all
the `bugs fixed
<https://github.com/hyperspy/hyperspy/issues?q=label%3A"type%3A+bug"+is%3Aclosed+milestone%3A1.0.1>`__.


v1.0
++++

This is a major release. Here we only list the highlist. A detailed list of
changes `is available in github
<https://github.com/hyperspy/hyperspy/issues?q=is%3Aclosed+milestone%3A1.0.0>`__.

NEW
---

* :ref:`roi-label`.
* :ref:`Robust PCA <mva.rpca>` (RPCA) and online RPCA algorithms.
* Numpy ufuncs can now :ref:`operate on HyperSpy's signals <ufunc-label>`.
* ComplexSignal and specialised subclasses to :ref:`operate on complex data <complex_data-label>`.
* Events :ref:`logging <logger-label>`.
* Query and :ref:`fetch spectra <eelsdb-label>` from `The EELS Database <https://eelsdb.eu/>`__.
* :ref:`interactive-label`.
* :ref:`events-label`.

Model
^^^^^

* :ref:`SAMFire-label`.
* Store :ref:`models in hdf5 files <storing_models-label>`.
* Add :ref:`fancy indexing <model_indexing-label>` to `Model`.
* :ref:`Two-dimensional model fitting <2D_model-label>`.

EDS
^^^
* :ref:`Z-factors quantification <eds_quantification-label>`.
* :ref:`Cross section quantification <eds_quantification-label>`.
* :ref:`EDS curve fitting <eds_fitting-label>`.
* X-ray :ref:`absorption coefficient database <eds_absorption_db-label>`.

IO
^^
* Support for reading certain files without :ref:`loading them to memory <load_to_memory-label>`.
* :ref:`Bruker's composite file (bcf) <bcf-format>` reading support.
* :ref:`Electron Microscopy Datasets (EMD) <emd-format>` read and write support.
* :ref:`SEMPER unf <unf-format>` read and write support.
* :ref:`DENS heat log <dens-format>` read support.
* :ref:`NanoMegas blockfile <blockfile-format>` read and write support.

Enhancements
------------
* More useful ``AxesManager`` repr string with html repr for Jupyter Notebook.
* Better progress bar (`tqdm <https://github.com/noamraph/tqdm>`__).
* Add support for :ref:`writing/reading scale and unit to tif files
  <tiff-format>` to be read with ImageJ or DigitalMicrograph.

Documentation
-------------

* The following sections of the User Guide were revised and largely overwritten:

  * :ref:`install-label`.
  * :ref:`ml-label`.
  * :ref:`eds-label`.
* New :ref:`dev_guide-label`.


API changes
-----------

* Split :ref:`components <model_components-label>` into `components1D` and `components2D`.
* Remove `record_by` from metadata.
* Remove simulation classes.
* The :py:class:`~._signals.signal1D.Signal1D`,
  :py:class:`~._signals.image.Signal2D` and :py:class:`~.signal.BaseSignal`
  classes deprecated the old `Spectrum` `Image` and `Signal` classes.



v0.8.5
++++++


This is a maintenance release. Follow the following links for details on all
the `bugs fixed
<https://github.com/hyperspy/hyperspy/issues?q=is%3Aissue+milestone%3A0.8.5+label%3A"type%3A+bug"+is%3Aclosed>`__,
`feature <https://github.com/hyperspy/hyperspy/issues?utf8=%E2%9C%93&q=milestone%3A0.8.5+is%3Aclosed++label%3A"type%3A+enhancement"+>`__
and `documentation
<https://github.com/hyperspy/hyperspy/pulls?utf8=%E2%9C%93&q=milestone%3A0.8.5+label%3Adocumentation+is%3Aclosed+>`__ enhancements.


It also includes a new feature and introduces an important API change that
will be fully enforced in Hyperspy 1.0.

New feature
-----------

* Widgets to interact with the model components in the Jupyter Notebook.
  See :ref:`here <notebook_interaction-label>` and
  `#1007 <https://github.com/hyperspy/hyperspy/pull/1007>`__ .

API changes
-----------

The new :py:class:`~.signal.BaseSignal`,
:py:class:`~._signals.signal1d.Signal1D` and
:py:class:`~._signals.signal2d.Signal2D` deprecate :py:class:`~.signal.Signal`,
:py:class:`~._signals.signal1D.Signal1D` and :py:class:`~._signals.image.Signal2D`
respectively. Also `as_signal1D`, `as_signal2D`, `to_signal1D` and `to_signal2D`
deprecate `as_signal1D`, `as_signal2D`, `to_spectrum` and `to_image`. See `#963
<https://github.com/hyperspy/hyperspy/pull/963>`__ and `#943
<https://github.com/hyperspy/hyperspy/issues/943>`__ for details.


v0.8.4
++++++

This release adds support for Python 3 and drops support for Python 2. In all
other respects it is identical to v0.8.3.

v0.8.3
++++++

This is a maintenance release that includes fixes for multiple bugs, some
enhancements, new features and API changes. This is set to be the last HyperSpy
release for Python 2. The release (HyperSpy 0.8.4) will support only Python 3.

Importantly, the way to start HyperSpy changes (again) in this release. Please
read carefully :ref:`importing_hyperspy-label` for details.

The broadcasting rules have also changed. See :ref:`signal.operations`
for details.

Follow the following links for details on all the `bugs fixed
<https://github.com/hyperspy/hyperspy/issues?page=1&q=is%3Aclosed+milestone%3A0.8.3+label%3A"type%3A+bug"&utf8=%E2%9C%93>`__,
`documentation enhancements
<https://github.com/hyperspy/hyperspy/issues?q=is%3Aclosed+milestone%3A0.8.3+label%3Adocumentation>`__,
`enhancements
<https://github.com/hyperspy/hyperspy/issues?q=is%3Aclosed+milestone%3A0.8.3+label%3A"type%3A+enhancement">`__,
`new features
<https://github.com/hyperspy/hyperspy/issues?q=is%3Aclosed+milestone%3A0.8.3+label%3ANew>`__
`and API changes
<https://github.com/hyperspy/hyperspy/issues?q=is%3Aclosed+milestone%3A0.8.3+label%3A"Api+change">`__


.. _changes_0.8.2:

v0.8.2
++++++

This is a maintenance release that fixes an issue with the Python installers. Those who have successfully installed v0.8.1 do not need to upgrade.

.. _changes_0.8.1:

v0.8.1
++++++

This is a maintenance release. Follow the following links for details on all
the `bugs fixed
<https://github.com/hyperspy/hyperspy/issues?page=1&q=is%3Aclosed+milestone%3A0.8.1+label%3A"type%3A+bug"&utf8=%E2%9C%93>`__,
`feature
<https://github.com/hyperspy/hyperspy/issues?utf8=%E2%9C%93&q=is%3Aclosed+milestone%3A0.8.1++label%3A"type%3A+enhancement"+>`__
and `documentation
<https://github.com/hyperspy/hyperspy/issues?q=is%3Aclosed+milestone%3A0.8.1+label%3Adocumentation>`__ enhancements.

Importantly, the way to start HyperSpy changes in this release. Read :ref:`importing_hyperspy-label` for details.

It also includes some new features and introduces important API changes that
will be fully enforced in Hyperspy 1.0.

New features
------------
* Support for IPython 3.0.
* ``%hyperspy`` IPython magic to easily and transparently import HyperSpy, matplotlib and numpy when using IPython.
* :py:class:`~._components.expression.Expression` model component to easily create analytical function components. More details
  :ref:`here <expression_component-label>`.
* :py:meth:`~.signal.Signal.unfolded` context manager.
* :py:meth:`~.signal.Signal.derivative` method.
* :ref:`syntax to access the components in the model <model_components-label>`
  that includes pretty printing of the components.

API changes
-----------

* :py:mod:`~.hyperspy.hspy` is now deprecated in favour of the new
  :py:mod:`~.hyperspy.api`. The new API renames and/or move several modules as
  folows:

    * ``hspy.components`` -> ``api.model.components``
    * ``hspy.utils``-> ``api``
    * ``hspy.utils.markers`` ``api.plot.markers``
    * ``hspy.utils.example_signals`` -> ``api.datasets.example_signals``


    In HyperSpy 0.8.1 the full content of :py:mod:`~.hyperspy.hspy` is still
    imported in the user namespace, but this can now be disabled in
    ``hs.preferences.General.import_hspy``. In Hyperspy 1.0 it will be
    disabled by default and the :py:mod:`~.hyperspy.hspy` module will be fully
    removed in HyperSpy 0.10. We encourage all users to migrate to the new
    syntax. For more details see :ref:`importing_hyperspy-label`.
* Indexing the :py:class:`~.signal.Signal` class is now deprecated. We encourage
  all users to use ``isig`` and ``inav`` instead for indexing.
* :py:func:`~.hyperspy.hspy.create_model` is now deprecated in favour of the new
  equivalent :py:meth:`~.signal.Signal.create_model` ``Signal`` method.
* :py:meth:`~.signal.Signal.unfold_if_multidim` is deprecated.


.. _changes_0.8:

v0.8
++++

New features
------------

Core
^^^^

* :py:meth:`~._signals.signal1D.Signal1D.spikes_removal_tool` displays derivative max value when used with
  GUI.
* Progress-bar can now be suppressed by passing ``show_progressbar`` argument to all functions that generate
  it.

IO
^^

* HDF5 file format now supports saving lists, tuples, binary strings and signals in metadata.


Plotting
^^^^^^^^

* New class,  :py:class:`~.drawing.marker.MarkerBase`, to plot markers with ``hspy.utils.plot.markers`` module.  See :ref:`plot.markers`.
* New method to plot images with the :py:func:`~.drawing.utils.plot_images` function in  ``hspy.utils.plot.plot_images``. See :ref:`plot.images`.
* Improved :py:meth:`~._signals.image.Signal2D.plot` method to customize the image. See :ref:`plot.customize_images`.

EDS
^^^

* New method for quantifying EDS TEM spectra using Cliff-Lorimer method, :py:meth:`~._signals.eds_tem.EDSTEMSpectrum.quantification`. See :ref:`eds_quantification-label`.
* New method to estimate for background subtraction, :py:meth:`~._signals.eds.EDSSpectrum.estimate_background_windows`. See :ref:`eds_background_subtraction-label`.
* New method to estimate the windows of integration, :py:meth:`~._signals.eds.EDSSpectrum.estimate_integration_windows`.
* New specific :py:meth:`~._signals.eds.EDSSpectrum.plot` method, with markers to indicate the X-ray lines, the window of integration or/and the windows for background subtraction. See :ref:`eds_plot_markers-label`.
* New examples of signal in the ``hspy.utils.example_signals`` module.

  + :py:func:`~.misc.example_signals_loading.load_1D_EDS_SEM_spectrum`
  + :py:func:`~.misc.example_signals_loading.load_1D_EDS_TEM_spectrum`

* New method to mask the vaccum, :py:meth:`~._signals.eds_tem.EDSTEMSpectrum.vacuum_mask` and a specific :py:meth:`~._signals.eds_tem.EDSTEMSpectrum.decomposition` method that incoroporate the vacuum mask

API changes
-----------

* :py:class:`~.component.Component` and :py:class:`~.component.Parameter` now inherit ``traits.api.HasTraits``
  that enable ``traitsui`` to modify these objects.
* :py:meth:`~.misc.utils.attrsetter` is added, behaving as the default python :py:meth:`setattr` with nested
  attributes.
* Several widget functions were made internal and/or renamed:
    + ``add_patch_to`` -> ``_add_patch_to``
    + ``set_patch`` -> ``_set_patch``
    + ``onmove`` -> ``_onmousemove``
    + ``update_patch_position`` -> ``_update_patch_position``
    + ``update_patch_size`` -> ``_update_patch_size``
    + ``add_axes`` -> ``set_mpl_ax``

v0.7.3
++++++

This is a maintenance release. A list of fixed issues is available in the
`0.7.3 milestone
<https://github.com/hyperspy/hyperspy/issues?milestone=6&page=1&state=closed>`__
in the github repository.

.. _changes_0.7.2:

v0.7.2
++++++

This is a maintenance release. A list of fixed issues is available in the
`0.7.2 milestone
<https://github.com/hyperspy/hyperspy/issues?milestone=5&page=1&state=closed>`__
in the github repository.

.. _changes_0.7.1:

v0.7.1
++++++

This is a maintenance release. A list of fixed issues is available in the
`0.7.1 milestone
<https://github.com/hyperspy/hyperspy/issues?milestone=4&page=1&state=closed>`__
in the github repository.


New features
------------

* Add suspend/resume model plot updating. See :ref:`model.visualization`.

v0.7
++++

New features
------------

Core
^^^^

* New syntax to index the :py:class:`~.axes.AxesManager`.
* New Signal methods to transform between Signal subclasses. More information
  :ref:`here <transforming.signal>`.

  + :py:meth:`~.signal.Signal.set_signal_type`
  + :py:meth:`~.signal.Signal.set_signal_origin`
  + :py:meth:`~.signal.Signal.as_signal2D`
  + :py:meth:`~.signal.Signal.as_signal1D`

* The string representation of the Signal class now prints the shape of the
  data and includes a separator between the navigation and the signal axes e.g
  (100, 10| 5) for a signal with two navigation axes of size 100 and 10 and one
  signal axis of size 5.
* Add support for RGBA data. See :ref:`signal.change_dtype`.
* The default toolkit can now be saved in the preferences.
* Added full compatibility with the Qt toolkit that is now the default.
* Added compatibility witn the the GTK and TK toolkits, although with no GUI
  features.
* It is now possible to run HyperSpy in a headless system.
* Added a CLI to :py:meth:`~.signal.Signal1DTools.remove_background`.
* New :py:meth:`~.signal.Signal1DTools.estimate_peak_width` method to estimate
  peak width.
* New methods to integrate over one axis:
  :py:meth:`~.signal.Signal.integrate1D` and
  :py:meth:`~.signal.Signal1DTools.integrate_in_range`.
* New :attr:`~signal.Signal.metadata` attribute, ``Signal.binned``. Several
  methods behave differently on binned and unbinned signals.
  See :ref:`signal.binned`.
* New :py:meth:`~.signal.Signal.map` method to easily transform the
  data using a function that operates on individual signals. See
  :ref:`signal.iterator`.
* New :py:meth:`~.signal.Signal.get_histogram` and
  :py:meth:`~.signal.Signal.print_summary_statistics` methods.
* The spikes removal tool has been moved to the :class:`~._signal.Signal1D`
  class so that it is available for all its subclasses.
* The :py:meth:`~.signal.Signal.split` method now can automatically split back
  stacked signals into its original part. See :ref:`signal.stack_split`.

IO
^^

* Improved support for FEI's emi and ser files.
* Improved support for Gatan's dm3 files.
* Add support for reading Gatan's dm4 files.

Plotting
^^^^^^^^

* Use the blitting capabilities of the different toolkits to
  speed up the plotting of images.
* Added several extra options to the Signal :py:meth:`~.signal.Signal.plot`
  method to customize the navigator. See :ref:`visualization-label`.
* Add compatibility with IPython's matplotlib inline plotting.
* New function, :py:func:`~.drawing.utils.plot_spectra`, to plot several
  spectra in the same figure. See :ref:`plot.spectra`.
* New function, :py:func:`~.drawing.utils.plot_signals`, to plot several
  signals at the same time. See :ref:`plot.signals`.
* New function, :py:func:`~.drawing.utils.plot_histograms`, to plot the histrograms
  of several signals at the same time. See :ref:`plot.signals`.

Curve fitting
^^^^^^^^^^^^^

* The chi-squared, reduced chi-squared and the degrees of freedom are
  computed automatically when fitting. See :ref:`model.fitting`.
* New functionality to plot the individual components of a model. See
  :ref:`model.visualization`.
* New method, :py:meth:`~.model.Model.fit_component`, to help setting the
  starting parameters. See :ref:`model.starting`.

Machine learning
^^^^^^^^^^^^^^^^

* The PCA scree plot can now be easily obtained as a Signal. See
  :ref:`mva.scree_plot`.
* The decomposition and blind source separation components can now be obtained
  as :py:class:`~.signal.Signal` instances. See :ref:`mva.get_results`.
* New methods to plot the decomposition and blind source separation results
  that support n-dimensional loadings. See :ref:`mva.visualization`.

Dielectric function
^^^^^^^^^^^^^^^^^^^

* New :class:`~.signal.Signal` subclass,
  :class:`~._signals.dielectric_function.DielectricFunction`.

EELS
^^^^

* New method,
  :meth:`~._signals.eels.EELSSpectrum.kramers_kronig_analysis` to calculate
  the dielectric function from low-loss electron energy-loss spectra based on
  the Kramers-Kronig relations. See :ref:`eels.kk`.
* New method to align the zero-loss peak,
  :meth:`~._signals.eels.EELSSpectrum.align_zero_loss_peak`.

EDS
^^^

* New signal, EDSSpectrum especialized in EDS data analysis, with subsignal
  for EDS with SEM and with TEM: EDSSEMSpectrum and EDSTEMSpectrum. See
  :ref:`eds-label`.
* New database of EDS lines available in the ``elements`` attribute of the
  ``hspy.utils.material`` module.
* Adapted methods to calibrate the spectrum, the detector and the microscope.
  See :ref:`eds_calibration-label`.
* Specific methods to describe the sample,
  :py:meth:`~._signals.eds.EDSSpectrum.add_elements` and
  :py:meth:`~._signals.eds.EDSSpectrum.add_lines`. See :ref:`eds_sample-label`
* New method to get the intensity of specific X-ray lines:
  :py:meth:`~._signals.eds.EDSSpectrum.get_lines_intensity`. See
  :ref:`eds_plot-label`

API changes
-----------

* hyperspy.misc has been reorganized. Most of the functions in misc.utils has
  been rellocated to specialized modules. misc.utils is no longer imported in
  hyperspy.hspy. A new hyperspy.utils module is imported instead.
* Objects that have been renamed

  + ``hspy.elements`` -> ``utils.material.elements``.
  + ``Signal.navigation_indexer`` -> ``inav``.
  + ``Signal.signal_indexer`` -> ``isig``.
  + ``Signal.mapped_parameters`` -> ``Signal.metadata``.
  + ``Signal.original_parameters`` -> ``Signal.original_metadata``.
* The metadata has been reorganized. See :ref:`metadata_structure`.
* The following signal methods now operate out-of-place:

  + :py:meth:`~.signal.Signal.swap_axes`
  + :py:meth:`~.signal.Signal.rebin`

.. _changes_0.6:

v0.6
++++

New features
------------

* Signal now supports indexing and slicing. See :ref:`signal.indexing`.
* Most arithmetic and rich arithmetic operators work with signal.
  See :ref:`signal.operations`.
* Much improved EELSSpectrum methods:
  :py:meth:`~._signals.eels.EELSSpectrum.estimate_zero_loss_peak_centre`,
  :py:meth:`~._signals.eels.EELSSpectrum.estimate_elastic_scattering_intensity` and
  :py:meth:`~._signals.eels.EELSSpectrum.estimate_elastic_scattering_threshold`.

* The axes can now be given using their name e.g. ``s.crop("x", 1,10)``
* New syntax to specify position over axes: an integer specifies the indexes
  over the axis and a floating number specifies the position in the axis units
  e.g. ``s.crop("x", 1, 10.)`` crops over the axis `x` (in meters) from index 1
  to value 10 meters. Note that this may make your old scripts behave in
  unexpected ways as just renaming the old \*_in_units and \*_in_values methods
  won't work in most cases.
* Most methods now use the natural order i.e. X,Y,Z.. to index the axes.
* Add padding to fourier-log and fourier-ratio deconvolution to fix the
  wrap-around problem and increase its performance.
* New
  :py:meth:`~.components.eels_cl_edge.EELSCLEdge.get_fine_structure_as_spectrum`
  EELSCLEdge method.
* New :py:class:`~.components.arctan.Arctan` model component.
* New
  :py:meth:`~.model.Model.enable_adjust_position`
  and :py:meth:`~.model.Model.disable_adjust_position`
  to easily change the position of components using the mouse on the plot.
* New Model methods
  :py:meth:`~.model.Model.set_parameters_value`,
  :py:meth:`~.model.Model.set_parameters_free` and
  :py:meth:`~.model.Model.set_parameters_not_free`
  to easily set several important component attributes of a list of components
  at once.
* New
  :py:func:`~.misc.utils.stack` function to stack signals.
* New Signal methods:
  :py:meth:`~.signal.Signal.integrate_simpson`,
  :py:meth:`~.signal.Signal.max`,
  :py:meth:`~.signal.Signal.min`,
  :py:meth:`~.signal.Signal.var`, and
  :py:meth:`~.signal.Signal.std`.
* New sliders window to easily navigate signals with navigation_dimension > 2.
* The Ripple (rpl) reader can now read rpl files produced by INCA.

API changes
-----------
* The following functions has been renamed or removed:

    * components.EELSCLEdge

        * knots_factor -> fine_structure_smoothing
        * edge_position -> onset_energy
        * energy_shift removed

    * components.Voigt.origin -> centre
    * signals.Signal1D

        * find_peaks_1D -> Signal.find_peaks1D_ohaver
        * align_1D -> Signal.align1D
        * shift_1D -> Signal.shift1D
        * interpolate_1D -> Signal.interpolate1D

    * signals.Signal2D.estimate_2D_translation -> Signal.estimate_shift2D
    * Signal

        * split_in -> split
        * crop_in_units -> crop
        * crop_in_pixels -> crop


* Change syntax to create Signal objects. Instead of a dictionary
  Signal.__init__ takes keywords e.g with  a new syntax .
  ``>>> s = signals.Signal1D(np.arange(10))`` instead of
  ``>>> s = signals.Signal1D({'data' : np.arange(10)})``



.. _changes_0.5.1:

v0.5.1
++++++

New features
------------
* New Signal method `get_current_signal` proposed by magnunor.
* New Signal `save` method keyword `extension` to easily change the saving format while keeping the same file name.
* New EELSSpectrum methods: estimate_elastic_scattering_intensity, fourier_ratio_deconvolution, richardson_lucy_deconvolution, power_law_extrapolation.
* New Signal1D method: hanning_taper.



Major bugs fixed
----------------
* The `print_current_values` Model method was raising errors when fine structure was enabled or when only_free = False.
*  The `load` function `signal_type` keyword was not passed to the readers.
* The spikes removal tool was unable to find the next spikes when the spike was detected close to the limits of the spectrum.
* `load` was raising an UnicodeError when the title contained non-ASCII characters.
* In Windows `HyperSpy Here` was opening in the current folder, not in the selected folder.
* The fine structure coefficients were overwritten with their std when charging values from the model.
* Storing the parameters in the maps and all the related functionality was broken for 1D spectrum.
* Remove_background was broken for 1D spectrum.




API changes
-----------
* EELSSPectrum.find_low_loss_centre was renamed to estimate_zero_loss_peak_centre.
* EELSSPectrum.calculate_FWHM was renamed to estimate_FWHM.

.. _changes_0.5:

v0.5
++++

New features
------------
* The documentation was thoroughly revised, courtesy of M. Walls.
* New user interface to remove spikes from EELS spectra.
* New align2D signals.Signal2D method to align image stacks.
* When loading image files, the data are now automatically converted to
  grayscale when all the color channels are equal.
* Add the possibility to load a stack memory mapped (similar to ImageJ
  virtual stack).
* Improved hyperspy starter script that now includes the possibility
  to start HyperSpy in the new IPython notebook.
* Add "HyperSpy notebook here" to the Windows context menu.
* The information displayed in the plots produced by Signal.plot have
  been enhanced.
* Added Egerton's sigmak3 and sigmal3 GOS calculations (translated
  from matlab by I. Iyengar) to the EELS core loss component.
* A browsable dictionary containing the chemical elements and
  their onset energies is now available in the user namespace under
  the variable name `elements`.
* The ripple file format now supports storing the beam energy, the collection and the convergence angle.


Major bugs fixed
----------------
* The EELS core loss component had a bug in the calculation of the
  relativistic gamma that produced a gamma that was always
  approximately zero. As a consequence the GOS calculation was wrong,
  especially for high beam energies.
* Loading msa files was broken when running on Python 2.7.2 and newer.
* Saving images to rpl format was broken.
* Performing BSS on data decomposed with poissonian noise normalization
  was failing when some columns or rows of the unfolded data were zero,
  what occurs often in EDX data for example.
* Importing some versions of scikits learn was broken
* The progress bar was not working properly in the new IPython notebook.
* The constrast of the image was not automatically updated.

API changes
-----------
* spatial_mask was renamed to navigation_mask.
* Signal1D and Signal2D are not loaded into the user namespace by default.
  The signals module is loaded instead.
* Change the default BSS algorithm to sklearn fastica, that is now
  distributed with HyperSpy and used in case that sklearn is not
  installed e.g. when using EPDFree.
* _slicing_axes was renamed to signal_axes.
* _non_slicing_axes to navigation_axes.
* All the Model \*_in_pixels methods  were renamed to to _*_in_pixel.
* EELSCLEdge.fs_state was renamed to fine_structure_active.
* EELSCLEdge.fslist was renamed to fine_structure_coeff.
* EELSCLEdge.fs_emax was renamed to fine_structure_width.
* EELSCLEdge.freedelta was renamed to free_energy_shift.
* EELSCLEdge.delta was renamed to energy_shift.
* A value of True in a mask now means that the item is masked all over
  HyperSpy.


.. _changes_0.4.1:

v0.4.1
++++++

New features
------------

 * Added TIFF 16, 32 and 64 bits support by using (and distributing) Christoph Gohlke's `tifffile library <http://www.lfd.uci.edu/~gohlke/code/tifffile.py.html>`__.
 * Improved UTF8 support.
 * Reduce the number of required libraries by making mdp and hdf5 not mandatory.
 * Improve the information returned by __repr__ of several objects.
 * DictionaryBrowser now has an export method, i.e. mapped parameters and original_parameters can be exported.
 * New _id_name attribute for Components and Parameters. Improvements in their __repr__ methods.
 * Component.name can now be overwriten by the user.
 * New Signal.__str__ method.
 * Include HyperSpy in The Python Package Index.


Bugs fixed
----------
 * Non-ascii characters breaking IO and print features fixed.
 * Loading of multiple files at once using wildcards fixed.
 * Remove broken hyperspy-gui script.
 * Remove unmantained and broken 2D peak finding and analysis features.

Syntax changes
--------------
 * In EELS automatic background feature creates a PowerLaw component, adds it to the model an add it to a variable in the user namespace. The variable has been renamed from `bg` to `background`.
 * pes_gaussian Component renamed to pes_core_line_shape.

.. _changes_0.4:

v0.4
++++

New features
------------
 * Add a slider to the filter ui.
 * Add auto_replot to sum.
 * Add butterworth filter.
 * Added centring and auto_transpose to the svd_pca algorithm.
 * Keep the mva_results information when changing the signal type.
 * Added sparse_pca and mini_batch_sparse_pca to decomposition algorithms.
 * Added TV to the smoothing algorithms available in BSS.
 * Added whitening to the mdp ICA preprocessing.
 * Add explained_variance_ratio.
 * Improvements in saving/loading mva data.
 * Add option to perform ICA on the scores.
 * Add orthomax FA algorithm.
 * Add plot methods to Component and Parameter.
 * Add plot_results to Model.
 * Add possibility to export the decomposition and bss results to a folder.
 * Add Signal method `change_dtype`.
 * Add the possibility to pass extra parameters to the ICA algorithm.
 * Add the possibility to reproject the data after a decomposition.
 * Add warning when decomposing a non-float signal.
 * adds a method to get the PCs as a Signal1D object and adds smoothing to the ICA preprocessing.
 * Add the possibility to select the energy range in which to perform spike removal operations.
 * the smoothings guis now offer differentiation and line color option. Smoothing now does not require a gui.
 * Fix reverse_ic which was not reversing the scores and improve the autoreversing method.
 * Avoid cropping when is not needed.
 * Changed criteria to reverse the ICs.
 * Changed nonans default to False for plotting.
 * Change the whitening algorithm to a svd based one and add sklearn fastica algorithm.
 * Clean the ummixing info after a new decomposition.
 * Increase the chances that similar independent components will have the same indexes.
 * Make savitzky-golay smoothing work without raising figures.
 * Make plot_decomposition* plot only the number of factors/scores determined by output_dimension.
 * make the Parameter __repr__ method print its name.
 * New contrast adjustment tool.
 * New export method for Model, Component and Parameter.
 * New Model method: print_current_values.
 * New signal, spectrum_simulation.
 * New smoothing algorithm: total variance denoising.
 * Plotting the components in the same or separate windows is now configurable in the preferences.
 * Plotting the spikes is now optional.
 * Return an error message when the decomposition algorithm is not recognised.
 * Store the masks in mva_results.
 * The free parameters are now automically updated on chaning the free attribute.

Bugs fixed
----------
 * Added missing keywords to plot_pca_factors and plot_ica_factors.
 * renamed incorrectly named exportPca and exportIca functions.
 * an error was raised when calling generate_data_from_model.
 * a signal with containing nans was failing to plot.
 * attempting to use any decomposition plotting method after loading with mva_results.load was raising an error.
 * a typo was causing in error in pca when normalize_variance = True.
 * a typo was raising an error when cropping the decomposition dimension.
 * commit 5ff3798105d6 made decomposition and other methods raise an error.
 * BUG-FIXED: the decomposition centering index was wrong.
 * ensure_directory was failing for the current directory.
 * model data forced to be 3D unnecessarily.
 * non declared variable was raising an error.
 * plot naming for peak char factor plots were messed up.
 * plot_RGB was broken.
 * plot_scores_2D was using the transpose of the shape to reshape the scores.
 * remove background was raising an error when the navigation dimension was 0.
 * saving the scores was sometimes transposing the shape.
 * selecting indexes while using the learning export functions was raising an error.
 * the calibrate ui was calculating wrongly the calibration the first time that Apply was pressed.
 * the offset estimation was summing instead of averaging.
 * the plot_explained_variance_ratio was actually plotting the cumulative, renamed.
 * the signal mask in decomposition and ica was not being raveled.
 * the slice attribute was not correctly set at init in some scenarios.
 * the smoothing and calibrabrion UIs were freezing when the plots where closed before closing the UI window.
 * to_spectrum was transposing the navigation dimension.
 * variance2one was operating in the wrong axis.
 * when closing the plots of a model, the UI object was not being destroyed.
 * when plotting an image the title was not displayed.
 * when the axis size was changed (e.g. after cropping) the set_signal_dimension method was not being called.
 * when using transform the data was being centered and the resulting scores were wrong.

Syntax changes
--------------

 * in decomposition V rename to explained_variance.
 * In FixedPattern, default interpolation changed to linear.
 * Line and parabole components deleted + improvements in the docstrings.
 * pca_V = variance.
 * mva_result renamed to learning_results.
 * pca renamed to decomposition.
 * pca_v and mva_results.v renamed to scores pc renamed to factors .
   pca_build_SI renamed to get_pca_model ica_build_SI renamed to get_ica_model.
 * plot_explained_variance renamed to plot_explained_variance_ratio.
 * principal_components_analysis renamed to decomposition.
 * rename eels_simulation to eels_spectrum_simulation.
 * Rename the output parameter of svd_pca and add scores.
 * Replace plot_lev by plot_explained_variance_ratio.
 * Scores renamed to loadings.
 * slice_bool renamed to navigate to make its function more explicit.
 * smoothing renamed to pretreatment and butter added.
 * variance2one renamed to normalize_variance.
 * w renamed to unmixing matrix and fixes a bug when loading a mva_result
   in which output_dimension = None.
 * ubshells are again availabe in the interactive session.
 * Several changes to the interface.
 * The documentation was updated to reflex the last changes.
 * The microscopes.csv file was updated so it no longer contains the
   Orsay VG parameters.
