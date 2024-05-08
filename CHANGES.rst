.. _changelog:

Changelog
*********

Changelog entries for the development version are available at
https://hyperspy.readthedocs.io/en/latest/changes.html

.. towncrier-draft-entries:: |release| [UNRELEASED]

.. towncrier release notes start

2.1.0 (2024-05-08)
==================

Enhancements
------------

- Add a dynamic navigator which updates when the number of navigation dimensions is greater than 3 (`#3199 <https://github.com/hyperspy/hyperspy/issues/3199>`_)
- Add an example to the gallery to show how to extract a line profile from an image using a :class:`~.api.roi.Line2DROI` (`#3227 <https://github.com/hyperspy/hyperspy/issues/3227>`_)
- Add an example showing how to handle RGB images. (`#3346 <https://github.com/hyperspy/hyperspy/issues/3346>`_)
- Use :func:`pint.get_application_registry` to get :class:`pint.UnitRegistry` and facilitate interoperability of pint quantity operation with other modules. (`#3357 <https://github.com/hyperspy/hyperspy/issues/3357>`_)
- :func:`~.api.plot.plot_roi_map` improvement:

  - Add ROIs on signal plot instead of adding them on a sum signal.
  - Add support for more than 3 ROIs.
  - Add ``color`` and ``cmap`` parameters to specify color and colormap.
  - Add ``single_figure`` parameter to enable plotting on a single figure using :func:`~.api.plot.plot_images` or :func:`~.api.plot.plot_spectra`.
  - Improve performance of updating ROI maps by a factor 2-4 when moving ROI.
  - Update images in :func:`~.api.plot.plot_images` on data change event.
  - Remove ROI widgets and disconnect interactive function when closing ROI map figures. (`#3364 <https://github.com/hyperspy/hyperspy/issues/3364>`_)
- Documentation improvement. (`#3365 <https://github.com/hyperspy/hyperspy/issues/3365>`_)


Bug Fixes
---------

- Fix ROI slicing of non-uniform axis (`#3328 <https://github.com/hyperspy/hyperspy/issues/3328>`_)
- Add the ability to save and load :class:`~.axes.BaseDataAxis` objects to a hyperspy file. (`#3342 <https://github.com/hyperspy/hyperspy/issues/3342>`_)
- Fix navigator event disconnection and fix plot when changing dtype from/to rbgx. (`#3346 <https://github.com/hyperspy/hyperspy/issues/3346>`_)
- Fix :func:`~.api.get_configuration_directory_path` function. (`#3349 <https://github.com/hyperspy/hyperspy/issues/3349>`_)
- Fix :func:`~.api.plot.plot_images` axis ticks discrepancy. (`#3361 <https://github.com/hyperspy/hyperspy/issues/3361>`_)
- Fixes in :func:`~.api.plot.plot_roi_map`:

  - Fix slicing signal when using CircleROI (`#3358 <https://github.com/hyperspy/hyperspy/issues/3358>`_).
  - Fix redundant events when using :class:`~.api.roi.SpanROI`, which was causing flickering of the colorbar in the ROI map figures (`#3364 <https://github.com/hyperspy/hyperspy/issues/3364>`_)
- Fix development version on ``RELEASE_next_minor`` branch. (`#3368 <https://github.com/hyperspy/hyperspy/issues/3368>`_)


API changes
-----------

- :func:`~.api.plot.plot_roi_map` doesn't return the sum of all ROI maps (``all_sum``) and the signals sliced by the ROIs (``roi_signals``), these can be obtained separately using the ``rois`` returned by :func:`~.api.plot.plot_roi_map` and :func:`~.api.interactive`. (`#3364 <https://github.com/hyperspy/hyperspy/issues/3364>`_)


Maintenance
-----------

- Ruff update:

  - Set the ``RELEASE_next_patch`` branch as target for the ``pre-commit.ci`` update to keep all branches in synchronisation.
  - Update ruff to version 0.3.3 and run ruff check/format on source code. (`#3335 <https://github.com/hyperspy/hyperspy/issues/3335>`_)
- Replace deprecated ``np.string_`` by ``np.bytes_``. (`#3338 <https://github.com/hyperspy/hyperspy/issues/3338>`_)
- Enable ruff isort and all pyflakes/Pycodestyle rules, except E501 to avoid confict with black formatting. (`#3348 <https://github.com/hyperspy/hyperspy/issues/3348>`_)
- Merge ``hyperspy.api.no`` and ``hyperspy.api.no_gui`` modules since the latter is not necessary anymore. (`#3349 <https://github.com/hyperspy/hyperspy/issues/3349>`_)
- Convert projet readme to markdown, fixes badges on github (`#3351 <https://github.com/hyperspy/hyperspy/issues/3351>`_)
- Simplify Azure Pipeline CI by removing build and uploading wheels, since this is now done on GitHub CI. (`#3356 <https://github.com/hyperspy/hyperspy/issues/3356>`_)
- Fix duplicated test and occasional test failure. (`#3365 <https://github.com/hyperspy/hyperspy/issues/3365>`_)
- Use lower case when checking matplotlib backend in the test suite. (`#3367 <https://github.com/hyperspy/hyperspy/issues/3367>`_)
- Add ``percentile_range`` traitsui attribute to ``ImageContrastEditor`` necessary for `hyperspy/hyperspy_gui_traitsui#76 <https://github.com/hyperspy/hyperspy_gui_traitsui/pull/76>`_. (`#3368 <https://github.com/hyperspy/hyperspy/issues/3368>`_)


2.0.1 (2024-02-26)
==================

Bug Fixes
---------

- Fix bug with side by side plotting of signal containing navigation dimension only. (`#3304 <https://github.com/hyperspy/hyperspy/issues/3304>`_)
- Fix getting release on some linux system wide install, e.g. Debian or Google colab (`#3318 <https://github.com/hyperspy/hyperspy/issues/3318>`_)
- Fix incorrect position of ``Texts`` marker when using mathtext. (`#3319 <https://github.com/hyperspy/hyperspy/issues/3319>`_)


Maintenance
-----------

- Update version switcher. (`#3291 <https://github.com/hyperspy/hyperspy/issues/3291>`_)
- Fix readme badges and fix broken web links. (`#3298 <https://github.com/hyperspy/hyperspy/issues/3298>`_)
- Use ruff to lint code. (`#3299 <https://github.com/hyperspy/hyperspy/issues/3299>`_)
- Use ruff to format code. (`#3300 <https://github.com/hyperspy/hyperspy/issues/3300>`_)
- Run test suite on osx arm64 on GitHub CI and speed running test suite using all available CPUs (3 or 4) instead of only 2. (`#3305 <https://github.com/hyperspy/hyperspy/issues/3305>`_)
- Fix API changes in scipy (:func:`scipy.signal.windows.tukey`) and scikit-image (:func:`skimage.restoration.unwrap_phase`). (`#3306 <https://github.com/hyperspy/hyperspy/issues/3306>`_)
- Fix deprecation warnings and warnings in the test suite (`#3320 <https://github.com/hyperspy/hyperspy/issues/3320>`_)
- Add documentation on how the documentation is updated and the required manual changes for minor and major releases. (`#3321 <https://github.com/hyperspy/hyperspy/issues/3321>`_)
- Add Google Analytics ID to learn more about documentation usage. (`#3322 <https://github.com/hyperspy/hyperspy/issues/3322>`_)
- Setup workflow to push development documentation automatically. (`#3297 <https://github.com/hyperspy/hyperspy/pull/3297>`_)

.. _changes_2.0:

2.0 (2023-12-20)
================

Release Highlights
------------------
- Hyperspy has split off some of the file reading/writing and domain specific functionalities into separate libraries!
  
  - `RosettaSciIO <https://hyperspy.org/rosettasciio>`_: A library for reading and writing scientific data files.
    See `RosettaSciIO release notes <https://hyperspy.org/rosettasciio/changes.html>`_ for new features and supported formats.
  - `exSpy <https://exspy.readthedocs.io>`_: A library for EELS and EDS analysis.
    See `exSpy release notes <https://hyperspy.org/exspy/changes.html>`_ for new features.
  - `HoloSpy <https://holospy.readthedocs.io>`_: A library for analysis of (off-axis) electron holography data.
    See `HoloSpy release notes <https://holospy.readthedocs.io/en/latest/changes.html>`_ for new features.

- The :py:mod:`~.api.plot.markers` API has been refactored

  - Lazy markers are now supported
  - Plotting many markers is now `much` faster
  - Added :py:class:`~.api.plot.markers.Polygons` marker

- The documentation has been restructured and improved!

  - Short example scripts are now included in the documentation
  - Improved guides for lazy computing as well as an improved developer guide

- Plotting is easier and more consistent:

  - Added horizontal figure layout choice when using the ``ipympl`` backend
  - Changing navigation coordinates using the keyboard arrow-keys has been removed.
    Use ``Crtl`` + ``Arrow`` instead.
  - Jump to navigation position using ``shift`` + click in the navigator figure.

- HyperSpy now works with Pyodide/Jupyterlite, checkout `hyperspy.org/jupyterlite-hyperspy <https://hyperspy.org/jupyterlite-hyperspy>`_!
- The deprecated API has removed: see the list of API changes and removal in the :ref:`sections below <hyperspy_2.0_api_changes>`.

New features
------------

- :py:meth:`~._signals.lazy.LazySignal.compute` will now pass keyword arguments to the dask :meth:`dask.array.Array.compute` method. This enables setting the scheduler and the number of computational workers. (`#2971 <https://github.com/hyperspy/hyperspy/issues/2971>`_)
- Changes to :meth:`~.api.signals.BaseSignal.plot`:
  
  - Added horizontal figure layout choice when using the ``ipympl`` backend. The default layour can be set in the plot section of the preferences GUI. (`#3140 <https://github.com/hyperspy/hyperspy/issues/3140>`_)
  
- Changes to :meth:`~.api.signals.Signal2D.find_peaks`:
  
  - Lazy signals return lazy peak signals
  - ``get_intensity`` argument added to get the intensity of the peaks
  - The signal axes are now stored in the ``metadata.Peaks.signal_axes`` attribute of the peaks' signal. (`#3142 <https://github.com/hyperspy/hyperspy/issues/3142>`_)
  
- Change the logging output so that logging messages are not displayed in red, to avoid confusion with errors. (`#3173 <https://github.com/hyperspy/hyperspy/issues/3173>`_)
- Added ``hyperspy.decorators.deprecated`` and ``hyperspy.decorators.deprecated_argument``:

  - Provide consistent and clean deprecation
  - Added a guide for deprecating code (`#3174 <https://github.com/hyperspy/hyperspy/issues/3174>`_)
  
- Add functionality to select navigation position using ``shift`` + click in the navigator. (`#3175 <https://github.com/hyperspy/hyperspy/issues/3175>`_)
- Added a ``plot_residual`` to :py:meth:`~.models.model1d.Model1D.plot`. When ``True``, a residual line (Signal - Model) appears in the model figure. (`#3186 <https://github.com/hyperspy/hyperspy/issues/3186>`_)
- Switch to :meth:`matplotlib.axes.Axes.pcolormesh` for image plots involving non-uniform axes.
  The following cases are covered: 2D-signal with arbitrary navigation-dimension, 1D-navigation and 1D-signal (linescan).
  Not covered are 2D-navigation images (still uses sliders). (`#3192 <https://github.com/hyperspy/hyperspy/issues/3192>`_)
- New :meth:`~.api.signals.BaseSignal.interpolate_on_axis` method to switch one axis of a signal. The data is interpolated in the process. (`#3214 <https://github.com/hyperspy/hyperspy/issues/3214>`_)
- Added :func:`~.api.plot.plot_roi_map`. Allows interactively using a set of ROIs to select regions of the signal axes of a signal and visualise how the signal varies in this range spatially. (`#3224 <https://github.com/hyperspy/hyperspy/issues/3224>`_)


Bug Fixes
---------

- Improve syntax in the `io` module. (`#3091 <https://github.com/hyperspy/hyperspy/issues/3091>`_)
- Fix behaviour of :py:class:`~.misc.utils.DictionaryTreeBrowser` setter with value of dictionary type (`#3094 <https://github.com/hyperspy/hyperspy/issues/3094>`_)
- Avoid slowing down fitting by optimising attribute access of model. (`#3155 <https://github.com/hyperspy/hyperspy/issues/3155>`_)
- Fix harmless error message when using multiple :class:`~.api.roi.RectangularROI`: check if resizer patches are drawn before removing them. Don't display resizers when adding the widget to the figure (widget in unselected state) for consistency with unselected state (`#3222 <https://github.com/hyperspy/hyperspy/issues/3222>`_)
- Fix keeping dtype in :py:meth:`~.api.signals.BaseSignal.rebin` when the endianess is specified in the dtype (`#3237 <https://github.com/hyperspy/hyperspy/issues/3237>`_)
- Fix serialization error due to ``traits.api.Property`` not being serializable if a dtype is specified.
  See #3261 for more details. (`#3262 <https://github.com/hyperspy/hyperspy/issues/3262>`_)
- Fix setting bounds for ``"trf"``, ``"dogbox"`` optimizer (`#3244 <https://github.com/hyperspy/hyperspy/issues/3244>`_)
- Fix bugs in new marker implementation:

  - Markers str representation fails if the marker isn't added to a signal
  - make :meth:`~.api.plot.markers.Markers.from_signal` to work with all markers - it was only working with :class:`~.api.plot.markers.Points` (`#3270 <https://github.com/hyperspy/hyperspy/issues/3270>`_)
- Documentation fixes:

  - Fix cross-references in documentation and enable sphinx "nitpicky" when building documentation to check for broken links.
  - Fix using mutable objects as default argument.
  - Change some :class:`~.component.Component` attributes to properties in order to include their docstrings in the API reference. (`#3273 <https://github.com/hyperspy/hyperspy/issues/3273>`_)




Improved Documentation
----------------------

- Restructure documentation:

  - Improve structure of the API reference
  - Improve introduction and overall structure of documentation
  - Add gallery of examples (`#3050 <https://github.com/hyperspy/hyperspy/issues/3050>`_)

- Add examples to the gallery to show how to use SpanROI and slice signal interactively (`#3221 <https://github.com/hyperspy/hyperspy/issues/3221>`_)
- Add a section on keeping a clean and sensible commit history to the developer guide. (`#3064 <https://github.com/hyperspy/hyperspy/issues/3064>`_)
- Replace ``sphinx.ext.imgmath`` by ``sphinx.ext.mathjax`` to fix the math rendering in the *ReadTheDocs* build (`#3084 <https://github.com/hyperspy/hyperspy/issues/3084>`_)
- Fix docstring examples in :class:`~.api.signals.BaseSignal` class.
  Describe how to test docstring examples in developer guide. (`#3095 <https://github.com/hyperspy/hyperspy/issues/3095>`_)
- Update intersphinx_mapping links of matplotlib, numpy and scipy. (`#3218 <https://github.com/hyperspy/hyperspy/issues/3218>`_)
- Add examples on creating signal from tabular data or reading from a simple text file (`#3246 <https://github.com/hyperspy/hyperspy/issues/3246>`_)
- Activate checking of example code in docstring and user guide using ``doctest`` and fix errors in the code. (`#3281 <https://github.com/hyperspy/hyperspy/issues/3281>`_)
- Update warning of "beta" state in big data section to be more specific. (`#3282 <https://github.com/hyperspy/hyperspy/issues/3282>`_)


Enhancements
------------

- Add support for passing ``**kwargs`` to :py:meth:`~.api.signals.Signal2D.plot` when using heatmap style in :py:func:`~.api.plot.plot_spectra` . (`#3219 <https://github.com/hyperspy/hyperspy/issues/3219>`_)
- Add support for pep 660 on editable installs for pyproject.toml based builds of extension (`#3252 <https://github.com/hyperspy/hyperspy/issues/3252>`_)
- Make HyperSpy compatible with pyodide (hence JupyterLite):
  
  - Set ``numba`` and ``numexpr`` as optional dependencies.
  - Replace ``dill`` by ``cloudpickle``.
  - Fallback to dask synchronous scheduler when running on pyodide.
  - Reduce packaging size to less than 1MB.
  - Add packaging test on GitHub CI. (`#3255 <https://github.com/hyperspy/hyperspy/issues/3255>`_)

.. _hyperspy_2.0_api_changes:

API changes
-----------

- RosettaSciIO was split out of the `HyperSpy repository <https://github.com/hyperspy/hyperspy>`_ on July 23, 2022. The IO-plugins and related functions so far developed in HyperSpy were moved to the `RosettaSciIO repository <https://github.com/hyperspy/rosettasciio>`__. (`#2972 <https://github.com/hyperspy/hyperspy/issues/2972>`_)
- Extend the IO functions to accept alias names for format ``name`` as defined in RosettaSciIO. (`#3009 <https://github.com/hyperspy/hyperspy/issues/3009>`_)
- Fix behaviour of :meth:`~hyperspy.model.BaseModel.print_current_values`, :meth:`~.component.Component.print_current_values`
  and :func:`~.api.print_known_signal_types`, which were not printing when running from a script - they were only printing when running in notebook or qtconsole. Now all print_* functions behave consistently: they all print the output instead of returning an object (string or html). The :func:`IPython.display.display` will pick a suitable rendering when running in an "ipython" context, for example notebook, qtconsole. (`#3145 <https://github.com/hyperspy/hyperspy/issues/3145>`_)
- The markers have been refactored - see the new :py:mod:`~.api.plot.markers` API and the :ref:`gallery of examples <gallery.markers>` for usage. The new :py:class:`~.api.plot.markers.Markers` uses :py:class:`matplotlib.collections.Collection`, is faster and more generic than the previous implementation and also supports lazy markers. Markers saved in HyperSpy files (``hspy``, ``zspy``) with HyperSpy < 2.0 are converted automatically when loading the file. (`#3148 <https://github.com/hyperspy/hyperspy/issues/3148>`_)
- For all functions with the ``rechunk`` parameter, the default has been changed from ``True`` to ``False``. This means HyperSpy will not automatically try to change the chunking for lazy signals. The old behaviour could lead to a reduction in performance when working with large lazy datasets, for example 4D-STEM data. (`#3166 <https://github.com/hyperspy/hyperspy/issues/3166>`_)
- Renamed ``Signal2D.crop_image`` to :meth:`~.api.signals.Signal2D.crop_signal` (`#3197 <https://github.com/hyperspy/hyperspy/issues/3197>`_)
- Changes and improvement of the map function:

  - Removes the ``parallel`` argument
  - Replace the ``max_workers`` with the ``num_workers`` argument to be consistent with ``dask``
  - Adds more documentation on setting the dask backend and how to use multiple cores
  - Adds ``navigation_chunk`` argument for setting the chunks with a non-lazy signal
  - Fix axes handling when the function to be mapped can be applied to the whole dataset - typically when it has the ``axis`` or ``axes`` keyword argument. (`#3198 <https://github.com/hyperspy/hyperspy/issues/3198>`_)
  
- Remove ``physics_tools`` since it is not used and doesn't fit in the scope of HyperSpy. (`#3235 <https://github.com/hyperspy/hyperspy/issues/3235>`_)
- Improve the readability of the code by replacing the ``__call__`` method of some objects with the more explicit ``_get_current_data``.
 
  - Rename ``__call__`` method of :py:class:`~.api.signals.BaseSignal` to ``_get_current_data``.
  - Rename ``__call__`` method of  :py:class:`hyperspy.model.BaseModel`  to ``_get_current_data``.
  - Remove ``__call__`` method of the :py:class:`hyperspy.component.Component` class. (`#3238 <https://github.com/hyperspy/hyperspy/issues/3238>`_)
  
- Rename ``hyperspy.api.datasets`` to :mod:`hyperspy.api.data` and simplify submodule structure:
  
  - ``hyperspy.api.datasets.artificial_data.get_atomic_resolution_tem_signal2d`` is renamed to :func:`hyperspy.api.data.atomic_resolution_image`
  - ``hyperspy.api.datasets.artificial_data.get_luminescence_signal`` is renamed to :func:`hyperspy.api.data.luminescence_signal`
  - ``hyperspy.api.datasets.artificial_data.get_wave_image`` is renamed to :func:`hyperspy.api.data.wave_image` (`#3253 <https://github.com/hyperspy/hyperspy/issues/3253>`_)


API Removal
-----------

As the HyperSpy API evolves, some of its parts are occasionally reorganized or removed.
When APIs evolve, the old API is deprecated and eventually removed in a major
release. The functions and methods removed in HyperSpy 2.0 are listed below along
with migration advises:

Axes
^^^^

- ``AxesManager.show`` has been removed, use :py:meth:`~.axes.AxesManager.gui` instead.
- ``AxesManager.set_signal_dimension`` has been removed, use :py:meth:`~.api.signals.BaseSignal.as_signal1D`,
  :py:meth:`~.api.signals.BaseSignal.as_signal2D` or :py:meth:`~.api.signals.BaseSignal.transpose` of the signal instance instead.

Components
^^^^^^^^^^

- The API of the :py:class:`~.api.model.components1D.Polynomial` has changed (it was deprecated in HyperSpy 1.5). The old API had a single parameters ``coefficients``, which has been replaced by ``a0``, ``a1``, etc.
- The ``legacy`` option (introduced in HyperSpy 1.6) for :class:`~.api.model.components1D.Arctan` has been removed, use :class:`exspy.components.EELSArctan` to use the old API.
- The ``legacy`` option (introduced in HyperSpy 1.6) for :class:`~.api.model.components1D.Voigt` has been removed, use :class:`exspy.components.PESVoigt` to use the old API.

Data Visualization
^^^^^^^^^^^^^^^^^^

- The ``saturated_pixels`` keyword argument of :py:meth:`~.api.signals.Signal2D.plot` has been removed, use ``vmin`` and/or ``vmax`` instead.
- The ``get_complex`` property of ``hyperspy.drawing.signal1d.Signal1DLine`` has been removed.
- The keyword argument ``line_style`` of :py:func:`~.api.plot.plot_spectra` has been renamed to ``linestyle``.
- Changing navigation coordinates using keyboard ``Arrow`` has been removed, use
  ``Crtl`` + ``Arrow`` instead.
- The ``markers`` submodules can not be imported from the :py:mod:`~.api` anymore, use :py:mod:`hyperspy.api.plot.markers`
  directly, i.e. :class:`hyperspy.api.plot.markers.Arrows`, instead.
- The creation of markers has changed to use their class name instead of aliases, for example,
  use ``m = hs.plot.markers.Lines`` instead of ``m = hs.plot.markers.line_segment``.

Loading and Saving data
^^^^^^^^^^^^^^^^^^^^^^^

The following deprecated keyword arguments have been removed during the
migration of the IO plugins to the `RosettaSciIO library
<https://hyperspy.org/rosettasciio/changes.html>`_:

- The arguments ``mmap_dir`` and ``load_to_memory`` of the :py:func:`~.api.load`
  function have been removed, use the ``lazy`` argument instead.
- :ref:`Bruker composite file (BCF) <bruker-format>`: The ``'spectrum'`` option for the
  ``select_type`` parameter was removed. Use ``'spectrum_image'`` instead.
- :ref:`Electron Microscopy Dataset (EMD) NCEM <emd_ncem-format>`: Using the
  keyword ``dataset_name`` was removed, use ``dataset_path`` instead.
- :ref:`NeXus data format <nexus-format>`: The ``dataset_keys``, ``dataset_paths``
  and ``metadata_keys`` keywords were removed. Use ``dataset_key``, ``dataset_path``
  and ``metadata_key`` instead.

Machine Learning
^^^^^^^^^^^^^^^^

- The ``polyfit`` keyword argument has been removed. Use ``var_func`` instead.
- The list of possible values for the ``algorithm`` argument of the :py:meth:`~.api.signals.BaseSignal.decomposition` method
  has been changed according to the following table:

  .. list-table:: Change of the ``algorithm`` argument
     :widths: 25 75
     :header-rows: 1

     * - hyperspy < 2.0
       - hyperspy >= 2.0
     * - fast_svd
       - SVD along with the argument svd_solver="randomized"
     * - svd
       - SVD
     * - fast_mlpca
       - MLPCA along with the argument svd_solver="randomized
     * - mlpca
       - MLPCA
     * - nmf
       - NMF
     * - RPCA_GoDec
       - RPCA

- The argument ``learning_rate`` of the ``ORPCA`` algorithm has been renamed to ``subspace_learning_rate``.
- The argument ``momentum`` of the ``ORPCA`` algorithm has been renamed to ``subspace_momentum``.
- The list of possible values for the ``centre`` keyword argument of the :py:meth:`~.api.signals.BaseSignal.decomposition` method
  when using the ``SVD`` algorithm has been changed according to the following table:

  .. list-table:: Change of the ``centre`` argument
     :widths: 50 50
     :header-rows: 1

     * - hyperspy < 2.0
       - hyperspy >= 2.0
     * - trials
       - navigation
     * - variables
       - signal
- For lazy signals, a possible value of the ``algorithm`` keyword argument of the
  :py:meth:`~._signals.lazy.LazySignal.decomposition` method has been changed
  from ``"ONMF"`` to ``"ORNMF"``.
- Setting the ``metadata`` and ``original_metadata`` attribute of signals is removed, use
  the :py:meth:`~.misc.utils.DictionaryTreeBrowser.set_item` and
  :py:meth:`~.misc.utils.DictionaryTreeBrowser.add_dictionary` methods of the
  ``metadata`` and ``original_metadata`` attribute instead.


Model fitting
^^^^^^^^^^^^^

- The ``iterpath`` default value has changed from ``'flyback'`` to ``'serpentine'``.
- Changes in the arguments of the :py:meth:`~hyperspy.model.BaseModel.fit` and :py:meth:`~hyperspy.model.BaseModel.multifit` methods:

  - The ``fitter`` argument has been renamed to ``optimizer``.
  - The list of possible values for the ``optimizer`` argument has been renamed according to the following table:

    .. list-table:: Renaming of the ``optimizer`` argument
       :widths: 50 50
       :header-rows: 1

       * - hyperspy < 2.0
         - hyperspy >= 2.0
       * - fmin
         - Nelder-Mead
       * - fmin_cg
         - CG
       * - fmin_ncg
         - Newton-CG
       * - fmin_bfgs
         - Newton-BFGS
       * - fmin_l_bfgs_b
         - L-BFGS-B
       * - fmin_tnc
         - TNC
       * - fmin_powell
         - Powell
       * - mpfit
         - lm
       * - leastsq
         - lm

    - ``loss_function="ml"`` has been renamed to ``loss_function="ML-poisson"``.
    - ``grad=True`` has been changed to ``grad="analytical"``.
    - The ``ext_bounding`` argument has been renamed to ``bounded``.
    - The ``min_function`` argument has been removed, use the ``loss_function`` argument instead.
    - The ``min_function_grad`` argument has been removed, use the ``grad`` argument instead.

- The following :py:class:`~hyperspy.model.BaseModel` methods have been removed:

  - ``hyperspy.model.BaseModel.set_boundaries``
  - ``hyperspy.model.BaseModel.set_mpfit_parameters_info``

- The arguments ``parallel`` and ``max_workers`` have been removed from the :py:meth:`~hyperspy.model.BaseModel.as_signal` methods.

- Setting the ``metadata``  attribute of a :py:class:`~.samfire.Samfire` has been removed, use
  the :py:meth:`~.misc.utils.DictionaryTreeBrowser.set_item` and
  :py:meth:`~.misc.utils.DictionaryTreeBrowser.add_dictionary` methods of the
  ``metadata`` attribute instead.

- The deprecated ``twin_function`` and ``twin_inverse_function`` have been privatized.
- Remove ``fancy`` argument of :meth:`~hyperspy.model.BaseModel.print_current_values` and :meth:`~.component.Component.print_current_values`,
  which wasn't changing the output rendering.
- The attribute ``channel_switches`` of :py:class:`~hyperspy.model.BaseModel` have been privatized, instead
  use the :py:meth:`~hyperspy.model.BaseModel.set_signal_range_from_mask` or any other methods to 
  set the signal range, such as :py:meth:`~.models.model1d.Model1D.set_signal_range`,
  :py:meth:`~.models.model1d.Model1D.add_signal_range` or :py:meth:`~.models.model1d.Model1D.remove_signal_range`
  and their :py:class:`~.models.model2d.Model2D` counterparts. 


Signal
^^^^^^

- ``metadata.Signal.binned`` is removed, use the ``is_binned`` axis attribute
  instead, e. g. ``s.axes_manager[-1].is_binned``.
- Some possible values for the ``bins`` argument of the :py:meth:`~.api.signals.BaseSignal.get_histogram`
  method have been changed according to the following table:

  .. list-table:: Change of the ``bins`` argument
     :widths: 50 50
     :header-rows: 1

     * - hyperspy < 2.0
       - hyperspy >= 2.0
     * - scotts
       - scott
     * - freedman
       - fd

- The ``integrate_in_range`` method has been removed, use :py:class:`~.roi.SpanROI`
  followed by :py:meth:`~.api.signals.BaseSignal.integrate1D` instead.
- The ``progressbar`` keyword argument of the :py:meth:`~._signals.lazy.LazySignal.compute` method
  has been removed, use ``show_progressbar`` instead.
- The deprecated ``comp_label`` argument of the methods :py:meth:`~.api.signals.BaseSignal.plot_decomposition_loadings`,
  :py:meth:`~.api.signals.BaseSignal.plot_decomposition_factors`, :py:meth:`~.api.signals.BaseSignal.plot_bss_loadings`,
  :py:meth:`~.api.signals.BaseSignal.plot_bss_factors`, :py:meth:`~.api.signals.BaseSignal.plot_cluster_distances`,
  :py:meth:`~.api.signals.BaseSignal.plot_cluster_labels` has been removed, use the ``title`` argument instead.
- The :py:meth:`~.api.signals.BaseSignal.set_signal_type` now raises an error when passing
  ``None`` to the ``signal_type`` argument. Use ``signal_type=""`` instead.
- Passing an "iterating over navigation argument" to the :py:meth:`~.api.signals.BaseSignal.map`
  method is removed, pass a HyperSpy signal with suitable navigation and signal shape instead.


Signal2D
^^^^^^^^

- :meth:`~.api.signals.Signal2D.find_peaks` now returns lazy signals in case of lazy input signal.


Preferences
^^^^^^^^^^^

- The ``warn_if_guis_are_missing`` HyperSpy preferences setting has been removed,
  as it is not necessary anymore.


Maintenance
-----------

- Pin third party GitHub actions and add maintenance guidelines on how to update them (`#3027 <https://github.com/hyperspy/hyperspy/issues/3027>`_)
- Drop support for python 3.7, update oldest supported dependencies and simplify code accordingly (`#3144 <https://github.com/hyperspy/hyperspy/issues/3144>`_)
- IPython and IParallel are now optional dependencies (`#3145 <https://github.com/hyperspy/hyperspy/issues/3145>`_)
- Fix Numpy 1.25 deprecation: implicit array to scalar conversion in :py:meth:`~.signals.Signal2D.align2D` (`#3189 <https://github.com/hyperspy/hyperspy/issues/3189>`_)
- Replace deprecated :mod:`scipy.misc` by :mod:`scipy.datasets` in documentation (`#3225 <https://github.com/hyperspy/hyperspy/issues/3225>`_)
- Fix documentation version switcher (`#3228 <https://github.com/hyperspy/hyperspy/issues/3228>`_)
- Replace deprecated :py:class:`scipy.interpolate.interp1d` with :py:func:`scipy.interpolate.make_interp_spline` (`#3233 <https://github.com/hyperspy/hyperspy/issues/3233>`_)
- Add support for python 3.12 (`#3256 <https://github.com/hyperspy/hyperspy/issues/3256>`_)
- Consolidate package metadata:

  - use ``pyproject.toml`` only
  - clean up unmaintained packaging files
  - use ``setuptools_scm`` to define version
  - add python 3.12 to test matrix (`#3268 <https://github.com/hyperspy/hyperspy/issues/3268>`_)
- Pin pytest-xdist to 3.5 as a workaround for test suite failure on Azure Pipeline (`#3274 <https://github.com/hyperspy/hyperspy/issues/3274>`_)



.. _changes_1.7.6:

1.7.6 (2023-11-17)
===================

Bug Fixes
---------

- Allows for loading of ``.hspy`` files saved with version 2.0.0 and greater and no unit or name set
  for some axis. (`#3241 <https://github.com/hyperspy/hyperspy/issues/3241>`_)


Maintenance
-----------

- Backport of 3189: fix Numpy1.25 deprecation: implicite array to scalar conversion in :py:meth:`~.api.signals.Signal2D.align2D` (`#3243 <https://github.com/hyperspy/hyperspy/issues/3243>`_)
- Pin pillow to <10.1 to avoid imageio error. (`#3251 <https://github.com/hyperspy/hyperspy/issues/3251>`_)


.. _changes_1.7.5:

1.7.5 (2023-05-04)
===================

Bug Fixes
---------

- Fix plotting boolean array with :py:func:`~.api.plot.plot_images` (`#3118 <https://github.com/hyperspy/hyperspy/issues/3118>`_)
- Fix test with scipy1.11 and update deprecated ``scipy.interpolate.interp2d`` in the test suite (`#3124 <https://github.com/hyperspy/hyperspy/issues/3124>`_)
- Use intersphinx links to fix links to scikit-image documentation (`#3125 <https://github.com/hyperspy/hyperspy/issues/3125>`_)


Enhancements
------------

- Improve performance of `model.multifit` by avoiding `axes.is_binned` repeated evaluation (`#3126 <https://github.com/hyperspy/hyperspy/issues/3126>`_)


Maintenance
-----------

- Simplify release workflow and replace deprecated ``actions/create-release`` action with ``softprops/action-gh-release``. (`#3117 <https://github.com/hyperspy/hyperspy/issues/3117>`_)
- Add support for python 3.11 (`#3134 <https://github.com/hyperspy/hyperspy/issues/3134>`_)
- Pin ``imageio`` to <2.28 (`#3138 <https://github.com/hyperspy/hyperspy/issues/3138>`_)


.. _changes_1.7.4:

1.7.4 (2023-03-16)
===================

Bug Fixes
---------

- Fixes an array indexing bug when loading a .sur file format spectra series. (`#3060 <https://github.com/hyperspy/hyperspy/issues/3060>`_)
- Speed up ``to_numpy`` function to avoid slow down when used repeatedly, typically during fitting (`#3109 <https://github.com/hyperspy/hyperspy/issues/3109>`_)


Improved Documentation
----------------------

- Replace ``sphinx.ext.imgmath`` by ``sphinx.ext.mathjax`` to fix the math rendering in the *ReadTheDocs* build (`#3084 <https://github.com/hyperspy/hyperspy/issues/3084>`_)


Enhancements
------------

- Add support for Phenom .elid revision 3 and 4 formats (`#3073 <https://github.com/hyperspy/hyperspy/issues/3073>`_)


Maintenance
-----------

- Add pooch as test dependency, as it is required to use scipy.dataset in latest scipy (1.10) and update plotting test. Fix warning when plotting non-uniform axis (`#3079 <https://github.com/hyperspy/hyperspy/issues/3079>`_)
- Fix matplotlib 3.7 and scikit-learn 1.4 deprecations (`#3102 <https://github.com/hyperspy/hyperspy/issues/3102>`_)
- Add support for new pattern to generate random numbers introduced in dask 2023.2.1. Deprecate usage of :py:class:`numpy.random.RandomState` in favour of :py:func:`numpy.random.default_rng`. Bump scipy minimum requirement to 1.4.0. (`#3103 <https://github.com/hyperspy/hyperspy/issues/3103>`_)
- Fix checking links in documentation for domain, which aren't compatible with sphinx linkcheck (`#3108 <https://github.com/hyperspy/hyperspy/issues/3108>`_)


.. _changes_1.7.3:

1.7.3 (2022-10-29)
===================

Bug Fixes
---------

- Fix error when reading Velox containing FFT with odd number of pixels (`#3040 <https://github.com/hyperspy/hyperspy/issues/3040>`_)
- Fix pint Unit for pint>=0.20 (`#3052 <https://github.com/hyperspy/hyperspy/issues/3052>`_)


Maintenance
-----------

- Fix deprecated import of scipy ``ascent`` in docstrings and the test suite (`#3032 <https://github.com/hyperspy/hyperspy/issues/3032>`_)
- Fix error handling when trying to convert a ragged signal to non-ragged for numpy >=1.24 (`#3033 <https://github.com/hyperspy/hyperspy/issues/3033>`_)
- Fix getting random state dask for dask>=2022.10.0 (`#3049 <https://github.com/hyperspy/hyperspy/issues/3049>`_)


.. _changes_1.7.2:

1.7.2 (2022-09-17)
===================

Bug Fixes
---------

- Fix some errors and remove unnecessary code identified by `LGTM
  <https://lgtm.com/projects/g/hyperspy/hyperspy/>`_. (`#2977 <https://github.com/hyperspy/hyperspy/issues/2977>`_)
- Fix error which occurs when guessing output size in the :py:meth:`~.api.signals.BaseSignal.map` function and using dask newer than 2022.7.1 (`#2981 <https://github.com/hyperspy/hyperspy/issues/2981>`_)
- Fix display of x-ray lines when using log norm and the intensity at the line is 0 (`#2995 <https://github.com/hyperspy/hyperspy/issues/2995>`_)
- Fix handling constant derivative in :py:meth:`~.api.signals.Signal1D.spikes_removal_tool` (`#3005 <https://github.com/hyperspy/hyperspy/issues/3005>`_)
- Fix removing horizontal or vertical line widget; regression introduced in hyperspy 1.7.0 (`#3008 <https://github.com/hyperspy/hyperspy/issues/3008>`_)


Improved Documentation
----------------------

- Add a note in the user guide to explain that when a file contains several datasets, :py:func:`~.api.load` returns a list of signals instead of a single signal and that list indexation can be used to access a single signal. (`#2975 <https://github.com/hyperspy/hyperspy/issues/2975>`_)


Maintenance
-----------

- Fix extension test suite CI workflow. Enable workflow manual trigger (`#2982 <https://github.com/hyperspy/hyperspy/issues/2982>`_)
- Fix deprecation warning and time zone test failing on windows (locale dependent) (`#2984 <https://github.com/hyperspy/hyperspy/issues/2984>`_)
- Fix external links in the documentation and add CI build to check external links (`#3001 <https://github.com/hyperspy/hyperspy/issues/3001>`_)
- Fix hyperlink in bibliography (`#3015 <https://github.com/hyperspy/hyperspy/issues/3015>`_)
- Fix matplotlib ``SpanSelector`` import for matplotlib 3.6 (`#3016 <https://github.com/hyperspy/hyperspy/issues/3016>`_)


.. _changes_1.7.1:

1.7.1 (2022-06-18)
===================

Bug Fixes
---------

- Fixes invalid file chunks when saving some signals to hspy/zspy formats. (`#2940 <https://github.com/hyperspy/hyperspy/issues/2940>`_)
- Fix issue where a TIFF image from an FEI FIB/SEM navigation camera image would not be read due to missing metadata (`#2941 <https://github.com/hyperspy/hyperspy/issues/2941>`_)
- Respect ``show_progressbar`` parameter in :py:meth:`~.api.signals.BaseSignal.map` (`#2946 <https://github.com/hyperspy/hyperspy/issues/2946>`_)
- Fix regression in :py:meth:`~hyperspy.models.model1d.Model1D.set_signal_range` which was raising an error when used interactively (`#2948 <https://github.com/hyperspy/hyperspy/issues/2948>`_)
- Fix :py:class:`~.api.roi.SpanROI` regression: the output of :py:meth:`~.roi.BaseInteractiveROI.interactive` was not updated when the ROI was changed. Fix errors with updating limits when plotting empty slice of data. Improve docstrings and test coverage. (`#2952 <https://github.com/hyperspy/hyperspy/issues/2952>`_)
- Fix stacking signals that contain their variance in metadata. Previously it was raising an error when specifying the stacking axis. (`#2954 <https://github.com/hyperspy/hyperspy/issues/2954>`_)
- Fix missing API documentation of several signal classes. (`#2957 <https://github.com/hyperspy/hyperspy/issues/2957>`_)
- Fix two bugs in :py:meth:`~.api.signals.BaseSignal.decomposition`:

  * The poisson noise normalization was not applied when giving a `signal_mask`
  * An error was raised when applying a ``signal_mask`` on a signal with signal dimension larger than 1. (`#2964 <https://github.com/hyperspy/hyperspy/issues/2964>`_)


Improved Documentation
----------------------

- Fix and complete docstrings of :py:meth:`~.api.signals.Signal2D.align2D` and :py:meth:`~.api.signals.Signal2D.estimate_shift2D`. (`#2961 <https://github.com/hyperspy/hyperspy/issues/2961>`_)


Maintenance
-----------

- Minor refactor of the EELS subshells in the ``elements`` dictionary. (`#2868 <https://github.com/hyperspy/hyperspy/issues/2868>`_)
- Fix packaging of test suite and tweak tests to pass on different platform of blas implementation (`#2933 <https://github.com/hyperspy/hyperspy/issues/2933>`_)


.. _changes_1.7.0:

1.7.0 (2022-04-26)
===================

New features
------------

- Add ``filter_zero_loss_peak`` argument to the ``hyperspy._signals.eels.EELSSpectrum.spikes_removal_tool`` method (`#1412 <https://github.com/hyperspy/hyperspy/issues/1412>`_)
- Add :py:meth:`~.api.signals.Signal2D.calibrate` method to :py:class:`~.api.signals.Signal2D` signal, which allows for interactive calibration (`#1791 <https://github.com/hyperspy/hyperspy/issues/1791>`_)
- Add ``hyperspy._signals.eels.EELSSpectrum.vacuum_mask`` method to: ``hyperspy._signals.eels.EELSSpectrum`` signal (`#2183 <https://github.com/hyperspy/hyperspy/issues/2183>`_)
- Support for :ref:`relative slicing <signal.indexing>` (`#2386 <https://github.com/hyperspy/hyperspy/issues/2386>`_)
- Implement non-uniform axes, not all hyperspy functionalities support non-uniform axes, see this `tracking issue <https://github.com/hyperspy/hyperspy/issues/2398>`_ for progress. (`#2399 <https://github.com/hyperspy/hyperspy/issues/2399>`_)
- Add (weighted) :ref:`linear least square fitting <linear_fitting-label>`. Close `#488 <https://github.com/hyperspy/hyperspy/issues/488>`_ and `#574 <https://github.com/hyperspy/hyperspy/issues/574>`_. (`#2422 <https://github.com/hyperspy/hyperspy/issues/2422>`_)
- Support for reading :external+rsciio:ref:`JEOL EDS data<jeol-format>` (`#2488 <https://github.com/hyperspy/hyperspy/issues/2488>`_)
- Plot overlayed images - see :ref:`plotting several images<plot.images>` (`#2599 <https://github.com/hyperspy/hyperspy/issues/2599>`_)
- Add initial support for :ref:`GPU computation<gpu_processing>` using cupy (`#2670 <https://github.com/hyperspy/hyperspy/issues/2670>`_)
- Add ``height`` property to the :py:class:`~._components.gaussian2d.Gaussian2D` component (`#2688 <https://github.com/hyperspy/hyperspy/issues/2688>`_)
- Support for reading and writing :external+rsciio:ref:`TVIPS image stream data<tvips-format>` (`#2780 <https://github.com/hyperspy/hyperspy/issues/2780>`_)
- Add in :external+rsciio:ref:`zspy format<zspy-format>`: hspy specification with the zarr format. Particularly useful to speed up loading and :ref:`saving large datasets<big_data.saving>` by using concurrency. (`#2825 <https://github.com/hyperspy/hyperspy/issues/2825>`_)
- Support for reading :external+rsciio:ref:`DENSsolutions Impulse data<dens-format>` (`#2828 <https://github.com/hyperspy/hyperspy/issues/2828>`_)
- Add lazy loading for :external+rsciio:ref:`JEOL EDS data<jeol-format>` (`#2846 <https://github.com/hyperspy/hyperspy/issues/2846>`_)
- Add :ref:`html representation<lazy._repr_html_>` for lazy signals and the
  :py:meth:`~._signals.lazy.LazySignal.get_chunk_size` method to get the chunk size
  of given axes (`#2855 <https://github.com/hyperspy/hyperspy/issues/2855>`_)
- Add support for Hamamatsu HPD-TA Streak Camera tiff files,
  with axes and metadata parsing. (`#2908 <https://github.com/hyperspy/hyperspy/issues/2908>`_)


Bug Fixes
---------

- Signals with 1 value in the signal dimension will now be :py:class:`~.api.signals.BaseSignal` (`#2773 <https://github.com/hyperspy/hyperspy/issues/2773>`_)
- :py:func:`exspy.material.density_of_mixture` now throws a Value error when the density of an element is unknown (`#2775 <https://github.com/hyperspy/hyperspy/issues/2775>`_)
- Improve error message when performing Cliff-Lorimer quantification with a single line intensity (`#2822 <https://github.com/hyperspy/hyperspy/issues/2822>`_)
- Fix bug for the hydrogenic gdos k edge (`#2859 <https://github.com/hyperspy/hyperspy/issues/2859>`_)
- Fix bug in axes.UnitConversion: the offset value was initialized by units. (`#2864 <https://github.com/hyperspy/hyperspy/issues/2864>`_)
- Fix bug where the :py:meth:`~.api.signals.BaseSignal.map` function wasn't operating properly when an iterating signal was larger than the input signal. (`#2878 <https://github.com/hyperspy/hyperspy/issues/2878>`_)
- In case the Bruker defined XML element node at SpectrumRegion contains no information on the
  specific selected X-ray line (if there is only single line available), suppose it is 'Ka' line. (`#2881 <https://github.com/hyperspy/hyperspy/issues/2881>`_)
- When loading Bruker Bcf, ``cutoff_at_kV=None`` does no cutoff (`#2898 <https://github.com/hyperspy/hyperspy/issues/2898>`_)
- Fix bug where the :py:meth:`~.api.signals.BaseSignal.map` function wasn't operating properly when an iterating signal was not an array. (`#2903 <https://github.com/hyperspy/hyperspy/issues/2903>`_)
- Fix bug for not saving ragged arrays with dimensions larger than 2 in the ragged dimension. (`#2906 <https://github.com/hyperspy/hyperspy/issues/2906>`_)
- Fix bug with importing some spectra from eelsdb and add progress bar (`#2916 <https://github.com/hyperspy/hyperspy/issues/2916>`_)
- Fix bug when the spikes_removal_tool would not work interactively for signal with 0-dimension navigation space. (`#2918 <https://github.com/hyperspy/hyperspy/issues/2918>`_)


Deprecations
------------

- Deprecate ``hyperspy.axes.AxesManager.set_signal_dimension`` in favour of using :py:meth:`~.api.signals.BaseSignal.as_signal1D`, :py:meth:`~.api.signals.BaseSignal.as_signal2D` or :py:meth:`~.api.signals.BaseSignal.transpose` of the signal instance instead. (`#2830 <https://github.com/hyperspy/hyperspy/issues/2830>`_)


Enhancements
------------

- :ref:`Region of Interest (ROI)<roi-label>` can now be created without specifying values (`#2341 <https://github.com/hyperspy/hyperspy/issues/2341>`_)
- mpfit cleanup (`#2494 <https://github.com/hyperspy/hyperspy/issues/2494>`_)
- Document reading Attolight data with the sur/pro format reader (`#2559 <https://github.com/hyperspy/hyperspy/issues/2559>`_)
- Lazy signals now caches the current data chunk when using multifit and when plotting, improving performance. (`#2568 <https://github.com/hyperspy/hyperspy/issues/2568>`_)
- Read cathodoluminescence metadata from digital micrograph files, amended in `PR #2894 <https://github.com/hyperspy/hyperspy/pull/2894>`_ (`#2590 <https://github.com/hyperspy/hyperspy/issues/2590>`_)
- Add possibility to search/access nested items in DictionaryTreeBrowser (metadata) without providing full path to item. (`#2633 <https://github.com/hyperspy/hyperspy/issues/2633>`_)
- Improve :py:meth:`~.api.signals.BaseSignal.map` function in :py:class:`~.api.signals.BaseSignal` by utilizing dask for both lazy and non-lazy signals. This includes adding a `lazy_output` parameter, meaning non-lazy signals now can output lazy results. See the :ref:`user guide<lazy_output-map-label>` for more information. (`#2703 <https://github.com/hyperspy/hyperspy/issues/2703>`_)
- :external+rsciio:ref:`NeXus<nexus-format>` file with more options when reading and writing (`#2725 <https://github.com/hyperspy/hyperspy/issues/2725>`_)
- Add ``dtype`` argument to :py:meth:`~.api.signals.BaseSignal.rebin` (`#2764 <https://github.com/hyperspy/hyperspy/issues/2764>`_)
- Add option to set output size when :external+rsciio:ref:`exporting images<image-format>` (`#2791 <https://github.com/hyperspy/hyperspy/issues/2791>`_)
- Add :py:meth:`~.axes.AxesManager.switch_iterpath` context manager to switch iterpath (`#2795 <https://github.com/hyperspy/hyperspy/issues/2795>`_)
- Add options not to close file (lazy signal only) and not to write dataset for hspy file format, see :external+rsciio:ref:`hspy-format` for details (`#2797 <https://github.com/hyperspy/hyperspy/issues/2797>`_)
- Add Github workflow to run test suite of extension from a pull request. (`#2824 <https://github.com/hyperspy/hyperspy/issues/2824>`_)
- Add :py:attr:`~.api.signals.BaseSignal.ragged` attribute to :py:class:`~.api.signals.BaseSignal` to clarify when a signal contains a ragged array. Fix inconsistency caused by ragged array and add a :ref:`ragged array<signal.ragged>` section to the user guide (`#2842 <https://github.com/hyperspy/hyperspy/issues/2842>`_)
- Import hyperspy submodules lazily to speed up importing hyperspy. Fix autocompletion `signals` submodule (`#2850 <https://github.com/hyperspy/hyperspy/issues/2850>`_)
- Add support for JEOL SightX tiff file (`#2862 <https://github.com/hyperspy/hyperspy/issues/2862>`_)
- Add new markers ``hyperspy.drawing._markers.arrow``, ``hyperspy.drawing._markers.ellipse`` and filled ``hyperspy.drawing._markers.rectangle``. (`#2871 <https://github.com/hyperspy/hyperspy/issues/2871>`_)
- Add metadata about the file-reading and saving operations to the Signals
  produced by :py:func:`~.api.load` and :py:meth:`~.api.signals.BaseSignal.save`
  (see the :ref:`metadata structure <general-file-metadata>` section of the user guide) (`#2873 <https://github.com/hyperspy/hyperspy/issues/2873>`_)
- expose Stage coordinates and rotation angle in metada for sem images in bcf reader. (`#2911 <https://github.com/hyperspy/hyperspy/issues/2911>`_)


API changes
-----------

- ``metadata.Signal.binned`` is replaced by an axis parameter, e. g. ``axes_manager[-1].is_binned`` (`#2652 <https://github.com/hyperspy/hyperspy/issues/2652>`_)
- * when loading Bruker bcf, ``cutoff_at_kV=None`` (default) applies no more automatic cutoff.
  * New acceptable values ``"zealous"`` and ``"auto"`` do automatic cutoff. (`#2910 <https://github.com/hyperspy/hyperspy/issues/2910>`_)
- Deprecate the ability to directly set ``metadata`` and ``original_metadata`` Signal
  attributes in favor of using :py:meth:`~.misc.utils.DictionaryTreeBrowser.set_item`
  and :py:meth:`~.misc.utils.DictionaryTreeBrowser.add_dictionary` methods or
  specifying metadata when creating signals (`#2913 <https://github.com/hyperspy/hyperspy/issues/2913>`_)


Maintenance
-----------

- Fix warning when build doc and formatting user guide (`#2762 <https://github.com/hyperspy/hyperspy/issues/2762>`_)
- Drop support for python 3.6 (`#2839 <https://github.com/hyperspy/hyperspy/issues/2839>`_)
- Continuous integration fixes and improvements; Bump minimal version requirement of dask to 2.11.0 and matplotlib to 3.1.3 (`#2866 <https://github.com/hyperspy/hyperspy/issues/2866>`_)
- Tweak tests tolerance to fix tests failure on aarch64 platform; Add python 3.10 build. (`#2914 <https://github.com/hyperspy/hyperspy/issues/2914>`_)
- Add support for matplotlib 3.5, simplify maintenance of ``RangeWidget`` and some signal tools. (`#2922 <https://github.com/hyperspy/hyperspy/issues/2922>`_)
- Compress some tiff tests files to reduce package size (`#2926 <https://github.com/hyperspy/hyperspy/issues/2926>`_)


.. _changes_1.6.5:

1.6.5 (2021-10-28)
===================

Bug Fixes
---------

- Suspend plotting during :meth:`exspy.models.EELSModel.smart_fit` call (`#2796 <https://github.com/hyperspy/hyperspy/issues/2796>`_)
- make :py:meth:`~.api.signals.BaseSignal.add_marker` also check if the plot is not active before plotting signal (`#2799 <https://github.com/hyperspy/hyperspy/issues/2799>`_)
- Fix irresponsive ROI added to a signal plot with a right hand side axis (`#2809 <https://github.com/hyperspy/hyperspy/issues/2809>`_)
- Fix :py:func:`~.api.plot.plot_histograms` drawstyle following matplotlib API change (`#2810 <https://github.com/hyperspy/hyperspy/issues/2810>`_)
- Fix incorrect :py:meth:`~.api.signals.BaseSignal.map` output size of lazy signal when input and output axes do not match (`#2837 <https://github.com/hyperspy/hyperspy/issues/2837>`_)
- Add support for latest h5py release (3.5) (`#2843 <https://github.com/hyperspy/hyperspy/issues/2843>`_)


Deprecations
------------

- Rename ``line_style`` to ``linestyle`` in :py:func:`~.api.plot.plot_spectra` to match matplotlib argument name (`#2810 <https://github.com/hyperspy/hyperspy/issues/2810>`_)


Enhancements
------------

- :py:meth:`~.roi.BaseInteractiveROI.add_widget` can now take a string or integer instead of tuple of string or integer (`#2809 <https://github.com/hyperspy/hyperspy/issues/2809>`_)


.. _changes_1.6.4:

1.6.4 (2021-07-08)
===================

Bug Fixes
---------

- Fix parsing EELS aperture label with unexpected value, for example 'Imaging' instead of '5 mm' (`#2772 <https://github.com/hyperspy/hyperspy/issues/2772>`_)
- Lazy datasets can now be saved out as blockfiles (blo) (`#2774 <https://github.com/hyperspy/hyperspy/issues/2774>`_)
- ComplexSignals can now be rebinned without error (`#2789 <https://github.com/hyperspy/hyperspy/issues/2789>`_)
- Method :py:meth:`~.api.model.components1D.Polynomial.estimate_parameters` of the :py:class:`~._components.polynomial.Polynomial` component now supports order
  greater than 10 (`#2790 <https://github.com/hyperspy/hyperspy/issues/2790>`_)
- Update minimal requirement of dependency importlib_metadata from
  >= 1.6.0 to >= 3.6 (`#2793 <https://github.com/hyperspy/hyperspy/issues/2793>`_)


Enhancements
------------

- When saving a dataset with a dtype other than
  `uint8 <https://numpy.org/doc/stable/user/basics.types.html>`_ to a blockfile
  (blo) it is now possible to provide the argument ``intensity_scaling`` to map
  the intensity values to the reduced range (`#2774 <https://github.com/hyperspy/hyperspy/issues/2774>`_)


Maintenance
-----------

- Fix image comparison failure with numpy 1.21.0 (`#2774 <https://github.com/hyperspy/hyperspy/issues/2774>`_)


.. _changes_1.6.3:

1.6.3 (2021-06-10)
===================

Bug Fixes
---------

- Fix ROI snapping regression (`#2720 <https://github.com/hyperspy/hyperspy/issues/2720>`_)
- Fix :py:meth:`~.api.signals.Signal1D.shift1D`, :py:meth:`~.api.signals.Signal1D.align1D` and ``hyperspy._signals.eels.EELSSpectrum.align_zero_loss_peak`` regression with navigation dimension larger than one (`#2729 <https://github.com/hyperspy/hyperspy/issues/2729>`_)
- Fix disconnecting events when closing figure and :py:meth:`~.api.signals.Signal1D.remove_background` is active (`#2734 <https://github.com/hyperspy/hyperspy/issues/2734>`_)
- Fix :py:meth:`~.api.signals.BaseSignal.map` regression of lazy signal with navigation chunks of size of 1 (`#2748 <https://github.com/hyperspy/hyperspy/issues/2748>`_)
- Fix unclear error message when reading a hspy file saved using blosc compression and ``hdf5plugin`` hasn't been imported previously (`#2760 <https://github.com/hyperspy/hyperspy/issues/2760>`_)
- Fix saving ``navigator`` of lazy signal (`#2763 <https://github.com/hyperspy/hyperspy/issues/2763>`_)


Enhancements
------------

- Use ``importlib_metadata`` instead of ``pkg_resources`` for extensions
  registration to speed up the import process and making it possible to install
  extensions and use them without restarting the python session (`#2709 <https://github.com/hyperspy/hyperspy/issues/2709>`_)
- Don't import hyperspy extensions when registering extensions (`#2711 <https://github.com/hyperspy/hyperspy/issues/2711>`_)
- Improve docstrings of various fitting methods (`#2724 <https://github.com/hyperspy/hyperspy/issues/2724>`_)
- Improve speed of :py:meth:`~.api.signals.Signal1D.shift1D` (`#2750 <https://github.com/hyperspy/hyperspy/issues/2750>`_)
- Add support for recent EMPAD file; scanning size wasn't parsed. (`#2757 <https://github.com/hyperspy/hyperspy/issues/2757>`_)


Maintenance
-----------

- Add drone CI to test arm64 platform (`#2713 <https://github.com/hyperspy/hyperspy/issues/2713>`_)
- Fix latex doc build on github actions (`#2714 <https://github.com/hyperspy/hyperspy/issues/2714>`_)
- Use towncrier to generate changelog automatically (`#2717 <https://github.com/hyperspy/hyperspy/issues/2717>`_)
- Fix test suite to support dask 2021.4.1 (`#2722 <https://github.com/hyperspy/hyperspy/issues/2722>`_)
- Generate changelog when building doc to keep the changelog of the development doc up to date on https://hyperspy.readthedocs.io/en/latest (`#2758 <https://github.com/hyperspy/hyperspy/issues/2758>`_)
- Use mamba and conda-forge channel on azure pipeline (`#2759 <https://github.com/hyperspy/hyperspy/issues/2759>`_)


.. _changes_1.6.2:

1.6.2 (2021-04-13)
===================

This is a maintenance release that adds support for python 3.9 and includes
numerous bug fixes and enhancements.
See `the issue tracker
<https://github.com/hyperspy/hyperspy/milestone/42?closed=1>`__
for details.

Bug Fixes
---------

* Fix disconnect event when closing navigator only plot (fixes `#996 <https://github.com/hyperspy/hyperspy/issues/996>`_), (`#2631 <https://github.com/hyperspy/hyperspy/pull/2631>`_)
* Fix incorrect chunksize when saving EMD NCEM file and specifying chunks (`#2629 <https://github.com/hyperspy/hyperspy/pull/2629>`_)
* Fix :py:meth:`~.api.signals.Signal2D.find_peaks` GUIs call with laplacian/difference of gaussian methods (`#2622 <https://github.com/hyperspy/hyperspy/issues/2622>`_ and `#2647 <https://github.com/hyperspy/hyperspy/pull/2647>`_)
* Fix various bugs with ``CircleWidget`` and ``Line2DWidget`` (`#2625 <https://github.com/hyperspy/hyperspy/pull/2625>`_)
* Fix setting signal range of model with negative axis scales (`#2656 <https://github.com/hyperspy/hyperspy/pull/2656>`_)
* Fix and improve mask handling in lazy decomposition; Close `#2605 <https://github.com/hyperspy/hyperspy/issues/2605>`_ (`#2657 <https://github.com/hyperspy/hyperspy/pull/2657>`_)
* Plot scalebar when the axis scales have different sign, fixes `#2557 <https://github.com/hyperspy/hyperspy/issues/2557>`_ (`#2657 <https://github.com/hyperspy/hyperspy/pull/2657>`_)
* Fix :py:meth:`~.api.signals.Signal1D.align1D` returning zeros shifts (`#2675 <https://github.com/hyperspy/hyperspy/pull/2675>`_)
* Fix finding dataset path for EMD NCEM file containing more than one dataset in a  group (`#2673 <https://github.com/hyperspy/hyperspy/pull/2673>`_)
* Fix squeeze function for multiple zero-dimensional entries, improved docstring, added to user guide. (`#2676 <https://github.com/hyperspy/hyperspy/pull/2676>`_)
* Fix error in Cliff-Lorimer quantification using absorption correction (`#2681 <https://github.com/hyperspy/hyperspy/pull/2681>`_)
* Fix ``navigation_mask`` bug in decomposition when provided as numpy array (`#2679 <https://github.com/hyperspy/hyperspy/pull/2679>`_)
* Fix closing image contrast tool and setting vmin/vmax values (`#2684 <https://github.com/hyperspy/hyperspy/pull/2684>`_)
* Fix range widget with matplotlib 3.4 (`#2684 <https://github.com/hyperspy/hyperspy/pull/2684>`_)
* Fix bug in :py:func:`~.api.interactive` with function returning `None`. Improve user guide example. (`#2686 <https://github.com/hyperspy/hyperspy/pull/2686>`_)
* Fix broken events when changing signal type `#2683 <https://github.com/hyperspy/hyperspy/pull/2683>`_
* Fix setting offset in rebin: the offset was changed in the wrong axis (`#2690 <https://github.com/hyperspy/hyperspy/pull/2690>`_)
* Fix reading XRF bruker file, close `#2689 <https://github.com/hyperspy/hyperspy/issues/2689>`_ (`#2694 <https://github.com/hyperspy/hyperspy/pull/2694>`_)


Enhancements
------------

* Widgets plotting improvement and add ``pick_tolerance`` to plot preferences (`#2615 <https://github.com/hyperspy/hyperspy/pull/2615>`_)
* Pass keyword argument to the image IO plugins (`#2627 <https://github.com/hyperspy/hyperspy/pull/2627>`_)
* Improve error message when file not found (`#2597 <https://github.com/hyperspy/hyperspy/pull/2597>`_)
* Add update instructions to user guide (`#2621 <https://github.com/hyperspy/hyperspy/pull/2621>`_)
* Improve plotting navigator of lazy signals, add ``navigator`` setter to lazy signals (`#2631 <https://github.com/hyperspy/hyperspy/pull/2631>`_)
* Use ``'dask_auto'`` when rechunk=True in :py:meth:`~._signals.lazy.LazySignal.change_dtype` for lazy signal (`#2645 <https://github.com/hyperspy/hyperspy/pull/2645>`_)
* Use dask chunking when saving lazy signal instead of rechunking and leave the user to decide what is the suitable chunking (`#2629 <https://github.com/hyperspy/hyperspy/pull/2629>`_)
* Added lazy reading support for FFT and DPC datasets in FEI emd datasets (`#2651 <https://github.com/hyperspy/hyperspy/pull/2651>`_).
* Improve error message when initialising SpanROI with left >= right (`#2604 <https://github.com/hyperspy/hyperspy/pull/2604>`_)
* Allow running the test suite without the pytest-mpl plugin (`#2624 <https://github.com/hyperspy/hyperspy/pull/2624>`_)
* Add Releasing guide (`#2595 <https://github.com/hyperspy/hyperspy/pull/2595>`_)
* Add support for python 3.9, fix deprecation warning with matplotlib 3.4.0 and bump minimum requirement to numpy 1.17.1 and dask 2.1.0. (`#2663 <https://github.com/hyperspy/hyperspy/pull/2663>`_)
* Use native endianess in numba jitted functions. (`#2678 <https://github.com/hyperspy/hyperspy/pull/2678>`_)
* Add option not to snap ROI when calling the :py:meth:`~.roi.BaseInteractiveROI.interactive` method of a ROI (`#2686 <https://github.com/hyperspy/hyperspy/pull/2686>`_)
* Make :py:class:`~.misc.utils.DictionaryTreeBrowser` lazy by default - see `#368 <https://github.com/hyperspy/hyperspy/issues/368>`_ (`#2623 <https://github.com/hyperspy/hyperspy/pull/2623>`_)
* Speed up setting CI on azure pipeline (`#2694 <https://github.com/hyperspy/hyperspy/pull/2694>`_)
* Improve performance issue with the map method of lazy signal (`#2617 <https://github.com/hyperspy/hyperspy/pull/2617>`_)
* Add option to copy/load original metadata in ``hs.stack`` and ``hs.load`` to avoid large ``original_metadata`` which can slowdown processing. Close `#1398 <https://github.com/hyperspy/hyperspy/issues/1398>`_, `#2045 <https://github.com/hyperspy/hyperspy/issues/2045>`_, `#2536 <https://github.com/hyperspy/hyperspy/issues/2536>`_ and `#1568 <https://github.com/hyperspy/hyperspy/issues/1568>`_. (`#2691 <https://github.com/hyperspy/hyperspy/pull/2691>`_)


Maintenance
-----------

* Fix warnings when building documentation (`#2596 <https://github.com/hyperspy/hyperspy/pull/2596>`_)
* Drop support for numpy<1.16, in line with NEP 29 and fix protochip reader for numpy 1.20 (`#2616 <https://github.com/hyperspy/hyperspy/pull/2616>`_)
* Run test suite against upstream dependencies (numpy, scipy, scikit-learn and scikit-image) (`#2616 <https://github.com/hyperspy/hyperspy/pull/2616>`_)
* Update external links in the loading data section of the user guide (`#2627 <https://github.com/hyperspy/hyperspy/pull/2627>`_)
* Fix various future and deprecation warnings from numpy and scikit-learn (`#2646 <https://github.com/hyperspy/hyperspy/pull/2646>`_)
* Fix ``iterpath`` VisibleDeprecationWarning when using :py:meth:`~.models.model1d.Model1D.fit_component` (`#2654 <https://github.com/hyperspy/hyperspy/pull/2654>`_)
* Add integration test suite documentation in the developer guide. (`#2663 <https://github.com/hyperspy/hyperspy/pull/2663>`_)
* Fix SkewNormal component compatibility with sympy 1.8 (`#2701 <https://github.com/hyperspy/hyperspy/pull/2701>`_)

.. _changes_1.6.1:

1.6.1 (2020-11-28)
===================

This is a maintenance release that adds compatibility with h5py 3.0 and includes
numerous bug fixes and enhancements.
See `the issue tracker
<https://github.com/hyperspy/hyperspy/milestone/41?closed=1>`__
for details.


.. _changes_1.6:

1.6.0 (2020-08-05)
===================

NEW
---

* Support for the following file formats:

  * :external+rsciio:ref:`digitalsurf-format`
  * :external+rsciio:ref:`elid-format`
  * :external+rsciio:ref:`nexus-format`
  * :external+rsciio:ref:`usid-format`
  * :external+rsciio:ref:`empad-format`
  * Prismatic EMD format, see :external+rsciio:ref:`emd-format`
* ``hyperspy._signals.eels.EELSSpectrum.print_edges_near_energy`` method
  that, if the `hyperspy-gui-ipywidgets package
  <https://github.com/hyperspy/hyperspy_gui_ipywidgets>`_
  is installed, includes an
  awesome interactive mode. See :external+exspy:ref:`eels_elemental_composition-label`.
* Model asymmetric line shape components:

  * :py:class:`~._components.doniach.Doniach`
  * :py:class:`~._components.split_voigt.SplitVoigt`
* :external+exspy:ref:`EDS absorption correction <eds_absorption-label>`.
* :ref:`Argand diagram for complex signals <complex.argand>`.
* :ref:`Multiple peak finding algorithms for 2D signals <peak_finding-label>`.
* :ref:`cluster_analysis-label`.

Enhancements
------------

* The :py:meth:`~.api.signals.BaseSignal.get_histogram` now uses numpy's
  `np.histogram_bin_edges()
  <https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html>`_
  and supports all of its ``bins`` keyword values.
* Further improvements to the contrast adjustment tool.
  Test it by pressing the ``h`` key on any image.
* The following components have been rewritten using
  :py:class:`~._components.expression.Expression`, boosting their
  speeds among other benefits.

  * :py:class:`~._components.arctan.Arctan`
  * :py:class:`~._components.voigt.Voigt`
  * :py:class:`~._components.heaviside.HeavisideStep`
* The model fitting :py:meth:`~hyperspy.model.BaseModel.fit` and
  :py:meth:`~hyperspy.model.BaseModel.multifit` methods have been vastly improved. See
  :ref:`model.fitting` and the API changes section below.
* New serpentine iteration path for multi-dimensional fitting.
  See :ref:`model.multidimensional-label`.
* The :py:func:`~.api.plot.plot_spectra`  function now listens to
  events to update the figure automatically.
  See :ref:`this example <sphx_glr_auto_examples_region_of_interest_ExtractLineProfile.py>`.
* Improve thread-based parallelism. Add ``max_workers`` argument to the
  :py:meth:`~.api.signals.BaseSignal.map` method, such that the user can directly
  control how many threads they launch.
* Many improvements to the :py:meth:`~.api.signals.BaseSignal.decomposition` and
  :py:meth:`~.api.signals.BaseSignal.blind_source_separation` methods, including support for
  scikit-learn like algorithms, better API and much improved documentation.
  See :ref:`ml-label` and the API changes section below.
* Add option to calculate the absolute thickness to the EELS
  ``hyperspy._signals.eels.EELSSpectrum.estimate_thickness`` method.
  See :external+exspy:ref:`eels_thickness-label`.
* Vastly improved performance and memory footprint of the
  :py:meth:`~.api.signals.Signal2D.estimate_shift2D` method.
* The :py:meth:`~.api.signals.Signal1D.remove_background` method can
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
* The :py:meth:`~.api.signals.Signal1D.plot` and
  :py:meth:`~.api.signals.Signal2D.plot` methods take a new keyword
  argument ``autoscale``. See :ref:`plot.customize_images` for details.
* The contrast editor and the decomposition methods can now operate on
  complex signals.
* The default colormap can now be set in
  :ref:`preferences <configuring-hyperspy-label>`.


API changes
-----------

* The :py:meth:`~.api.signals.Signal2D.plot` keyword argument
  ``saturated_pixels`` is deprecated. Please use
  ``vmin`` and/or ``vmax`` instead.
* The :py:func:`~.api.load` keyword argument ``dataset_name`` has been
  renamed to ``dataset_path``.
* The :py:meth:`~.api.signals.BaseSignal.set_signal_type` method no longer takes
  ``None``. Use the empty string ``""`` instead.
* The :py:meth:`~.api.signals.BaseSignal.get_histogram` ``bins`` keyword values
  have been renamed as follows for consistency with numpy:

  * ``"scotts"`` -> ``"scott"``,
  * ``"freedman"`` -> ``"fd"``
* Multiple changes to the syntax of the :py:meth:`~hyperspy.model.BaseModel.fit`
  and :py:meth:`~hyperspy.model.BaseModel.multifit` methods:

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

* The following :py:class:`~hyperspy.model.BaseModel` methods are now private:

  * ``hyperspy.model.BaseModel.set_boundaries``
  * ``hyperspy.model.BaseModel.set_mpfit_parameters_info``

* The ``comp_label`` keyword of the machine learning plotting functions
  has been renamed to ``title``.
* The :py:class:`~hyperspy.learn.rpca.orpca` constructor's ``learning_rate``
  keyword has been renamed to ``subspace_learning_rate``
* The :py:class:`~hyperspy.learn.rpca.orpca` constructor's ``momentum``
  keyword has been renamed to ``subspace_momentum``
* The :py:class:`~hyperspy.learn.svd_pca.svd_pca` constructor's ``centre`` keyword
  values have been renamed as follows:

  * ``"trials"`` -> ``"navigation"``
  * ``"variables"`` -> ``"signal"``
* The ``bounds`` keyword argument of the
  :py:meth:`~._signals.lazy.LazySignal.decomposition` is deprecated and will be removed.
* Several syntax changes in the :py:meth:`~.api.signals.BaseSignal.decomposition` method:

  * Several ``algorithm`` keyword values have been renamed as follows:

    * ``"svd"``: ``"SVD"``,
    * ``"fast_svd"``: ``"SVD"``,
    * ``"nmf"``: ``"NMF"``,
    * ``"fast_mlpca"``: ``"MLPCA"``,
    * ``"mlpca"``: ``"MLPCA"``,
    * ``"RPCA_GoDec"``: ``"RPCA"``,
  * The ``polyfit`` argument has been deprecated and will be removed.
    Use ``var_func`` instead.


.. _changes_1.5.2:


1.5.2 (2019-09-06)
===================

This is a maintenance release that adds compatibility with Numpy 1.17 and Dask
2.3.0 and fixes a bug in the Bruker reader. See `the issue tracker
<https://github.com/hyperspy/hyperspy/issues?q=label%3A"type%3A+bug"+is%3Aclosed+milestone%3Av1.5.2>`__
for details.


.. _changes_1.5.1:

1.5.1 (2019-07-28)
===================

This is a maintenance release that fixes some regressions introduced in v1.5.
Follow the following links for details on all the `bugs fixed
<https://github.com/hyperspy/hyperspy/issues?q=label%3A"type%3A+bug"+is%3Aclosed+milestone%3Av1.5.1>`__.


.. _changes_1.5:

1.5.0 (2019-07-27)
===================

NEW
---

* New method :py:meth:`hyperspy.component.Component.print_current_values`. See
  :ref:`the User Guide for details <Component.print_current_values>`.
* New :py:class:`hyperspy._components.skew_normal.SkewNormal` component.
* New :py:meth:`hyperspy.api.signals.BaseSignal.apply_apodization` method and
  ``apodization`` keyword for :py:meth:`hyperspy.api.signals.BaseSignal.fft`. See
  :ref:`signal.fft` for details.
* Estimation of number of significant components by the elbow method.
  See :ref:`mva.scree_plot`.

Enhancements
------------

* The contrast adjustment tool has been hugely improved. Test it by pressing the ``h`` key on any image.
* The :ref:`Developer Guide <dev_guide>` has been extended, enhanced and divided into
  chapters.
* Signals with signal dimension equal to 0 and navigation dimension 1 or 2 are
  automatically transposed when using
  :py:func:`hyperspy.api.plot.plot_images`
  or :py:func:`hyperspy.api.plot.plot_spectra` respectively. This is
  specially relevant when plotting the result of EDS quantification. See
  :external+exspy:ref:`eds-label` for examples.
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
  * :py:class:`exspy.components.VolumePlasmonDrude`
  * :py:class:`exspy.components.DoublePowerLaw`
  * The ``hyperspy._components.polynomial_deprecated.Polynomial``
    component will be deprecated in HyperSpy 2.0 in favour of the new
    :py:class:`hyperspy._components.polynomial.Polynomial` component, that is based on
    :py:class:`hyperspy._components.expression.Expression` and has an improved API. To
    start using the new component pass the ``legacy=False`` keyword to the
    the ``hyperspy._components.polynomial_deprecated.Polynomial`` component
    constructor.


For developers
--------------
* Drop support for python 3.5
* New extension mechanism that enables external packages to register HyperSpy
  objects. See :ref:`writing_extensions-label` for details.


.. _changes_1.4.2:

1.4.2 (2019-06-19)
===================

This is a maintenance release. Among many other fixes and enhancements, this
release fixes compatibility issues with Matplotlib v 3.1. Follow the
following links for details on all the `bugs fixed
<https://github.com/hyperspy/hyperspy/issues?q=label%3A"type%3A+bug"+is%3Aclosed+milestone%3Av1.4.2>`__
and `enhancements
<https://github.com/hyperspy/hyperspy/issues?q=is%3Aclosed+milestone%3Av1.4.2+label%3A"type%3A+enhancement">`__.


.. _changes_1.4.1:

1.4.1 (2018-10-23)
===================

This is a maintenance release. Follow the following links for details on all
the `bugs fixed
<https://github.com/hyperspy/hyperspy/issues?q=label%3A"type%3A+bug"+is%3Aclosed+milestone%3Av1.4.1>`__
and `enhancements
<https://github.com/hyperspy/hyperspy/issues?q=is%3Aclosed+milestone%3Av1.4.1+label%3A"type%3A+enhancement">`__.

This release fixes compatibility issues with Python 3.7.


.. _changes_1.4:

1.4.0 (2018-09-02)
===================

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

  * Reading FEI's Velox EMD file format based on the HDF5 open standard. See :external+rsciio:ref:`emd_fei-format`.
  * Reading Bruker's SPX format. See :external+rsciio:ref:`bruker-format`.
  * Reading and writing the mrcz open format. See :external+rsciio:ref:`mrcz-format`.
* New ``hyperspy.datasets.artificial_data`` module which contains functions for generating
  artificial data, for use in things like docstrings or for people to test
  HyperSpy functionalities. See :ref:`example-data-label`.
* New :meth:`~.api.signals.BaseSignal.fft` and :meth:`~.api.signals.BaseSignal.ifft` signal methods. See :ref:`signal.fft`.
* New :meth:`holospy.signals.HologramImage.statistics` method to compute useful hologram parameters. See :external+holospy:ref:`holography.stats-label`.
* Automatic axes units conversion and better units handling using `pint <https://pint.readthedocs.io/en/latest/>`__.
  See :ref:`quantity_and_converting_units`.
* New :class:`~.roi.Line2DROI` :meth:`~.roi.Line2DROI.angle` method. See :ref:`roi-label` for details.

Enhancements
------------

* :py:func:`~.api.plot.plot_images` improvements (see :ref:`plot.images` for details):

  * The ``cmap`` option of :py:func:`~.api.plot.plot_images`
    supports iterable types, allowing the user to specify different colormaps
    for the different images that are plotted by providing a list or other
    generator.
  * Clicking on an individual image updates it.
* New customizable keyboard shortcuts to navigate multi-dimensional datasets. See :ref:`visualization-label`.
* The :py:meth:`~.api.signals.Signal1D.remove_background` method now operates much faster
  in multi-dimensional datasets and adds the options to interatively plot the remainder of the operation and
  to set the removed background to zero. See :ref:`signal1D.remove_background` for details.
* The  :py:meth:`~.api.signals.Signal2D.plot` method now takes a ``norm`` keyword that can be "linear", "log",
  "auto"  or a matplotlib norm. See :ref:`plot.customize_images` for details.
  Moreover, there are three new extra keyword
  arguments, ``fft_shift`` and ``power_spectrum``, that are useful when plotting fourier transforms. See
  :ref:`signal.fft`.
* The :py:meth:`~.api.signals.Signal2D.align2D` and :py:meth:`~.api.signals.Signal2D.estimate_shift2D`
  can operate with sub-pixel accuracy using skimage's upsampled matrix-multiplication DFT. See :ref:`signal2D.align`.


.. _changes_1.3.2:

1.3.2 (2018-07-03)
===================

This is a maintenance release. Follow the following links for details on all
the `bugs fixed
<https://github.com/hyperspy/hyperspy/issues?q=label%3A"type%3A+bug"+is%3Aclosed+milestone%3Av1.3.2>`__
and `enhancements <https://github.com/hyperspy/hyperspy/issues?q=is%3Aclosed+milestone%3Av1.3.2+label%3A"type%3A+enhancement">`__.


.. _changes_1.3.1:

1.3.1 (2018-04-19)
===================

This is a maintenance release. Follow the following links for details on all
the `bugs fixed
<https://github.com/hyperspy/hyperspy/issues?q=label%3A"type%3A+bug"+is%3Aclosed+milestone%3Av1.3.1>`__
and `enhancements <https://github.com/hyperspy/hyperspy/issues?q=is%3Aclosed+milestone%3Av1.3.1+label%3A"type%3A+enhancement">`__.

Starting with this version, the HyperSpy WinPython Bundle distribution is
no longer released in sync with HyperSpy. For HyperSpy WinPython Bundle
releases see https://github.com/hyperspy/hyperspy-bundle


.. _changes_1.3:

1.3.0 (2017-05-27)
===================

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
* :py:meth:`~.api.signals.BaseSignal.rebin` supports upscaling and rebinning to
  arbitrary sizes through linear interpolation. See :ref:`rebin-label`. It also runs faster if `numba <http://numba.pydata.org/>`__ is installed.
* :py:attr:`~.axes.AxesManager.signal_extent` and :py:attr:`~.axes.AxesManager.navigation_extent` properties to easily get the extent of each space.
* New IPywidgets Graphical User Interface (GUI) elements for the `Jupyter Notebook <http://jupyter.org>`__.
  See the new `hyperspy_gui_ipywidgets <https://github.com/hyperspy/hyperspy_gui_ipywidgets>`__ package.
  It is not installed by default, see :ref:`install-label` for details.
* All the :ref:`roi-label` now have a ``gui`` method to display a GUI if
  at least one of HyperSpy's GUI packgages are installed.

Enhancements
------------
* Creating many markers is now much faster.
* New "Stage" metadata node. See :ref:`metadata_structure` for details.
* The Brucker file reader now supports the new version of the format. See :external+rsciio:ref:`bruker-format`.
* HyperSpy is now compatible with all matplotlib backends, including the nbagg which is
  particularly convenient for interactive data analysis in the
  `Jupyter Notebook <http://jupyter.org>`__ in combination with the new
  `hyperspy_gui_ipywidgets <https://github.com/hyperspy/hyperspy_gui_ipywidgets>`__ package.
  See :ref:`importing_hyperspy-label`.
* The ``vmin`` and ``vmax`` arguments of the
  :py:func:`~.api.plot.plot_images` function now accept lists to enable
  setting these parameters for each plot individually.
* The :py:meth:`~.api.signals.BaseSignal.plot_decomposition_results` and
  :py:meth:`~.api.signals.BaseSignal.plot_bss_results` methods now makes a better
  guess of the number of navigators (if any) required to visualise the
  components. (Previously they were always plotting four figures by default.)
* All functions that take a signal range can now take a :py:class:`~.roi.SpanROI`.
* The following ROIs can now be used for indexing or slicing (see :ref:`here <roi-slice-label>` for details):

  * :py:class:`~.api.roi.Point1DROI`
  * :py:class:`~.api.roi.Point2DROI`
  * :py:class:`~.api.roi.SpanROI`
  * :py:class:`~.api.roi.RectangularROI`


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

  * :py:meth:`~.api.signals.BaseSignal.plot_decomposition_results`.
  * :py:meth:`~.api.signals.BaseSignal.plot_decomposition_factors`.

* The default extension when saving to HDF5 following HyperSpy's specification
  is now ``hspy`` instead of ``hdf5``. See :external+rsciio:ref:`hspy-format`.

* The following methods are deprecated and will be removed in HyperSpy 2.0

  * ``.axes.AxesManager.show``. Use :py:meth:`~.axes.AxesManager.gui`
    instead.
  * All ``notebook_interaction`` method. Use the equivalent ``gui`` method
    instead.
  * ``hyperspy.api.signals.Signal1D.integrate_in_range``.
    Use :py:meth:`~.api.signals.BaseSignal.integrate1D` instead.

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
* The new ``hyperspy.ui_registry`` enables easy connection of external
  GUI elements to HyperSpy. This is the mechanism used to split the traitsui
  and ipywidgets GUI elements.


.. _changes_1.2:

1.2.0 (2017-02-02)
===================

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
* Parallel :py:meth:`~.api.signals.BaseSignal.map` and all the functions that use
  it internally (a good fraction of HyperSpy's functionaly). See
  :ref:`map-label`.
* :external+holospy:ref:`electron-holography-label` reconstruction.
* Support for reading :external+rsciio:ref:`edax-format` files.
* New signal methods :py:meth:`~.api.signals.BaseSignal.indexmin` and
  :py:meth:`~.api.signals.BaseSignal.valuemin`.

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
* :py:meth:`~.api.signals.BaseSignal.map` can now take functions that return
  differently-shaped arrays or arbitrary objects, see :ref:`map-label`.
* Add support for stacking multi-signal files. See :ref:`load-multiple-label`.
* Markers can now be saved to hdf5 and creating many markers is easier and
  faster. See :ref:`plot.markers`.
* Add option to save to HDF5 file using the ".hspy" extension instead of
  ".hdf5". See :external+rsciio:ref:`hspy-format`. This will be the default extension in
  HyperSpy 1.3.

For developers
--------------
* Most of HyperSpy plotting features are now covered by unittests. See
  :ref:`plot-test-label`.
* unittests migrated from nose to pytest. See :ref:`testing-label`.


.. _changes_1.1.2:

1.1.2 (2079-01-12)
===================

This is a maintenance release. Follow the following links for details on all
the `bugs fixed
<https://github.com/hyperspy/hyperspy/issues?q=label%3A"type%3A+bug"+is%3Aclosed+milestone%3Av1.1.2>`__
and `enhancements <https://github.com/hyperspy/hyperspy/issues?q=is%3Aclosed+milestone%3Av1.1.2+label%3A"type%3A+enhancement">`__.


.. _changes_1.1.1:

1.1.1 (2016-08-24)
===================

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

1.1.0 (2016-08-03)
===================

This is a minor release. Follow the following links for details on all
the `bugs fixed
<https://github.com/hyperspy/hyperspy/issues?q=label%3A"type%3A+bug"+is%3Aclosed+milestone%3A1.1>`__.

NEW
---

* :ref:`signal.transpose`.
* :external+rsciio:ref:`protochips-format` reader.

Enhancements
------------


* :py:meth:`~hyperspy.model.BaseModel.fit` takes a new algorithm, the global optimizer
  `differential evolution`.
* :py:meth:`~hyperspy.model.BaseModel.fit` algorithm, `leastsq`, inherits SciPy's bound
  constraints support (requires SciPy >= 0.17).
* :py:meth:`~hyperspy.model.BaseModel.fit` algorithm names changed to be consistent
  `scipy.optimze.minimize()` notation.



1.0.1 (2016-07-27)
===================

This is a maintenance release. Follow the following links for details on all
the `bugs fixed
<https://github.com/hyperspy/hyperspy/issues?q=label%3A"type%3A+bug"+is%3Aclosed+milestone%3A1.0.1>`__.


1.0.0 (2016-07-14)
===================

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
* Query and fetch spectr from :func:`exspy.data.eelsdb` from `The EELS Database <https://eelsdb.eu/>`__.
* :ref:`interactive-label`.
* :ref:`events-label`.

Model
^^^^^

* :ref:`SAMFire-label`.
* Store :ref:`models in hdf5 files <storing_models-label>`.
* Add :ref:`fancy indexing <model_indexing-label>` to `Model`.
* :ref:Two-dimensional model fitting (:py:class:`~.models.model2d.Model2D`).


EDS
^^^
* :external+exspy:ref:`Z-factors quantification <eds_quantification-label>`.
* :external+exspy:ref:`Cross section quantification <eds_quantification-label>`.
* :external+exspy:ref:`EDS curve fitting <eds_fitting-label>`.
* X-ray :external+exspy:ref:`absorption coefficient database <eds_absorption_db-label>`.

IO
^^
* Support for reading certain files without :ref:`loading them to memory <load_to_memory-label>`.
* :external+rsciio:ref:`Bruker's composite file (bcf) <bruker-format>` reading support.
* :external+rsciio:ref:`Electron Microscopy Datasets (EMD) <emd-format>` read and write support.
* :external+rsciio:ref:`SEMPER unf <semper-format>` read and write support.
* :external+rsciio:ref:`DENS heat log <dens-format>` read support.
* :external+rsciio:ref:`NanoMegas blockfile <blockfile-format>` read and write support.

Enhancements
------------
* More useful ``AxesManager`` repr string with html repr for Jupyter Notebook.
* Better progress bar (`tqdm <https://github.com/noamraph/tqdm>`__).
* Add support for :external+rsciio:ref:`writing/reading scale and unit to tif files
  <tiff-format>` to be read with ImageJ or DigitalMicrograph.

Documentation
-------------

* The following sections of the User Guide were revised and largely overwritten:

  * :ref:`install-label`.
  * :ref:`ml-label`.
  * :external+exspy:ref:`eds-label`.
* New :ref:`dev_guide`.


API changes
-----------

* Split :ref:`components <model_components-label>` into ``components1D`` and ``components2D``.
* Remove ``record_by`` from metadata.
* Remove simulation classes.
* The :py:class:`~.api.signals.Signal1D`,
  ``hyperspy._signals.image.Signal2D`` and :py:class:`~.api.signals.BaseSignal`
  classes deprecated the old `Spectrum` `Image` and `Signal` classes.



0.8.5 (2016-07-02)
===================


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

The new :py:class:`~.api.signals.BaseSignal`,
:py:class:`~.api.signals.Signal1D` and
:py:class:`~.api.signals.Signal2D` deprecate ``hyperspy.signal.Signal``,
:py:class:`~.api.signals.Signal1D` and :py:class:`~.api.signals.Signal2D`
respectively. Also ``as_signal1D``, ``as_signal2D```, ``to_signal1D`` and ``to_signal2D``
deprecate ``as_signal1D``, ``as_signal2D``, ``to_spectrum`` and ``to_image``. See `#963
<https://github.com/hyperspy/hyperspy/pull/963>`__ and `#943
<https://github.com/hyperspy/hyperspy/issues/943>`__ for details.


0.8.4 (2016-03-04)
===================

This release adds support for Python 3 and drops support for Python 2. In all
other respects it is identical to 0.8.3.

0.8.3 (2016-03-04)
===================

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

0.8.2 (2015-08-13)
===================

This is a maintenance release that fixes an issue with the Python installers. Those who have successfully installed 0.8.1 do not need to upgrade.

.. _changes_0.8.1:

0.8.1 (2015-08-12)
===================

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
* ``hyperspy.signal.Signal.unfolded`` context manager.
* ``hyperspy.signal.Signal.derivative`` method.
* :ref:`syntax to access the components in the model <model_components-label>`
  that includes pretty printing of the components.

API changes
-----------

* ``hyperspy.hspy`` is now deprecated in favour of the new
  :py:mod:`hyperspy.api`. The new API renames and/or move several modules as
  folows:

    * ``hspy.components`` -> ``hyperspy.api.model.components``
    * ``hspy.utils``-> ``hyperspy.api``
    * ``hspy.utils.markers`` ``hyperspy.api.plot.markers``
    * ``hspy.utils.example_signals`` -> ``hyperspy.api.datasets.example_signals``


    In HyperSpy 0.8.1 the full content of ``hyperspy.hspy`` is still
    imported in the user namespace, but this can now be disabled in
    ``hs.preferences.General.import_hspy``. In Hyperspy 1.0 it will be
    disabled by default and the ``hyperspy.hspy`` module will be fully
    removed in HyperSpy 0.10. We encourage all users to migrate to the new
    syntax. For more details see :ref:`importing_hyperspy-label`.
* Indexing the ``hyperspy.signal.Signal`` class is now deprecated. We encourage
  all users to use ``isig`` and ``inav`` instead for indexing.
* ``hyperspy.hspy.create_model`` is now deprecated in favour of the new
  equivalent ``hyperspy.signal.Signal.create_model`` ``Signal`` method.
* ``hyperspy.signal.Signal.unfold_if_multidim`` is deprecated.


.. _changes_0.8:

0.8.0 (2015-04-07)
===================

New features
------------

Core
^^^^

* :py:meth:`~.api.signals.Signal1D.spikes_removal_tool` displays derivative max value when used with
  GUI.
* Progress-bar can now be suppressed by passing ``show_progressbar`` argument to all functions that generate
  it.

IO
^^

* HDF5 file format now supports saving lists, tuples, binary strings and signals in metadata.


Plotting
^^^^^^^^

* New class, ``hyperspy.drawing.marker.MarkerBase``, to plot markers with ``hspy.utils.plot.markers`` module.  See :ref:`plot.markers`.
* New method to plot images with the :py:func:`~.api.plot.plot_images` function in  ``hspy.utils.plot.plot_images``. See :ref:`plot.images`.
* Improved ``hyperspy._signals.image.Signal2D.plot`` method to customize the image. See :ref:`plot.customize_images`.

EDS
^^^

* New method for quantifying EDS TEM spectra using Cliff-Lorimer method, ``hyperspy._signals.eds_tem.EDSTEMSpectrum.quantification``. See :external+exspy:ref:`eds_quantification-label`.
* New method to estimate for background subtraction, ``hyperspy._signals.eds.EDSSpectrum.estimate_background_windows``. See :external+exspy:ref:`eds_background_subtraction-label`.
* New method to estimate the windows of integration, ``hyperspy._signals.eds.EDSSpectrum.estimate_integration_windows``.
* New specific ``hyperspy._signals.eds.EDSSpectrum.plot`` method, with markers to indicate the X-ray lines, the window of integration or/and the windows for background subtraction. See :external+exspy:ref:`eds_plot_markers-label`.
* New examples of signal in the ``hspy.utils.example_signals`` module.

  + ``hyperspy.misc.example_signals_loading.load_1D_EDS_SEM_spectrum``
  + ``hyperspy.misc.example_signals_loading.load_1D_EDS_TEM_spectrum``

* New method to mask the vaccum, ``hyperspy._signals.eds_tem.EDSTEMSpectrum.vacuum_mask`` and a specific ``hyperspy._signals.eds_tem.EDSTEMSpectrum.decomposition`` method that incoroporate the vacuum mask

API changes
-----------

* :py:class:`~hyperspy.component.Component` and :py:class:`~hyperspy.component.Parameter` now inherit ``traits.api.HasTraits``
  that enable ``traitsui`` to modify these objects.
* ``hyperspy.misc.utils.attrsetter`` is added, behaving as the default python ``setattr`` with nested
  attributes.
* Several widget functions were made internal and/or renamed:
    + ``add_patch_to`` -> ``_add_patch_to``
    + ``set_patch`` -> ``_set_patch``
    + ``onmove`` -> ``_onmousemove``
    + ``update_patch_position`` -> ``_update_patch_position``
    + ``update_patch_size`` -> ``_update_patch_size``
    + ``add_axes`` -> ``set_mpl_ax``

0.7.3 (2015-08-22)
===================

This is a maintenance release. A list of fixed issues is available in the
`0.7.3 milestone
<https://github.com/hyperspy/hyperspy/issues?milestone=6&page=1&state=closed>`__
in the github repository.

.. _changes_0.7.2:

0.7.2 (2015-08-22)
===================

This is a maintenance release. A list of fixed issues is available in the
`0.7.2 milestone
<https://github.com/hyperspy/hyperspy/issues?milestone=5&page=1&state=closed>`__
in the github repository.

.. _changes_0.7.1:

0.7.1 (2015-06-17)
===================

This is a maintenance release. A list of fixed issues is available in the
`0.7.1 milestone
<https://github.com/hyperspy/hyperspy/issues?milestone=4&page=1&state=closed>`__
in the github repository.


New features
------------

* Add suspend/resume model plot updating. See :ref:`model.visualization`.

0.7.0 (2014-04-03)
===================

New features
------------

Core
^^^^

* New syntax to index the :py:class:`~.axes.AxesManager`.
* New Signal methods to transform between Signal subclasses. More information
  :ref:`here <transforming_signal-label>`.

  + ``hyperspy.signal.Signal.set_signal_type``
  + ``hyperspy.signal.Signal.set_signal_origin``
  + ``hyperspy.signal.Signal.as_signal2D``
  + ``hyperspy.signal.Signal.as_signal1D``

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
* Added a CLI to ``hyperspy.signal.Signal1DTools.remove_background``.
* New ``hyperspy.signal.Signal1DTools.estimate_peak_width`` method to estimate
  peak width.
* New methods to integrate over one axis:
  ``hyperspy.signal.Signal.integrate1D`` and
  ``hyperspy.signal.Signal1DTools.integrate_in_range``.
* New ``hyperspy.signal.Signal.metadata`` attribute, ``Signal.binned``. Several
  methods behave differently on binned and unbinned signals.
  See :ref:`signal.binned`.
* New ``hyperspy.signal.Signal.map`` method to easily transform the
  data using a function that operates on individual signals. See
  :ref:`signal.iterator`.
* New ``hyperspy.signal.Signal.get_histogram`` and
  ``hyperspy.signal.Signal.print_summary_statistics`` methods.
* The spikes removal tool has been moved to the :class:`~.api.signals.Signal1D`
  class so that it is available for all its subclasses.
* The ``hyperspy.signal.Signal.split``` method now can automatically split back
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
* Added several extra options to the Signal ``hyperspy.signal.Signal.plot``
  method to customize the navigator. See :ref:`visualization-label`.
* Add compatibility with IPython's matplotlib inline plotting.
* New function, :py:func:`~.api.plot.plot_spectra`, to plot several
  spectra in the same figure. See :ref:`plot.spectra`.
* New function, :py:func:`~.api.plot.plot_signals`, to plot several
  signals at the same time. See :ref:`plot.signals`.
* New function, :py:func:`~.api.plot.plot_histograms`, to plot the histrograms
  of several signals at the same time. See :ref:`plot.signals`.

Curve fitting
^^^^^^^^^^^^^

* The chi-squared, reduced chi-squared and the degrees of freedom are
  computed automatically when fitting. See :ref:`model.fitting`.
* New functionality to plot the individual components of a model. See
  :ref:`model.visualization`.
* New method, ``hyperspy.model.Model.fit_component``, to help setting the
  starting parameters. See :ref:`model.starting`.

Machine learning
^^^^^^^^^^^^^^^^

* The PCA scree plot can now be easily obtained as a Signal. See
  :ref:`mva.scree_plot`.
* The decomposition and blind source separation components can now be obtained
  as ``hyperspy.signal.Signal`` instances. See :ref:`mva.get_results`.
* New methods to plot the decomposition and blind source separation results
  that support n-dimensional loadings. See :ref:`mva.visualization`.

Dielectric function
^^^^^^^^^^^^^^^^^^^

* New ``hyperspy.signal.Signal`` subclass,
  ``hyperspy._signals.dielectric_function.DielectricFunction``.

EELS
^^^^

* New method,
  ``hyperspy._signals.eels.EELSSpectrum.kramers_kronig_analysis`` to calculate
  the dielectric function from low-loss electron energy-loss spectra based on
  the Kramers-Kronig relations. See :external+exspy:ref:`eels.kk`.
* New method to align the zero-loss peak,
  ``hyperspy._signals.eels.EELSSpectrum.align_zero_loss_peak``.

EDS
^^^

* New signal, EDSSpectrum especialized in EDS data analysis, with subsignal
  for EDS with SEM and with TEM: EDSSEMSpectrum and EDSTEMSpectrum. See
  :external+exspy:ref:`eds-label`.
* New database of EDS lines available in the ``elements`` attribute of the
  ``hspy.utils.material`` module.
* Adapted methods to calibrate the spectrum, the detector and the microscope.
  See :external+exspy:ref:`eds_calibration-label`.
* Specific methods to describe the sample,
  ``hyperspy._signals.eds.EDSSpectrum.add_elements`` and
  ``hyperspy._signals.eds.EDSSpectrum.add_lines``. See :external+exspy:ref:`eds_sample-label`
* New method to get the intensity of specific X-ray lines:
  ``hyperspy._signals.eds.EDSSpectrum.get_lines_intensity``. See
  :external+exspy:ref:`eds_sample-label`

API changes
-----------

* hyperspy.misc has been reorganized. Most of the functions in misc.utils has
  been rellocated to specialized modules. misc.utils is no longer imported in
  ``hyperspy.hspy``. A new ``hyperspy.utils`` module is imported instead.
* Objects that have been renamed

  + ``hspy.elements`` -> ``utils.material.elements``.
  + ``Signal.navigation_indexer`` -> ``inav``.
  + ``Signal.signal_indexer`` -> ``isig``.
  + ``Signal.mapped_parameters`` -> ``Signal.metadata``.
  + ``Signal.original_parameters`` -> ``Signal.original_metadata``.
* The metadata has been reorganized. See :ref:`metadata_structure`.
* The following signal methods now operate out-of-place:

  + ``hyperspy.signal.Signal.swap_axes``
  + ``hyperspy.signal.Signal.rebin``

.. _changes_0.6:

0.6.0 (2013-05-25)
===================

New features
------------

* Signal now supports indexing and slicing. See :ref:`signal.indexing`.
* Most arithmetic and rich arithmetic operators work with signal.
  See :ref:`signal.operations`.
* Much improved EELSSpectrum methods:
  ``hyperspy._signals.eels.EELSSpectrum.estimate_zero_loss_peak_centre``,
  ``hyperspy._signals.eels.EELSSpectrum.estimate_elastic_scattering_intensity`` and
  ``hyperspy._signals.eels.EELSSpectrum.estimate_elastic_scattering_threshold``.

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
  ``hyperspy.components.eels_cl_edge.EELSCLEdge.get_fine_structure_as_spectrum``
  EELSCLEdge method.
* New ``hyperspy.components.arctan.Arctan`` model component.
* New
  ``hyperspy.model.Model.enable_adjust_position``
  and ``hyperspy.model.Model.disable_adjust_position``
  to easily change the position of components using the mouse on the plot.
* New Model methods
  ``hyperspy.model.Model.set_parameters_value``,
  ``hyperspy.model.Model.set_parameters_free`` and
  ``hyperspy.model.Model.set_parameters_not_free``
  to easily set several important component attributes of a list of components
  at once.
* New
  :py:func:`~.api.stack` function to stack signals.
* New Signal methods:
  ``hyperspy.signal.Signal.integrate_simpson``,
  ``hyperspy.signal.Signal.max``,
  ``hyperspy.signal.Signal.min``,
  ``hyperspy.signal.Signal.var``, and
  ``hyperspy.signal.Signal.std``.
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

0.5.1 (2012-09-28)
===================

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
* EELSSpectrum.find_low_loss_centre was renamed to estimate_zero_loss_peak_centre.
* EELSSpectrum.calculate_FWHM was renamed to estimate_FWHM.

.. _changes_0.5:

0.5.0 (2012-09-07)
===================

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

0.4.1 (2012-04-16)
===================

New features
------------

 * Added TIFF 16, 32 and 64 bits support by using (and distributing) Christoph Gohlke's `tifffile library <https://pypi.org/project/tifffile/>`_.
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

0.4.0 (2012-02-29)
===================

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
