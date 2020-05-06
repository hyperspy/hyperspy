

.. _testing-label:

Running and writing tests
=========================

Writing tests
^^^^^^^^^^^^^

Every new function that is written in to HyperSpy needs to be tested and
documented. HyperSpy uses the `pytest <http://doc.pytest.org/>`_ library
for testing. The tests reside in the ``hyperspy.tests`` module.

Tests are short functions, found in ``hyperspy/tests``, that call your functions
under some known conditions and check the outputs against known values. They
should depend on as few other features as possible so that when they break
we know exactly what caused it. Ideally, the tests should be written at the
same time than the code itself, as they are very convenient to run to check
outputs when coding. Writing tests can seem laborious but you'll probably
soon find that they're very important as they force you to sanity check all
you do.

**Useful hints on testing:**

* When comparing integers, it's fine to use ``==``
* When comparing floats, be sure to use :py:meth:`np.testing.assert_almost_equal`
  or :py:meth:`np.testing.assert_allclose()`
* :py:meth:`np.testing.assert_allclose()` is also convenient for comparing
  numpy arrays
* The ``hyperspy.misc.test_utils.py`` contains a few useful functions for
  testing
* ``@pytest.mark.parametrize()`` is a very convenient decorator to test several
  parameters of the same function without having to write to much repetitive
  code, which is often error-prone. See `pytest documentation
  <http://doc.pytest.org/en/latest/parametrize.html>`_ for more details.
* It is good to check that the tests does not use too much of memory after
  creating new tests. If you need to explicitly delete your objects and free
  memory, you can do the following to release the memory associated to the
  ``s`` object, for example:

.. code:: python

   >>> del s
   >>> gc.collect()


Running tests
^^^^^^^^^^^^^

First ensure pytest and its plugins are installed by:

.. code:: bash

   # If using a standard hyperspy install
   pip install hyperspy[test]

   # Or, from a hyperspy local development directory
   pip install -e .[test]

   # Or just installing the dependencies using conda
   conda install -c conda-forge pytest pytest-mpl

To run them:

.. code:: bash

   pytest --mpl --pyargs hyperspy

Or, from HyperSpy's project folder, simply:

.. code:: bash

   pytest

.. note::

  pytest configuration options are set in the ``setup.cfg`` file, under the
  ``[tool:pytest]`` section. See the `pytest configuration documentation
  <https://docs.pytest.org/en/latest/customize.html>`_ for more details.

Test coverage
^^^^^^^^^^^^^

Once, you have pushed your pull request to the official HyperSpy repository,
it can be useful to check the coverage of your tests using the
`codecov.io <https://codecov.io/gh/hyperspy/hyperspy>`_ check of
your PR. There should be a link to it at the bottom of your PR on the Github
PR page. This service can help you to find how well your code is being tested
and exactly which parts are not currently tested.

You can also measure code coverage locally. If you have installed ``pytest-cov``,
you can run (from HyperSpy's project folder):

.. code:: bash

   pytest --cov=hyperspy

Configuration options for code coverage are also set in the ``setup.cfg`` file,
under the ``[coverage:run]`` and ``[coverage:report]`` sections. See the `coverage
documentation <https://coverage.readthedocs.io/en/coverage-5.1/config.html>`_
for more details.

Continuous integration (CI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The test suite is run using continuous integration services provided by
`Travis CI <https://travis-ci.org/github/hyperspy/hyperspy>`_ and
`Azure Pipeline <https://dev.azure.com/franciscode-la-pena-manchon/hyperspy/_build>`_.
The CI helper scripts are pulled from the
`ci-scripts <https://github.com/hyperspy/ci-scripts>`_ repository.

The testing matrix is as follow:

- **Travis CI**: test all supported python versions on Linux; all dependencies
  are pulled from `pypi <https://pypi.org>`_,
- **Azure Pipeline**: test a range of python version on Linux, MacOS and Windows;
  all dependencies are pulled from `anaconda cloud <https://anaconda.org/>`_
  using the `Anaconda "defaults" <https://anaconda.org/anaconda>`_ and the
  `"conda-forge" <https://anaconda.org/conda-forge>`_ channel (in this order of
  priority)

This testing matrix has been designed to be simple and easy to maintain and also
to ensure that packages from pypi and Anaconda cloud are not mixed in order to
avoid red herring failures of the test suite caused by application binary
interface (ABI) incompatibility between dependencies.

The most recent versions of packages will be available first on pypi and later
on anaconda cloud. It means that if a recent release of a dependency breaks the
test suite, it should happen first on travis - usual suspect would be
matplotlib, numpy, scipy, etc. Similarly, deprecation warning should appear
first on Travis CI.

The build of the doc is done on Travis CI and it is worth checking that no new
warnings have been introduced when writing documentation in the user guide or
in the docstring.


.. _plot-test-label:

Plot testing
^^^^^^^^^^^^
Plotting is tested using the ``@pytest.mark.mpl_image_compare`` decorator of
the `pytest mpl plugin <https://pypi.python.org/pypi/pytest-mpl>`_.  This
decorator uses reference images to compare with the generated output during the
tests. The reference images are located in the folder defined by the argument
``baseline_dir`` of the ``@pytest.mark.mpl_image_compare`` decorator.

To run plot tests, you simply need to add the option ``--mpl``:
::

    pytest --mpl

If you don't use ``--mpl``, the code of the tests will be executed, but the
images will not be compared to the references images.

If you need to add or change some plots, follow the workflow below:

    1. Write the tests using appropriate decorators such as
       ``@pytest.mark.mpl_image_compare``.
    2. If you need to generate a new reference image in the folder
       ``plot_test_dir``, for example, run: ``pytest
       --mpl-generate-path=plot_test_dir``
    3. Run again the tests and this time they should pass.
    4. Use ``git add`` to put the new file in the git repository.

When the plotting tests are failing, it is possible to download the figure
comparison images generated by pytest-mpl in the artifacts tabs of the
corresponding build on azure pipeline:

.. figure:: ../user_guide/images/azure_pipeline_artifacts.png


The plotting tests are tested on azure pipeline against a specific version of
matplotlib defined in ``conda_environment_dev.yml`` since small changes in the
way matplotlib generates the figure can make the tests fail.

For plotting tests, the matplotlib backend is set to ``agg`` by setting
the ``MPLBACKEND`` environment variable to ``agg``. At the first import of
``matplotlib.pyplot``, matplotlib will look at the ``MPLBACKEND`` environment
variable and accordingly set the backend.

Exporting pytest results as HTML
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
With ``pytest-html`` it is possible to export the results of running pytest
for easier viewing. It can be installed by conda:

.. code:: bash

   conda install pytest-html

and run by:

.. code:: bash

   pytest --mpl --html=report.html

See `pytest-mpl <https://pypi.python.org/pypi/pytest-mpl>`_ for more details.


