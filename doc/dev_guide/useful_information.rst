
.. _useful_information-label:

Useful information
==================

NEP 29 — Recommend Python and Numpy version support
---------------------------------------------------

Abstract
^^^^^^^^

`NEP 29 <https://numpy.org/neps/nep-0029-deprecation_policy.html>`_
(NumPy Enhancement Proposals) recommends that all projects across the
Scientific Python ecosystem adopt a common “time window-based” policy for
support of Python and NumPy versions. Standardizing a recommendation for
project support of minimum Python and NumPy versions will improve downstream
project planning.

Implementation recommendation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This project supports:

* All minor versions of Python released 42 months prior to the project, and
  at minimum the two latest minor versions.
* All minor versions of ``numpy`` released in the 24 months prior to the project,
  and at minimum the last three minor versions.

In ``setup.py``, the ``python_requires`` variable should be set to the minimum
supported version of Python. All supported minor versions of Python should be
in the test matrix and have binary artifacts built for the release.

Minimum Python and NumPy version support should be adjusted upward on every
major and minor release, but never on a patch release.

Conda-forge packaging
---------------------

The feedstock for the conda package lives in the conda-forge organisation on
github: `conda-forge/hyperspy-feedstock <https://github.com/conda-forge/hyperspy-feedstock>`_.

Monitoring version distribution
-------------------------------

Download metrics are available from pypi and Anaconda cloud but the reliability
of these numbers is poor for the following reason:

* hyperspy is distributed by other means: the
  `hyperspy-bundle <https://github.com/hyperspy/hyperspy-bundle>`_, or by
  various linux distribution (Arch-Linux, openSUSE)
* these packages may be used by continuous integration of other python libraries

However, distribution of download version can be useful to identify
issues, such as version pinning or library incompatibilities. Various service
processing the `pypi data <https://packaging.python.org/guides/analyzing-pypi-package-downloads/>`_
are available online:

* `pepy.tech <https://pepy.tech/project/hyperspy>`_
* `libraries.io <https://libraries.io/pypi/hyperspy/usage>`_
* `pypistats.org <https://pypistats.org/packages/hyperspy>`_
