.. -*- mode: rst -*-

|Travis|_ |AppVeyor|_ |Coveralls|_ |pypi_version|_  |rtd|_ |gitter|_ |saythanks|_

.. |Travis| image:: https://api.travis-ci.org/hyperspy/hyperspy.png?branch=RELEASE_next_minor
.. _Travis: https://travis-ci.org/hyperspy/hyperspy

.. |AppVeyor| image:: https://ci.appveyor.com/api/projects/status/github/hyperspy/hyperspy?svg=true&branch=RELEASE_next_minor
.. _AppVeyor: https://ci.appveyor.com/project/hyperspy/hyperspy/branch/RELEASE_next_minor

.. |Coveralls| image:: https://coveralls.io/repos/hyperspy/hyperspy/badge.svg
.. _Coveralls: https://coveralls.io/r/hyperspy/hyperspy

.. |pypi_version| image:: http://img.shields.io/pypi/v/hyperspy.svg?style=flat
.. _pypi_version: https://pypi.python.org/pypi/hyperspy

.. |rtd| image:: https://readthedocs.org/projects/hyperspy/badge/?version=latest
.. _rtd: https://readthedocs.org/projects/hyperspy/?badge=latest

.. |gitter| image:: https://badges.gitter.im/Join%20Chat.svg
.. _gitter: https://gitter.im/hyperspy/hyperspy?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

.. |saythanks| image:: https://img.shields.io/badge/say%20-thanks!-orange.svg
.. _saythanks: https://saythanks.io/to/hyperspy 


HyperSpy is an open source Python library which provides tools to facilitate
the interactive data analysis of multidimensional datasets that can be
described as multidimensional arrays of a given signal (e.g. a 2D array of
spectra a.k.a spectrum image).

HyperSpy aims at making it easy and natural to apply analytical procedures that
operate on an individual signal to multidimensional arrays, as well as
providing easy access to analytical tools that exploit the multidimensionality
of the dataset.

Its modular structure makes it easy to add features to analyze different kinds
of signals. Currently there are specialized tools to analyze electron
energy-loss spectroscopy (EELS) and energy dispersive X-rays (EDX) data.

HyperSpy is released under the GPL v3 license.

.. warning::

    **Since version 0.8.4 HyperSpy only supports Python 3. If you need to install
    HyperSpy in Python 2.7 install HyperSpy 0.8.3.**

Cite
----

|DOI|_

.. |DOI| image:: https://zenodo.org/badge/doi/10.5281/zenodo.240660.svg
.. _DOI: http://dx.doi.org/10.5281/zenodo.240660
