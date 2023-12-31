
What is HyperSpy
================

HyperSpy is an open source Python library which provides tools to facilitate
the interactive data analysis of multidimensional datasets that can be
described as multidimensional arrays of a given signal (e.g. a 2D array of
spectra a.k.a spectrum image).

HyperSpy aims at making it easy and natural to apply analytical procedures
that operate on an individual signal to multidimensional datasets of any
size, as well as providing easy access to analytical tools that exploit their
multidimensionality.

.. versionadded:: 1.5
  External packages can extend HyperSpy by registering signals,
  components and widgets.


The functionality of HyperSpy can be extended by external packages, e.g. to
implement features for analyzing a particular sort of data (usually related to a
specific set of experimental methods). A `list of packages that extend HyperSpy
<https://github.com/hyperspy/hyperspy-extensions-list>`_ is curated in a
dedicated repository. For details on how to register extensions see
:ref:`writing_extensions-label`.


.. versionchanged:: 2.0
    HyperSpy was split into a core package (HyperSpy) that provides the common
    infrastructure for multidimensional datasets and the dedicated IO package
    :external+rsciio:doc:`RosettaSciIO <index>`. Signal classes focused on
    specific types of data previously included in HyperSpy (EELS, EDS, Holography)
    were moved to specialized `HyperSpy extensions
    <https://github.com/hyperspy/hyperspy-extensions-list>`_.


HyperSpy's character
====================

HyperSpy has been written by a subset of the people who use it, a particularity
that sets its character:

* To us, this program is a research tool, much like a screwdriver or a Green's
  function. We believe that the better our tools are, the better our research
  will be. We also think that it is beneficial for the advancement of knowledge
  to share our research tools and to forge them in a collaborative way. This is
  because by collaborating we advance faster, mainly by avoiding reinventing the
  wheel. Idealistic as it may sound, many other people think like this and it is
  thanks to them that this project exists.

* Not surprisingly, we care about making it easy for others to contribute to
  HyperSpy. In other words,
  we aim at minimising the “user becomes developer” threshold.
  Do you want to contribute already? No problem, see the :ref:`dev_guide`
  for details.

* The main way of interacting with the program is through scripting.
  This is because `Jupyter <https://jupyter.org/>`_ exists, making your
  interactive data analysis productive, scalable, reproducible and,
  most importantly, fun. That said, widgets to interact with HyperSpy
  elements are provided where there
  is a clear productivity advantage in doing so. See the
  `hyperspy-gui-ipywidgets <https://github.com/hyperspy/hyperspy_gui_ipywidgets>`_
  and
  `hyperspy-gui-traitsui <https://github.com/hyperspy/hyperspy_gui_traitsui>`_
  packages for details. Not enough? If you
  need a full, standalone GUI, `HyperSpyUI <https://hyperspy.org/hyperspyUI/>`_
  is for you.

Learning resources
==================

.. grid:: 2 3 3 3
  :gutter: 2

  .. grid-item-card::
    :link: user_guide/install
    :link-type: doc

    :octicon:`rocket;2em;sd-text-info` Getting Started
    ^^^

    New to HyperSpy or Python? The getting started guide provides an
    introduction on basic usage of HyperSpy and how to install it.

  .. grid-item-card::
    :link: user_guide/index
    :link-type: doc

    :octicon:`book;2em;sd-text-info` User Guide
    ^^^

    The user guide provides in-depth information on key concepts of HyperSpy
    and how to use it along with background information and explanations.

  .. grid-item-card::
    :link: reference/index
    :link-type: doc

    :octicon:`code-square;2em;sd-text-info` Reference
    ^^^

    Documentation of the metadata specification and of the Application Progamming Interface (API),
    which describe how HyperSpy functions work and which parameters can be used.

  .. grid-item-card::
    :link: auto_examples/index
    :link-type: doc

    :octicon:`zap;2em;sd-text-info` Examples
    ^^^

    Gallery of short examples illustrating simple tasks that can be performed with HyperSpy.

  .. grid-item-card::
    :link: https://github.com/hyperspy/hyperspy-demos

    :octicon:`workflow;2em;sd-text-info` Tutorials
    ^^^

    Tutorials in form of Jupyter Notebooks to learn how to
    process multi-dimensional data using HyperSpy.

  .. grid-item-card::
    :link: dev_guide/index
    :link-type: doc

    :octicon:`people;2em;sd-text-info` Contributing
    ^^^

    HyperSpy is a community project maintained for and by its users.
    There are many ways you can help!

Citing HyperSpy
================

If HyperSpy has been significant to a project that leads to an academic
publication, please acknowledge that fact by citing it. The DOI in the
badge below is the `Concept DOI <https://help.zenodo.org/faq/#versioning>`_ of
HyperSpy. It can be used to cite the project without referring to a specific
version. If you are citing HyperSpy because you have used it to process data,
please use the DOI of the specific version that you have employed. You can
find iy by clicking on the DOI badge below.

.. image:: https://zenodo.org/badge/doi/10.5281/zenodo.592838.svg
   :target: https://doi.org/10.5281/zenodo.592838

HyperSpy's citation in the scientific literature
------------------------------------------------

Given the increasing number of articles that cite HyperSpy we do not maintain a list of
articles citing HyperSpy. For an up to date list search for
HyperSpy in a scientific database e.g. `Google Scholar
<https://scholar.google.co.uk/scholar?q=hyperspy>`_.

.. Note::
    Articles published before 2012 may mention the HyperSpy project under
    its old name, `EELSLab`.

Credits
=======

.. include:: ../AUTHORS.txt
