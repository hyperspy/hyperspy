Introduction
============

What is HyperSpy
----------------

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


.. _hyperspy_extensions-label:

External packages can extend HyperSpy to e.g. implement features to analyse a
particular sort of data. For details on how to register extensions see
:ref:`writing_extensions-label`. For a list of packages that extend HyperSpy
follow `this link <https://github.com/hyperspy/hyperspy-extensions-list>`_.

.. note::
    From version 2.0, HyperSpy will be split into a core package (HyperSpy)
    that will provide the common infrastructure and a number of HyperSpy
    extensions specialized in the analysis of different types of data.

HyperSpy's character
--------------------

HyperSpy has been written by a subset of the people who use it, a particularity
that sets its character:

* To us this program is a research tool, much like a screwdriver or a Green's
  function. We believe that the better our tools are, the better our research
  will be. We also think that it is beneficial for the advancement of knowledge
  to share our research tools and to forge them in a collaborative way. This is
  because by collaborating we advance faster, mainly by avoiding reinventing the
  wheel. Idealistic as it may sound, many other people think like this and it is
  thanks to them that this project exists.

* Not surprisingly, we care about making it easy for others to contribute to
  HyperSpy. In other words,
  we aim at minimising the “user becomes developer” threshold.
  Do you want to contribute already? No problem, see the :ref:`dev_guide-label`
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
  need a full, standalone GUI, `HyperSpyUI <http://hyperspy.org/hyperspyUI/>`_
  is for you.
