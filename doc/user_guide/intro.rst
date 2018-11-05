Introduction
============

What is HyperSpy
----------------

HyperSpy is an open source Python library which provides tools to facilitate
the interactive data analysis of multidimensional datasets that can be
described as multidimensional arrays of a given signal (e.g. a 2D array of
spectra a.k.a spectrum image).

HyperSpy aims at making it easy and natural to apply analytical procedures that
operate on an individual signal to multidimensional arrays, as well as
providing easy access to analytical tools that exploit the multidimensionality
of the dataset.

Its modular structure makes it easy to add features to analyze different kinds
of signals.

HyperSpy extensions
-------------------

In this document we refer to external programs that build on HyperSpy as
"HyperSpy extensions". Those programs can e.g. provide extra data analysis tools
file formats, add graphical user interfaces, provide
extra blind source separation algorithms etc.

There are multiple HyperSpy extensions. For a list of extensions hosted
publicly in GitHub search for the GitHub topic `hyperspy-extension <https://github.com/topics/hyperspy-extension>`_

.. note::
    From version 2.0, HyperSpy will be split into a core package (HyperSpy)
    that will provide the common infrastructure and a number of HyperSpy
    extensions.

Our vision
----------

To us this program is a research tool, much like a screwdriver or a Green's
function. We believe that the better our tools are, the better our research
will be. We also think that it is beneficial for the advancement of knowledge
to share our research tools and to forge them in a collaborative way. This is
because by collaborating we advance faster, mainly by avoiding reinventing the
wheel. Idealistic as it may sound, many other people think like this and it is
thanks to them that this project exists.

HyperSpy's character
--------------------

HyperSpy has been written by a subset of the people who use it, a particularity
that sets its character:

* The main way of interacting with the program is through the command line.
  This is because:

    * Our command line interface is agreeable to use thanks to `IPython
      <http://ipython.org/>`_.
    * With a command line interface it is very easy
      to automate the data analysis, and therefore boost productivity. Of
      course the drawback is that the learning curve is steeper, but we have
      tried to keep it as gentle as possible.

* That said, Graphical User Interface (GUI) elements are provided where there
  is a clear productivity advantage in doing so. See the
  `jupyter widgets GUI <https://github.com/hyperspy/hyperspy_gui_ipywidgets>`_
  and the
  `traitsui GUI <https://github.com/hyperspy/hyperspy_gui_traitsui>`_. If you
  need a full, standalone GUI, `HyperSpyUI <http://hyperspy.org/hyperspyUI/>`_
  is for you.
* We see HyperSpy as a collaborative project, and therefore we care
  about making it easy for others to contribute to it. In other words,
  we want to minimise the “user becomes developer” threshold. To achieve this
  goal we:

    * Use an open-source license, the `GPL v3
      <http://www.gnu.org/licenses/gpl-3.0-standalone.html>`_.
    * Try to keep the code as simple and easy to understand as possible.
    * Have chosen to write in `Python <http://www.python.org/>`_, a high level
      programming language with `high quality scientific libraries
      <http://www.scipy.org/>`_ and which is very easy to learn.
