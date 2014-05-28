Introduction
============

What is HyperSpy
----------------

HyperSpy is an open source Python library which provides tools to facilitate
the interactive data analysis of multidimensional datasets that can be
described as multidimensional arrays of a given signal (e.g. a 2D array of
spectra a.k.a spectrum image).

Hyperpsy aims at making it easy and natural to apply analytical procedures that
operate on an individual signal to multidimensional arrays, as well as
providing easy access to analytical tools that exploit the multidimensionality
of the dataset.

Its modular structure makes it easy to add features to analyze different kinds
of signals. Currently there are specialized tools to analyze electron
energy-loss spectroscopy (EELS) and energy dispersive X-rays (EDX) data. 

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
    * Writing and maintaining user
      interfaces (UIs) require time from the developers and the current ones
      prefer to spend their time adding new features. Maybe in the future we
      will provide a fully featured GUI, but HyperSpy will always remain fully
      scriptable.

* That said, UIs are provided where there is a clear productivity advantage in
  doing so.
  For example, there are UIs to perform windows quantification, data smoothing,
  adjusting the preferences, loading data...
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



