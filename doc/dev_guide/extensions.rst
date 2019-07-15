

.. _writing_extensions-label:

Writing packages that extend HyperSpy
=====================================

.. versionadded:: 1.5
  External packages can extend HyperSpy by registering signals,
  components and widgets.

External packages can extend HyperSpy by registering signals, components and
widgets. Models can be provided by external packages too but don't need to 
be registered. Instead, they are returned by the ``create_model`` method of the
relevant signal subclass. Objects registered by external packages can be
used, saved and loaded like any of those objects shipped with HyperSpy.

It is good practice to add all packages that extend HyperSpy 
`to the list of known extensions
<https://github.com/hyperspy/hyperspy-extensions-list>`_ regardless their
maturity level. In this way we can avoid duplication of efforts and issues
arising from naming conflicts.

At this point it is worth noting that HyperSpy's main strength is its amazing
community of users and developers. We trust that the developers of packages
that extend HyperSpy will play by the same rules that have made the Python
scientific ecosystem in general, and HyperSpy in particular, successful. In
particular, avoiding duplication of efforts and being good community players
by contributing code to the best matching project are essential for the
sustainability of our software ecosystem.

Creating new HyperSpy BaseSignal subclasses
-------------------------------------------

When to create a new ``BaseSignal`` subclass
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

HyperSpy provides most of its functionality through the different
``BaseSignal`` subclasses. A HyperSpy `Signal` is a class that contains data
for analysis and functions to perform the analysis in the form of class
methods. Functions that are useful for the analysis of most datasets are in
the ``BaseSignal`` class. Functions that are useful only to the analysis of
e.g. images belong to the ``Signal2D`` ``BaseSignal`` subclass. So, say that you
are missing a certain function for the analysis of your data in HyperSpy. The
first questions that you should ask yourself is: is that function of general
interest for all datasets? If the anIf it is specific, say, to spectral analysis, the
next question to ask is: is it useful for any sort of spectrum? If yes, you
could consider contributing it to HyperSpy's ``Signal1D`` subclass. If not,
you could check the different specific signals that ship with HyperSpy and
those in `https://github.com/hyperspy/hyperspy-extensions-list`_ to verify if
there is already a class for that sort of data. If yes, please consider
contributing it to those packages. If not,

.. mermaid::

   graph TD
    A(New function needed)
    B{Is it useful for data of any type and dimensions?}
    C(Contribute it to BaseSignal)
    D{Does an SignalxD for the required dimension exist in HyperSpy?}
    E[Contribute new SignalxD to HyperSpy]
    F{Is the function useful only for some sort of data?}
    G(Contribute it to SignalxD)
    H{Does an signal for that sort of data exists?}
    I(Contribute to package providing the relevant signal)
    J(Create you own package and signal subclass to host the funtion)
    A-->B
    B-- Yes -->C
    B-- No  -->D
    D-- Yes -->F
    D-- No  -->E
    E-->F
    F-- Yes -->H
    F-- No  -->G
    H-- Yes -->I
    H-- No -->J





Writing HyperSpy extensions
===========================



Add "extension" mechanism to register components, signals and models provided by external projects.

In this implementation declaring hyperspy extensions involve:

1. Creating a Python package that includes new signals, models or components.
2. Declaring the objects in a ``hyperspy_extension.yaml`` file locate in the Python package root. For an example [check this](https://github.com/hyperspy/hyperspy_sample_extension/blob/master/hspy_ext/hyperspy_extension.yaml).
3. Installing the Python package that provides the extensions
4. Enabling the extensions by adding the name of the module to the ` ~/.hyperspy/hspy_extensions.yaml` file. This can be automated with ``conda`` but not with ``pip``.

This is part of the splitting HyperSpy effort (see #821 and #1599).

### Progress of the PR
- [ ] Change implemented (can be split into several points)
    - [x] External components
    - [x] External models
    - [x] External signals
    - [x] External GUIs
        - [x] Implement for  ``hyperspy_gui_ipywidgets``
        - [x] Implement for  ``hyperspy_gui_traitsui``
    - [x] ~~Create extension register script (currently the extension must be registered manually, see below)~~ Automatically register extensions on installation
    - [x] Store info about the object provider to raise meaningful errors when attempting to load a file that requires installing an external extension.
    - [x] Add comments to example specs file (thanks @thomasaarholt)
    - [ ] Add Component2D example
    - [x] Implement informative error when external package not installed
- [ ] update docstring (if appropriate),
- [x] add tests,
- [ ] ready for review.

### How to test it

1. Install / checkout hyperspy from this branch
2. Install https://github.com/hyperspy/hyperspy_sample_extension/

```python
>>> import hyperspy.api as hs
>>> s = hs.signals.BaseSignal([0])
>>> s.set_signal_type("MySignal")
>>> print(s)
<MySignal, title: , dimensions: (|1)>
>>> m = s.create_model()
>>> print(m)
<MyModel>
>>> from hspy_ext.component import MyComponent
>>> m.append(MyComponent())
>>> m.save("test_extension")
>>> sr = hs.load("test_extension.hspy")
>>> m = sr.models.restore("a")
>>> print(m)
<MyModel>
>>> print(m.components)
   # |      Attribute Name |      Component Name |      Component Type
---- | ------------------- | ------------------- | -------------------
   0 |         MyComponent |         MyComponent |         MyComponent
```




