

.. _writing_extensions-label:

Writing HyperSpy extensions
===========================

External packages can extend HyperSpy by registering signals, components and
widgets. Objects registered by external packages can be used, saved and
loaded like any of those objects shipped with HyperSpy. For details on how to
register extensions refer to the :ref:`writing_extensions-label`.

There are multiple HyperSpy extensions. We maintain a list of extension in the
following GitHub repository:
`https://github.com/hyperspy/hyperspy-extensions-list`_


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




