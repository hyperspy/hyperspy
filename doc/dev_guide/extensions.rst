

.. _writing_extensions-label:

Writing packages that extend HyperSpy
=====================================

.. versionadded:: 1.5
  External packages can extend HyperSpy by registering signals,
  components and widgets.

External packages can extend HyperSpy by registering signals, components and
widgets. Objects registered by external packages are "first-class citizens" i.e.
they can be used, saved and loaded like any of those objects shipped with
HyperSpy. Because of HyperSpy's structure, we anticipate that most packages
registering HyperSpy extensions will provide support for specific sorts of
data.

Models can be provided by external packages too but don't need to 
be registered. Instead, they are returned by the ``create_model`` method of
the relevant :py:class:`hyperspy.signal.BaseSignal` subclass.

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

Registering extensions
----------------------

In order to register HyperSpy extensions you need to:

1. Add the following line to your package's ``setup.py``:

    .. code-block:: python

        entry_points={'hyperspy.extensions': 'your_package_name =
        your_package_name'},
2. Create a ``hyperspy_extension.yaml`` configuration file in your
   module's root directory.
3. Declare all new HyperSpy objects provided by your package in the
   ``hyperspy_extension.yaml`` file.

For a full example on how to create a package that extends HyperSpy see
`the HyperSpy Sample Extension package
<https://github.com/hyperspy/hyperspy_sample_extension>`_.


Creating new HyperSpy BaseSignal subclasses
-------------------------------------------

When and where create a new ``BaseSignal`` subclass
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

HyperSpy provides most of its functionality through the different
:py:class:`hyperspy.signal.BaseSignal`
subclasses. A HyperSpy "signal" is a class that contains data for analysis
and functions to perform the analysis in the form of class methods. Functions
that are useful for the analysis of most datasets are in the
:py:class:`hyperspy.signal.BaseSignal` class. All other functions are in
specialized subclasses.

The flowchart below can help you decide where to add
a new data analysis function. Notice that only if no suitable package exists
for your function you should consider creating your own.

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

Registering a new BaseSignal subclass
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Creating new HyperSpy model components
--------------------------------------

When and where create a new components 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

HyperSpy provides the :py:class:`hyperspy._components.expression.Expression`
component that enables easy creation of 1D and 2D components from
mathematical expressions. Therefore, strictly speaking, we only need to
create new components when they cannot be expressed as simple mathematical
equations. However, HyperSpy is all about simplifying the interactive data
processing workflow. Therefore, we consider that functions that are commonly
used for model fitting, in general or specific domains, are worth adding to
HyperSpy itself (if they are of common interest) or to specialized external
packages extending HyperSpy.

The flowchart below can help you decide when and where to add
a new hyperspy model :py:class:`hyperspy.component.Component`.
for your function you should consider creating your own.

.. mermaid::

   graph TD

     A(New component needed)
     B{Can it be declared using Expression?}
     C{Can it be useful to other users?}
     D(Just use Expression)
     E[Create new component using Expression]
     F[Create new component from the scratch]
     G{Is it useful for general users?}
     H(Contribute it to HyperSpy)
     I{Does a suitable package for it exist?}
     J[Contribute it to the relevant package]
     K[Create your own package to host it]

     A-->B
     B-- Yes -->C
     B-- No  -->F
     C-- No  -->D
     C-- Yes -->E
     E-->G
     F-->G
     G-- Yes --> H
     G-- No  --> I
     I-- Yes --> J
     I-- No  --> K








