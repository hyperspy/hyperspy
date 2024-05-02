

.. _writing_extensions-label:

Writing packages that extend HyperSpy
=====================================

.. versionadded:: 1.5
  External packages can extend HyperSpy by registering signals,
  components and widgets.

.. warning::
  The mechanism to register extensions is in beta state. This means that it can
  change between minor and patch versions. Therefore, if you maintain a package
  that registers HyperSpy extensions, please verify that it works properly with
  any future HyperSpy release. We expect it to reach maturity with the release
  of HyperSpy 2.0.

External packages can extend HyperSpy by registering signals, components and
widgets. Objects registered by external packages are "first-class citizens" i.e.
they can be used, saved and loaded like any of those objects shipped with
HyperSpy. Because of HyperSpy's structure, we anticipate that most packages
registering HyperSpy extensions will provide support for specific sorts of
data.

Models can also be provided by external packages, but don't need to
be registered. Instead, they are returned by the ``create_model`` method of
the relevant :class:`~.api.signals.BaseSignal` subclass, see for example,
the :meth:`exspy.signals.EDSTEMSpectrum.create_model` of the
:class:`exspy.signals.EDSTEMSpectrum`.

It is good practice to add all packages that extend HyperSpy
`to the list of known extensions
<https://github.com/hyperspy/hyperspy-extensions-list>`_ regardless of their
maturity level. In this way, we can avoid duplication of efforts and issues
arising from naming conflicts. This repository also runs an `integration test
suite <https://github.com/hyperspy/hyperspy-extensions-list/actions>`__ daily,
which runs the test suite of all extensions to check the status of
the ecosystem. See the :ref:`corresponding section <integration_test_suite-label>`
for more details.

At this point, it is worth noting that HyperSpy's main strength is its amazing
community of users and developers. We trust that the developers of packages
that extend HyperSpy will play by the same rules that have made the Python
scientific ecosystem successful. In particular, avoiding duplication of
efforts and being good community players by contributing code to the best
matching project are essential for the sustainability of our open software
ecosystem.

Registering extensions
----------------------

In order to register HyperSpy extensions, you need to:

1. Add the following line to your package's ``setup.py``:

   .. code-block:: python

      entry_points={'hyperspy.extensions': 'your_package_name = your_package_name'},
2. Create a ``hyperspy_extension.yaml`` configuration file in your
   module's root directory.
3. Declare all new HyperSpy objects provided by your package in the
   ``hyperspy_extension.yaml`` file.

For a full example on how to create a package that extends HyperSpy, see
`the HyperSpy Sample Extension package
<https://github.com/hyperspy/hyperspy_sample_extension>`_.


Creating new HyperSpy BaseSignal subclasses
-------------------------------------------

When and where to create a new ``BaseSignal`` subclass
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

HyperSpy provides most of its functionality through the different
:class:`~.api.signals.BaseSignal`
subclasses. A HyperSpy "signal" is a class that contains data for analysis
and functions to perform the analysis in the form of class methods. Functions
that are useful for the analysis of most datasets are in the
:class:`~.api.signals.BaseSignal` class. All other functions are in
specialized subclasses.

The flowchart below can help you decide where to add
a new data analysis function. Notice that only if no suitable package exists
for your function, you should consider creating your own.

..  This is the original mermaid code. It produces a nicer looking diagram
    with the defaults, but, as of version 0.3.1, it raises an exception in
    ReadTheDocs, so we use graphviz below instead.

    .. mermaid::

       graph TD

         A(New function needed!)
         B{Is it useful for data of any type and dimensions?}
         C(Contribute it to BaseSignal)
         D{Does a SignalxD for the required dimension exist in HyperSpy?}
         E[Contribute new SignalxD to HyperSpy]
         F{Is the function useful for a specific type of data only?}
         G(Contribute it to SignalxD)
         H{Does a signal for that sort of data exists?}
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


.. graphviz::

    digraph G {
         A [label="New function needed!"]
         B [label="Is it useful for data of any type and dimensions?",shape="diamond"]
         C [label="Contribute it to BaseSignal"]
         D [label="Does a SignalxD for the required dimension exist in HyperSpy?",shape="diamond"]
         E [label="Contribute new SignalxD to HyperSpy"]
         F [label="Is the function useful for a specific type of data only?",shape="diamond"]
         G [label="Contribute it to SignalxD"]
         H [label="Does a signal for that sort of data exist?",shape="diamond"]
         I [label="Contribute to package providing the relevant signal"]
         J [label="Create you own package and signal subclass to host the funtion"]
         A->B
         B->C [label="Yes"]
         B->D [label="No"]
         D->F [label="Yes"]
         D->E [label="No"]
         E->F
         F->H [label="Yes"]
         F->G [label="No"]
         H->I [label="Yes"]
         H->J [label="No"]

    }


Registering a new BaseSignal subclass
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To register a new :class:`~.api.signals.BaseSignal` subclass you must add it to the
``hyperspy_extension.yaml`` file, as in the following example:

.. code-block:: yaml

    signals:
        MySignal:
            signal_type: "MySignal"
            signal_type_aliases:
            - MS
            - ThisIsMySignal
            # The dimension of the signal subspace. For example, 2 for images, 1 for
            # spectra. If the signal can take any signal dimension, set it to -1.
            signal_dimension: 1
            # The data type, "real" or "complex".
            dtype: real
            # True for LazySignal subclasses
            lazy: False
            # The module where the signal is located.
            module: my_package.signal


Note that HyperSpy uses ``signal_type`` to determine which class is the most
appropriate to deal with a particular sort of data. Therefore, the signal type
must be specific enough for HyperSpy to find a single signal subclass
match for each sort of data.

.. warning::
    HyperSpy assumes that only one signal
    subclass exists for a particular ``signal_type``. It is up to external
    package developers to avoid ``signal_type`` clashes, typically by collaborating
    in developing a single package per data type.

The optional ``signal_type_aliases`` are used to determine the most appropriate
signal subclass when using
:meth:`~.api.signals.BaseSignal.set_signal_type`.
For example, if the ``signal_type`` ``Electron Energy Loss Spectroscopy``
has an ``EELS`` alias, setting the signal type to ``EELS`` will correctly assign
the signal subclass with ``Electron Energy Loss Spectroscopy`` signal type.
It is good practice to choose a very explicit ``signal_type`` while leaving
acronyms for ``signal_type_aliases``.

Creating new HyperSpy model components
--------------------------------------

When and where to create a new component
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

HyperSpy provides the :class:`hyperspy._components.expression.Expression`
component that enables easy creation of 1D and 2D components from
mathematical expressions. Therefore, strictly speaking, we only need to
create new components when they cannot be expressed as simple mathematical
equations. However, HyperSpy is all about simplifying the interactive data
processing workflow. Therefore, we consider that functions that are commonly
used for model fitting, in general or specific domains, are worth adding to
HyperSpy itself (if they are of common interest) or to specialized external
packages extending HyperSpy.

The flowchart below can help you decide when and where to add
a new hyperspy model :class:`hyperspy.component.Component`
for your function, should you consider creating your own.

..  This is the original mermaid code. It produces a nicer looking diagram
    with the defaults, but, as of version 0.3.1, it raises an exception in
    ReadTheDocs, so we use graphviz below instead.


    .. mermaid::

       graph TD

         A(New component needed!)
         B{Can it be declared using Expression?}
         C{Can it be useful to other users?}
         D(Just use Expression)
         E[Create new component using Expression]
         F[Create new component from scratch]
         G{Is it useful for general users?}
         H(Contribute it to HyperSpy)
         I{Does a suitable package exist?}
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


.. graphviz::

    digraph G {


        A [label="New component needed!"]
        B [label="Can it be declared using Expression?",shape="diamond"]
        C [label="Can it be useful to other users?",shape="diamond"]
        D [label="Just use Expression"]
        E [label="Create new component using Expression"]
        F [label="Create new component from scratch"]
        G [label="Is it useful for general users?",shape="diamond"]
        H [label="Contribute it to HyperSpy"]
        I [label="Does a suitable package exist?",shape="diamond"]
        J [label="Contribute it to the relevant package"]
        K [label="Create your own package to host it"]

        A->B
        B->C [label="Yes"]
        B->F [label="No"]
        C->E [label="Yes"]
        C->D [label="No"]
        E->G
        F->G
        G->H [label="Yes"]
        G->I [label="No"]
        I->J [label="Yes"]
        I->K [label="No"]
    }


Registering new components
^^^^^^^^^^^^^^^^^^^^^^^^^^

All new components must be a subclass of
:class:`hyperspy._components.expression.Expression`. To register a new
1D component add  it to the ``hyperspy_extension.yaml`` file as in the following
example:

.. code-block:: yaml

    components1D:
      # _id_name of the component. It must be a UUID4. This can be generated
      # using ``uuid.uuid4()``. Also, many editors can automatically generate
      # UUIDs. The same UUID must be stored in the components ``_id_name`` attribute.
      fc731a2c-0a05-4acb-91df-d15743b531c3:
        # The module where the component class is located.
        module: my_package.components
        # The actual class of the component
        class: MyComponent1DClass

Equivalently, to add a new component 2D:

.. code-block:: yaml

    components2D:
      # _id_name of the component. It must be a UUID4. This can be generated
      # using ``uuid.uuid4()``. Also, many editors can automatically generate
      # UUIDs. The same UUID must be stored in the components ``_id_name`` attribute.
      2ffbe0b5-a991-4fc5-a089-d2818a80a7e0:
        # The module where the component is located.
        module: my_package.components
        class: MyComponent2DClass

.. note::

  HyperSpy's legacy components use their class name instead of a UUID as
  ``_id_name``. This is for compatibility with old versions of the software.
  New components (including those provided through the extension mechanism) 
  must use a UUID4 in order to i) avoid name clashes ii) make it easy to find
  the component online if e.g. the package is renamed or the component
  relocated.


Creating and registering new widgets and toolkeys
-------------------------------------------------

To generate GUIs of specific methods and functions, HyperSpy uses widgets and
toolkeys:

* *widgets* (typically ipywidgets or traitsui objects) generate GUIs,
* *toolkeys* are functions which associate widgets to a signal method 
  or to a module function.

An extension can declare new toolkeys and widgets. For example, the
`hyperspy-gui-traitsui <https://github.com/hyperspy/hyperspy_gui_traitsui>`_ and
`hyperspy-gui-ipywidgets <https://github.com/hyperspy/hyperspy_gui_ipywidgets>`_
provide widgets for toolkeys declared in HyperSpy.

Registering toolkeys
^^^^^^^^^^^^^^^^^^^^
To register a new toolkey:

1. Declare a new toolkey, *e. g.* by adding the ``hyperspy.ui_registry.add_gui_method``
   decorator to the function you want to assign a widget to.
2. Register a new toolkey that you have declared in your package by adding it to
   the ``hyperspy_extension.yaml`` file, as in the following example:


.. code-block:: yaml

    GUI:
      # In order to assign a widget to a function, that function must declare
      # a `toolkey`. The `toolkeys` list contains a list of all the toolkeys
      # provided by extensions. In order to avoid name clashes, by convention,
      # toolkeys must start with the name of the package that provides them.
      toolkeys:
        - my_package.MyComponent


Registering widgets
^^^^^^^^^^^^^^^^^^^

In the example below, we register a new ``ipywidget`` widget for the
``my_package.MyComponent`` toolkey of the previous example. The ``function``
simply returns the widget to display. The key *module* defines where the functions
resides.

.. code-block:: yaml

    GUI:
      widgets:
        ipywidgets:
          # Each widget is declared using a dictionary with two keys, `module` and `function`.
          my_package.MyComponent:
            # The function that creates the widget
            function: get_mycomponent_widget
            # The module where the function resides.
            module: my_package.widgets


.. _integration_test_suite-label:

Integration test suite
----------------------

The `integration test suite <https://github.com/hyperspy/hyperspy-extensions-list/actions>`__
runs the test suite of HyperSpy and of all registered HyperSpy extensions on a daily basis against both the
release and development versions. The build matrix is as follows:

.. list-table:: Build matrix of the integration test suite
   :widths: 25 25 25
   :header-rows: 1

   * - HyperSpy
     - Extension
     - Dependencies
   * - Release
     - Release
     - Release
   * - Release
     - Development
     - Release
   * - RELEASE_next_patch
     - Release
     - Release
   * - RELEASE_next_patch
     - Development
     - Release
   * - RELEASE_next_minor
     - Release
     - Release
   * - RELEASE_next_minor
     - Development
     - Release
   * - RELEASE_next_minor
     - Development
     - Development
   * - RELEASE_next_minor
     - Development
     - Pre-release

The development packages of the dependencies are provided by the
`scipy-wheels-nightly <https://pypi.anaconda.org/scipy-wheels-nightly/simple>`_
repository, which provides ``scipy``, ``numpy``, ``scikit-learn`` and ``scikit-image``
at the time of writing.
The pre-release packages are obtained from `PyPI <https://pypi.org>`_ and these
will be used for any dependency which provides a pre-release package on PyPI.

A similar `Integration test  <https://github.com/hyperspy/hyperspy/actions/workflows/tests_extension.yml>`__
workflow can run from pull requests (PR) to the
`hyperspy <https://github.com/hyperspy/hyperspy>`_ repository when the label
``run-extension-tests`` is added to a PR or when a PR review is edited.

