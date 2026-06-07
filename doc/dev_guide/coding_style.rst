.. _ruff: https://docs.astral.sh/ruff
.. _coding_style-label:

Coding style
============

HyperSpy follows the Style Guide for Python Code - these are rules
for code consistency that you can read all about in the `Python Style Guide
<https://www.python.org/dev/peps/pep-0008/>`_. You can use the `ruff`_ code
formatter to automatically fix the style of your code using pre-commit hooks.

Linting errors can be suppressed in the code using the ``# noqa`` marker,
more information in the `ruff documentation <https://docs.astral.sh/ruff/linter/#error-suppression>`_.

Pre-commit hooks
================
Code linting and formatting is checked continuously using `ruff`_ pre-commit hooks.

These can be run locally by using `pre-commit <https://pre-commit.com>`__.
Alternatively, the comment ``pre-commit.ci autofix`` can be added to a PR to fix the formatting
using `pre-commit.ci <https://pre-commit.ci>`_.

Deprecations
============
HyperSpy follows `semantic versioning <https://semver.org>`_ where changes follow such that:

1. MAJOR version when you make incompatible API changes
2. MINOR version when you add functionality in a backward compatible manner
3. PATCH version when you make backward compatible bug fixes

This means that as little, ideally no, functionality should break between minor releases.
Deprecation warnings are raised whenever possible and feasible for functions/methods/properties/arguments,
so that users get a heads-up one (minor) release before something is removed or changes, with a possible
alternative to be used.

A deprecation decorator should be placed right above the object signature to be deprecated:

.. code-block:: python

    @deprecated(since=1.7.0, removal=2.0.0, alternative="bar")
    def foo(self, n):
        return n + 1

    @deprecated_argument(since=1.7.0, removal=2.0.0,name="x", alternative="y")
    def this_property(y):
        return y

This will update the docstring, and print a visible deprecation warning telling
the user to use the alternative function or argument.

These deprecation wrappers are inspired by those in ``kikuchipy``.

Traits conventions
==================

HyperSpy uses `Enthought traits <https://docs.enthought.com/traits/>`_ (not Jupyter ``traitlets``) for
observable attributes on interactive tools, components, and axes. All trait handlers must use the modern
``@observe`` pattern. The minimum supported version is traits 7.0.

Handler pattern
---------------

Use ``@t.observe("trait_name")`` decorator with ``(self, event=None)`` signature:

.. code-block:: python

    import traits.api as t

    class MyTool(t.HasTraits):
        threshold = t.Range(0.0, 1.0, value=0.5)

        @t.observe("threshold", post_init=True)
        def _threshold_changed(self, event=None):
            self.update_plot()

The ``event`` parameter has ``.old``, ``.new``, and ``.name`` attributes. Use ``post_init=True`` for
handlers that should not fire during ``__init__``.

Plain classes (not ``HasTraits``)
---------------------------------

If a class does not inherit from ``t.HasTraits``, the ``@t.observe`` decorator will not register.
Use ``self.observe()`` in ``__init__`` instead:

.. code-block:: python

    class SpikesRemoval:
        def __init__(self):
            if hasattr(self, "observe"):
                self.observe(self._index_changed, "index")

        def _index_changed(self, event=None):
            ...

ROI validation/revert
---------------------

Handlers that validate and revert invalid values must use ``self.trait_setq()`` to prevent
infinite recursion:

.. code-block:: python

    @t.observe("left")
    def _left_changed(self, event=None):
        if event.new < 0:
            self.trait_setq(left=0)  # does NOT trigger observer again

List item observation
---------------------

When observing traits on items in a ``t.List``, use the expression API with ``optional=True``
to handle items that may not have the trait:

.. code-block:: python

    from traits.observation.api import trait as trait_expr

    expr = trait_expr("_axes").list_items().trait("scale", optional=True)
    self.observe(handler, expr)

Deprecated patterns
-------------------

The following patterns are deprecated and must not be used in new code:

- ``self.on_trait_change(handler, "name")`` — use ``self.observe(handler, "name")``
- ``def _name_changed(self, old, new)`` — use ``@t.observe("name")`` with ``(self, event=None)``
- ``t.Unicode()`` — use ``t.Str()``
- ``t.Either([...])`` — use ``t.Union(...)``
- ``Property(depends_on="x")`` — use ``Property(observe="x")`` with ``@cached_property``
