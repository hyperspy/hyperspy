# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

"""HyperSpy-specific additions to anyplotlib's traits-to-traitlets shim.

This file is NOT imported by HyperSpy.  Its source is spliced verbatim into
the traits-shim block of anywidget_bridge.js by
hyperspy_anywidget._patch_bridge, and runs inside Pyodide, where the names
sys, _types, _tr (traitlets), _traits_root, _traits_api and _on_trait_change
are already bound by the surrounding shim.

It exists because the upstream shim stops at traits.api, while HyperSpy also
reaches for traits.observation.api -- the expression-builder flavour of
observe() introduced in Traits 6::

    from traits.observation.api import trait
    self.observe(handler, trait("_axes").list_items().trait("index"))

Everything here should move upstream into anyplotlib; see the note on
hyperspy_anywidget.BRIDGE_PATCHES.

Style constraint: no backticks and no backslashes anywhere in this file, and
no JavaScript template-literal placeholders (a dollar sign followed by a
brace).  The whole file is embedded in a template literal, and
hyperspy_anywidget enforces this at build time.
"""

# --- traits.observation expression builder ---------------------------------


class _ObservePath:
    """The subset of Traits' observation expressions HyperSpy actually uses.

    Traits builds an observer graph; the shim's _on_trait_change instead
    understands a single "list_trait.child_attr" string and does the per-item
    bookkeeping itself.  So all this needs to do is remember the attribute
    names in order and join them with dots.  list_items() adds no segment: in
    the shim's vocabulary the list step is implied by the parent being a list
    trait.
    """

    def __init__(self, names=(), optional=False):
        self._names = tuple(names)
        self.optional = optional

    def trait(self, name, optional=False):
        return _ObservePath(self._names + (name,), optional=optional or self.optional)

    def list_items(self, optional=False, notify=True):
        return _ObservePath(self._names, optional=optional or self.optional)

    def dict_items(self, optional=False, notify=True):
        return self.list_items(optional=optional)

    def set_items(self, optional=False, notify=True):
        return self.list_items(optional=optional)

    def __str__(self):
        return ".".join(self._names)

    def __repr__(self):
        return "_ObservePath({0!r})".format(str(self))

    def __hash__(self):
        return hash((self._names, self.optional))

    def __eq__(self, other):
        return isinstance(other, _ObservePath) and self._names == other._names


def _observation_trait(name, optional=False, notify=True):
    return _ObservePath((name,), optional=optional)


_traits_observation = _types.ModuleType("traits.observation")  # noqa: F821
_traits_observation.__path__ = []
_traits_observation_api = _types.ModuleType("traits.observation.api")  # noqa: F821
_traits_observation_api.trait = _observation_trait
_traits_observation_api.match = _observation_trait
_traits_observation_api.anytrait = _observation_trait
_traits_observation.api = _traits_observation_api

# traits was registered as a plain module, so "import traits.observation"
# fails with "traits is not a package" until it grows a __path__.
_traits_root.__path__ = []  # noqa: F821
_traits_root.observation = _traits_observation  # noqa: F821
sys.modules["traits.observation"] = _traits_observation  # noqa: F821
sys.modules["traits.observation.api"] = _traits_observation_api  # noqa: F821


# --- HasTraits.observe: accept expressions and remove= ----------------------

_hs_orig_observe = _tr.HasTraits.observe  # noqa: F821


def _hs_observe(self, handler, names=_tr.All, type="change", remove=False):  # noqa: F821,A002
    """traits-compatible observe().

    Traits' signature is observe(handler, expression, remove=False, ...) and
    its expressions can reach through a list trait into its items.  traitlets
    only knows flat trait names on self, so anything with a dot in it is
    handed to the shim's _on_trait_change, which walks the list and re-wires
    itself when the list changes.
    """
    if isinstance(names, _ObservePath):
        names = str(names)

    if isinstance(names, str) and "." in names:
        return _on_trait_change(self, handler, names, remove=remove)  # noqa: F821

    if remove:
        try:
            return self.unobserve(handler, names=names, type=type)
        except Exception:
            # traits tolerates removing an observer that was never added.
            return None

    return _hs_orig_observe(self, handler, names=names, type=type)


_tr.HasTraits.observe = _hs_observe  # noqa: F821


# --- List traits must notify on in-place mutation ---------------------------
#
# This is the difference that actually breaks HyperSpy.  A Traits List fires a
# change event when it is mutated in place; a traitlets List only fires when
# the whole list is reassigned.  So AxesManager._remove_one_axis, which ends
# in "self._axes.remove(axis)", silently skipped _update_attributes(), leaving
# signal_dimension stale.  The visible symptom was that summing away both
# signal axes returned something still calling itself a Signal2D, which then
# failed to deep-copy.  Restore the Traits behaviour for every List trait
# rather than patching HyperSpy's individual mutation sites.


class _NotifyingList(list):
    """A list that reports in-place mutation to the HasTraits that owns it."""

    _owner = None
    _trait_name = None

    def _hs_notify(self):
        owner = self._owner
        if owner is None or self._trait_name is None:
            return
        try:
            # old is new: the observers HyperSpy registers only care that
            # something changed, and traitlets has no before-image to offer.
            owner._notify_trait(self._trait_name, self, self)
        except Exception:
            # A mutation can land mid-__init__, before the owner is complete.
            # Those early notifications are redundant anyway -- AxesManager
            # calls _update_attributes() itself once construction finishes.
            pass


def _hs_make_mutator(name):
    parent = getattr(list, name)

    def _mutator(self, *args, **kwargs):
        result = parent(self, *args, **kwargs)
        self._hs_notify()
        return result

    _mutator.__name__ = name
    return _mutator


for _hs_name in (
    "append",
    "extend",
    "insert",
    "remove",
    "pop",
    "clear",
    "sort",
    "reverse",
    "__setitem__",
    "__delitem__",
    "__iadd__",
    "__imul__",
):
    setattr(_NotifyingList, _hs_name, _hs_make_mutator(_hs_name))


_hs_orig_list_validate = _TraitsList.validate  # noqa: F821


def _hs_list_validate(self, obj, value):
    validated = _hs_orig_list_validate(self, obj, value)
    wrapped = _NotifyingList(validated)
    wrapped._owner = obj
    wrapped._trait_name = self.name
    return wrapped


_TraitsList.validate = _hs_list_validate  # noqa: F821


# --- Enum: accept Traits' variadic spelling ---------------------------------
#
# The upstream shim only handles Enum([a, b, c]).  Traits also accepts
# Enum(a, b, c), which is the spelling HyperSpy uses in model1d, the
# background-removal and contrast-editor tools, and the baseline tool.  In
# Traits the default is always the first value; a default= keyword is plain
# metadata, not the default, so it is passed through untouched.


class _HsEnum(_tr.TraitType):  # noqa: F821
    def __init__(self, *values, **kw):
        if len(values) == 1 and isinstance(values[0], (list, tuple)):
            allowed = list(values[0])
        else:
            allowed = list(values)
        self._allowed = allowed
        super().__init__(allowed[0] if allowed else None, **kw)

    def validate(self, obj, value):
        if value in self._allowed:
            return value
        raise _tr.TraitError(  # noqa: F821
            "The value {0!r} is not in {1!r}".format(value, self._allowed)
        )


_traits_api.Enum = _HsEnum  # noqa: F821
_traits_api.BaseEnum = _HsEnum  # noqa: F821


# --- Union: Traits takes the member traits variadically ---------------------
#
# traitlets.Union wants a single list, Traits wants *args.  HyperSpy's
# Component declares "value = t.Property(t.Union(t.CFloat(0), Array()))", so
# importing hyperspy.component under the upstream mapping raised
# "Union.__init__() takes 2 positional arguments but 3 were given".


class _HsUnion(_tr.Union):  # noqa: F821
    def __init__(self, *trait_types, **kw):
        if len(trait_types) == 1 and isinstance(trait_types[0], (list, tuple)):
            members = list(trait_types[0])
        else:
            members = list(trait_types)
        super().__init__(members, **kw)


_traits_api.Union = _HsUnion  # noqa: F821
_traits_api.Either = _HsUnion  # noqa: F821


# --- trait_setq: assign without notifying -----------------------------------
#
# The Traits API pairs trait_set() with trait_setq(), the quiet version.
# HyperSpy's ROIs use it to seed geometry without re-triggering the update
# they are in the middle of, so plot_roi_map died on the missing attribute.
# traitlets has no quiet setter -- hold_trait_notifications() only defers
# notifications, it still delivers them -- so write the validated value
# straight into the trait store, which is what Traits does.


def _hs_trait_setq(self, **kw):
    for name, value in kw.items():
        trait = self.traits().get(name)
        # A Property has no stored value to write; its setter is the only way
        # in, and it is the setter's own business whether it notifies.
        if trait is None or isinstance(trait, _HsProperty):
            setattr(self, name, value)
            continue
        try:
            value = trait._validate(self, value)
        except Exception:
            pass
        self._trait_values[name] = value


if not hasattr(_tr.HasTraits, "trait_setq"):  # noqa: F821
    _tr.HasTraits.trait_setq = _hs_trait_setq  # noqa: F821


# --- Property: a computed trait, not a stored one ---------------------------
#
# traits.Property routes attribute access to _get_<name>/_set_<name> on the
# instance.  The upstream shim degrades it to a plain Any defaulting to None,
# which is silent but fatal for HyperSpy's model code: Parameter.value,
# .free, .twin, .bmin, .bmax and Component.name/.active are all Properties,
# so every one of them read back as None and model examples died with
# "'NoneType' object has no attribute 'value'".


class _HsProperty(_tr.TraitType):  # noqa: F821
    """Delegates to _get_<name>/_set_<name>, as Traits does."""

    allow_none = True

    def __init__(self, *args, fget=None, fset=None, depends_on=None, **kw):
        self._hs_fget = fget
        self._hs_fset = fset
        # The positional argument is the value trait Traits would validate
        # against.  The getter/setter own validation here, so it is unused.
        super().__init__(**kw)

    def __get__(self, obj, cls=None):
        if obj is None:
            return self
        if self._hs_fget is not None:
            return self._hs_fget(obj)
        getter = getattr(obj, "_get_" + str(self.name), None)
        if getter is None:
            return None
        return getter()

    def __set__(self, obj, value):
        if self._hs_fset is not None:
            self._hs_fset(obj, value)
            return
        setter = getattr(obj, "_set_" + str(self.name), None)
        if setter is None:
            raise _tr.TraitError(  # noqa: F821
                "cannot set read-only property {0!r}".format(self.name)
            )
        setter(value)


_traits_api.Property = _HsProperty  # noqa: F821


# --- trait_property_changed: notify for a trait with no stored value --------
#
# A Property has nothing for traitlets to diff, so Traits gives the setter a
# way to announce the change itself.  HyperSpy's Component._set_name and
# Parameter's setters call it; without it, creating any component raised
# "'Gaussian' object has no attribute 'trait_property_changed'".


def _hs_trait_property_changed(self, name, old_value, new_value=None):
    if new_value is None:
        new_value = getattr(self, name, None)
    self.notify_change(
        _tr.Bunch(  # noqa: F821
            name=name,
            old=old_value,
            new=new_value,
            owner=self,
            type="change",
        )
    )


if not hasattr(_tr.HasTraits, "trait_property_changed"):  # noqa: F821
    _tr.HasTraits.trait_property_changed = _hs_trait_property_changed  # noqa: F821


# --- add_trait must not discard a value that is already set -----------------
#
# Component.init_parameters does setattr(self, name, parameter) and only then
# self.add_trait(name, t.Instance(Parameter)).  Under Traits the assigned
# value survives.  Under traitlets, add_traits() installs a data descriptor
# that takes precedence over the instance __dict__, so every parameter read
# back as None and constructing any component failed at
# "setattr(getattr(self, kwarg), 'value', value)".  Carry the value across.


def _hs_add_trait(self, name, trait_class_or_inst):
    if isinstance(trait_class_or_inst, type):
        inst = trait_class_or_inst()
    else:
        inst = trait_class_or_inst

    missing = object()
    existing = self.__dict__.get(name, missing)
    self.add_traits(**{name: inst})
    if existing is not missing:
        try:
            setattr(self, name, existing)
        except Exception:
            # Keep the plain attribute rather than losing the value entirely.
            self.__dict__[name] = existing


_tr.HasTraits.add_trait = _hs_add_trait  # noqa: F821


# --- Float/CFloat must accept Undefined as "not set yet" --------------------
#
# Every ROI declares its geometry as t.CFloat(t.Undefined) -- see
# hyperspy/roi.py, e.g. "left, right = (t.CFloat(t.Undefined),) * 2" -- using
# Undefined as the unset sentinel, which Traits allows.  traitlets' Float
# rejects it, so merely constructing a RectangularROI raised "expected a
# float, not the Sentinel traitlets.Undefined" and took out every ROI example.
# The upstream shim already does exactly this for Str; Float needs it too.
# Coercion is kept, because Traits' CFloat accepts "7.5".


class _HsCFloat(_tr.Any):  # noqa: F821
    def __init__(self, default_value=0.0, **kw):
        kw.setdefault("allow_none", True)
        super().__init__(**kw)
        # Kept separately because traitlets reads Undefined as "no default was
        # given" and would substitute Any's own default of None -- which is
        # precisely the sentinel HyperSpy is testing for with
        # "if t.Undefined in tuple(self)".
        self._hs_default = default_value

    def get(self, obj, cls=None):
        try:
            return obj._trait_values[self.name]
        except KeyError:
            obj._trait_values[self.name] = self._hs_default
            return self._hs_default

    def validate(self, obj, value):
        if value is None or value is _tr.Undefined:  # noqa: F821
            return value
        try:
            return float(value)
        except (TypeError, ValueError):
            return value


_traits_api.CFloat = _HsCFloat  # noqa: F821
_traits_api.Float = _HsCFloat  # noqa: F821
_traits_api.BaseFloat = _HsCFloat  # noqa: F821
_traits_api.BaseCFloat = _HsCFloat  # noqa: F821


# --- One trait instance per attribute name ----------------------------------
#
# HyperSpy declares ROI geometry compactly:
#
#     top, bottom, left, right = (t.CFloat(t.Undefined),) * 4
#
# That binds the *same* trait object to four names.  Traits clones a trait
# definition per name, so it means what it looks like.  traitlets does not: a
# TraitType is a descriptor that stores its own .name and reads
# obj._trait_values[self.name], so all four names ended up sharing one slot.
# Setting left then right then top then bottom left every one of them holding
# the last value written -- left == right, so every ROI had zero width, the
# slice came out empty, and the Live FFT example died with "Invalid number of
# FFT data points (0)".
#
# Give each name its own copy before traitlets binds them.  A shallow copy is
# enough and is safer than deepcopy, which would try to follow class objects
# held by Instance traits.

import copy as _hs_copy  # noqa: E402

_HsMetaHasTraits = type(_tr.HasTraits)  # noqa: F821
_hs_orig_setup_class = _HsMetaHasTraits.setup_class


def _hs_setup_class(cls, classdict, **kwargs):
    first_use = {}
    for name, value in list(classdict.items()):
        if not isinstance(value, _tr.TraitType):  # noqa: F821
            continue
        if id(value) in first_use:
            duplicate = _hs_copy.copy(value)
            classdict[name] = duplicate
            setattr(cls, name, duplicate)
        else:
            first_use[id(value)] = name
    return _hs_orig_setup_class(cls, classdict, **kwargs)


_HsMetaHasTraits.setup_class = _hs_setup_class


# --- Do not auto-register methods traitlets already observes ----------------
#
# Traits fires a method called _<name>_changed automatically, so the upstream
# shim walks the class and registers every such method as an observer.  But
# HyperSpy's BaseDataAxis._index_changed is *also* decorated with
# @traits.observe("index"), and traitlets wires decorated handlers itself.
# The method therefore got registered twice, and the shim's copy guessed the
# wrong arity for the decorator object (signature() fails on it, so it fell
# back to the four-argument form) -- dragging a navigator raised
# "_index_changed() takes from 1 to 2 positional arguments but 4 were given".
#
# Rebuild the patched __init__ on top of the pristine traitlets one the shim
# stashed in _orig_HasTraits_init, skipping anything traitlets owns.


def _hs_is_event_handler(obj):
    """True for traitlets EventHandler descriptors (@observe, @validate, ...)."""
    return hasattr(obj, "func") and hasattr(obj, "instance_init")


def _hs_required_arg_count(method):
    import inspect

    try:
        sig = inspect.signature(method)
    except (ValueError, TypeError):
        return 3
    return len(
        [
            p
            for p in sig.parameters.values()
            if p.name != "self"
            and p.default is inspect.Parameter.empty
            and p.kind
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
    )


def _hs_make_changed_bridge(instance, method, nargs):
    def _bridge(change):
        if nargs == 0:
            method(instance)
        elif nargs == 1:
            method(instance, change["new"])
        elif nargs == 2:
            method(instance, change["name"], change["new"])
        else:
            method(instance, change["name"], change["old"], change["new"])

    return _bridge


def _hs_has_traits_init(self, *args, **kwargs):
    _orig_HasTraits_init(self, *args, **kwargs)  # noqa: F821
    cls = type(self)
    traits = cls.class_traits()
    for attr_name in dir(cls):
        if not attr_name.endswith("_changed"):
            continue
        inner = attr_name[1:-8]
        if not inner or inner not in traits:
            continue
        method = getattr(cls, attr_name, None)
        if method is None or isinstance(method, property):
            continue
        if _hs_is_event_handler(method):
            # traitlets already registered it from its decorator.
            continue
        if not callable(method):
            continue
        self.observe(
            _hs_make_changed_bridge(self, method, _hs_required_arg_count(method)),
            names=[inner],
        )


_tr.HasTraits.__init__ = _hs_has_traits_init  # noqa: F821

print("[hyperspy] traits.observation shim installed")
