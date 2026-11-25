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

import contextvars
from contextlib import contextmanager
from copy import deepcopy
from operator import attrgetter

import cloudpickle

from hyperspy.exceptions import UnsafeLegacyContent
from hyperspy.misc import utils
from hyperspy.misc._utils import slugify


def check_that_flags_make_sense(flags):
    # one of: fn, id, sig
    def do_error(f1, f2):
        raise ValueError('The flags "%s" and "%s" are not compatible' % (f1, f2))

    if "fn" in flags:
        if "id" in flags:
            do_error("fn", "id")
        if "sig" in flags:
            do_error("fn", "sig")
    if "id" in flags:
        # fn done previously
        if "sig" in flags:
            do_error("id", "sig")
        if "init" in flags:
            do_error("id", "init")
    # all sig cases already covered


VALID_FLAGS = ("", "init", "fn", "id", "sig")


def parse_flag_string(flags):
    """Parse a whitelist flag string into a list of flags.

    Reject flags that are not a string or that name no known flag, so that
    a malformed saved dictionary fails with a clear error instead of an
    unhandled error from the string methods or a silent default.
    """
    if not isinstance(flags, str):
        raise ValueError(f"Invalid flags {flags!r}: expected a flag string")
    parsed = flags.replace(" ", "").split(",")
    unknown = [flag for flag in parsed if flag not in VALID_FLAGS]
    if unknown:
        raise ValueError(
            f"Invalid flags {flags!r}: unknown flag(s) {', '.join(map(repr, unknown))}"
        )
    return parsed


_TRUSTED = contextvars.ContextVar("hyperspy_trusted_restore", default=False)


@contextmanager
def _trust_scope(trusted):
    """Run the block with pickled content allowed only when *trusted* is True.

    Any other value leaves the current scope untouched: it never enables trust
    from a non-True value and never disables an enclosing trusted scope.
    """
    if trusted is True:
        token = _TRUSTED.set(True)
        try:
            yield
        finally:
            _TRUSTED.reset(token)
    else:
        yield


def _is_trusted():
    """Return True only inside an explicitly trusted restore scope."""
    return _TRUSTED.get() is True


def _loads(data, what):
    """Unpickle *data* only inside a trusted restore scope.

    Parameters
    ----------
    data : bytes
        Pickled content stored in a file.
    what : str
        Short description of the stored item for the error message.
    """
    if not _is_trusted():
        raise UnsafeLegacyContent(what)
    return cloudpickle.loads(data)


def _is_pickled_fn_marker(marker):
    """Return True for a marker written for a pickled function."""
    if marker is True:
        return True
    if isinstance(marker, str):
        return marker == "True"
    if isinstance(marker, bytes):
        return marker == b"True"
    return False


def _is_pickled_fn_value(value):
    """Return True for the stored (pickled marker, bytes) function form."""
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and _is_pickled_fn_marker(value[0])
        and isinstance(value[1], bytes)
    )


def _pickled_whitelist_key(item):
    """Return a whitelist key whose stored function value needs unpickling."""
    whitelist = item.get("_whitelist")
    if not isinstance(whitelist, dict):
        return None
    for key, flags_str in whitelist.items():
        if not isinstance(flags_str, str):
            continue
        if "fn" not in parse_flag_string(flags_str):
            continue
        if _is_pickled_fn_value(item.get(key)):
            return key
    return None


def _preflight_pickled_content(component_dicts):
    """Reject pickled content before a model starts loading it.

    The scan is deliberately tolerant: malformed shapes are skipped and the
    guarded unpickling sinks still protect anything the scan does not
    recognize. Only :class:`~hyperspy.exceptions.UnsafeLegacyContent` is
    raised, and only when the current scope is not trusted.
    """
    if _is_trusted() or not isinstance(component_dicts, list):
        return
    # Imported lazily because hyperspy.model imports this module at import
    # time; a module-level import would create a cycle.
    from hyperspy.model import _COMPONENTS

    for component in component_dicts:
        if not isinstance(component, dict):
            continue
        _id = component.get("_id_name")
        if (
            isinstance(_id, str)
            and _id not in _COMPONENTS
            and "_class_dump" in component
        ):
            raise UnsafeLegacyContent(f"the stored class of component {_id!r}")
        key = _pickled_whitelist_key(component)
        if key is not None:
            raise UnsafeLegacyContent(f"the stored function {key!r}")
        parameters = component.get("parameters")
        if not isinstance(parameters, list):
            continue
        for parameter in parameters:
            if not isinstance(parameter, dict):
                continue
            key = _pickled_whitelist_key(parameter)
            if key is not None:
                raise UnsafeLegacyContent(f"the stored function {key!r}")


def export_to_dictionary(target, whitelist, dic, fullcopy=True):
    """
    Export attributes of target from whitelist.keys() to dictionary dic
    All values are references only by default.

    Parameters
    ----------
    target : object
        Must contain the (nested) attributes of the whitelist.keys().
    whitelist : dict
        A dictionary, keys of which are used as attributes for exporting.
        Key 'self' is only available with tag 'id', when the id of the
        target is saved. The values are either None, or a tuple, where:

        * the first item a string, which containts flags, separated by
          commas.
        * the second item is None if no 'init' flag is given, otherwise
          the object required for the initialization.

        The flag conventions are as follows:

        * 'init': object used for initialization of the target. The object is
          saved in the tuple in whitelist
        * 'fn': the targeted attribute is a function, and may be pickled. A
          tuple of (thing, value) will be exported to the dictionary,
          where thing is None if function is passed as-is, and True if
          cloudpickle package is used to pickle the function, with the value as
          the result of the pickle.
        * 'id': the id of the targeted attribute is exported (e.g. id(target.name))
        * 'sig': The targeted attribute is a signal, and will be converted to a
          dictionary if fullcopy=True

    dict
        A dictionary where the object will be exported.
    bool
        Copies of objects are stored, not references. If any found,
        functions will be pickled and signals converted to dictionaries.
    """
    whitelist_flags = {}
    for key, value in whitelist.items():
        if value is None:
            # No flags and/or values are given, just save the target
            thing = attrgetter(key)(target)
            if fullcopy:
                thing = deepcopy(thing)
            dic[key] = thing
            whitelist_flags[key] = ""
            continue

        flags_str, value = value
        flags = parse_flag_string(flags_str)
        check_that_flags_make_sense(flags)
        if key == "self":
            if "id" not in flags:
                raise ValueError('Key "self" is only available with flag "id" given')
            value = id(target)
        else:
            if "id" in flags:
                value = id(attrgetter(key)(target))

        # here value is either id(thing), or None (all others except 'init'),
        # or something for init
        if "init" not in flags and value is None:
            value = attrgetter(key)(target)
        # here value either id(thing), or an actual target to export
        if "sig" in flags:
            if fullcopy:
                from hyperspy.signal import BaseSignal

                if isinstance(value, BaseSignal):
                    value = value._to_dictionary()
                    value["data"] = deepcopy(value["data"])
        elif "fn" in flags:
            if fullcopy:
                value = (True, cloudpickle.dumps(value))
            else:
                value = (None, value)
        elif fullcopy:
            value = deepcopy(value)

        dic[key] = value
        whitelist_flags[key] = flags_str

    if "_whitelist" not in dic:
        dic["_whitelist"] = {}
    # the saved whitelist does not have any values, as they are saved in the
    # original dictionary. Have to restore then when loading from dictionary,
    # most notably all with 'init' flags!!
    dic["_whitelist"].update(whitelist_flags)


def load_from_dictionary(target, dic):
    """
    Loads attributes of target to dictionary dic
    The attribute list is read from dic['_whitelist'].keys()

    Parameters
    ----------
    target : obj
        must contain the (nested) attributes of the whitelist.keys()
    dic : dict
        A dictionary, containing field '_whitelist', which is a dictionary
        with all keys that were exported, with values being flag strings.
        The convention of the flags is as follows:

        * 'init': object used for initialization of the target. Will be
          copied to the _whitelist after loading
        * 'fn': the targeted attribute is a function, and may have been
          pickled (preferably with cloudpickle package).
        * 'id': the id of the original object was exported and the
          attribute will not be set. The key has to be '_id_'
        * 'sig': The targeted attribute was a signal, and may have been
          converted to a dictionary if fullcopy=True

    Pickled content is only loaded when this function runs inside a public
    restore call made with ``trusted=True``; otherwise
    :class:`~hyperspy.exceptions.UnsafeLegacyContent` is raised.

    """
    new_whitelist = {}
    for key, flags_str in dic["_whitelist"].items():
        # Whitelist keys are file-controlled dotted attribute paths: reject
        # non-string and dunder parts before any traversal or mutation to
        # prevent class pollution (for example
        # `__class__.__base__.__getattribute__`). Attribute traversal
        # slugifies each part, so a part such as `!__class__` still reaches a
        # dunder: reject the normalized parts too.
        if not isinstance(key, str):
            raise ValueError(f"Refusing to set the non-string whitelist key {key!r}")
        if any(
            part.startswith("__")
            or slugify(part, valid_variable_name=True).startswith("__")
            for part in key.split(".")
        ):
            raise ValueError(f"Refusing to set the dunder whitelist key {key!r}")
        if key not in dic:
            raise ValueError(
                f"Invalid `_whitelist`: key {key!r} is missing from the dictionary"
            )
        value = dic[key]
        flags = parse_flag_string(flags_str)
        check_that_flags_make_sense(flags)
        if "id" not in flags:
            value = reconstruct_object(flags, value)
            if "init" in flags:
                new_whitelist[key] = (flags_str, value)
            else:
                utils.attrsetter(target, key, value)
                if len(flags_str):
                    new_whitelist[key] = (flags_str, None)
                else:
                    new_whitelist[key] = None
    if hasattr(target, "_whitelist"):
        if isinstance(target._whitelist, dict):
            target._whitelist.update(new_whitelist)
    else:
        utils.attrsetter(target, "_whitelist", new_whitelist)


def reconstruct_object(flags, value):
    """Reconstructs the value (if necessary) after having saved it in a
    dictionary.

    Pickled content is only loaded when this function runs inside a public
    restore call made with ``trusted=True``; otherwise
    :class:`~hyperspy.exceptions.UnsafeLegacyContent` is raised.
    """
    if not isinstance(flags, list):
        flags = parse_flag_string(flags)
    if "sig" in flags:
        if isinstance(value, dict):
            from hyperspy.signal import BaseSignal

            value = BaseSignal(**value)
            value._assign_subclass()
        return value
    if "fn" in flags:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(
                f'Invalid value for the "fn" flag {value!r}: expected a '
                "(cloudpickle marker, pickled function) pair"
            )
        ifcloudpickle, thing = value
        if ifcloudpickle is None:
            return thing
        if _is_pickled_fn_marker(ifcloudpickle):
            return _loads(thing, "a stored function")
        # should not be reached
        raise ValueError("The object format is not recognized")
    if utils.is_dask_array(value):
        value = value.compute()
    return value
