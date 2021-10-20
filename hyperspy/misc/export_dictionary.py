# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.

from operator import attrgetter
from hyperspy.misc.utils import attrsetter
from copy import deepcopy
import dill
from dask.array import Array


def check_that_flags_make_sense(flags):
    # one of: fn, id, sig
    def do_error(f1, f2):
        raise ValueError(
            'The flags "%s" and "%s" are not compatible' %
            (f1, f2))
    if 'fn' in flags:
        if 'id' in flags:
            do_error('fn', 'id')
        if 'sig' in flags:
            do_error('fn', 'sig')
    if 'id' in flags:
        # fn done previously
        if 'sig' in flags:
            do_error('id', 'sig')
        if 'init' in flags:
            do_error('id', 'init')
    # all sig cases already covered


def parse_flag_string(flags):
    return flags.replace(' ', '').split(',')


def export_to_dictionary(target, whitelist, dic, fullcopy=True):
    """ Exports attributes of target from whitelist.keys() to dictionary dic
    All values are references only by default.

    Parameters
    ----------
        target : object
            must contain the (nested) attributes of the whitelist.keys()
        whitelist : dictionary
            A dictionary, keys of which are used as attributes for exporting.
            Key 'self' is only available with tag 'id', when the id of the
            target is saved. The values are either None, or a tuple, where:
                - the first item a string, which containts flags, separated by
                commas.
                - the second item is None if no 'init' flag is given, otherwise
                the object required for the initialization.
            The flag conventions are as follows:
            * 'init':
                object used for initialization of the target. The object is
                saved in the tuple in whitelist
            * 'fn':
                the targeted attribute is a function, and may be pickled. A
                tuple of (thing, value) will be exported to the dictionary,
                where thing is None if function is passed as-is, and True if
                dill package is used to pickle the function, with the value as
                the result of the pickle.
            * 'id':
                the id of the targeted attribute is exported (e.g.
                id(target.name))
            * 'sig':
                The targeted attribute is a signal, and will be converted to a
                dictionary if fullcopy=True
        dic : dictionary
            A dictionary where the object will be exported
        fullcopy : bool
            Copies of objects are stored, not references. If any found,
            functions will be pickled and signals converted to dictionaries

    """
    whitelist_flags = {}
    for key, value in whitelist.items():
        if value is None:
            # No flags and/or values are given, just save the target
            thing = attrgetter(key)(target)
            if fullcopy:
                thing = deepcopy(thing)
            dic[key] = thing
            whitelist_flags[key] = ''
            continue

        flags_str, value = value
        flags = parse_flag_string(flags_str)
        check_that_flags_make_sense(flags)
        if key is 'self':
            if 'id' not in flags:
                raise ValueError(
                    'Key "self" is only available with flag "id" given')
            value = id(target)
        else:
            if 'id' in flags:
                value = id(attrgetter(key)(target))

        # here value is either id(thing), or None (all others except 'init'),
        # or something for init
        if 'init' not in flags and value is None:
            value = attrgetter(key)(target)
        # here value either id(thing), or an actual target to export
        if 'sig' in flags:
            if fullcopy:
                from hyperspy.signal import BaseSignal
                if isinstance(value, BaseSignal):
                    value = value._to_dictionary()
                    value['data'] = deepcopy(value['data'])
        elif 'fn' in flags:
            if fullcopy:
                value = (True, dill.dumps(value))
            else:
                value = (None, value)
        elif fullcopy:
            value = deepcopy(value)

        dic[key] = value
        whitelist_flags[key] = flags_str

    if '_whitelist' not in dic:
        dic['_whitelist'] = {}
    # the saved whitelist does not have any values, as they are saved in the
    # original dictionary. Have to restore then when loading from dictionary,
    # most notably all with 'init' flags!!
    dic['_whitelist'].update(whitelist_flags)


def load_from_dictionary(target, dic):
    """ Loads attributes of target to dictionary dic
    The attribute list is read from dic['_whitelist'].keys()

    Parameters
    ----------
        target : object
            must contain the (nested) attributes of the whitelist.keys()
        dic : dictionary
            A dictionary, containing field '_whitelist', which is a dictionary
            with all keys that were exported, with values being flag strings.
            The convention of the flags is as follows:
            * 'init':
                object used for initialization of the target. Will be copied to
                the _whitelist after loading
            * 'fn':
                the targeted attribute is a function, and may have been pickled
                (preferably with dill package).
            * 'id':
                the id of the original object was exported and the attribute
                will not be set. The key has to be '_id_'
            * 'sig':
                The targeted attribute was a signal, and may have been converted
                to a dictionary if fullcopy=True

    """
    new_whitelist = {}
    for key, flags_str in dic['_whitelist'].items():
        value = dic[key]
        flags = parse_flag_string(flags_str)
        if 'id' not in flags:
            value = reconstruct_object(flags, value)
            if 'init' in flags:
                new_whitelist[key] = (flags_str, value)
            else:
                attrsetter(target, key, value)
                if len(flags_str):
                    new_whitelist[key] = (flags_str, None)
                else:
                    new_whitelist[key] = None
    if hasattr(target, '_whitelist'):
        if isinstance(target._whitelist, dict):
            target._whitelist.update(new_whitelist)
    else:
        attrsetter(target, '_whitelist', new_whitelist)


def reconstruct_object(flags, value):
    """ Reconstructs the value (if necessary) after having saved it in a
    dictionary
    """
    if not isinstance(flags, list):
        flags = parse_flag_string(flags)
    if 'sig' in flags:
        if isinstance(value, dict):
            from hyperspy.signal import BaseSignal
            value = BaseSignal(**value)
            value._assign_subclass()
        return value
    if 'fn' in flags:
        ifdill, thing = value
        if ifdill is None:
            return thing
        if ifdill in [True, 'True', b'True']:
            return dill.loads(thing)
        # should not be reached
        raise ValueError("The object format is not recognized")
    if isinstance(value, Array):
        value = value.compute()
    return value
