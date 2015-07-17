# -*- coding: utf-8 -*-
# Copyright 2007-2014 The HyperSpy developers
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
try:
    import dill
    dill_avail = True
except ImportError:
    dill_avail = False
    import types
    import marshal


def set_attr(target, attrs, value):
    """ Like operator.attrgetter, but for setattr - supports "nested" attributes.

        Parameters
        ----------
            target : object
            attrs : string
            value : object

        """
    where = attrs.rfind('.')
    if where != -1:
        target = attrgetter(attrs[:where])(target)
    setattr(target, attrs[where + 1:], value)


def export_to_dictionary(target, whitelist, dic, picklable=False):
    """ Exports attributes of target from whitelist.keys() to dictionary dic
        All values are references only by default.
        If picklable=True, the functions are copies (hence allows arbitrary closure)

        Parameters
        ----------
            target : object
                must contain the (nested) attributes of the whitelist.keys()
            whitelist : dictionary
                A dictionary, keys of which are used as attributes for exporting.
                For easier loading afterwards, highly advisable to contain key '_whitelist'.
                The convention is as follows:
                * key starts with '_init_' (e.g. key = '_init_volume'):
                    object of the whitelist[key] is saved, used for initialization of the target
                * key starts with '_fn_' (e.g. key = '_fn_twin_function'):
                    the targeted attribute is a function, and may be pickled (preferably with dill package).
                    A tuple of (thing, value) is exported, where thing is None if function is passed as-is, and bool if
                    dill package is used to pickle the function,
                    and value is the result.
                * key is '_id_' (e.g. key = '_id_'):
                    the id of the target is exported (e.g. id(target) )
            dic : dictionary
                A dictionary where the object will be exported
    """
    for key, value in whitelist.iteritems():
        if key.startswith('_init_'):
            dic[key] = value
        elif key.startswith('_fn_'):
            if picklable:
                if dill_avail:
                    dic[key] = (True, dill.dumps(attrgetter(key[4:])(target)))
                else:
                    dic[key] = (
                        False, marshal.dumps(attrgetter(key[4:])(target).func_code))
            else:
                dic[key] = (None, attrgetter(key[4:])(target))
        elif key == '_id_':
            dic[key] = id(target)
        else:
            dic[key] = attrgetter(key)(target)


def load_from_dictionary(target, dic):
    """ Loads attributes of target to dictionary dic
        The attribute list is read from dic['_whitelist'].keys()
        All values are references only

        Parameters
        ----------
            target : object
                must contain the (nested) attributes of the whitelist.keys()
            dic : dictionary
                A dictionary, containing field '_whitelist', which is a dictionary with all keys that were exported
                The convention is as follows:
                * key starts with '_init_' (e.g. key = '_init_volume'):
                    object had to be used for initialization of the target
                * key starts with '_fn_' (e.g. key = '_fn_twin_function'):
                    the value is a tuple of (thing, picked_function), the targeted attribute is assigned
                    unpickled (preferably with dill package) function.
                    thing {Bool, None} shows whether if the function was pickled and if using the dill package
                * key is '_id_' (e.g. key = '_id_'):
                    skipped.
    """
    for key in dic['_whitelist'].keys():
        value = dic[key]
        if key.startswith('_fn_'):
            if value[0] is None:
                set_attr(target, key[4:], value[1])
            else:
                if value[0] and not dill_avail:
                    raise ValueError(
                        "the dictionary was constructed using \"dill\" package, which is not available on the system")
                elif dill_avail:
                    set_attr(target, key[4:], dill.loads(value[1]))

                else:
                    set_attr(
                        target, key[
                            4:], types.FunctionType(
                            marshal.loads(
                                value[1]), globals()))
        elif key.startswith('_init_') or key.startswith('_id_'):
            pass
        else:
            set_attr(target, key, value)
