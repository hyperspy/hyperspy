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
    where = attrs.rfind('.')
    if where != -1:
        target = attrgetter(attrs[:where])(target)
    setattr(target, attrs[where + 1:], value)


def export_to_dictionary(target, whitelist, dic):
    for key, value in whitelist.iteritems():
        if key.startswith('_init_'):
            dic[key] = value
        elif key.startswith('_fn_'):
            if dill_avail:
                dic[key] = (True, dill.dumps(attrgetter(key[4:])(target)))
            else:
                dic[key] = (
                    False, marshal.dumps(attrgetter(key[4:])(target).func_code))
        elif key == '_id_':
            dic[key] = id(target)
        else:
            dic[key] = attrgetter(key)(target)


def load_from_dictionary(target, dic):
    for key in dic['_whitelist'].keys():
        value = dic[key]
        if key.startswith('_fn_'):
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
