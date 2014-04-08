# -*- coding: utf-8 -*-
# Copyright 2007-2011 The Hyperspy developers
#
# This file is part of  Hyperspy.
#
#  Hyperspy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  Hyperspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  Hyperspy.  If not, see <http://www.gnu.org/licenses/>.


from functools import partial

class CmpPartial(partial):
    def __init__(self, *args, **kwargs):
       super(partial, self).__init__(args, kwargs)

    def __eq__(self, other):
        return hasattr(other, "func") and hasattr(other, "args") and self.args == other.args and self.func == other.func \
            and self.keywords == other.keywords

    def __ne__(self, other):
        return (not hasattr(other, "func")) or (not hasattr(args, "func")) or self.args != other.args or self.func != other.func \
            or self.keywords != other.keywords

    def __str__(self):
        return self.func.func_name + "(" + str(self.args) + ")"

    @property
    def func_code(self):
        return self.func.func_code

    @property
    def func_name(self):
        return self.func.func_name

    @property
    def func_defaults(self):
        return self.func.func_defaults

    @property
    def func_dict(self):
        return self.func.func_dict

    @property
    def func_doc(self):
        return self.func.func_doc

    @property
    def func_closure(self):
        return self.func.func_closure

    @property
    def func_globals(self):
        return self.func.func_globals
