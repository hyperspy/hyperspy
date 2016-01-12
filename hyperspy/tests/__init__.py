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


import os
import nose


@nose.tools.nottest
def test(args=[], no_path_adjustment=False):
    """Run tests.

       args : list of strings
           a list of options that will be passed to nosetests
       no_path_adjustment : bool
           If True it the --no-path-adjustment option wil be passed to nosetests
    """
    mod_loc = os.path.dirname(__file__)
    totest = os.path.join(mod_loc, 'io', 'test_dm3.py')

    if no_path_adjustment is not None:
        args.append('--no-path-adjustment')
    args.insert(0, totest)
    return nose.run(argv=args)
