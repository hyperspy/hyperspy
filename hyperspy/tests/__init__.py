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
import warnings

from hyperspy.defaults_parser import preferences
preferences.General.show_progressbar = False

# Check if we should fail on external deprecation messages
fail_on_external = os.environ.pop('FAIL_ON_EXTERNAL_DEPRECATION', False)
if isinstance(fail_on_external, str):
    fail_on_external = (fail_on_external.lower() in
                        ['true', 't', '1', 'yes', 'y', 'set'])

if fail_on_external:
    warnings.filterwarnings(
        'error', category=DeprecationWarning)
    # Travis setup has these warnings, so ignore:
    warnings.filterwarnings(
        'ignore',
        r"BaseException\.message has been deprecated as of Python 2\.6",
        DeprecationWarning)
    # Don't care about warnings in hyperspy in this mode!
    warnings.filterwarnings('default', module="hyperspy")
else:
    # Fall-back filter: Error
    warnings.simplefilter('error')
    warnings.filterwarnings(
        'ignore', "Failed to import the optional scikit image package",
        UserWarning)
    # We allow extrernal warnings:
    warnings.filterwarnings('default',
                            module="(?!hyperspy)")
