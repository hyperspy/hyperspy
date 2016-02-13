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
import os.path
import shutil
from hyperspy import messages

config_files = list()
data_path = os.sep.join([os.path.dirname(__file__), '..', 'data'])

if os.name == 'posix':
    config_path = os.path.join(os.path.expanduser('~'), '.hyperspy')
    os_name = 'posix'
elif os.name in ['nt', 'dos']:
    ##    appdata = os.environ['APPDATA']
    config_path = os.path.expanduser('~/.hyperspy')
# if os.path.isdir(appdata) is False:
# os.mkdir(appdata)
##    config_path = os.path.join(os.environ['APPDATA'], 'hyperspy')
    os_name = 'windows'
else:
    messages.warning_exit('Unsupported operating system: %s' % os.name)

if os.path.isdir(config_path) is False:
    messages.information("Creating config directory: %s" % config_path)
    os.mkdir(config_path)

for file in config_files:
    templates_file = os.path.join(data_path, file)
    config_file = os.path.join(config_path, file)
    if os.path.isfile(config_file) is False:
        messages.information("Setting configuration file: %s" % file)
        shutil.copy(templates_file, config_file)
