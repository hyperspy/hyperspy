# -*- coding: utf-8 -*-
# Copyright © 2007 Francisco Javier de la Peña
#
# This file is part of EELSLab.
#
# EELSLab is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# EELSLab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EELSLab; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  
# USA

import os
import os.path
import shutil
import messages

config_files = ['eelslabrc', 'microscopes.csv', 'edges_db.csv']
data_path = os.sep.join([os.path.dirname(__file__), 'data'])

if os.name == 'posix':
    config_path = os.path.join(os.path.expanduser('~'),'.eelslab')
    os_name = 'posix'
elif os.name in ['nt','dos']:
##    appdata = os.environ['APPDATA']
    config_path = os.path.expanduser('~/.eelslab')
##    if os.path.isdir(appdata) is False:
##        os.mkdir(appdata)
##    config_path = os.path.join(os.environ['APPDATA'], 'eelslab')
    os_name = 'windows'
else:
    messages.warning_exit('Unsupported operating system:', os.name)

if os.path.isdir(config_path) is False:
    messages.warning_exit("Creating config directory: ", config_path)
    os.mkdir(config_path)
    
for file in config_files:
    templates_file = os.path.join(data_path, file)
    config_file = os.path.join(config_path, file)
    if os.path.isfile(config_file) is False:
        print "Setting configuration file: ", file
        shutil.copy(templates_file, config_file)
        

        
