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

import csv
import os
import os.path

from config_dir import config_path
from defaults_parser import defaults

microscopes_file = os.path.join(config_path, 'microscopes.csv')

class Microscope(object):
    microscopes = {}
    name = None
    E0 = None
    alpha = None
    beta = None
    pppc = None
    correlation_factor = None
    def __init__(self):
        self.load_microscopes()
        self.set_microscope(defaults.microscope)
    
    def load_microscopes(self):
        column_labels = ['name', 'E0', 'alpha', 'beta', 'pppc', 
        'correlation_factor'] 
        f = open(microscopes_file, 'r')
        db = csv.reader(f)
        for row in db:
            i = 1
            if row != '' and row[0] != 'Syntax' and row[0][0] != '#': 
                self.microscopes[row[0]] = {}
                for column in row[1:]:
                    self.microscopes[row[0]][column_labels[i]] = float(row[i])
                    i += 1
        f.close()
        
    def get_available_microscope_names(self):
        for microscope in self.microscopes.keys():
            print microscope
    
    def set_microscope(self, microscope_name):
        for key in self.microscopes[microscope_name]:
            exec('self.%s = self.microscopes[\'%s\'][\'%s\']' % (key, microscope_name, key))
        self.name = microscope_name
    
    def __repr__(self):
        info = '''
        Microscope parameters:
        -----------------------------
        
        Microscope: %s
        Convergence angle: %1.2f mrad
        Collection angle: %1.2f mrad
        Beam energy: %1.2E eV
        pppc: %1.2f
        Correlation factor: %1.2f
        ''' % (self.name, self.alpha, self.beta, self.E0, 
        self.pppc, self.correlation_factor)
        return info

microscope = Microscope()


