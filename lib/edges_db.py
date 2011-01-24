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

from config_dir import config_path


file_path = os.path.join(config_path, 'edges_db.csv') 
f = open(file_path, 'r')
reader = csv.reader(f)
edges_dict = {}
for row in reader:
    twin_subshell = None
    element, subshell = row[0].split('.')
    Z = row[1]
    if edges_dict.has_key(element) is not True :
        edges_dict[element]={}
        edges_dict[element]['subshells'] = {}
        edges_dict[element]['Z'] = Z
    if row[3] is not '':
        if subshell == "L3":
            twin_subshell = "L2"
            factor = 0.5
        if subshell == "M3":
            twin_subshell = "M2"
            factor = 0.5
        if subshell == "M5":
            twin_subshell = "M4"
            factor = 4/6.
        if subshell == "N3":
            twin_subshell = "N2"
            factor = 2/4.
        if subshell == "N5":
            twin_subshell = "N4"
            factor = 4/6.
        if subshell == "N7":
            twin_subshell = "N6"
            factor = 6/8.
        if subshell == "O5":
            twin_subshell = "O4"
            factor = 4/6.
            
    edges_dict[element]['subshells'][subshell] = {}
    edges_dict[element]['subshells'][subshell]['onset_energy'] = float(row[2])
    edges_dict[element]['subshells'][subshell]['filename'] = row[0]
    edges_dict[element]['subshells'][subshell]['relevance'] = row[4]
    edges_dict[element]['subshells'][subshell]['factor'] = 1
    
    if twin_subshell is not None :
        edges_dict[element]['subshells'][twin_subshell] = {}
        edges_dict[element]['subshells'][twin_subshell]['onset_energy'] = \
        float(row[3])
        edges_dict[element]['subshells'][twin_subshell]['filename'] = row[0]
        edges_dict[element]['subshells'][twin_subshell]['relevance'] = row[4]
        edges_dict[element]['subshells'][twin_subshell]['factor'] = factor
