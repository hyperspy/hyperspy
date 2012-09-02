import os
import csv

from hyperspy.misc.config_dir import config_path


file_path = os.path.join(config_path, 'edges_db.csv') 
f = open(file_path, 'r')
reader = csv.reader(f)
edges_dict = {}

for row in reader:
    twin_subshell = None
    element, subshell = row[0].split('.')
    Z = int(row[1])
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
