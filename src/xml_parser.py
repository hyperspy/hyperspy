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

from xml.dom import minidom

import Experiments

def get_data(node):
    if node.firstChild is not None:
        return node.firstChild.data
    else:
        return None

def get_elements(xmldoc):
    elements = set()
    for element in xmldoc.getElementsByTagName('element'):
        elements.add(element.getElementsByTagName('name')[0].firstChild.data)
    return elements

def get_spectrum(xmldoc, section):
    cl = xmldoc.getElementsByTagName(section)[0]
    eels = cl.getElementsByTagName('eels')[0]
    spectrum = eels.getElementsByTagName('spectrum-image')[0]
    return spectrum
    
def get_filename(xmldoc, section):
    spectrum = get_spectrum(xmldoc, section)
    filename_node = spectrum.getElementsByTagName('file_name')[0]
    return get_data(filename_node)

def get_gain(xmldoc, section):
    spectrum = get_spectrum(xmldoc, section)
    gain = spectrum.getElementsByTagName('gain_correction')[0]
    spectrum = gain.getElementsByTagName('spectrum-image')[0]
    filename_node = spectrum.getElementsByTagName('file_name')[0]
    return get_data(filename_node)

def action_active(xmldoc, action):
    bg_action = xmldoc.getElementsByTagName(action)[0]
    return bool(bg_action.getAttribute('active'))
    
def xml_parser(filename):
    xmldoc = minidom.parse(filename)
    
    elements = get_elements(xmldoc)
    hl = get_filename(xmldoc, 'core_loss')
    ll = get_filename(xmldoc, 'low_loss')
    gain = get_gain(xmldoc, 'core_loss')
    gain_ll = get_gain(xmldoc, 'low_loss')
    
    eds = Experiments(hl = hl, ll = ll, gain = gain, gain_ll = gain_ll)
    eds.add_elements(elements)
    
    if action_active(xmldoc, 'remove_background'):
        eds.hl.remove_background()
    if action_active(xmldoc, 'save_spectrum_image'):
        plot_filename = filename[:-4]+'_cl'
        print plot_filename
        eds.hl.plot_spectrum(filename = plot_filename)
