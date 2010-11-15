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

import eelslib.Release
import eelslib.components as components

from eelslib.spectrum import Spectrum
from eelslib.image import Image
from eelslib.experiments import Experiments
from eelslib.model import Model
from eelslib.file_io import load
from eelslib.edges_db import edges_dict

__version__ = eelslib.Release.version
__revision__ = eelslib.Release.revision

print(eelslib.Release.info)
