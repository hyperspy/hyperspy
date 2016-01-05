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


def load_1D_EDS_SEM_spectrum():
    """
    Load an EDS-SEM spectrum

    - Sample: EDS-TM002 provided by BAM (www.webshop.bam.de)
    - SEM Microscope: Nvision40 Carl Zeiss
    - EDS Detector: X-max 80 from Oxford Instrument
    """
    from hyperspy.io import load
    file_path = os.sep.join([os.path.dirname(__file__), 'eds',
                             'example_signals', '1D_EDS_SEM_Spectrum.hdf5'])
    return load(file_path)


def load_1D_EDS_TEM_spectrum():
    """
    Load an EDS-TEM spectrum

    - Sample: FePt bimetallic nanoparticles
    - SEM Microscope: Tecnai Osiris 200 kV D658 AnalyticalTwin
    - EDS Detector: Super-X 4 detectors Brucker
    """
    from hyperspy.io import load
    file_path = os.sep.join([os.path.dirname(__file__), 'eds',
                             'example_signals', '1D_EDS_TEM_Spectrum.hdf5'])
    return load(file_path)
