# -*- coding: utf-8 -*-
# Copyright 2007-2011 The Hyperspy developers
#
# This file is part of  Hyperspy.
#
#  Hyperspy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  Hyperspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  Hyperspy.  If not, see <http://www.gnu.org/licenses/>.


from hyperspy.Release import version as __version__
from hyperspy import components
from hyperspy import signals
from hyperspy.io import load
from hyperspy.defaults_parser import preferences
from hyperspy import utils
from hyperspy.misc.eels.elements import elements_db as elements
from hyperspy.misc.eds.elements import elements as elements_eds

def get_configuration_directory_path():
    import hyperspy.misc.config_dir
    return hyperspy.misc.config_dir.config_path 
        
def create_model(signal, *args, **kwargs):
    """Create a model object
    
    Any extra argument is passes to the Model constructor.
    
    Parameters
    ----------    
    signal: A signal class
    
    If the signal is an EELS signal the following extra parameters
    are available:
    
    auto_background : boolean
        If True, and if spectrum is an EELS instance adds automatically 
        a powerlaw to the model and estimate the parameters by the 
        two-area method.
    auto_add_edges : boolean
        If True, and if spectrum is an EELS instance, it will 
        automatically add the ionization edges as defined in the 
        Spectrum instance. Adding a new element to the spectrum using
        the components.EELSSpectrum.add_elements method automatically
        add the corresponding ionisation edges to the model.
    ll : {None, EELSSpectrum}
        If an EELSSPectrum is provided, it will be assumed that it is
        a low-loss EELS spectrum, and it will be used to simulate the 
        effect of multiple scattering by convolving it with the EELS
        spectrum.
    GOS : {'hydrogenic', 'Hartree-Slater', None}
        The GOS to use when auto adding core-loss EELS edges.
        If None it will use the Hartree-Slater GOS if 
        they are available, otherwise it will use the hydrogenic GOS.
    
    Returns
    -------
    
    A Model class
    
    """
    
    from hyperspy.signals.eels import EELSSpectrum
    from hyperspy.models.eelsmodel import EELSModel
    from hyperspy.model import Model
    if isinstance(signal, EELSSpectrum):
        return EELSModel(signal, *args, **kwargs)
    else:
        return Model(signal, *args, **kwargs)
        
