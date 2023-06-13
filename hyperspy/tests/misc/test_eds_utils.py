# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

import pytest
import numpy as np

from hyperspy.signals import Signal2D
from hyperspy.misc.eds.utils import _get_element_and_line
from hyperspy.misc.eds.utils import Wpercent
from hyperspy.misc.eds.utils import MeanZ
from hyperspy.misc.eds.utils import Mucoef
from hyperspy.misc.eds.utils import Cabsorption
from hyperspy.misc.eds.utils import Windowabsorption
from hyperspy.datasets.example_signals import EDS_SEM_Spectrum



def test_get_element_and_line():
    assert _get_element_and_line('Mn_Ka') == ('Mn', 'Ka')

    with pytest.raises(ValueError):
        _get_element_and_line('MnKa') == -1


def test_Wpercent():
                
    s = EDS_SEM_Spectrum()
    m = s.create_model()
    g=Wpercent(m,None)
    np.testing.assert_allclose([i for i in g], [ 9.32,  9.86, 26.61, 26.77, 27.43 ], atol=0.01) #test the quantification 
    
    # test the 2d dimension
    nav = Signal2D([[1.,0.8]]).T # define navigation space
    s = s * nav
    m = s.create_model()
    g=Wpercent(m,None)
    assert g.shape==(1,2,len(s.metadata.Sample.elements))#The number of elements have to be equal to the number of Xray_lines
    
    # test the 3d dimension
    nav = Signal2D([[1.,0.8], [0.6, 0.4]]).T # define navigation space
    s = s * nav
    m = s.create_model()
    g=Wpercent(m,None)
    assert g.shape==(2,2,len(s.metadata.Sample.elements))#The number of elements have to be equal to the number of Xray_lines
    
    #test mean composition
    s = EDS_SEM_Spectrum()
    m = s.create_model()
    g=Wpercent(m,'Mean')
    nav = Signal2D([[1.,0.8], [0.6, 0.4]]).T # define navigation space
    s = s * nav
    assert g.shape[0]==(len(s.metadata.Sample.elements))
    np.testing.assert_allclose([i for i in g], [ 9.32,  9.86, 26.61, 26.77, 27.43 ], atol=0.01) #test the quantification
    
    #test input as an array
    s = EDS_SEM_Spectrum()
    quanti=np.array([ 9.32,  9.86, 26.61, 26.77 , 27.43 ])
    g=Wpercent(m,quanti)
    assert g.shape[0]==(len(s.metadata.Sample.elements))
    np.testing.assert_allclose([i for i in g], [ 9.32,  9.86, 26.61, 26.77, 27.43 ], atol=0.01)
    
    
def test_MeanZ():
    s = EDS_SEM_Spectrum()
    m = s.create_model()
    g=Wpercent(m,None)
    np.testing.assert_allclose(MeanZ(m,g),27.18,atol=0.01)
    
    
def test_Mucoef():
    s = EDS_SEM_Spectrum()
    m = s.create_model()
    g=Wpercent(m,None)
    assert Mucoef(m,g).shape[0]==len(s.data)
    nav = Signal2D([[1.,0.8], [0.6, 0.4]]).T # define navigation space
    s = s * nav
    m = s.create_model()
    g=Wpercent(m,'Mean')
    assert Mucoef(m,g).size==m.axis.size
    
def test_Cabsorption():
    s = EDS_SEM_Spectrum()
    m = s.create_model()
    assert Cabsorption(m).size==m.axis.size
    
def test_Windowabsorption():  
    s = EDS_SEM_Spectrum()
    m = s.create_model()
    detector='Polymer_C'
    assert Windowabsorption(m,detector).size==m.axis.size
    
    