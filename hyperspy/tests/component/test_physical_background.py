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


import numpy as np

from hyperspy.signals import Signal2D
from hyperspy.datasets.example_signals import EDS_SEM_Spectrum

def test_initialize():
    s = EDS_SEM_Spectrum()
    m = s.create_model(auto_background=False)
    m.add_physical_background()
    m.components.Physical_background.initialize()
    assert(m.components.Physical_background.a.value==0.001)
    assert(m.components.Physical_background.b.value==0.0)
    assert(m.components.Physical_background._dic['E0']==10.0)
    assert(m.components.Physical_background.mt.value==100.0)
    np.testing.assert_allclose(m.components.Physical_background._dic['teta'],1.66,atol=0.01)
    nav = Signal2D([[1.,0.8], [0.6, 0.4]]).T # define navigation space
    s = s * nav
    m = s.create_model(auto_background=False)
    m.add_physical_background()
    m.components.Physical_background.initialize()
    assert(m.components.Physical_background.quanti.map.shape==(2,2))
        

def test_function():
    s = EDS_SEM_Spectrum()
    m = s.create_model(auto_background=False)
    m.add_physical_background()
    m.components.Physical_background.initialize()
    m.fit_background(bounded=True)
    assert(np.mean((s.data-m.as_signal().data))<50)
    assert(m.components.Physical_background.b.value==0.0)
    assert(len(m.components.Physical_background._dic['Mu'])==m.axis.size)
    assert(len(m.components.Physical_background._dic['Window_absorption'])==m.axis.size)
    assert(m.components.Physical_background._dic['Window_absorption'].all()<=1)
