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


import os

import numpy as np

from nose.tools import assert_true, assert_equal, assert_not_equal
from hyperspy.signals import EELSSpectrum
from hyperspy.components import VolumePlasmonDrude, Lorentzian 

class Test1D:
    def setUp(self):
        """ To test the kramers_kronig_transform we will generate 3
        EELSSpectrum instances. First a model energy loss function(ELF),
        in our case following the Drude bulk plasmon peak. Second, we 
        simulate the inelastic scattering to generate a model scattering
        distribution (SPC). Finally, we use a lorentzian peak with 
        integral equal to 1 to simulate a ZLP.
        """
        # Create an empty spectrum
        s = EELSSpectrum(np.zeros((32,2048)))
        s.set_microscope_parameters(
                    beam_energy=300.0,
                    convergence_angle=14.0,
                    collection_angle=10.0)
        # Duplicate it
        z = s.deepcopy()
        ejeE = s.axes_manager.signal_axes[0]
        ejeE.scale = 0.02

        ejeE_z = z.axes_manager.signal_axes[0]
        ejeE_z.scale = ejeE.scale
        ejeE_z.offset = -10

        vpm = VolumePlasmonDrude()
        zlp = Lorentzian()

        zlp.A.value = 1.0
        zlp.gamma.value = 0.2
        zlp.centre.value = 0.0

        rnd=np.random.random
        ij=s.axes_manager
        vpm.intensity.value = 1.0
        vpm.plasmon_linewidth.value = 2.0
        
        # angular correction (energy dependent integral)
        me=511.06        # Electron rest mass in [keV/c2]
        e0=s.mapped_parameters.TEM.beam_energy 
        beta=s.mapped_parameters.TEM.EELS.collection_angle *1e-3
        tgt=e0*(2*me+e0)/(me+e0)
        thetaE=(ejeE.axis+1e-3) / (tgt * 1e3)
        angular_correction = np.log(1+(beta/thetaE)**2)

        # thickness and kinetic (constant correction)
        thk=50 # the film is 50 nm thick!
        t=e0*(1+e0/2/me)/(1+e0/me)**2
        bohr=5.292e-2  # bohradius [nm]
        pi= 3.141592654   # Pi
        constant_correction = thk / (2*pi*bohr*t*1e3)

        correction = angular_correction * constant_correction
        
        elf = s.deepcopy()
        for i in enumerate(s):
            vpm.plasmon_energy.value = 10 + (rnd() - 0.5) * 5
            # The ZLP
            elf.data[ij.coordinates] = vpm.function(ejeE.axis)
            s.data[ij.coordinates] = vpm.function(ejeE.axis) * correction
            z.data[ij.coordinates] = zlp.function(ejeE_z.axis)
        
        self.signal = {'ELF': elf, 'SPC': s, 'ZLP': z}
        
    def test_01(self):
        """ The kramers kronig transform method applied to the signal we
        have just designed above will return the CDF for the Drude bulk
        plasmon. Hopefully, we recover the signal by inverting the CDF.
        """
        items = self.signal
        elf = items['ELF']
        spc = items['SPC']
        zlp = items['ZLP']
        # i use n=1000 to simulate a metal (enormous n)
        cdf=spc.kramers_kronig_transform(zlp=zlp,iterations=1,n=1000.)[0]
        assert_true(np.allclose(np.imag(-1/cdf.data),elf.data,rtol=0.1))
        
    def test_02(self):
        """ After obtaining the CDF from KKA of the input Drude model, 
        we can calculate the two Bethe f-sum rule integrals: one for 
        imag(CDF) and other for imag(-1/CDF).
        
        First condition: neff(imag(-1/CDF)) and neff(imag(CDF)) should 
        have close values (nearly equal at higher energies).
        """
        items = self.signal
        elf = items['ELF']
        spc = items['SPC']
        zlp = items['ZLP']
        cdf, thk = spc.kramers_kronig_transform(zlp=zlp,iterations=1,n=1000.)
        neff1 = elf.bethe_f_sum()
        neff2 = cdf.bethe_f_sum()
        assert_true(np.allclose(neff1.data,neff2.data,rtol = 0.2))
        
    def test_03(self): 
        """ Second condition: neff1 should remain less than neff2.
        items = self.signal"""
        items = self.signal
        elf = items['ELF']
        spc = items['SPC']
        zlp = items['ZLP']
        cdf, thk =spc.kramers_kronig_transform(zlp=zlp,iterations=1,n=1000.)
        # the crop is because there is an overestimation of the plasmon tail
        elf.crop_spectrum(None,10.)
        cdf.crop_spectrum(None,10.)
        neff1 = elf.bethe_f_sum()
        neff2 = cdf.bethe_f_sum()
        assert_true(np.alltrue((neff2.data-neff1.data) >= 0)) 
        
    def test_04(self): 
        """ Kramers kronig analysis gives a rough estimation of sample
        thickness. As we have predefined sample thickness for our 
        scattering distribution, we can use it for testing putposes."""
        items = self.signal
        spc = items['SPC']
        zlp = items['ZLP']
        cdf,thk=spc.kramers_kronig_transform(zlp=zlp,iterations=1,n=1000.)
        thk0 = 50. * np.ones(len(thk.data))
        assert_true(np.allclose(thk0, thk.data, rtol=1.)) 
