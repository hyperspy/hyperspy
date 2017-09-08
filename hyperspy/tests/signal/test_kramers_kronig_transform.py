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


import numpy as np

import pytest

from hyperspy.components1d import VolumePlasmonDrude, Lorentzian
from hyperspy.misc.eels.tools import eels_constant
import hyperspy.api as hs


class Test2D:

    def setup_method(self, method):
        """To test the kramers_kronig_analysis we will generate 3
        EELSSpectrum instances. First a model energy loss function(ELF),
        in our case following the Drude bulk plasmon peak. Second, we
        simulate the inelastic scattering to generate a model scattering
        distribution (SPC). Finally, we use a lorentzian peak with
        integral equal to 1 to simulate a ZLP.

        """

        # Parameters
        i0 = 1.
        t = hs.signals.BaseSignal(np.arange(10, 70, 10).reshape((2, 3)))
        t = t.transpose(signal_axes=0)
        scale = 0.02

        # Create an 3x2x2048 spectrum with Drude plasmon
        s = hs.signals.EELSSpectrum(np.zeros((2, 3, 2 * 2048)))
        s.set_microscope_parameters(
            beam_energy=300.0,
            convergence_angle=5,
            collection_angle=10.0)
        s.axes_manager.signal_axes[0].scale = scale
        k = eels_constant(s, i0, t)

        vpm = VolumePlasmonDrude()
        m = s.create_model(auto_background=False)
        m.append(vpm)
        vpm.intensity.map['values'][:] = 1
        vpm.plasmon_energy.map['values'] = np.array([[8., 18.4, 15.8],
                                                     [16.6, 4.3, 3.7]])
        vpm.fwhm.map['values'] = np.array([[2.3, 4.8, 0.53],
                                           [3.7, 0.3, 0.3]])
        vpm.intensity.map['is_set'][:] = True
        vpm.plasmon_energy.map['is_set'][:] = True
        vpm.fwhm.map['is_set'][:] = True
        s.data = (m.as_signal(show_progressbar=None) * k).data

        # Create ZLP
        z = s.deepcopy()
        z.axes_manager.signal_axes[0].scale = scale
        z.axes_manager.signal_axes[0].offset = -10
        zlp = Lorentzian()
        zlp.A.value = i0
        zlp.gamma.value = 0.2
        zlp.centre.value = 0.0
        z.data[:] = zlp.function(z.axes_manager[-1].axis).reshape((1, 1, -1))
        z.data *= scale
        self.s = s
        self.thickness = t
        self.k = k
        self.zlp = z

    def test_df_given_n(self):
        """The kramers kronig analysis method applied to the signal we
        have just designed above will return the CDF for the Drude bulk
        plasmon. Hopefully, we recover the signal by inverting the CDF.

        """
        # i use n=1000 to simulate a metal (enormous n)
        cdf = self.s.kramers_kronig_analysis(zlp=self.zlp,
                                             iterations=1,
                                             n=1000.)
        s = cdf.get_electron_energy_loss_spectrum(self.zlp, self.thickness)
        assert np.allclose(s.data,
                           self.s.data[..., 1:],
                           rtol=0.01)

    def test_df_given_thickness(self):
        """The kramers kronig analysis method applied to the signal we
        have just designed above will return the CDF for the Drude bulk
        plasmon. Hopefully, we recover the signal by inverting the CDF.

        """
        cdf = self.s.kramers_kronig_analysis(zlp=self.zlp,
                                             iterations=1,
                                             t=self.thickness)
        s = cdf.get_electron_energy_loss_spectrum(self.zlp, self.thickness)
        assert np.allclose(s.data,
                           self.s.data[..., 1:],
                           rtol=0.01)

    def test_bethe_sum_rule(self):
        df = self.s.kramers_kronig_analysis(zlp=self.zlp,
                                            iterations=1,
                                            n=1000.)
        neff1, neff2 = df.get_number_of_effective_electrons(nat=50e27,
                                                            cumulative=False)
        assert np.allclose(neff1.data,
                           np.array([[0.91187657, 4.72490711, 3.60594653],
                                     [3.88077047, 0.26759741, 0.19813647]]))
        assert np.allclose(neff2.data,
                           np.array([[0.91299039, 4.37469112, 3.41580094],
                                     [3.64866394, 0.15693674, 0.11146413]]))

    def test_thickness_estimation(self):
        """Kramers kronig analysis gives a rough estimation of sample
        thickness. As we have predefined sample thickness for our
        scattering distribution, we can use it for testing putposes.

        """
        cdf, output = self.s.kramers_kronig_analysis(zlp=self.zlp,
                                                     iterations=1,
                                                     n=1000.,
                                                     full_output=True)
        assert np.allclose(
            self.thickness.data,
            output['thickness'].data,
            rtol=0.01)

    def test_thicness_input_array(self):
        with pytest.raises(ValueError):
            self.s.kramers_kronig_analysis(zlp=self.zlp,
                                           iterations=1,
                                           t=self.thickness.data)
