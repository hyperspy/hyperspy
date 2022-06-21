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

import io
import contextlib
import numpy as np
import pytest

import hyperspy.api as hs
from hyperspy.decorators import lazifyTestClass


# Dask does not always work nicely with np.errstate,
# see: https://github.com/dask/dask/issues/3245, so
# filter out divide-by-zero warnings that only appear
# when the test is lazy. When the test is not lazy,
# internal use of np.errstate means the warnings never
# appear in the first place.
@pytest.mark.filterwarnings("ignore:invalid value encountered in subtract:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:divide by zero encountered in log:RuntimeWarning")
@lazifyTestClass
class TestCreateEELSModel:

    def setup_method(self, method):
        s = hs.signals.EELSSpectrum(np.zeros(200))
        s.set_microscope_parameters(100, 10, 10)
        s.axes_manager[-1].offset = 150
        s.add_elements(("B", "C"))
        self.s = s

    def test_create_eelsmodel(self):
        from hyperspy.models.eelsmodel import EELSModel
        assert isinstance(self.s.create_model(), EELSModel)

    def test_create_eelsmodel_no_md(self):
        s = self.s
        del s.metadata.Acquisition_instrument
        with pytest.raises(ValueError):
            s.create_model()

    def test_auto_add_edges_true(self):
        m = self.s.create_model(auto_add_edges=True)
        cnames = [component.name for component in m]
        assert "B_K" in cnames and "C_K" in cnames

    def test_gos(self):
        m = self.s.create_model(auto_add_edges=True, GOS="hydrogenic")
        assert m["B_K"].GOS._name == "hydrogenic"

    def test_auto_add_background_true(self):
        m = self.s.create_model(auto_background=True)
        from hyperspy.components1d import PowerLaw
        is_pl_instance = [isinstance(c, PowerLaw) for c in m]
        assert True in is_pl_instance

    def test_auto_add_edges_false(self):
        m = self.s.create_model(auto_background=False)
        from hyperspy.components1d import PowerLaw
        is_pl_instance = [isinstance(c, PowerLaw) for c in m]
        assert not True in is_pl_instance

    def test_auto_add_edges_false_names(self):
        m = self.s.create_model(auto_add_edges=False)
        cnames = [component.name for component in m]
        assert not "B_K" in cnames or "C_K" in cnames

    def test_low_loss(self):
        ll = self.s.deepcopy()
        ll.axes_manager[-1].offset = -20
        m = self.s.create_model(ll=ll)
        assert m.low_loss is ll
        assert m.convolved

    def test_low_loss_bad_shape(self):
        ll = hs.stack([self.s] * 2)
        with pytest.raises(ValueError):
            _ = self.s.create_model(ll=ll)


@lazifyTestClass
class TestEELSModel:

    def setup_method(self, method):
        s = hs.signals.EELSSpectrum(np.ones(200))
        s.set_microscope_parameters(100, 10, 10)
        s.axes_manager[-1].offset = 150
        s.add_elements(("B", "C"))
        self.s = s
        self.m = s.create_model()

    def test_suspend_auto_fsw(self):
        m = self.m
        m["B_K"].fine_structure_width = 140.
        m.suspend_auto_fine_structure_width()
        m.enable_fine_structure()
        m.resolve_fine_structure()
        assert 140 == m["B_K"].fine_structure_width

    def test_resume_fsw(self):
        m = self.m
        m["B_K"].fine_structure_width = 140.
        m.suspend_auto_fine_structure_width()
        m.resume_auto_fine_structure_width()
        window = (m["C_K"].onset_energy.value -
                  m["B_K"].onset_energy.value - m._preedge_safe_window_width)
        m.enable_fine_structure()
        m.resolve_fine_structure()
        assert window == m["B_K"].fine_structure_width

    def test_disable_fine_structure(self):
        self.m.components.C_K.fine_structure_active = True
        self.m.components.B_K.fine_structure_active = True
        self.m.disable_fine_structure()
        assert not self.m.components.C_K.fine_structure_active
        assert not self.m.components.B_K.fine_structure_active

    def test_get_first_ionization_edge_energy_C_B(self):
        assert (self.m._get_first_ionization_edge_energy() ==
                self.m["B_K"].onset_energy.value)

    def test_get_first_ionization_edge_energy_C(self):
        self.m["B_K"].active = False
        assert (self.m._get_first_ionization_edge_energy() ==
                self.m["C_K"].onset_energy.value)

    def test_get_first_ionization_edge_energy_None(self):
        self.m["B_K"].active = False
        self.m["C_K"].active = False
        assert self.m._get_first_ionization_edge_energy() is None

    def test_two_area_powerlaw_estimation_BC(self):
        self.m.signal.data = 2. * self.m.axis.axis ** (-3)  # A= 2, r=3
        #self.m.signal.axes_manager[-1].is_binned = False
        self.m.two_area_background_estimation()
        np.testing.assert_allclose(
            self.m._background_components[0].A.value,
            2.1451237089380295)
        np.testing.assert_allclose(
            self.m._background_components[0].r.value,
            3.0118980767392736)

    def test_two_area_powerlaw_estimation_C(self):
        self.m["B_K"].active = False
        self.m.signal.data = 2. * self.m.axis.axis ** (-3)  # A= 2, r=3
        #self.m.signal.axes_manager[-1].is_binned = False
        self.m.two_area_background_estimation()
        np.testing.assert_allclose(
            self.m._background_components[0].A.value,
            2.3978438900878087)
        np.testing.assert_allclose(
            self.m._background_components[0].r.value,
            3.031884021065014)

    def test_two_area_powerlaw_estimation_no_edge(self):
        self.m["B_K"].active = False
        self.m["C_K"].active = False
        self.m.signal.data = 2. * self.m.axis.axis ** (-3)  # A= 2, r=3
        print(self.m.signal.axes_manager[-1].is_binned)
        #self.m.signal.axes_manager[-1].is_binned = False
        self.m.two_area_background_estimation()
        np.testing.assert_allclose(
            self.m._background_components[0].A.value,
            2.6598803469440986)
        np.testing.assert_allclose(
            self.m._background_components[0].r.value,
            3.0494030409062058)

    def test_get_start_energy_none(self):
        assert (self.m._get_start_energy() ==
                150)

    def test_get_start_energy_above(self):
        assert (self.m._get_start_energy(170) ==
                170)

    def test_get_start_energy_below(self):
        assert (self.m._get_start_energy(100) ==
                150)

    def test_remove_components(self):
        comp = self.m[1]
        assert len(self.m) == 3
        self.m.remove(comp)
        assert len(self.m) == 2

    def test_fit_wrong_kind(self):
        with pytest.raises(ValueError):
            self.m.fit(kind="wrongkind")

    def test_enable_background(self):
        self.m.components.PowerLaw.active = False
        self.m.enable_background()
        assert self.m.components.PowerLaw.active

    def test_disable_background(self):
        self.m.components.PowerLaw.active = True
        self.m.disable_background()
        assert not self.m.components.PowerLaw.active

    def test_signal1d_property(self):
        assert self.s == self.m.signal1D
        s_new = hs.signals.EELSSpectrum(np.ones(200))
        s_new.set_microscope_parameters(100, 10, 10)
        self.m.signal1D = s_new
        assert self.m.signal1D == s_new

    def test_signal1d_property_wrong_value_setter(self):
        m = self.m
        s = hs.signals.Signal1D(np.ones(200))
        with pytest.raises(ValueError):
            self.m.signal1D = s

    def test_remove(self):
        m = self.m
        c_k = m.components.C_K
        assert c_k in m
        m.remove(c_k)
        assert c_k not in m

    def test_quantify(self):
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            self.m.quantify()
        out = f.getvalue()
        assert out == '\nAbsolute quantification:\nElem.\tIntensity\nB\t1.000000\nC\t1.000000\n'

    def test_enable_edges(self):
        m = self.m
        m.components.B_K.active = False
        m.components.C_K.active = False
        m.enable_edges(edges_list=[m.components.B_K])
        assert m.components.B_K.active
        assert not m.components.C_K.active
        m.enable_edges()
        assert m.components.B_K.active
        assert m.components.C_K.active

    def test_disable_edges(self):
        m = self.m
        m.components.B_K.active = True
        m.components.C_K.active = True
        m.disable_edges(edges_list=[m.components.B_K])
        assert not m.components.B_K.active
        assert m.components.C_K.active
        m.disable_edges()
        assert not m.components.B_K.active
        assert not m.components.C_K.active

    def test_set_all_edges_intensities_positive(self):
        m = self.m
        m.components.B_K.intensity.ext_force_positive = False
        m.components.B_K.intensity.ext_bounded = False
        m.components.C_K.intensity.ext_force_positive = False
        m.components.C_K.intensity.ext_bounded = False
        m.set_all_edges_intensities_positive()
        assert m.components.B_K.intensity.ext_force_positive
        assert m.components.B_K.intensity.ext_bounded
        assert m.components.C_K.intensity.ext_force_positive
        assert m.components.C_K.intensity.ext_bounded

    def test_unset_all_edges_intensities_positive(self):
        m = self.m
        m.components.B_K.intensity.ext_force_positive = True
        m.components.B_K.intensity.ext_bounded = True
        m.components.C_K.intensity.ext_force_positive = True
        m.components.C_K.intensity.ext_bounded = True
        m.unset_all_edges_intensities_positive()
        assert not m.components.B_K.intensity.ext_force_positive
        assert not m.components.B_K.intensity.ext_bounded
        assert not m.components.C_K.intensity.ext_force_positive
        assert not m.components.C_K.intensity.ext_bounded

    def test_fix_edges(self):
        m = self.m
        m.components.B_K.onset_energy.free = True
        m.components.B_K.intensity.free = True
        m.components.B_K.fine_structure_coeff.free = True
        m.components.C_K.onset_energy.free = True
        m.components.C_K.intensity.free = True
        m.components.C_K.fine_structure_coeff.free = True
        m.fix_edges(edges_list=[m.components.B_K])
        assert not m.components.B_K.onset_energy.free
        assert not m.components.B_K.intensity.free
        assert not m.components.B_K.fine_structure_coeff.free
        assert m.components.C_K.onset_energy.free
        assert m.components.C_K.intensity.free
        assert m.components.C_K.fine_structure_coeff.free
        m.fix_edges()
        assert not m.components.B_K.onset_energy.free
        assert not m.components.B_K.intensity.free
        assert not m.components.B_K.fine_structure_coeff.free
        assert not m.components.C_K.onset_energy.free
        assert not m.components.C_K.intensity.free
        assert not m.components.C_K.fine_structure_coeff.free

    def test_free_edges(self):
        m = self.m
        m.components.B_K.onset_energy.free = False
        m.components.B_K.intensity.free = False
        m.components.B_K.fine_structure_coeff.free = False
        m.components.C_K.onset_energy.free = False
        m.components.C_K.intensity.free = False
        m.components.C_K.fine_structure_coeff.free = False
        m.free_edges(edges_list=[m.components.B_K])
        assert m.components.B_K.onset_energy.free
        assert m.components.B_K.intensity.free
        assert m.components.B_K.fine_structure_coeff.free
        assert not m.components.C_K.onset_energy.free
        assert not m.components.C_K.intensity.free
        assert not m.components.C_K.fine_structure_coeff.free
        m.free_edges()
        assert m.components.B_K.onset_energy.free
        assert m.components.B_K.intensity.free
        assert m.components.B_K.fine_structure_coeff.free
        assert m.components.C_K.onset_energy.free
        assert m.components.C_K.intensity.free
        assert m.components.C_K.fine_structure_coeff.free

    def test_fix_fine_structure(self):
        m = self.m
        m.components.B_K.fine_structure_coeff.free = True
        m.components.C_K.fine_structure_coeff.free = True
        m.fix_fine_structure(edges_list=[m.components.B_K])
        assert not m.components.B_K.fine_structure_coeff.free
        assert m.components.C_K.fine_structure_coeff.free
        m.fix_fine_structure()
        assert not m.components.B_K.fine_structure_coeff.free
        assert not m.components.C_K.fine_structure_coeff.free

    def test_free_fine_structure(self):
        m = self.m
        m.components.B_K.fine_structure_coeff.free = False
        m.components.C_K.fine_structure_coeff.free = False
        m.free_fine_structure(edges_list=[m.components.B_K])
        assert m.components.B_K.fine_structure_coeff.free
        assert not m.components.C_K.fine_structure_coeff.free
        m.free_fine_structure()
        assert m.components.B_K.fine_structure_coeff.free
        assert m.components.C_K.fine_structure_coeff.free


@lazifyTestClass
class TestEELSModelFitting:

    def setup_method(self, method):
        data = np.zeros(200)
        data[25:] = 100
        s = hs.signals.EELSSpectrum(data)
        s.set_microscope_parameters(100, 10, 10)
        s.axes_manager[-1].offset = 150
        s.add_elements(("B", ))
        self.m = s.create_model(auto_background=False)

    def test_free_edges(self):
        m = self.m
        m.enable_fine_structure()
        intensity = m.components.B_K.intensity.value
        onset_energy = m.components.B_K.onset_energy.value
        fine_structure_coeff = m.components.B_K.fine_structure_coeff.value
        m.free_edges()
        m.multifit()
        assert intensity != m.components.B_K.intensity.value
        assert onset_energy != m.components.B_K.onset_energy.value
        assert fine_structure_coeff != m.components.B_K.fine_structure_coeff.value

    def test_fix_edges(self):
        m = self.m
        m.enable_fine_structure()
        intensity = m.components.B_K.intensity.value
        onset_energy = m.components.B_K.onset_energy.value
        fine_structure_coeff = m.components.B_K.fine_structure_coeff.value
        m.free_edges()
        m.fix_edges()
        m.multifit()
        assert intensity == m.components.B_K.intensity.value
        assert onset_energy == m.components.B_K.onset_energy.value
        assert fine_structure_coeff == m.components.B_K.fine_structure_coeff.value


@lazifyTestClass
class TestFitBackground:

    def setup_method(self, method):
        s = hs.signals.EELSSpectrum(np.ones(200))
        s.set_microscope_parameters(100, 10, 10)
        s.axes_manager[-1].offset = 150
        CE = hs.material.elements.C.Atomic_properties.Binding_energies.K.onset_energy_eV
        BE = hs.material.elements.B.Atomic_properties.Binding_energies.K.onset_energy_eV
        s.isig[BE:] += 1
        s.isig[CE:] += 1
        s.add_elements(("Be", "B", "C"))
        self.m = s.create_model(auto_background=False)
        self.m.append(hs.model.components1D.Offset())

    def test_fit_background_B_C(self):
        self.m.fit_background()
        np.testing.assert_allclose(self.m["Offset"].offset.value,
                        1)
        assert self.m["B_K"].active
        assert self.m["C_K"].active

    def test_fit_background_C(self):
        self.m["B_K"].active = False
        self.m.fit_background()
        np.testing.assert_allclose(self.m["Offset"].offset.value,
                        1.7142857)
        assert not self.m["B_K"].active
        assert self.m["C_K"].active

    def test_fit_background_no_edge(self):
        self.m["B_K"].active = False
        self.m["C_K"].active = False
        self.m.fit_background()
        np.testing.assert_allclose(self.m["Offset"].offset.value,
                        2.14)
        assert not self.m["B_K"].active
        assert not self.m["C_K"].active


@lazifyTestClass
class TestFitBackground2D:

    def setup_method(self):
        pl = hs.model.components1D.PowerLaw()
        data = np.empty((2, 250))
        data[0] = pl.function(np.arange(150, 400))
        pl.r.value = 1.5
        data[1] = pl.function(np.arange(150, 400))
        s = hs.signals.EELSSpectrum(data)
        s.set_microscope_parameters(100, 10, 10)
        s.axes_manager[-1].offset = 150
        self.s = s
        self.m = s.create_model()

    def test_only_current_false(self):
        self.m.fit_background(only_current=False)
        residual = self.s - self.m.as_signal()
        assert pytest.approx(residual.data) == 0
