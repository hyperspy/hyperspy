import numpy as np

import pytest
from numpy.testing import assert_allclose

import hyperspy.api as hs


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
        ll = self.s.deepcopy()
        ll.axes_manager[-1].offset = -20
        ll.axes_manager.navigation_shape = (123,)
        with pytest.raises(ValueError):
            m = self.s.create_model(ll=ll)


class TestEELSModel:

    def setup_method(self, method):
        s = hs.signals.EELSSpectrum(np.zeros(200))
        s.set_microscope_parameters(100, 10, 10)
        s.axes_manager[-1].offset = 150
        s.add_elements(("B", "C"))
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
                  m["B_K"].onset_energy.value -
                  hs.preferences.EELS.preedge_safe_window_width)
        m.enable_fine_structure()
        m.resolve_fine_structure()
        assert window == m["B_K"].fine_structure_width

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
        self.m.signal.metadata.Signal.binned = False
        self.m.two_area_background_estimation()
        assert_allclose(
            self.m._background_components[0].A.value,
            2.1451237089380295)
        assert_allclose(
            self.m._background_components[0].r.value,
            3.0118980767392736)

    def test_two_area_powerlaw_estimation_C(self):
        self.m["B_K"].active = False
        self.m.signal.data = 2. * self.m.axis.axis ** (-3)  # A= 2, r=3
        self.m.signal.metadata.Signal.binned = False
        self.m.two_area_background_estimation()
        assert_allclose(
            self.m._background_components[0].A.value,
            2.3978438900878087)
        assert_allclose(
            self.m._background_components[0].r.value,
            3.031884021065014)

    def test_two_area_powerlaw_estimation_no_edge(self):
        self.m["B_K"].active = False
        self.m["C_K"].active = False
        self.m.signal.data = 2. * self.m.axis.axis ** (-3)  # A= 2, r=3
        self.m.signal.metadata.Signal.binned = False
        self.m.two_area_background_estimation()
        assert_allclose(
            self.m._background_components[0].A.value,
            2.6598803469440986)
        assert_allclose(
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
        assert_allclose(self.m["Offset"].offset.value,
                        1)
        assert self.m["B_K"].active
        assert self.m["C_K"].active

    def test_fit_background_C(self):
        self.m["B_K"].active = False
        self.m.fit_background()
        assert_allclose(self.m["Offset"].offset.value,
                        1.71212121212)
        assert not self.m["B_K"].active
        assert self.m["C_K"].active

    def test_fit_background_no_edge(self):
        self.m["B_K"].active = False
        self.m["C_K"].active = False
        self.m.fit_background()
        assert_allclose(self.m["Offset"].offset.value,
                        2.13567839196)
        assert not self.m["B_K"].active
        assert not self.m["C_K"].active
