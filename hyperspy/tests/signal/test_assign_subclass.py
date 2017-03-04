from collections import namedtuple

import numpy as np

import hyperspy.api as hs
from hyperspy.io import assign_signal_subclass


testcase = namedtuple('testcase', ['dtype', 'sig_dim', 'sig_type', 'cls'])

subclass_cases = (testcase('float', 1000, '', 'BaseSignal'),
                  testcase('float', 1, '', 'Signal1D'),
                  testcase('float', 2, '', 'Signal2D'),
                  testcase('float', 1, 'EELS', 'EELSSpectrum'),
                  testcase('float', 1, 'EDS_SEM', 'EDSSEMSpectrum'),
                  testcase('float', 1, 'EDS_TEM', 'EDSTEMSpectrum'),
                  testcase('complex', 1, 'DielectricFunction',
                           'DielectricFunction'),
                  testcase('complex', 1, 'dielectric function',
                           'DielectricFunction'),
                  testcase('complex', 1000, '', 'ComplexSignal'),
                  testcase('complex', 1, '', 'ComplexSignal1D'),
                  testcase('complex', 2, '', 'ComplexSignal2D'),
                  testcase('float', 1000, 'weird', 'BaseSignal'),
                  testcase('float', 1, 'weird', 'Signal1D'),
                  testcase('complex', 1000, 'weird', 'ComplexSignal'),
                  )


def test_assignment_class():
    for case in subclass_cases:
        assert (
            assign_signal_subclass(
                dtype=np.dtype(case.dtype),
                signal_dimension=case.sig_dim,
                signal_type=case.sig_type,
                lazy=False) is
            getattr(hs.signals, case.cls))
        lazyclass = 'Lazy' + case.cls if case.cls is not 'BaseSignal' \
            else 'LazySignal'
        assert (
            assign_signal_subclass(
                dtype=np.dtype(case.dtype),
                signal_dimension=case.sig_dim,
                signal_type=case.sig_type,
                lazy=True) is
            getattr(hs.signals, lazyclass))


class TestConvertBaseSignal:

    def setup_method(self, method):
        self.s = hs.signals.BaseSignal(np.zeros((3, 3)))

    def test_base_to_lazy(self):
        assert not self.s._lazy
        self.s._lazy = True
        self.s._assign_subclass()
        assert isinstance(self.s, hs.signals.LazySignal)
        assert self.s._lazy

    def test_base_to_1d(self):
        self.s.axes_manager.set_signal_dimension(1)
        self.s._assign_subclass()
        assert isinstance(self.s, hs.signals.Signal1D)
        self.s.metadata.Signal.record_by = ''
        self.s._assign_subclass()
        assert isinstance(self.s, hs.signals.BaseSignal)

    def test_base_to_2d(self):
        self.s.axes_manager.set_signal_dimension(2)
        self.s._assign_subclass()
        assert isinstance(self.s, hs.signals.Signal2D)

    def test_base_to_complex(self):
        self.s.change_dtype(complex)
        assert isinstance(self.s, hs.signals.ComplexSignal)
        # Going back from ComplexSignal to BaseSignal is not possible!
        # If real data is required use `real`, `imag`, `amplitude` or `phase`!


class TestConvertSignal1D:

    def setup_method(self, method):
        self.s = hs.signals.Signal1D([0])

    def test_lazy_to_eels_and_back(self):
        self.s = self.s.as_lazy()
        self.s.set_signal_type("EELS")
        assert isinstance(self.s, hs.signals.LazyEELSSpectrum)
        self.s.set_signal_type("")
        assert isinstance(self.s, hs.signals.LazySignal1D)

    def test_signal1d_to_eels(self):
        self.s.set_signal_type("EELS")
        assert isinstance(self.s, hs.signals.EELSSpectrum)
        self.s.set_signal_type("")
        assert isinstance(self.s, hs.signals.Signal1D)

    def test_signal1d_to_eds_tem(self):
        self.s.set_signal_type("EDS_TEM")
        assert isinstance(self.s, hs.signals.EDSTEMSpectrum)
        self.s.set_signal_type("")
        assert isinstance(self.s, hs.signals.Signal1D)

    def test_signal1d_to_eds_sem(self):
        self.s.set_signal_type("EDS_SEM")
        assert isinstance(self.s, hs.signals.EDSSEMSpectrum)
        self.s.set_signal_type("")
        assert isinstance(self.s, hs.signals.Signal1D)


class TestConvertComplexSignal:

    def setup_method(self, method):
        self.s = hs.signals.ComplexSignal(np.zeros((3, 3)))

    def test_complex_to_complex1d(self):
        self.s.axes_manager.set_signal_dimension(1)
        self.s._assign_subclass()
        assert isinstance(self.s, hs.signals.ComplexSignal1D)

    def test_complex_to_complex2d(self):
        self.s.axes_manager.set_signal_dimension(2)
        self.s._assign_subclass()
        assert isinstance(self.s, hs.signals.ComplexSignal2D)


class TestConvertComplexSignal1D:

    def setup_method(self, method):
        self.s = hs.signals.ComplexSignal1D([0])

    def test_complex_to_dielectric_function(self):
        self.s.set_signal_type("DielectricFunction")
        assert isinstance(self.s, hs.signals.DielectricFunction)
        self.s.set_signal_type("")
        assert isinstance(self.s, hs.signals.ComplexSignal1D)


if __name__ == '__main__':

    import pytest
    pytest.main(__name__)
