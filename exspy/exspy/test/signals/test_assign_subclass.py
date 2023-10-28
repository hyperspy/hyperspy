
import numpy as np
import pytest

import hyperspy.api as hs
from hyperspy import _lazy_signals
from hyperspy.decorators import lazifyTestClass

import exspy


class TestConvertSigna:

    def setup_method(self, method):
        self.s = hs.signals.Signal1D([0, 1])

    def test_lazy_to_eels_and_back(self):
        self.s = self.s.as_lazy()
        self.s.set_signal_type("EELS")
        assert isinstance(self.s, exspy.signals.LazyEELSSpectrum)
        self.s.set_signal_type("")
        assert isinstance(self.s, _lazy_signals.LazySignal1D)

    def test_signal1d_to_eels(self):
        self.s.set_signal_type("EELS")
        assert isinstance(self.s, exspy.signals.EELSSpectrum)
        self.s.set_signal_type("")
        assert isinstance(self.s, hs.signals.Signal1D)

    def test_signal1d_to_eds_tem(self):
        self.s.set_signal_type("EDS_TEM")
        assert isinstance(self.s, exspy.signals.EDSTEMSpectrum)
        self.s.set_signal_type("")
        assert isinstance(self.s, hs.signals.Signal1D)

    def test_signal1d_to_eds_sem(self):
        self.s.set_signal_type("EDS_SEM")
        assert isinstance(self.s, exspy.signals.EDSSEMSpectrum)
        self.s.set_signal_type("")
        assert isinstance(self.s, hs.signals.Signal1D)

    def test_error_None(self):
        with pytest.raises(TypeError):
            self.s.set_signal_type(None)


class TestConvertComplexSignal1D:

    def setup_method(self, method):
        self.s = hs.signals.ComplexSignal1D([0, 1])

    def test_complex_to_dielectric_function(self):
        self.s.set_signal_type("DielectricFunction")
        assert isinstance(self.s, exspy.signals.DielectricFunction)
        self.s.set_signal_type("")
        assert isinstance(self.s, hs.signals.ComplexSignal1D)


def test_lazy_to_eels_and_back():
    s = hs.signals.Signal1D([0, 1])
    s = s.as_lazy()
    s.set_signal_type("EELS")
    assert isinstance(s, exspy.signals.LazyEELSSpectrum)
    s.set_signal_type("")
    assert isinstance(s, _lazy_signals.LazySignal1D)


def test_complex_to_dielectric_function():
    s = hs.signals.ComplexSignal1D([0, 1])
    s.set_signal_type("DielectricFunction")
    assert isinstance(s, exspy.signals.DielectricFunction)
    s.set_signal_type("")
    assert isinstance(s, hs.signals.ComplexSignal1D)


@lazifyTestClass
class Test1d:

    def setup_method(self, method):
        self.s = hs.signals.BaseSignal([0, 1, 2])

    def test_set_EELS(self):
        s = self.s.as_signal1D(0)
        s.set_signal_type("EELS")
        assert s.metadata.Signal.signal_type == "EELS"
        if s._lazy:
            _class = exspy.signals.LazyEELSSpectrum
        else:
            _class = exspy.signals.EELSSpectrum
        assert isinstance(s, _class)


@lazifyTestClass
class Test2d:

    def setup_method(self, method):
        self.s = hs.signals.BaseSignal(np.random.random((2, 3)))  # (|3, 2)

    def test_s2EELS2im2s(self):
        pytest.importorskip("exspy")
        s = self.s.as_signal1D(0)
        s.set_signal_type("EELS")
        im = s.as_signal2D((1, 0))
        assert im.metadata.Signal.signal_type == "EELS"
        s = im.as_signal1D(0)
        assert s.metadata.Signal.signal_type == "EELS"
        if s._lazy:
            from exspy.signals import LazyEELSSpectrum
            _class = LazyEELSSpectrum
        else:
            from exspy.signals import EELSSpectrum
            _class = EELSSpectrum
        assert isinstance(s, _class)
