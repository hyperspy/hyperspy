import numpy as np
import pytest

from hyperspy.misc.utils import find_subclasses
from hyperspy.signal import BaseSignal
import hyperspy.signals


@pytest.mark.parametrize("signal",
                         find_subclasses(hyperspy.signals, BaseSignal))
def test_lazy_signal_inheritance(signal):
    bs = getattr(hyperspy.signals, signal)
    s = bs(np.empty((2,) * bs._signal_dimension))
    ls = s.as_lazy()
    assert isinstance(ls, bs)
