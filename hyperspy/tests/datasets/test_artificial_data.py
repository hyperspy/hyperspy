import numpy as np
import hyperspy.datasets.artificial_data as ad


def test_get_low_loss_eels_signal():
    s = ad.get_low_loss_eels_signal()
    assert s.metadata.Signal.signal_type == 'EELS'


def test_get_core_loss_eels_signal():
    s = ad.get_core_loss_eels_signal(add_powerlaw=False)
    assert s.metadata.Signal.signal_type == 'EELS'
    s1 = ad.get_core_loss_eels_signal(add_powerlaw=True)
    assert s1.metadata.Signal.signal_type == 'EELS'
    assert s1.data.sum() > s.data.sum()

    np.random.seed(seed=10)
    s2 = ad.get_core_loss_eels_signal()
    np.random.seed(seed=10)
    s3 = ad.get_core_loss_eels_signal()
    assert (s2.data == s3.data).all()


def test_get_core_loss_eels_model():
    m = ad.get_core_loss_eels_model(add_powerlaw=False)
    assert m.signal.metadata.Signal.signal_type == 'EELS'
    m1 = ad.get_core_loss_eels_model(add_powerlaw=True)
    assert m1.signal.metadata.Signal.signal_type == 'EELS'
    assert m1.signal.data.sum() > m.signal.data.sum()


def test_get_low_loss_eels_line_scan_signal():
    s = ad.get_low_loss_eels_line_scan_signal()
    assert s.metadata.Signal.signal_type == 'EELS'


def test_get_core_loss_eels_line_scan_signal():
    s = ad.get_core_loss_eels_line_scan_signal()
    assert s.metadata.Signal.signal_type == 'EELS'


def test_get_atomic_resolution_tem_signal2d():
    s = ad.get_atomic_resolution_tem_signal2d()
    assert s.axes_manager.signal_dimension == 2
