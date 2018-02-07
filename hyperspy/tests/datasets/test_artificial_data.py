import hyperspy.datasets.artificial_data as ad


def test_get_low_loss_eels_signal():
    s = ad.get_low_loss_eels_signal()
    assert s.metadata.Signal.signal_type == 'EELS'


def test_get_core_loss_eels_signal():
    s = ad.get_core_loss_eels_signal()
    assert s.metadata.Signal.signal_type == 'EELS'


def test_get_core_loss_eels_model():
    m = ad.get_core_loss_eels_model()
    assert m.signal.metadata.Signal.signal_type == 'EELS'


def test_get_low_loss_eels_line_scan_signal():
    s = ad.get_low_loss_eels_line_scan_signal()
    assert s.metadata.Signal.signal_type == 'EELS'


def test_get_core_loss_eels_line_scan_signal():
    s = ad.get_core_loss_eels_line_scan_signal()
    assert s.metadata.Signal.signal_type == 'EELS'


def test_get_atomic_resolution_tem_signal2d():
    s = ad.get_atomic_resolution_tem_signal2d()
    assert s.axes_manager.signal_dimension == 2
