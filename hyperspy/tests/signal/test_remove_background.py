import numpy as np

from hyperspy import signals
from hyperspy import components1d
from hyperspy.decorators import lazifyTestClass
from hyperspy.misc.test_utils import update_close_figure

from hyperspy._signals.signal1d import BackgroundRemoval

@lazifyTestClass
class TestRemoveBackground1DGaussian:

    def setup_method(self, method):
        gaussian = components1d.Gaussian()
        gaussian.A.value = 10
        gaussian.centre.value = 10
        gaussian.sigma.value = 1
        self.signal = signals.Signal1D(
            gaussian.function(np.arange(0, 20, 0.01)))
        self.signal.axes_manager[0].scale = 0.01
        self.signal.metadata.Signal.binned = False

    def test_background_remove_gaussian(self):
        s1 = self.signal.remove_background(
            signal_range=(None, None),
            background_type='Gaussian',
            show_progressbar=None)
        assert np.allclose(s1.data, np.zeros(len(s1.data)))

    def test_background_remove_gaussian_full_fit(self):
        s1 = self.signal.remove_background(
            signal_range=(None, None),
            background_type='Gaussian',
            fast=False)
        assert np.allclose(s1.data, np.zeros(len(s1.data)))


@lazifyTestClass
class TestRemoveBackground1DPowerLaw:

    def setup_method(self, method):
        pl = components1d.PowerLaw()
        pl.A.value = 1e10
        pl.r.value = 3
        self.signal = signals.Signal1D(
            pl.function(np.arange(100, 200)))
        self.signal.axes_manager[0].offset = 100
        self.signal.metadata.Signal.binned = False

    def test_background_remove_pl(self):
        s1 = self.signal.remove_background(
            signal_range=(None, None),
            background_type='PowerLaw',
            show_progressbar=None)
        assert np.allclose(s1.data, np.zeros(len(s1.data)), atol=60)

    def test_background_remove_pl_int(self):
        self.signal.change_dtype("int")
        s1 = self.signal.remove_background(
            signal_range=(None, None),
            background_type='PowerLaw',
            show_progressbar=None)
        assert np.allclose(s1.data, np.zeros(len(s1.data)), atol=60)

#@update_close_figure
def test_BackgroundRemoval_gui():
    # create a signal
    axd = {'name':'some',
           'units':'axis',
           'scale':0.1,
           'offset':-10.,
           'size':512}
    s = signals.Signal1D(np.random.rand(axd['size']), axes=[axd,])
    s = signals.BaseSignal(np.ones((2,2))).T * s

    m = s.create_model()
    vpm = components1d.VolumePlasmonDrude()
    zlp = components1d.Gaussian(1000.)
    m.append(vpm)
    m.append(zlp)
    s = m.as_signal(show_progressbar=False)

    # Remove Background using CLI
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bck_cli = s.remove_background(signal_range=(1., 2.))

    # Remove Background GUI
    br = BackgroundRemoval(s,
                           background_type='Power Law',
                           polynomial_order=2,
                           fast=True,
                           plot_remainder=True,
                           show_progressbar=None)

    # Interact with GUI...
    Ei = 1.
    Ef = 2.
    br.ss_left_value  = Ei
    br.ss_right_value = Ef
    br.span_selector_changed()

    # ... extracting remainder line
    bck_gui = bck_cli.deepcopy()
    for si in s:
        idx = s.axes_manager.indices
        bck_gui.data[idx] = np.nan_to_num(
                                s._plot.signal_plot.ax_lines[2].data_function())
    bck_cli.crop_signal1D(Ei)
    bck_gui.crop_signal1D(Ei)

    # compare
    assert np.allclose(bck_cli.data, bck_gui.data)
