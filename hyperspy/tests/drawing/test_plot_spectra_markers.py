
import logging

from hyperspy.datasets.artificial_data import get_core_loss_eels_line_scan_signal
from hyperspy.datasets.example_signals import EDS_TEM_Spectrum

import pytest

from hyperspy.misc.test_utils import update_close_figure
from hyperspy.utils import stack

default_tol = 2.0
baseline_dir = 'plot_spectra_markers'
style_pytest_mpl = 'default'


class TestEDSMarkers:

    @pytest.mark.mpl_image_compare(
        baseline_dir=baseline_dir, tolerance=default_tol, style=style_pytest_mpl)
    def test_plot_eds_lines(self,):
        a = EDS_TEM_Spectrum()
        s = stack([a, a * 5])
        s.plot(True)
        s.axes_manager.navigation_axes[0].index = 1
        return s._plot.signal_plot.figure


    @pytest.mark.parametrize("norm", [None, "log", "auto", "linear"])
    def test_plot_eds_lines_norm(self, norm):
        a = EDS_TEM_Spectrum()
        s = stack([a, a * 5])
        # When norm is None, don't specify (use default)
        # otherwise use specify value
        kwargs = {"norm":norm} if norm else {}
        s.plot(True, **kwargs)


    @pytest.mark.mpl_image_compare(
        baseline_dir=baseline_dir, tolerance=default_tol, style=style_pytest_mpl,
        filename='test_plot_eds_lines.png')
    def test_plot_xray_lines(self,):
        # It should be the same image as with previous test (test_plot_eds_lines)
        a = EDS_TEM_Spectrum()
        s = stack([a, a * 5])
        s.plot()
        s._plot_xray_lines(xray_lines=True)
        s.axes_manager.navigation_axes[0].index = 1
        return s._plot.signal_plot.figure


    def test_plot_eds_lines_not_in_range(self, caplog):
        s = EDS_TEM_Spectrum().isig[5.0:8.0]
        s.plot()
        with caplog.at_level(logging.WARNING):
            s._plot_xray_lines(xray_lines=['Pt_Ka'])

        assert "Pt_Ka is not in the data energy range." in caplog.text


    def test_plot_eds_lines_background(self,):
        s = EDS_TEM_Spectrum().isig[5.0:8.0]
        s.plot()
        bw = s.estimate_background_windows()
        s._plot_xray_lines(background_windows=bw)

    def test_plot_add_background_windows(self,):
        s = EDS_TEM_Spectrum().isig[5.0:8.0]
        s.plot()
        bw = s.estimate_background_windows()
        s._add_background_windows_markers(bw)
        # Add integration windows
        iw = s.estimate_integration_windows(windows_width=2.0, xray_lines=['Fe_Ka'])
        s._add_vertical_lines_groups(iw, linestyle='--')

    def test_plot_eds_markers_no_energy(self,):
        s = EDS_TEM_Spectrum()
        del s.metadata.Acquisition_instrument.TEM.beam_energy
        s.plot(True)

class TestEELSMarkers:
    @pytest.mark.mpl_image_compare(
        baseline_dir=baseline_dir, tolerance=default_tol, style=style_pytest_mpl)
    def test_plot_eels_labels(self,):
        s = get_core_loss_eels_line_scan_signal(True, add_noise=False)
        s.add_elements(['Cr'])
        s.plot(plot_edges=True)
        return s._plot.signal_plot.figure

    def test_plot_eels_labels_nav(self,):
        s = get_core_loss_eels_line_scan_signal(True, add_noise=False)
        s.add_elements(['Cr', 'Fe'])
        s.plot(plot_edges=True)
        s.axes_manager.indices = (10,)
        s._plot.close()


@update_close_figure()
def test_plot_eds_markers_close():
    s = EDS_TEM_Spectrum()
    s.plot(True)
    return s