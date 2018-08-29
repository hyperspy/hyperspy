import pytest
from pytest import approx
import numpy as np
import hyperspy.misc.eels.tools as tools
from hyperspy._components.arctan import Arctan


class TestGetEdgeOnset:

    def setup_method(self):
        self.x_axis = np.linspace(0, 100, 10000, endpoint=True)
        self.data = self.x_axis

    @pytest.mark.parametrize("start,end,output", [
        (0, 100, 50), (5, 95, 50), (20, 80, 50), (0, 80, 40),
        (10, 50, 30)])
    def test_start_end(self, start, end, output):
        onset = tools.get_edge_onset(
                self.data, start, end, self.x_axis, percent_position=0.5)
        assert approx(onset, abs=0.01) == output

    @pytest.mark.parametrize("percent_position,output", [
        (0.1, 10), (0.3, 30), (0.5, 50), (0.7, 70), (0.9, 90)])
    def test_percent_position(self, percent_position, output):
        onset = tools.get_edge_onset(
                self.data, 0, 100, self.x_axis,
                percent_position=percent_position)
        assert approx(onset, abs=0.01) == output

    def test_different_x_axis(self):
        x_axis = np.arange(100, 200, 0.01)
        onset = tools.get_edge_onset(
                self.data, 110, 190, x_axis, percent_position=0.5)
        assert approx(onset, abs=0.01) == 150

    def test_arctan_onset_with_noise(self):
        x_axis = np.linspace(100, 200, 10000, endpoint=True)
        arctan = Arctan(A=300, k=1., x0=150, minimum_at_zero=True)
        data = arctan.function(x_axis)
        data[50:200] = 600
        onset = tools.get_edge_onset(data, 100, 200, x_axis, 0.5)
        assert approx(onset, abs=0.01) == 150

    def test_percent_position_wrong_input(self):
        with pytest.raises(ValueError):
            tools.get_edge_onset(self.data, 10, 90, self.x_axis, 1.1)
        with pytest.raises(ValueError):
            tools.get_edge_onset(self.data, 10, 90, self.x_axis, -0.1)
