import numpy as np

from hyperspy.component import Component
from hyperspy.axes import AxesManager
from unittest import mock


class TestMultidimensionalActive:

    def setup_method(self, method):
        self.c = Component(["parameter"])
        self.c._axes_manager = AxesManager([{"size": 3,
                                             "navigate": True},
                                            {"size": 2,
                                             "navigate": True}])

    def test_enable_pixel_switching_current_on(self):
        c = self.c
        c._axes_manager.indices = (1, 1)
        c.active = True
        c.active_is_multidimensional = True
        assert np.all(c._active_array)

    def test_enable_pixel_switching_current_off(self):
        c = self.c
        c._axes_manager.indices = (1, 1)
        c.active = False
        c.active_is_multidimensional = True
        assert not self.c.active

    def test_disable_pixel_switching(self):
        c = self.c
        c.active = True
        c.active_is_multidimensional = True
        c.active_is_multidimensional = False
        assert c._active_array is None

    def test_disable_pixel_switching_current_on(self):
        c = self.c
        c._axes_manager.indices = (1, 1)
        c.active = True
        c.active_is_multidimensional = True
        c.active_is_multidimensional = False
        assert c.active

    def test_disable_pixel_switching_current_off(self):
        c = self.c
        c._axes_manager.indices = (1, 1)
        c.active = False
        c.active_is_multidimensional = True
        c.active_is_multidimensional = False
        assert not c.active


def test_update_number_free_parameters():
    c = Component(['one', 'two', 'three'])
    c.one.free = False
    c.two.free = True
    c.three.free = True
    c.two._number_of_elements = 2
    c.three._number_of_elements = 3
    c._nfree_param = 0
    c._update_free_parameters()
    assert c._nfree_param == 5
    # check that only the correct parameters are in the list _AND_ the list is
    # name-ordered
    assert [c.three, c.two] == c.free_parameters


class TestGeneralMethods:

    def setup_method(self, method):
        self.c = Component(["one", "two"])
        self.c.one.free = False
        self.c.two.free = True
        self.c.one._number_of_elements = 1
        self.c.two._number_of_elements = 2

    def test_export_free(self):
        c = self.c
        c.one.export = mock.MagicMock()
        c.two.export = mock.MagicMock()
        c.free_parameters = {c.two, }
        call_args = {'folder': 'folder1',
                     'format': 'format1',
                     'save_std': 'save_std1'}
        c.export(only_free=True, **call_args)
        assert c.two.export.call_args[1] == call_args
        assert not c.one.export.called

    def test_export_all_no_twins(self):
        c = self.c
        c.one.export = mock.MagicMock()
        c.two.export = mock.MagicMock()
        c.free_parameters = {c.two, }
        call_args = {'folder': 'folder1',
                     'format': 'format1',
                     'save_std': 'save_std1'}
        c.export(only_free=False, **call_args)
        assert c.two.export.call_args[1] == call_args
        assert c.one.export.call_args[1] == call_args

    def test_export_all_twins(self):
        c = self.c
        c.one.export = mock.MagicMock()
        c.two.export = mock.MagicMock()
        c.two.twin = c.one
        c.free_parameters = {c.two, }
        call_args = {'folder': 'folder1',
                     'format': 'format1',
                     'save_std': 'save_std1'}
        c.export(only_free=False, **call_args)
        assert c.one.export.call_args[1] == call_args
        assert not c.two.export.called

    def test_update_number_parameters(self):
        self.c.nparam = 0
        self.c.update_number_parameters()
        assert self.c.nparam == 3

    def test_fetch_from_array(self):
        arr = np.array([30, 20, 10])
        arr_std = np.array([30.5, 20.5, 10.5])
        self.c.fetch_values_from_array(arr, p_std=arr_std, onlyfree=False)
        assert self.c.one.value == 30
        assert self.c.one.std == 30.5
        assert self.c.two.value == (20, 10)
        assert self.c.two.std == (20.5, 10.5)

    def test_fetch_from_array_free(self):
        arr = np.array([30, 20, 10])
        arr_std = np.array([30.5, 20.5, 10.5])
        self.c.one.value = 1.
        self.c.one.std = np.nan
        self.c.fetch_values_from_array(arr, p_std=arr_std, onlyfree=True)
        assert self.c.one.value == 1
        assert self.c.one.std is np.nan
        assert self.c.two.value == (30, 20)
        assert self.c.two.std == (30.5, 20.5)

    def test_fetch_stored_values_fixed(self):
        c = self.c
        c.one.fetch = mock.MagicMock()
        c.two.fetch = mock.MagicMock()
        c.fetch_stored_values(only_fixed=True)
        assert c.one.fetch.called
        assert not c.two.fetch.called

    def test_fetch_stored_values_all(self):
        c = self.c
        c.one.fetch = mock.MagicMock()
        c.two.fetch = mock.MagicMock()
        c.fetch_stored_values()
        assert c.one.fetch.called
        assert c.two.fetch.called

    def test_fetch_stored_values_all_twinned_bad(self):
        c = self.c
        c.one._twin = 1.
        c.one.fetch = mock.MagicMock()
        c.two.fetch = mock.MagicMock()
        c.fetch_stored_values()
        assert c.one.fetch.called
        assert c.two.fetch.called

    def test_fetch_stored_values_all_twinned(self):
        c = self.c
        c.one.twin = c.two
        c.one.fetch = mock.MagicMock()
        c.two.fetch = mock.MagicMock()
        c.fetch_stored_values()
        assert not c.one.fetch.called
        assert c.two.fetch.called

    def test_set_parameters_free_all(self):
        self.c.set_parameters_free()
        assert self.c.one.free
        assert self.c.two.free

    def test_set_parameters_free_name(self):
        self.c.set_parameters_free(['one'])
        assert self.c.one.free
        assert self.c.two.free

    def test_set_parameters_not_free_all(self):
        self.c.set_parameters_not_free()
        assert not self.c.one.free
        assert not self.c.two.free

    def test_set_parameters_not_free_name(self):
        self.c.one.free = True
        self.c.set_parameters_not_free(['two'])
        assert self.c.one.free
        assert not self.c.two.free


class TestCallMethods:

    def setup_method(self, method):
        self.c = Component(["one", "two"])
        c = self.c
        c.model = mock.MagicMock()
        c.model.channel_switches = np.array([True, False, True])
        c.model.axis.axis = np.array([0.1, 0.2, 0.3])
        c.function = mock.MagicMock()
        c.function.return_value = np.array([1.3, ])
        c.model.signal.axes_manager.signal_axes = [mock.MagicMock(), ]
        c.model.signal.axes_manager.signal_axes[0].scale = 2.

    def test_call(self):
        c = self.c
        assert 1.3 == c()
        np.testing.assert_array_equal(c.function.call_args[0][0],
                                      np.array([0.1, 0.3]))

    def test_plotting_not_active_component(self):
        c = self.c
        c.active = False
        c.model.signal.metadata.Signal.binned = False
        res = c._component2plot(c.model.axes_manager, out_of_range2nans=False)
        assert np.isnan(res).all()

    def test_plotting_active_component_notbinned(self):
        c = self.c
        c.active = True
        c.model.signal.metadata.Signal.binned = False
        res = c._component2plot(c.model.axes_manager, out_of_range2nans=False)
        np.testing.assert_array_equal(res, np.array([1.3, ]))

    def test_plotting_active_component_binned(self):
        c = self.c
        c.active = True
        c.model.signal.metadata.Signal.binned = True
        res = c._component2plot(c.model.axes_manager, out_of_range2nans=False)
        np.testing.assert_array_equal(res, 2. * np.array([1.3, ]))

    def test_plotting_active_component_out_of_range(self):
        c = self.c
        c.active = True
        c.model.signal.metadata.Signal.binned = False
        c.function.return_value = np.array([1.1, 1.3])
        res = c._component2plot(c.model.axes_manager, out_of_range2nans=True)
        np.testing.assert_array_equal(res, np.array([1.1, np.nan, 1.3]))
