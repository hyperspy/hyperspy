import nose.tools as nt
import numpy as np

from hyperspy.misc.utils import DictionaryTreeBrowser
from hyperspy.signal import BaseSignal


class TestDictionaryBrowser:

    def setUp(self):
        tree = DictionaryTreeBrowser(
            {
                "Node1": {"leaf11": 11,
                          "Node11": {"leaf111": 111},
                          },
                "Node2": {"leaf21": 21,
                          "Node21": {"leaf211": 211},
                          },
            })
        self.tree = tree

    def test_add_dictionary(self):
        self.tree.add_dictionary({
            "Node1": {"leaf12": 12,
                      "Node11": {"leaf111": 222,
                                 "Node111": {"leaf1111": 1111}, },
                      },
            "Node3": {
                "leaf31": 31},
        })
        nt.assert_equal(
            {"Node1": {"leaf11": 11,
                       "leaf12": 12,
                       "Node11": {"leaf111": 222,
                                  "Node111": {
                                      "leaf1111": 1111},
                                  },
                       },
             "Node2": {"leaf21": 21,
                       "Node21": {"leaf211": 211},
                       },
             "Node3": {"leaf31": 31},
             }, self.tree.as_dictionary())

    def test_add_signal_in_dictionary(self):
        tree = self.tree
        s = BaseSignal([1., 2, 3])
        s.axes_manager[0].name = 'x'
        s.axes_manager[0].units = 'ly'
        tree.add_dictionary({"_sig_signal name": s._to_dictionary()})
        nt.assert_is_instance(tree.signal_name, BaseSignal)
        np.testing.assert_array_equal(tree.signal_name.data, s.data)
        nt.assert_dict_equal(tree.signal_name.metadata.as_dictionary(),
                             s.metadata.as_dictionary())
        nt.assert_equal(tree.signal_name.axes_manager._get_axes_dicts(),
                        s.axes_manager._get_axes_dicts())

    def test_signal_to_dictionary(self):
        tree = self.tree
        s = BaseSignal([1., 2, 3])
        s.axes_manager[0].name = 'x'
        s.axes_manager[0].units = 'ly'
        tree.set_item('Some name', s)
        d = tree.as_dictionary()
        np.testing.assert_array_equal(d['_sig_Some name']['data'], s.data)
        d['_sig_Some name']['data'] = 0
        nt.assert_dict_equal(
            {
                "Node1": {
                    "leaf11": 11,
                    "Node11": {
                        "leaf111": 111},
                },
                "Node2": {
                    "leaf21": 21,
                    "Node21": {
                        "leaf211": 211},
                },
                "_sig_Some name": {
                    'axes': [
                        {
                            'name': 'x',
                            'navigate': False,
                                    'offset': 0.0,
                                    'scale': 1.0,
                                    'size': 3,
                                    'units': 'ly'}],
                    'data': 0,
                    'learning_results': {},
                    'metadata': {
                        'General': {
                            'title': ''},
                        'Signal': {
                            'binned': False,
                            'record_by': '',
                            'signal_origin': '',
                            'signal_type': ''},
                        '_HyperSpy': {
                            'Folding': {
                                'original_axes_manager': None,
                                'original_shape': None,
                                'unfolded': False,
                                'signal_unfolded': False}}},
                    'original_metadata': {},
                    'tmp_parameters': {}}},
            d)
