import nose.tools

from hyperspy.misc.utils import DictionaryTreeBrowser


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
        nose.tools.assert_equal(
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
