from hyperspy.utils import print_known_signal_types


class TestPrintKnownSignalTypes:
    def test_text_output(self):
        obj = print_known_signal_types()
        assert obj.__repr__()

    def test_html_output(self):
        obj = print_known_signal_types()
        assert obj._repr_html_()
