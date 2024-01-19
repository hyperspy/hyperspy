from hyperspy.utils import print_known_signal_types


def test_text_output(capsys):
    print_known_signal_types()
    captured = capsys.readouterr()
    assert "signal_type" in captured.out
    # the output will be str, not html
    assert "<p>" not in captured.out
