try:
    from prettytable import TableStyle

    MARKDOWN = TableStyle.MARKDOWN
except ImportError:
    # Deprecated in prettytable 3.12.0
    from prettytable import MARKDOWN

from hyperspy.utils import print_known_signal_types


def test_text_output(capsys):
    print_known_signal_types()
    captured = capsys.readouterr()
    assert "signal_type" in captured.out
    # the output will be str, not html
    assert "<p>" not in captured.out


def test_style(capsys):
    print_known_signal_types(style=MARKDOWN)
    captured = capsys.readouterr()

    assert "signal_type" in captured.out
    # the output will be markdown, not ascii
    assert ":--" in captured.out  # markdown
    assert "<p>" not in captured.out  # not html
    assert "+--" not in captured.out  # not ascii
