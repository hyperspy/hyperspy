import nose.tools


@nose.tools.nottest
def test_travis():
    nose.tools.Ski
    nose.tools.assert_true(False)
