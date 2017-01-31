

from hyperspy.misc.utils import slugify


def test_slugify():
    assert slugify('ab !@#_ ja &(]') == 'ab___ja'
