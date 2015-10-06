__author__ = 'to266'


from hyperspy._samfire_utils.strategy import segmenter_strategy
from hyperspy._samfire_utils._segmenters.histogram import Histogram_segmenter


class histogram_strategy(segmenter_strategy):

    def __init__(self, bins='freedman'):
        segmenter_strategy.__init__(self, 'Histogram segmenter strategy')
        self.segmenter = Histogram_segmenter(bins)
