__author__ = 'to266'


from hyperspy._samfire_utils.strategy import SegmenterStrategy
from hyperspy._samfire_utils._segmenters.histogram import HistogramSegmenter


class HistogramStrategy(SegmenterStrategy):

    def __init__(self, bins='freedman'):
        SegmenterStrategy.__init__(self, 'Histogram segmenter strategy')
        self.segmenter = HistogramSegmenter(bins)
