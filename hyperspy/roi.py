import traits.api as t
from hyperspy.events import Events, Event


class BaseROI(t.HasTraits):
    def __init__(self):
        super(BaseROI, self).__init__()
        self.events = Events()
        self.events.roi_changed = Event()


class RectangularROI(BaseROI):
    top, bottom, left, right = (t.CFloat(t.Undefined),) * 4

    def __init__(self, top, bottom, left, right):
        super(RectangularROI, self).__init__()
        self.top, self.bottom, self.left, self.right = top, bottom, left, right

    def _top_changed(self, old, new):
        if self.bottom is not t.Undefined and new >= self.bottom:
            self.top = old
        else:
            self.update()

    def _bottom_changed(self, old, new):
        if self.top is not t.Undefined and new <= self.top:
            self.bottom = old
        else:
            self.update()

    def _right_changed(self, old, new):
        if self.left is not t.Undefined and new <= self.left:
            self.right = old
        else:
            self.update()

    def _left_changed(self, old, new):
        if self.right is not t.Undefined and new >= self.right:
            self.left = old
        else:
            self.update()

    def update(self):
        if t.Undefined not in (self.top, self.bottom, self.left, self.right):
            self.events.roi_changed.trigger()

    def __call__(self, signal, out=None):
        if out is None:
            roi = signal[self.left:self.right, self.top:self.bottom]
            return roi
        else:
            signal.__getitem__((slice(self.left, self.right),
                                slice(self.top, self.bottom)),
                               out=out)

    def __repr__(self):
        return "%s(top=%f, bottom=%f, left=%f, right=%f)" % (
            self.__class__.__name__,
            self.top,
            self.bottom,
            self.left,
            self.right)
