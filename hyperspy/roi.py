import traits.api as t
from hyperspy.events import Events, Event


class RectangularROI(t.HasTraits):
    top, bottom, left, right = (t.CInt(t.Undefined),) * 4

    def __init__(self, top, bottom, left, right):
        super(RectangularROI, self).__init__()
        self._signals = []
        self.events = Events()
        self.events.roi_changed = Event()
        self.top, self.bottom, self.left, self.right = top, bottom, left, right

    def register_signals(self, signals):
        for signal in signals:
            self._signals.append((signal,
                                 signal._deepcopy_with_new_data(signal.data)))

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
        if (not self._signals or
                t.Undefined in (self.top, self.bottom, self.left, self.right)):
            return
        for signal, out in self._signals:
            self(signal, out=out)
            out.events.data_changed.trigger()
        self.events.roi_changed.trigger()

    def __call__(self, sig, register=False, out=None):
        if out is None:
            roi = sig[self.left:self.right, self.top:self.bottom]
            if register:
                self._signals.append((sig, roi))
            return roi
        else:
            sig.__getitem__((slice(self.left, self.right),
                             slice(self.top, self.bottom)),
                            out=out)

    def __repr__(self):
        return "%s(top=%f, bottom=%f, left=%f, right=%f)" % (
            self.__class__.__name__,
            self.top,
            self.bottom,
            self.left,
            self.right)
