class Interactive:
    def __init__(self, obj, f, event, 
                 recompute_out_event=None,
                 *args, **kwargs):
        self.f = f
        self.source = obj
        self.args = args
        self.kwargs = kwargs
        if kwargs.has_key('out'):
            self.f(*args, **kwargs)
            self.out = kwargs.pop('out')
        else:
            self.out = self.f(*args, **kwargs)
            if recompute_out_event:
                recompute_out_event.connect(self._recompute_out)
        event.connect(self.update)
        self.out.events.data_changed.connect(lambda x=None: self.out.update_plot())
        self.out.events.axes_changed.connect(lambda x=None: self.out._replot())

    def _recompute_out(self):
        out = self.f(*self.args, **self.kwargs)
        self.out.data = out.data
        changes = self.out._update_calibration_from(self.source.axes_manager)
        if changes:
            self.out.events.axes_changed.trigger(self.out)

    def update(self, *args, **kwargs):
        self.f(out=self.out, *self.args, **self.kwargs)


def interactive(obj, f, event, *args, **kwargs):
    """Update operation result when a given event is triggered.

    Parameters
    ----------
    obj: anything
        The target of the operation.
    f: function or method
        A function that operates on `obj` and that can place the result in an
        object given through the `out` keyword.
    event: Event.
        Update the result of the operation when the event is triggered.

    *args, **kwargs
        Arguments and keyword arguments to be passed to `f`.


    """

    cls = Interactive(obj, f, event, *args, **kwargs)
    return cls.out
