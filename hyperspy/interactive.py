class Interactive:
    def __init__(self, f, event,
                 recompute_out_event=None,
                 *args, **kwargs):
        self.f = f
        self.args = args
        self.kwargs = kwargs
        if 'out' in kwargs:
            self.f(*args, **kwargs)
            self.out = kwargs.pop('out')
        else:
            self.out = self.f(*args, **kwargs)
            if recompute_out_event:
                recompute_out_event.connect(self._recompute_out)
        if event:
            event.connect(self.update, 0)

    def _recompute_out(self):
        out = self.f(*self.args, **self.kwargs)
        self.out.data = out.data
        changes = self.out.axes_manager.update_from(out.axes_manager)
        if changes:
            self.out.events.axes_changed.trigger(self.out)

    def update(self):
        self.f(out=self.out, *self.args, **self.kwargs)


def interactive(f, event, recompute_out_event=None, *args, **kwargs):
    """Update operation result when a given event is triggered.

    Parameters
    ----------
    f: function or method
        A function that returns an object and that optionally can place the
        result in an object given through the `out` keyword.
    event: {Event | None}
        Update the result of the operation when the event is triggered.
        Optional.
    recompute_out_event: {Event | None}
        Optional argument. If supplied, this event causes a full recomputation
        of a new object. Both the data and axes of the new object are then
        copied over to the existing `out` object. Only useful for `Signal` or
        other objects that have an attribute `axes_manager`.

    *args, **kwargs
        Arguments and keyword arguments to be passed to `f`.


    """

    cls = Interactive(f, event, recompute_out_event, *args, **kwargs)
    return cls.out
