import sys

class EventSuppressionContext(object):
    """
    Context manager for event suppression. When passed an Events class,
    it will suppress all the events in that container when activated by
    using it in a 'with' statement. The previous suppression state will be 
    restored when the 'with' block completes.
    """
    def __init__(self, events):
        self.events = events
        self.old = {}
        
    def __enter__(self):
        self.old = {}
        try:
            for e in self.events.__dict__.itervalues():
                self.old[e] = e.suppress
                e.suppress = True
        except e:
            self.__exit__(*sys.exc_info())
            raise
        return self
        
    def __exit__(self, type, value, tb):
        for e, oldval in self.old.iteritems():
            e.suppress = oldval
        # Never suppress events

class Events(object):
    """
    Events container.

    All available events are attributes of this class.

    """

    @property
    def suppress(self):
        """
        Use this property with a 'with' statement to temporarily suppress all
        events in the container. When the 'with' vlock completes, the old 
        suppression values will be restored.
        
        Example usage pattern:
        with obj.events.suppress:
            obj.val_a = a
            obj.val_b = b
        obj.events.values_changed.trigger()
        """
        return EventSuppressionContext(self)


class Event(object):

    def __init__(self):
        self.connected = set()
        self.suppress = False

    def connect(self, function):
        if not callable(function):
            raise TypeError("Only callables can be registered")
        self.connected.add(function)

    def disconnect(self, function):
        self.connected.remove(function)

    def trigger(self, *args, **kwargs):
        if not self.suppress:
            for f in self.connected:
                f(*args, **kwargs)

    def __deepcopy__(self, memo):
        dc = type(self)()
        memo[id(self)] = dc
        return dc
