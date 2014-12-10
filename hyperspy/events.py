class Events(object):
    """
    Events container.

    All available events are attributes of this class.

    """

    pass


class Event(object):
    connected = set()

    def connect(self, function):
        if not callable(function):
            raise TypeError("Only callables can be registered")
        self.connected.add(function)

    def disconnect(self, function):
        self.connected.remove(function)

    def trigger(self, *args, **kwargs):
        for f in self.connected:
            f(*args, **kwargs)
