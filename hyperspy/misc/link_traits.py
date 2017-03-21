import traitlets
import traits.api as t


class link_traits(traitlets.link):
    """Link traits from different objects together so they remain in sync.

    Parameters
    ----------
    source : (object / attribute name) pair
    target : (object / attribute name) pair

    Examples
    --------

    >>> c = link((src, 'value'), (tgt, 'value'))
    >>> src.value = 5  # updates other objects as well
    """

    def __init__(self, source, target):
        # _validate_link(source, target)
        self.source, self.target = source, target
        try:
            setattr(target[0], target[1], getattr(source[0], source[1]))
        finally:
            if isinstance(source[0], t.HasTraits):
                source[0].on_trait_change(
                    self._update_target_traits, name=source[1])
            elif isinstance(source[0], traitlets.HasTraits):
                source[0].observe(self._update_target, names=source[1])
            else:
                raise ValueError(
                    "source must contains either traits or traitlets.")
            if isinstance(target[0], t.HasTraits):
                target[0].on_trait_change(
                    self._update_source_traits, name=target[1])
            elif isinstance(target[0], traitlets.HasTraits):
                target[0].observe(self._update_source, names=target[1])
            else:
                raise ValueError(
                    "target must contains either traits or traitlets.")

    def _update_target_traits(self, new):
        if self.updating:
            return
        with self._busy_updating():
            setattr(self.target[0], self.target[1], new)

    def _update_source_traits(self, new):
        if self.updating:
            return
        with self._busy_updating():
            setattr(self.source[0], self.source[1], new)

    def unlink(self):
        if isinstance(self.source[0], t.HasTraits):
            self.source[0].on_trait_change(
                self._update_target_traits, name=self.source[1], remove=True)
        else:
            self.source[0].unobserve(self._update_target, names=self.source[1])
        if isinstance(self.target[0], t.HasTraits):
            self.target[0].on_trait_change(
                self._update_source_traits, name=self.target[1], remove=True)
        else:
            self.target[0].unobserve(self._update_source, names=self.target[1])
        self.source, self.target = None, None
