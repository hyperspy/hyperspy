import traitlets
import traits.api as t


def has_traits(obj):
    return isinstance(obj, t.HasTraits)


def has_traitlets(obj):
    return isinstance(obj, traitlets.HasTraits)


class link_traits(traitlets.link):
    """Link traits from different objects together so they remain in sync.

    This is a sublclass of traitlets.links that adds support for linking
    traitlets traits and enthought traits.

    Parameters
    ----------
    source : (object / attribute name) pair
    target : (object / attribute name) pair

    Examples
    --------

    >>> c = link((src, 'value'), (tgt, 'value'))
    >>> src.value = 5  # updates other objects as well

    See Also
    --------
    directional_link

    """

    def __init__(self, source, target):
        # _validate_link(source, target)
        self.source, self.target = source, target
        try:
            source_value = getattr(source[0], source[1])
            if source_value not in (t.Undefined, traitlets.Undefined, None):
                setattr(target[0], target[1], source_value)
        finally:
            if has_traits(source[0]):
                source[0].on_trait_change(
                    self._update_target_traits, name=source[1])
            elif has_traitlets(source[0]):
                source[0].observe(self._update_target, names=source[1])
            else:
                raise ValueError(
                    "source must contains either traits or traitlets.")
            if has_traits(target[0]):
                target[0].on_trait_change(
                    self._update_source_traits, name=target[1])
            elif has_traitlets(target[0]):
                target[0].observe(self._update_source, names=target[1])
            else:
                raise ValueError(
                    "target must contains either traits or traitlets.")

    def _update_target_traits(self, new):
        if self.updating:
            return
        with self._busy_updating():
            if new is t.Undefined and has_traitlets(self.target[0]):
                new = self.target[0].traits()[self.target[1]].default_value
            setattr(self.target[0], self.target[1], new)

    def _update_source_traits(self, new):
        if self.updating:
            return
        with self._busy_updating():
            if new is t.Undefined and has_traitlets(self.source[0]):
                new = self.source[0].traits()[self.source[1]].default_value
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


class directional_link(traitlets.dlink):
    """Link the trait of a source object with traits of target objects.

    This is a sublclass of traitlets.directional_link that adds support for
    linking traitlets traits and enthought traits.

    Parameters
    ----------
    source : (object, attribute name) pair
    target : (object, attribute name) pair
    transform: callable (optional)
        Data transformation between source and target.
    Examples
    --------
    >>> c = directional_link((src, 'value'), (tgt, 'value'))
    >>> src.value = 5  # updates target objects
    >>> tgt.value = 6  # does not update source object

    See Also
    --------
    link

    """

    def __init__(self, source, target, transform=None):
        self._transform = transform if transform else lambda x: x
        self.source, self.target = source, target
        try:
            source_value = getattr(source[0], source[1])
            if source_value not in (t.Undefined, traitlets.Undefined, None):
                setattr(target[0], target[1], self._transform(source_value))
        finally:
            if has_traits(source[0]):
                source[0].on_trait_change(
                    self._update_traits, name=source[1])
            elif has_traitlets(source[0]):
                source[0].observe(self._update, names=source[1])
            else:
                raise ValueError(
                    "source must contains either enthought traits or "
                    "traitlets.")

    def _update_traits(self, name, new):
        if self.updating:
            return
        with self._busy_updating():
            if new is t.Undefined and has_traitlets(self.target[0]):
                new = self.target[0].traits()[self.target[1]].default_value
            setattr(self.target[0], self.target[1], self._transform(new))

    def unlink(self):
        if isinstance(self.source[0], t.HasTraits):
            self.source[0].on_trait_change(
                self._update_target_traits, name=self.source[1], remove=True)
        else:
            self.source[0].unobserve(self._update_target, names=self.source[1])
        self.source, self.target = None, None
