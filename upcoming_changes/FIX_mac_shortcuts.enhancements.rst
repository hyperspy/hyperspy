Add ``platform`` keyword argument to :func:`~.utils.show_keybindings`.
When set, overrides the platform detection used for rendering shortcut
symbols. The ``None`` default now respects the ``platform_shortcuts``
preference, fixing display for remote clients (e.g. macOS client
connected to a Linux server).

Fix ``config2template`` config-file loading order so that
``platform_shortcuts`` is applied before individual ``modifier_dims_*``
keys. This allows a preset such as ``"macos"`` to serve as a baseline
that the user can selectively override in ``hyperspyrc``; previously
alphabetical iteration would apply the preset last and silently
overwrite any individual modifier settings.
