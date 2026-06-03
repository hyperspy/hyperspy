Improve keyboard shortcut support for macOS:

- Add ``super`` (Command/Windows key) to the modifier key enum.
- On macOS, change the default modifier for navigation dimensions 4–5
  from ``alt`` to ``ctrl+alt`` (``Cmd+Option``) to avoid conflicts with
  the Option key producing special characters.
- All previously hardcoded drawing shortcuts (``e``, ``h``, ``l``,
  ``+``, ``-``, ``x``, ``c``, ``y``, ``u``, ``PageUp``, ``PageDown``)
  are now configurable via plot preferences.
- Log-scale toggle (``l`` key) is now configurable independently of the
  matplotlib default (which HyperSpy already suppresses); users can remap
  it without affecting matplotlib behaviour.
- Add documentation for macOS shortcut defaults and how to configure
  shortcuts when connecting to a remote HyperSpy instance from macOS.
- Add keyboard shortcuts for 1D model plots: ``a`` toggles adjust
  position, ``w`` toggles component visibility, ``t`` toggles the
  residual line. All three are configurable via preferences.
